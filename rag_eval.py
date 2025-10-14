import argparse, json, re, os
from collections import Counter


# --------------------- Normalization & SQuAD-style metrics --------------------- #
def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, golds: list[str]) -> int:
    p = _normalize(pred)
    return max(1 if p == _normalize(g) else 0 for g in golds)


def f1_score(pred: str, golds: list[str]) -> float:
    def f1_pair(p: str, g: str) -> float:
        p_toks = _normalize(p).split()
        g_toks = _normalize(g).split()
        if len(p_toks) == 0 and len(g_toks) == 0:
            return 1.0
        common = Counter(p_toks) & Counter(g_toks)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        prec = num_same / max(1, len(p_toks))
        rec = num_same / max(1, len(g_toks))
        return 2 * prec * rec / (prec + rec + 1e-12)

    return max(f1_pair(pred, g) for g in golds)


# --------------------- Retriever metrics from contexts --------------------- #
def _contains_any_answer(text: str, golds: list[str]) -> bool:
    t = " ".join(text.lower().split())
    for g in golds:
        gnorm = " ".join(g.lower().split())
        if gnorm and gnorm in t:
            return True
    return False


def hit_at_k(contexts: list[str], golds: list[str]) -> int:
    """1 if any of the top-k contexts contains a gold answer substring, else 0."""
    return 1 if any(_contains_any_answer(c, golds) for c in contexts) else 0


def mrr_at_k(contexts: list[str], golds: list[str]) -> float:
    """
    MRR@k with binary relevance: relevance = 1 if context contains the answer.
    Reciprocal rank of the FIRST relevant hit; 0 if none.
    """
    for rank, c in enumerate(contexts, start=1):
        if _contains_any_answer(c, golds):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(contexts: list[str], golds: list[str]) -> float:
    """
    nDCG@k with binary gains (1 if context contains answer, else 0).
    DCG = sum_i (gain_i / log2(i+1)), IDCG = ideal DCG with all relevant first.
    With binary relevance and unknown total relevant, IDCG is 1.0 (best-case single hit)
    if at least one relevant exists in the top-k; else both DCG and IDCG = 0.
    """
    gains = [1 if _contains_any_answer(c, golds) else 0 for c in contexts]
    dcg = 0.0
    for i, g in enumerate(gains, start=1):
        if g:
            # log2(i+1) = ln(i+1)/ln(2); use 1.0 if i==1 -> denominator log2(2)=1
            denom = (
                (i + 1).bit_length() - 1 if i in (1,) else ((i + 1).bit_length() - 1)
            )  # fast approx
            # safer exact:
            import math

            denom = math.log2(i + 1)
            dcg += g / denom
    # Ideal DCG: put a single 1 at rank 1 if any relevant exists; else 0
    idcg = 1.0 if any(gains) else 0.0
    return (dcg / idcg) if idcg > 0 else 0.0


# --------------------- IO --------------------- #
def load_preds(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, default="runs/preds_base.jsonl")
    ap.add_argument("--out", type=str, default="runs/metrics_base.txt")
    ap.add_argument(
        "--compute_retrieval",
        action="store_true",
        help="also compute retrieval metrics (Hit@k, MRR@k, nDCG@k) using contexts",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = load_preds(args.preds)
    ems, f1s = [], []

    # Optional retrieval stats
    hits, mrrs, ndcgs = [], [], []
    k_seen = None

    for r in rows:
        pred = r["prediction"]
        golds = r["gold_answers"]
        ems.append(exact_match(pred, golds))
        f1s.append(f1_score(pred, golds))

        if args.compute_retrieval:
            ctxs = r.get("contexts", [])
            if k_seen is None:
                k_seen = len(ctxs)
            # If for some reason contexts missing, skip retrieval metrics for that row
            if ctxs:
                hits.append(hit_at_k(ctxs, golds))
                mrrs.append(mrr_at_k(ctxs, golds))
                ndcgs.append(ndcg_at_k(ctxs, golds))

    em = sum(ems) / max(1, len(ems))
    f1 = sum(f1s) / max(1, len(f1s))

    # Console
    print(f"Count={len(rows)}  EM={em:.4f}  F1={f1:.4f}")
    if args.compute_retrieval and hits:
        print(
            f"Retrieval@{k_seen}: Hit={sum(hits)/len(hits):.4f}  MRR={sum(mrrs)/len(mrrs):.4f}  nDCG={sum(ndcgs)/len(ndcgs):.4f}"
        )

    # File
    lines = [
        f"Preds file: {args.preds}",
        f"Count: {len(rows)}",
        f"EM: {em:.6f}",
        f"F1: {f1:.6f}",
    ]
    if args.compute_retrieval and hits:
        lines += [
            f"Hit@{k_seen}: {sum(hits)/len(hits):.6f}",
            f"MRR@{k_seen}: {sum(mrrs)/len(mrrs):.6f}",
            f"nDCG@{k_seen}: {sum(ndcgs)/len(ndcgs):.6f}",
        ]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved metrics to: {args.out}")


if __name__ == "__main__":
    main()
