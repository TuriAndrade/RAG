import argparse, os, json, faiss, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer as QATok, AutoModelForQuestionAnswering
from tqdm.auto import tqdm


def load_store(store_dir):
    index = faiss.read_index(f"{store_dir}/faiss_ip.index")
    passages = []
    with open(f"{store_dir}/passages.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line)["text"])
    return index, passages


@torch.no_grad()
def encode(texts, tok, enc, device, max_len=256):
    batch = tok(
        texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    ).to(device)
    out = enc(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1)
    emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype("float32")


@torch.no_grad()
def answer_with_reader(question, contexts, qa_tok, qa, device, max_length=384):
    best_span, best_score = "", -1e9
    for c in contexts:
        inpt = qa_tok(
            question,
            c,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(device)
        out = qa(**inpt)
        start = out.start_logits.argmax(-1).item()
        end = out.end_logits.argmax(-1).item()
        if end < start:
            continue
        score = out.start_logits[0, start].item() + out.end_logits[0, end].item()
        span = qa_tok.decode(
            inpt["input_ids"][0][start : end + 1], skip_special_tokens=True
        ).strip()
        if score > best_score:
            best_score, best_span = score, span
    return best_span if best_span else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--encoder_q", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    ap.add_argument(
        "--tokenizer_q", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    ap.add_argument("--store", type=str, default="./rag_store/base")
    ap.add_argument("--qa_model", type=str, default="deepset/roberta-base-squad2")
    ap.add_argument("--qa_max_length", type=int, default=384)
    ap.add_argument("--top_m", type=int, default=5)
    ap.add_argument("--out", type=str, default="./runs/preds_base.jsonl")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--max_eval", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Retrieval encoder
    qtok = AutoTokenizer.from_pretrained(args.tokenizer_q, use_fast=True)
    qenc = AutoModel.from_pretrained(args.encoder_q).to(device).eval()
    index, passages = load_store(args.store)

    # QA reader
    qa_tok = QATok.from_pretrained(args.qa_model, use_fast=True)
    qa = AutoModelForQuestionAnswering.from_pretrained(args.qa_model).to(device).eval()

    ds = load_dataset("squad")[args.split]
    n = min(args.max_eval, len(ds))

    with open(args.out, "w", encoding="utf-8") as f, tqdm(
        total=n, desc="RAG QA", dynamic_ncols=True, leave=False
    ) as pbar:
        for i in range(n):
            ex = ds[i]
            q = ex["question"]
            gold_texts = ex["answers"]["text"]

            # retrieve
            qv = encode([q], qtok, qenc, device)
            sims, ids = index.search(qv, k=args.k)
            ctxs = [passages[int(pid)] for pid in ids[0]]

            # extractive answer (score top_m contexts)
            pred = answer_with_reader(
                q, ctxs[: args.top_m], qa_tok, qa, device, max_length=args.qa_max_length
            )

            rec = {
                "id": ex["id"],
                "question": q,
                "prediction": pred,
                "gold_answers": gold_texts,
                "contexts": ctxs[: args.top_m],
                "k": args.k,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                pbar.set_postfix(k=args.k, top_m=args.top_m)
            pbar.update(1)

    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
