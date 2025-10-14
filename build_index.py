# build_index.py
# Create a FAISS index of MiniLM embeddings over SQuAD v1.1 passages (low-level).
# Usage examples:
#   # train-only corpus
#   python build_index.py --splits train --max-examples 500 --chunk-tokens 180 --chunk-stride 60 \
#       --out-dir ./rag_store_train
#
#   # train + validation corpus (valid for evaluating on held-out validation questions)
#   python build_index.py --splits train,validation --max-examples 10000 --chunk-tokens 180 --chunk-stride 60 \
#       --out-dir ./rag_store_all

import argparse, json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import faiss
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


# ------------------------- Chunking by tokens (no extra deps) ------------------------- #
def chunk_by_tokens(text: str, tokenizer, chunk_tokens: int, stride: int) -> List[str]:
    """Split a long string into overlapping token windows using the encoder tokenizer."""
    # Use tokenizer call API; disable truncation here (we window manually after)
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + chunk_tokens, len(ids))
        sub_ids = ids[start:end]
        if len(sub_ids) == 0:
            break
        chunk = tokenizer.decode(sub_ids, skip_special_tokens=True)
        chunks.append(chunk)
        if end == len(ids):
            break
        start += max(1, chunk_tokens - stride)  # overlap = stride
    return chunks


# ------------------------- Mean pooling + L2 normalization ------------------------- #
@torch.no_grad()
def encode_text_batch(
    texts: List[str], tokenizer, model, device, max_len: int = 256
) -> np.ndarray:
    """Low-level encoding: tokenize, forward, mean-pool with attention mask, L2-normalize."""
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)

    outputs = model(**batch)  # last_hidden_state (B, L, H)
    last_hidden = outputs.last_hidden_state
    attn = batch["attention_mask"].unsqueeze(-1)  # (B, L, 1)
    summed = (last_hidden * attn).sum(dim=1)  # (B, H)
    counts = attn.sum(dim=1).clamp(min=1)  # (B, 1)
    emb = summed / counts  # mean-pool

    # L2-normalize so dot product == cosine similarity
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype("float32")


def iter_batches(items: List[Any], bs: int):
    for i in range(0, len(items), bs):
        yield items[i : i + bs]


def collect_passages_from_split(
    ds_split, tokenizer, chunk_tokens: int, chunk_stride: int, max_examples: int | None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Build passages list and metadata from a single split:
      passages[i] -> text
      meta[i] -> {article_id, squad_idx, offset_chunk, original_context[:220], title?, ...}
    """
    passages, metas = [], []
    n = len(ds_split) if max_examples is None else min(max_examples, len(ds_split))
    for i in range(n):
        ex = ds_split[i]
        context = ex["context"]
        title = ex.get("title", None)
        chunks = chunk_by_tokens(
            context, tokenizer, chunk_tokens=chunk_tokens, stride=chunk_stride
        )
        for j, ch in enumerate(chunks):
            passages.append(ch)
            metas.append(
                {
                    "squad_idx": i,
                    "chunk_idx": j,
                    "title": title,
                    "context_preview": context[:220].replace("\n", " "),
                }
            )
    return passages, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    # New: multiple splits (comma-separated). Example: "train,validation"
    ap.add_argument(
        "--splits",
        type=str,
        default="train,validation",
        help="Comma-separated SQuAD splits to index (e.g., 'train' or 'train,validation').",
    )
    # Interpreted as a PER-SPLIT cap for simplicity
    ap.add_argument(
        "--max-examples", type=int, default=10000, help="Per-split cap on SQuAD rows."
    )
    ap.add_argument("--chunk-tokens", type=int, default=180)
    ap.add_argument("--chunk-stride", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer max length at encode time.",
    )
    ap.add_argument("--out-dir", type=str, default="./rag_store/base")
    ap.add_argument("--use-fp16", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SQuAD v1.1...")
    squad = load_dataset("squad")
    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
    print("Will index splits:", split_list)

    print("Loading tokenizer/model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Silence spurious long-sequence warnings during chunking:
    tokenizer.model_max_length = 10**9

    model = AutoModel.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if args.use_fp16 and device.type == "cuda":
        model.half()

    # Collect passages from all requested splits
    print("Chunking contexts into passages...")
    all_passages: List[str] = []
    all_metas: List[Dict[str, Any]] = []
    for split_name in split_list:
        ds = squad[split_name]
        passages, metas = collect_passages_from_split(
            ds_split=ds,
            tokenizer=tokenizer,
            chunk_tokens=args.chunk_tokens,
            chunk_stride=args.chunk_stride,
            max_examples=args.max_examples,
        )
        print(f"  Split '{split_name}': {len(passages)} passages before dedup")
        all_passages.extend(passages)
        # Tag the split name on metadata
        for m in metas:
            m["split"] = split_name
        all_metas.extend(metas)

    print(f"Total passages (all splits, pre-dedup): {len(all_passages)}")

    # Deduplicate identical passages (optional but nice to shrink index)
    print("Deduplicating passages...")
    uniq_map: Dict[str, int] = {}
    passages: List[str] = []
    metas: List[Dict[str, Any]] = []
    for i, p in enumerate(all_passages):
        if p not in uniq_map:
            uniq_map[p] = len(passages)
            passages.append(p)
            metas.append(all_metas[i])
    print(f"After dedup: {len(passages)} passages")

    # Encode in batches
    print("Encoding passages -> MiniLM embeddings...")
    vecs = []
    for batch_texts in iter_batches(passages, args.batch_size):
        emb = encode_text_batch(
            batch_texts, tokenizer, model, device, max_len=args.max_length
        )
        vecs.append(emb)
    X = np.vstack(vecs) if vecs else np.zeros((0, 384), dtype="float32")  # (N, d)
    if X.size == 0:
        raise RuntimeError("No passages to index. Check your splits/limits.")
    d = X.shape[1]
    print("Embeddings shape:", X.shape)

    # Build FAISS index (cosine via normalized vectors -> inner product)
    print("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(d)
    index.add(X)
    print("FAISS ntotal:", index.ntotal)

    # Persist everything
    faiss.write_index(index, str(out_dir / "faiss_ip.index"))
    with open(out_dir / "passages.jsonl", "w", encoding="utf-8") as f:
        for i, text in enumerate(passages):
            rec = {"id": i, "text": text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for i, m in enumerate(metas):
            m2 = {"id": i, **m}
            f.write(json.dumps(m2, ensure_ascii=False) + "\n")

    # Quick self-check: query a few things
    print("Self-check: querying a couple of terms...")
    demo_queries = [
        "Who is the subject of the paragraph?",
        "When did the event happen?",
    ]
    # Encode queries with the same pipeline
    q_vecs = encode_text_batch(
        demo_queries, tokenizer, model, device, max_len=args.max_length
    )
    sims, idxs = index.search(q_vecs, k=3)  # top-3 for each query
    for qi, q in enumerate(demo_queries):
        print(f"\nQ: {q}")
        for rank, (score, pid) in enumerate(zip(sims[qi], idxs[qi])):
            prev = passages[pid][:100].replace("\n", " ")
            print(f"  {rank+1}) score={score:.3f}  text[:100]={prev!r}")

    print(f"\nDone. Wrote index + store to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
