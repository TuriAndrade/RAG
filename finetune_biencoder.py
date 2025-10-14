# finetune_biencoder.py
# Dual-encoder fine-tuning on SQuAD with InfoNCE (in-batch negatives).

import argparse, os
import random
import csv

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def mean_pool(last_hidden, attn_mask):
    # last_hidden: (B, L, H), attn_mask: (B, L)
    attn = attn_mask.unsqueeze(-1).type_as(last_hidden)  # (B, L, 1)
    summed = (last_hidden * attn).sum(dim=1)  # (B, H)
    counts = attn.sum(dim=1).clamp(min=1)  # (B, 1)
    emb = summed / counts  # (B, H)
    return torch.nn.functional.normalize(emb, p=2, dim=1)  # (B, H)


class SquadQP(Dataset):
    """
    Produces (question, positive_passage) pairs:
      - passage = window around the answer span (crude char-based heuristic).
    """

    def __init__(self, split="train", max_examples=2000, seed=0):
        self.ds = load_dataset("squad")[split]
        self.max_examples = min(max_examples, len(self.ds))
        random.seed(seed)
        self.indices = list(range(self.max_examples))
        random.shuffle(self.indices)

    def __len__(self):
        return self.max_examples

    def __getitem__(self, idx):
        ex = self.ds[self.indices[idx]]
        q = ex["question"]
        ctx = ex["context"]
        ans = ex["answers"]["text"][0]
        # Window around first occurrence of the answer
        pos_start = ctx.lower().find(ans.lower())
        if pos_start == -1:
            p = ctx
        else:
            L = 300
            p = ctx[max(pos_start - L, 0) : pos_start + L]
        return q, p


def collate(batch, tokenizer, max_len_q=64, max_len_p=256, device="cpu"):
    qs, ps = zip(*batch)
    q_tok = tokenizer(
        list(qs),
        padding=True,
        truncation=True,
        max_length=max_len_q,
        return_tensors="pt",
    )
    p_tok = tokenizer(
        list(ps),
        padding=True,
        truncation=True,
        max_length=max_len_p,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in q_tok.items()}, {
        k: v.to(device) for k, v in p_tok.items()
    }


@torch.no_grad()
def evaluate_epoch(enc_q, enc_p, loader, ce_loss, temperature: float) -> float:
    """Return average InfoNCE loss over the validation loader."""
    enc_q.eval()
    enc_p.eval()
    total, count = 0.0, 0
    for q_tok, p_tok in loader:
        q_out = enc_q(**q_tok)
        p_out = enc_p(**p_tok)
        q_emb = mean_pool(q_out.last_hidden_state, q_tok["attention_mask"])  # (B, H)
        p_emb = mean_pool(p_out.last_hidden_state, p_tok["attention_mask"])  # (B, H)
        logits = (q_emb @ p_emb.T) / temperature  # (B, B)
        targets = torch.arange(logits.size(0), device=logits.device)
        loss = ce_loss(logits, targets)
        bsz = logits.size(0)
        total += loss.item() * bsz
        count += bsz
    enc_q.train()
    enc_p.train()
    return total / max(1, count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    ap.add_argument("--out-dir", type=str, default="./pretrained_encoders")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max-examples", type=int, default=10000)
    ap.add_argument("--val-max-examples", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-fp16", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    enc_q = AutoModel.from_pretrained(args.model).to(device)
    enc_p = AutoModel.from_pretrained(args.model).to(device)
    if args.use_fp16 and device.type == "cuda":
        enc_q.half()
        enc_p.half()

    # Datasets / Loaders
    train_set = SquadQP(
        split=args.split, max_examples=args.max_examples, seed=args.seed
    )
    val_set = SquadQP(
        split="validation", max_examples=args.val_max_examples, seed=args.seed + 1
    )

    def _collate(b):
        return collate(b, tokenizer, device=device)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
        drop_last=True,
    )

    params = list(enc_q.parameters()) + list(enc_p.parameters())
    opt = AdamW(params, lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    ce = nn.CrossEntropyLoss()
    enc_q.train()
    enc_p.train()

    # Metrics storage
    train_losses, val_losses = [], []

    # Progress bar over total train steps
    global_pbar = tqdm(
        total=total_steps, desc="Training", leave=False, dynamic_ncols=True
    )
    step = 0

    for ep in range(1, args.epochs + 1):
        # -------- Train epoch --------
        running_loss, running_count = 0.0, 0

        for batch in train_loader:
            q_tok, p_tok = batch

            # Forward
            q_out = enc_q(**q_tok)
            p_out = enc_p(**p_tok)
            q_emb = mean_pool(
                q_out.last_hidden_state, q_tok["attention_mask"]
            )  # (B, H)
            p_emb = mean_pool(
                p_out.last_hidden_state, p_tok["attention_mask"]
            )  # (B, H)

            # InfoNCE with in-batch negatives
            logits = (q_emb @ p_emb.T) / args.temperature  # (B, B)
            targets = torch.arange(logits.size(0), device=logits.device)
            loss = ce(logits, targets)

            # Optim
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            # Accumulate train loss
            bsz = logits.size(0)
            running_loss += loss.item() * bsz
            running_count += bsz

            step += 1
            if step % 50 == 0:
                with torch.no_grad():
                    acc = (logits.argmax(dim=1) == targets).float().mean().item()
                    lr = (
                        sched.get_last_lr()[0]
                        if hasattr(sched, "get_last_lr")
                        else opt.param_groups[0]["lr"]
                    )
                global_pbar.set_postfix(
                    epoch=ep,
                    loss=f"{loss.item():.4f}",
                    acc=f"{acc:.3f}",
                    lr=f"{lr:.2e}",
                )
            global_pbar.update(1)

        avg_train = running_loss / max(1, running_count)
        train_losses.append(avg_train)

        # -------- Validation epoch --------
        avg_val = evaluate_epoch(
            enc_q, enc_p, val_loader, ce_loss=ce, temperature=args.temperature
        )
        val_losses.append(avg_val)

    global_pbar.close()

    # -------- Save checkpoints --------
    enc_q_dir = os.path.join(args.out_dir, "encoder_q")
    enc_p_dir = os.path.join(args.out_dir, "encoder_p")
    os.makedirs(enc_q_dir, exist_ok=True)
    os.makedirs(enc_p_dir, exist_ok=True)

    enc_q.save_pretrained(enc_q_dir)
    enc_p.save_pretrained(enc_p_dir)

    # Save tokenizer both at root (for convenience) and inside encoders (so build_index can find it)
    tokenizer.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(enc_p_dir)
    tokenizer.save_pretrained(enc_q_dir)

    # -------- Save metrics --------
    metrics_csv = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
            w.writerow([i, tr, va])

    # -------- Plot and save loss curve --------
    png_path = os.path.join(args.out_dir, "loss_curve.png")
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print("Saved fine-tuned encoders to:", args.out_dir)
    print("Saved metrics to:", metrics_csv)
    print("Saved loss curve to:", png_path)


if __name__ == "__main__":
    main()
