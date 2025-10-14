# 🧠 Retrieval-Augmented Generation (RAG) on SQuAD v1.1

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline on the **SQuAD v1.1** dataset, comparing a **base retriever** (MiniLM) and a **fine-tuned retriever** trained with contrastive learning on question–context pairs.

---

## 📖 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a hybrid architecture that combines:
- A **retriever** that fetches relevant documents from a knowledge base using dense embeddings.
- A **reader/generator** that uses those retrieved documents to generate or extract answers.

This approach improves factual accuracy and reduces hallucinations by grounding the model’s answers in retrieved evidence.

**Pipeline overview:**
1. Encode all corpus passages → build FAISS index.
2. For each question:
   - Retrieve top-𝑘 relevant passages.
   - Concatenate question + retrieved passages.
   - Use a QA model (generative or extractive) to produce the answer.
3. Evaluate using Exact Match (EM), F1, and retrieval metrics.

---

## ⚙️ What Was Done

We compared two setups:
1. **Base retriever:** Using the pretrained `sentence-transformers/all-MiniLM-L6-v2`.
2. **Fine-tuned retriever:** MiniLM fine-tuned on SQuAD using **InfoNCE** (in-batch contrastive learning) to align questions and passages.

Both use the same:
- **Reader model:** `deepset/roberta-base-squad2`
- **Knowledge base:** SQuAD train + validation contexts
- **Evaluation set:** SQuAD validation questions (300 samples)

---

## 📊 Results

| Model | EM | F1 | Hit@5 | MRR@5 | nDCG@5 |
|:------|----:|----:|----:|----:|----:|
| **Base** | 0.6033 | 0.6453 | 0.7800 | 0.6342 | 1.0311 |
| **Fine-tuned** | **0.6467** | **0.6895** | **0.8000** | **0.6460** | **1.0722** |

✅ **Improvements across all metrics**, showing that fine-tuning the retriever leads to better question–passage alignment and more accurate answers.

---

## 🚀 Reproduction Guide

Follow these steps to reproduce the entire pipeline.

---

### 1️⃣ (Optional) Fine-tune the Bi-Encoder Retriever

```bash
python finetune_biencoder.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --out-dir ./pretrained_encoders \
    --split train \
    --max-examples 10000 \
    --val-max-examples 2000 \
    --batch-size 128 \
    --epochs 10 \
    --lr 2e-5 \
    --warmup_steps 200 \
    --temperature 0.05 \
    --seed 0 \
    --use-fp16
```

This script performs contrastive learning (InfoNCE loss) on SQuAD questions and passages.
Outputs:
- `./pretrained_encoders/encoder_q/` and `./pretrained_encoders/encoder_p/` (query and passage encoders)
- `./pretrained_encoders/metrics.csv`, `./pretrained_encoders/loss_curve.png`

### 2️⃣ Build the Retrieval Index

```bash
python build_index.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --splits train,validation \
    --max-examples 10000 \
    --chunk-tokens 180 \
    --chunk-stride 60 \
    --batch-size 256 \
    --max-length 256 \
    --out-dir ./rag_store/base \
    --use-fp16
```

This:
- Loads SQuAD train + validation contexts.
- Chunks long texts into overlapping passages.
- Encodes them with MiniLM.
- Builds a FAISS cosine-similarity index.
- Saves passages and metadata to `./rag_store/base`.
- To use the pretrained enconder replace model and out-dir with:
```bash
    --model `./pretrained_encoders/encoder_p/` \
    --out-dir ./rag_store/pretrained \
```

### 3️⃣ Run RAG Inference

```bash
python rag_generate.py \
    --encoder_q sentence-transformers/all-MiniLM-L6-v2 \
    --tokenizer_q sentence-transformers/all-MiniLM-L6-v2 \
    --store ./rag_store/base \
    --qa_model deepset/roberta-base-squad2 \
    --qa_max_length 384 \
    --top_m 5 \
    --out ./runs/preds_base.jsonl \
    --split validation \
    --max_eval 300 \
    --k 5
```

This:
- Retrieves top-5 passages for each question.
- Passes them to the extractive reader to generate the final answer.
- Stores predictions, gold answers, and retrieved contexts.
- To use the pretrained enconder replace encoder_q, tokenizer_q and out with:
```bash
    --encoder_q `./pretrained_encoders/encoder_q/` \
    --tokenizer_q `./pretrained_encoders/` \
    --out ./runs/preds_pretrained.jsonl \
```

### 4️⃣ Evaluate Results
```bash
python rag_eval.py \
    --preds runs/preds_base.jsonl \
    --out runs/metrics_base.txt \
    --compute_retrieval
```

This computes:
- **Exact Match (EM)** – proportion of predictions exactly matching any gold answer.
- **F1** – harmonic mean of token-level precision and recall.
- **Hit@k** – proportion of queries whose relevant document is in top-k retrieved passages.
- **MRR@k** – Mean Reciprocal Rank: how early the correct passage appears.
- **nDCG@k** – normalized Discounted Cumulative Gain: weighted rank quality measure.
- To evaluate the pretrained replace preds and out with:
```bash
    --preds runs/preds_pretrained.jsonl \
    --out runs/metrics_pretrained.txt \
```