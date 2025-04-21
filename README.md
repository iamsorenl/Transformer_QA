# Robustness and Domain Adaptation in Transformer-Based Question Answering

This repository explores the **robustness** and **domain adaptation** of transformer-based models—specifically **RoBERTa fine-tuned on SQuAD 2.0**—within the context of Question Answering (QA) tasks.

The work focuses on evaluating how well QA models generalize to out-of-domain data and how adapter-based fine-tuning methods like **LoRA (Low-Rank Adaptation)** can be used to improve performance efficiently.

---

## Project Overview

### Part 1: Robustness Testing
- Evaluated how easily RoBERTa can be misled by:
  - **In-domain** passages with ambiguous or misleading phrasing
  - **Edited passages** introducing absurdity or false information
  - **Out-of-domain** data from social media, fiction, biomedical texts, etc.
- Results show that RoBERTa relies heavily on word surface forms and often fails under noisy or illogical input.

### Part 2: Domain Adaptation (Covid-QA)
- Used `roberta-base-squad2` to evaluate out-of-domain performance on the **Covid-QA** biomedical dataset
- Reported **Exact Match (EM)** and **F1 Score** on both dev and test splits
- Baseline performance revealed difficulty adapting to complex domain-specific terminology and reasoning

### Part 3: LoRA Adapter Training
- Implemented **LoRA** to fine-tune only a small subset of the model's parameters
- Used Hugging Face’s `Trainer` with:
  - LoRA rank: 8
  - Alpha: 16
  - Dropout: 0.1
  - CLS token for unanswerable questions
  - Sliding window context processing for long passages
- Achieved significant improvements over baseline with reduced training cost

---

## Performance Summary

| Model                     | Dataset | Exact Match (EM) | F1 Score |
|--------------------------|---------|------------------|----------|
| RoBERTa-SQuAD2 (Baseline) | Dev     | 27.09            | 46.38    |
| RoBERTa-SQuAD2 (Baseline) | Test    | 24.27            | 43.79    |
| LoRA-RoBERTa (Adapter)    | Dev     | 33.50            | 61.22    |
| LoRA-RoBERTa (Adapter)    | Test    | 29.60            | 54.33    |

---

## Setup Instructions

### 1. Clone and install dependencies
```bash
git clone https://github.com/your-username/transformer_qa.git
cd transformer_qa
pip install -r requirements.txt
```

### 2. (Optional) Use a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

---

## Requirements

The project uses Python 3.8+ and the following libraries:

```
torch
transformers
datasets
adapter-transformers
evaluate
numpy
pandas
matplotlib
tqdm
```

You’ll need a GPU or Apple MPS-compatible machine to efficiently run training jobs.

---
