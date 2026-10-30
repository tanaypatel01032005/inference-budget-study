# Reproducibility Guide

This guide describes how to replicate the entire Inference Budget Study workflow from initial environment setup through dataset preparation, batch generation, submission, and analysis.

---

## 1. Environment Setup

### 1.1 Python Virtual Environment

Python 3.9+ is recommended. On Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Dataset Preparation

Download the official GSM8K test split and extract the fixed 1,000-question sample:

```powershell
python scripts/download_dataset.py
```

- **Output File**: `data/processed/gsm8k_test_1000.csv`
- **Metadata**: `data/metadata/dataset_metadata.json`
- **Seed**: `42` (ensures identical problem sampling across environments)
- **Sample count**: Exactly 1,000 problems

*Note: If the processed CSV is already present, the script prevents accidental overwrite unless `--force` is provided.*

---

## 3. Sheshnag Configuration

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Edit `.env` to supply live credentials:

```dotenv
SHESHNAG_BASE_URL=https://<sheshnag-deployment-host>/v1
SHESHNAG_API_KEY=<your-api-key>
SHESHNAG_MODEL_ID=<live-model-id>
```

Verify connection and catalogue:

```powershell
python scripts/submit_sheshnag.py --check-models
```

---

## 4. Batch Generation & Validation

Generate the 50,000 JSONL requests (1,000 questions × 50 independent generations):

```powershell
python scripts/create_batch.py
```

Validate format, counts, uniqueness, and absence of credentials or invalid parameters ($n > 1$, streaming):

```powershell
python scripts/validate_batch.py
```

---

## 5. Batch Submission & Job Execution

Submit the validated batch to Sheshnag:

```powershell
python scripts/submit_sheshnag.py
```

To monitor an existing submission without blocking:

```powershell
python scripts/submit_sheshnag.py --status <BATCH_ID>
```

To download completed results:

```powershell
python scripts/download_results.py --batch-id <BATCH_ID>
```

---

## 6. Evaluation & Voting

Parse responses, execute deterministic plurality voting for $N \in \{1, 3, 5, 10, 20, 50\}$, and evaluate exact matches:

```powershell
python scripts/evaluate.py
```

- **Outputs**:
  - `results/parsed/generations.csv`
  - `results/final/evaluation.csv`
  - `results/final/evaluation_long.csv`

---

## 7. Analysis & Visualizations

Compute bootstrap confidence intervals, marginal gains, and output figures:

```powershell
python scripts/analyze.py
```

- **Outputs**:
  - `results/analysis/summary.csv`
  - `results/analysis/summary.json`
  - `results/analysis/marginal_gains.csv`
  - `results/analysis/figures/accuracy_vs_generations.png`
  - `results/analysis/figures/marginal_gain.png`
  - `results/analysis/figures/accuracy_vs_budget.png`
  - `reports/results_summary.md` (updated)

---

## 8. Local Verification (No GPU / No Sheshnag Required)

Run unit test suite:

```powershell
pytest tests/ -v
```

Execute end-to-end mock pipeline:

```powershell
python scripts/run_mock.py --questions 20 --generations 50
```

