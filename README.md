# Inference Budget Study

> **A systematic empirical study investigating inference-time computation vs. reasoning quality: How many sampled generations are enough?**

---

## Research Question

> **How does increasing the number of independent generations affect the accuracy of a language model on mathematical reasoning, and at what point do additional generations provide diminishing returns?**

---

## Experiment at a Glance

| Parameter | Specification |
|:---|:---|
| **Benchmark** | GSM8K (`main`, official `test` split) |
| **Questions** | Exactly 1,000 questions (randomly sampled with seed `42`) |
| **Generations / Question** | 50 independent completions |
| **Total Workload** | 50,000 Sheshnag batch requests |
| **Inference Budgets ($N$)** | $N \in \{1, 3, 5, 10, 20, 50\}$ |
| **Aggregation Rule** | Deterministic plurality/majority voting over normalized answers |
| **Primary Metric** | Exact-answer accuracy (%) |
| **Secondary Metric** | Marginal accuracy gain per additional generation |
| **Statistical Rigor** | 95% bootstrap confidence intervals (10,000 resamples, question-level) |

---

## Pipeline Workflow

```text
GSM8K Test Split (1,319 questions)
  │
  ▼  (Random sample without replacement, seed=42)
Fixed 1,000-Question Dataset (data/processed/gsm8k_test_1000.csv)
  │
  ▼  (50 independent requests per question, n=1, prompt template)
50,000 Sheshnag Batch Requests (batch/generated/sheshnag_batch.jsonl)
  │
  ▼  (Upload & Poll Sheshnag OpenAI-compatible Batch API)
Raw Completions (results/raw/sheshnag_results.jsonl)
  │
  ▼  (Multi-method answer parsing & mathematical normalization)
Parsed Generations (results/parsed/generations.csv)
  │
  ▼  (Deterministic shuffle seed=42+q_id, plurality voting at N=1,3,5,10,20,50)
Evaluation Tables (results/final/evaluation.csv & evaluation_long.csv)
  │
  ▼  (Bootstrap CIs, marginal gain deltas, publication figures)
Analysis & Plots (results/analysis/ & reports/results_summary.md)
```

---

## Quickstart

### 1. Environment Setup

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Local Pipeline (Offline Mock Mode)

Run the full pipeline using simulated completions without needing Sheshnag access:

```powershell
# Run unit tests
pytest tests/ -v

# Run end-to-end mock run
python scripts/run_mock.py
```

### 3. Prepare Dataset

```powershell
python scripts/download_dataset.py
```

This downloads GSM8K and writes `data/processed/gsm8k_test_1000.csv` and `data/metadata/dataset_metadata.json`.

---

## Sheshnag Setup Tomorrow

When Sheshnag access is granted, only three values need to be configured in `.env`:

```dotenv
SHESHNAG_BASE_URL=https://<sheshnag-deployment-host>/v1
SHESHNAG_API_KEY=<your-personal-api-key>
SHESHNAG_MODEL_ID=<model-id-from-catalogue>
```

### Live Run Commands:

1. **Check available models**:
   ```powershell
   python scripts/submit_sheshnag.py --check-models
   ```
2. **Generate the 50,000 JSONL batch**:
   ```powershell
   python scripts/create_batch.py
   ```
3. **Validate batch integrity**:
   ```powershell
   python scripts/validate_batch.py
   ```
4. **Submit and monitor**:
   ```powershell
   python scripts/submit_sheshnag.py
   ```
5. **Evaluate & analyze results**:
   ```powershell
   python scripts/evaluate.py
   python scripts/analyze.py
   ```

Refer to `SHESHNAG_HANDOFF.txt` for the complete operator guide.

---

## Repository Structure

```text
inference-budget-study/
│
├── README.md                      # Overview and instructions
├── SHESHNAG_HANDOFF.txt           # Operator handoff guide for Sheshnag execution
├── REPRODUCIBILITY.md             # Detailed reproducibility specifications
├── requirements.txt               # Pinned dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
│
├── config/
│   └── config.yaml                # Centralized hyperparameters & filepaths
│
├── prompts/
│   └── math_reasoning.txt         # Fixed mathematical reasoning prompt template
│
├── data/
│   ├── raw/                       # Staging for raw downloads
│   ├── processed/                 # Processed 1,000-question GSM8K sample
│   └── metadata/                  # Dataset sampling metadata & random seed record
│
├── batch/
│   ├── generated/                 # 50,000-line JSONL batch file for Sheshnag
│   └── results/                   # Sheshnag batch job metadata and status receipts
│
├── src/
│   └── inference_budget/
│       ├── __init__.py            # Package root
│       ├── config.py              # Configuration & environment variable loaders
│       ├── dataset.py             # GSM8K loader, question selector, gold parser
│       ├── prompting.py           # Prompt template renderer
│       ├── parsing.py             # Multi-strategy answer extractor & normalizer
│       ├── voting.py              # Deterministic plurality voter & tie-breaker
│       ├── batch.py               # Batch request builder & strict validator
│       ├── sheshnag.py            # OpenAI-compatible Batch API client
│       ├── evaluation.py          # Multi-budget evaluation engine
│       └── analysis.py            # Bootstrap CIs, marginal gain analysis, plots
│
├── scripts/
│   ├── download_dataset.py        # Downloads & samples GSM8K test set
│   ├── create_batch.py            # Generates the 50,000-request JSONL batch
│   ├── validate_batch.py          # Validates batch size, structure, parameters
│   ├── submit_sheshnag.py         # Submits, polls, and monitors Sheshnag batch
│   ├── download_results.py        # Downloads completed batch results
│   ├── evaluate.py                # Parses answers and computes accuracy across N
│   ├── analyze.py                 # Produces CIs, marginal gain tables, and plots
│   └── run_mock.py                # Standalone offline pipeline simulator
│
├── tests/
│   ├── __init__.py
│   ├── test_parsing.py            # Unit tests for normalization and extraction
│   ├── test_voting.py             # Unit tests for voting and tie-breaking
│   ├── test_dataset.py            # Unit tests for gold answer extraction
│   ├── test_batch.py              # Unit tests for batch generator & validator
│   └── test_evaluation.py         # Unit tests for accuracy calculation & budgets
│
├── results/
│   ├── raw/                       # Raw downloaded JSONL results from Sheshnag
│   ├── parsed/                    # Structured CSV of extracted model answers
│   ├── final/                     # Wide & long evaluation CSVs
│   ├── analysis/                  # Summary CSVs, JSON, and publication figures
│   └── mock/                      # Outputs from offline mock test suite
│
└── reports/
    ├── research_report.md         # Full scientific research paper
    ├── methodology.md             # Formal methodology documentation
    └── results_summary.md         # Empirical results and marginal gain tables
```
