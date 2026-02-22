# llm-abstract-screening

Reproducible code for LLM-assisted abstract screening and evaluation.

## Data availability note
The exported Embase title/abstract text used in the study cannot be redistributed due to licensing and copyright restrictions.
This repository therefore **does not include raw title/abstract text**, and the default pipeline **does not store** raw text in outputs.

Input spreadsheet (user-provided) expected columns:
- `title`
- `abstract`
Optional columns:
- `human expert` (abstract-level human decision)
- `final` (full-text inclusion decision)

## Quickstart

### Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### Set API key
```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

### Run screening
```bash
python scripts/run_screening.py \
  --dataset cholangio \
  --input path/to/input.xlsx \
  --model gpt-5-mini-2025-08-07 \
  --seed 12345 \
  --temperature 1 \
  --top-p 1 \
  --outdir outputs
```

Outputs:
- `outputs/<run_name>.jsonl` : per-record results (no raw text)
- `outputs/<run_name>.xlsx`  : spreadsheet summary (no raw text)
- `outputs/<run_name>.run_meta.json` : reproducibility metadata (prompt hashes, token totals, etc.)

### Run evaluation (optional)
```bash
python scripts/run_evaluation.py \
  --xlsx path/to/NSCLC_Results.xlsx \
  --sheets 4.0 5.0 5.0-mini \
  --human-col final \
  --out outputs/eval_results.xlsx
```


## Prompt management
Prompts are stored as versioned text files under `prompts/`.

Example:
```bash
python scripts/run_screening.py --dataset nsclc --input input.xlsx --model gpt-5-mini-2025-08-07 --system-version v1 --ontology-version v1
```

## Default sampling settings
By default, all presets run with `temperature=1.0` and `top_p=1.0` (unless you override via CLI flags). These values are recorded in `*.run_meta.json` for traceability.

## Configuration

Key runtime parameters are controlled via CLI flags (single place to manage environment factors):

- `--model` (recommend using a snapshot ID)
- `--temperature` (default: 1.0)
- `--top-p` (default: 1.0)
- `--seed`
- Optional pricing for cost estimation in `*.run_meta.json`:
  - `--input-per-1m`
  - `--output-per-1m`

Example:
```bash
python scripts/run_screening.py --dataset nsclc --input data/private.xlsx --model gpt-5-mini-2025-08-07 \
  --temperature 1.0 --top-p 1.0 --seed 12345 --input-per-1m 0.25 --output-per-1m 2.00
```


## Quick reproducibility

1) Prepare an input spreadsheet using `templates/input_template.xlsx` (keep real datasets private).
2) Set `OPENAI_API_KEY` in `.env`.
3) Run screening (example):

```bash
python scripts/run_screening.py       --dataset nsclc       --input data/my_private_export.xlsx       --model gpt-5-mini-2025-08-07       --temperature 1.0 --top-p 1.0 --seed 12345       --input-per-1m 0.25 --output-per-1m 2.00
```

Outputs (JSONL/XLSX/run_meta) will appear under `outputs/`.
