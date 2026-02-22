# Data (not included in the public repository)

This repository is designed to run on **private** bibliographic exports (e.g., Embase), which may be subject to copyright/licensing restrictions.
Therefore, **do not commit raw title/abstract datasets** to GitHub.

## Expected input spreadsheet

The screening runner expects an `.xlsx` file with at least:

| column | required | description |
|---|---:|---|
| `title` | ✅ | Record title |
| `abstract` | ✅ | Record abstract |

### Optional columns (for evaluation / benchmarking)

If you want to compare LLM outputs against human decisions, you may add:

| column | meaning |
|---|---|
| `human expert` | Decision at the **abstract-screening** stage (e.g., yes/no) |
| `final` | Decision after **full-text review** (reference standard in many workflows) |

The public repo does not include these data. You can keep your real files locally under `data/` (gitignored).

## Example (template only)

Use a synthetic template or an empty file with headers only. For example:

- `templates/input_template.xlsx` (recommended to share)
- Your private file: `data/my_private_export.xlsx` (do not share)
