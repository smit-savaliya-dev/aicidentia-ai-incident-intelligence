# AICIDENTIA

> **Discovering Patterns. Understanding AI Harm. Learning from Failure.**

**Learning from Failure: A Data-Driven Analysis of Reported Real-World AI Incidents**

MSc Data Analytics - Domain Applications project, National College of Ireland (individual work, due 19 Nov 2026).
Dashboard: **AICIDENTIA Intelligence** (Streamlit). Process framework: **CRISP-DM**.

## Overview

AICIDENTIA analyses the [AI Incident Database (AIID)](https://incidentdatabase.ai/) to learn from reported
real-world AI failures.

| RQ | Question | Methods |
|----|----------|---------|
| RQ1 | Risk themes and their trends over time | LDA vs NMF vs BERTopic; Mann-Kendall trend tests |
| RQ2 | Network of incidents, organisations, harmed groups and risk domains | Louvain vs Leiden vs label propagation; centrality |
| RQ3 | Classify incidents into risk domains; audit AI-generated labels against 150 hand labels | TF-IDF + logistic regression/SVM vs sentence embeddings vs fine-tuned DistilBERT; Cohen's kappa |
| RQ4 (stretch) | Similar-incident retrieval | TF-IDF vs BM25 vs sentence embeddings; precision@5, MRR |

**Status:** project scaffold only. Module functions are stubs (`NotImplementedError`) and the dashboard
pages are placeholders.

## Reproducibility

- Python is pinned in `.python-version` (3.13.12); every dependency is pinned in `uv.lock`
  (`requirements.txt` is an export of the same pins).
- The global seed (`random_seed: 42` in `config/config.yaml`) is applied with `aicidentia.config.set_seeds()`.
- The data is a frozen snapshot: file names carry the export date, and `scripts/check_data.py` records
  SHA-256 hashes in `docs/data_manifest.md`.
- `docs/environment.md` lists the exact OS, Python and package versions of the machine used.

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/) and git.

```bash
# 1. Create .venv, install Python 3.13.12 (if needed) and every dependency from uv.lock
uv sync --all-extras

# 2. Strip notebook outputs on commit (once per clone)
uv run nbstripout --install

# 3. (Optional) regenerate docs/environment.md
uv run python scripts/generate_environment_doc.py
```

Activate the environment (optional - `uv run <cmd>` works without activating):

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

**torch / GPU.** On Windows and Linux, `pyproject.toml` pulls the CUDA 13.2 build of torch from the PyTorch
index. For a CPU-only machine, change the index URL in `[[tool.uv.index]]` from `.../whl/cu132` to
`.../whl/cpu`, then run `uv lock && uv sync --all-extras`. DistilBERT fine-tuning can also run on Google Colab.

Without uv, use `python -m venv .venv`, activate it, then
`pip install -r requirements.txt && pip install -e . --no-deps`.

**VS Code.** Open this folder (`aicidentia-ai-incident-intelligence`) as the workspace and accept the
recommended extensions. The interpreter is `.venv`, Ruff formats on save, and notebooks use the `.venv` kernel.

## Data

Download manually (nothing is downloaded automatically) and place in `data/raw/`:

| File | Required | Used for |
|------|----------|----------|
| `AIID_Excel_Export-YYYYMMDD.xlsx` | yes | Main dataset (sheets: Incidents, Reports, Entities, Data Dictionary, Coverage Map) |
| `backup-YYYYMMDDhhmmss.tar.bz2` | optional | Full article text and editor-marked similar incidents |

Then set `snapshot_date` in `config/config.yaml` and verify:

```bash
uv run python scripts/check_data.py
```

This prints each sheet with its row and column counts, writes SHA-256 hashes to `docs/data_manifest.md`
and safely extracts the snapshot into `data/raw/snapshot_<date>/`.

Put your 150 hand-labelled incidents in `data/manual/` (tracked in git). Optional OECD AIM exports go in
`data/external/`.

## How to run everything

| Task | Command |
|------|---------|
| Check the data | `uv run python scripts/check_data.py` |
| Open Jupyter | `uv run jupyter lab` |
| Run the dashboard | `uv run streamlit run app/Home.py` |
| Run the tests | `uv run pytest` |
| Lint / format | `uv run ruff check .` / `uv run ruff format .` |
| Re-export requirements.txt | `uv run python scripts/export_requirements.py` |

Notebooks run in order (CRISP-DM): `01_data_audit` -> `02_cleaning_eda` -> `03_rq1_themes_trends` ->
`04_rq2_network` -> `05_rq3_classification_label_audit` -> `06_rq4_similarity`.

## Project structure

```
config/           config.yaml - paths, seed, model names, year range
data/             raw (git-ignored), interim, processed, external, manual (tracked)
notebooks/        01-06, one per CRISP-DM stage / research question
src/aicidentia/   installable package: data, features, models, evaluation, viz
app/              Streamlit dashboard (Home.py + pages/)
scripts/          check_data.py, generate_environment_doc.py, export_requirements.py
models/           saved models (git-ignored)
reports/          figures/ and tables/
docs/             environment.md, project_log.md, data_manifest.md
tests/            smoke tests
```

## Notes

- NLTK corpora (stopwords, WordNet, ...) and Hugging Face models are downloaded on first use, not during setup.
- BERTopic installs the `hdbscan` package as a hard dependency; our code uses scikit-learn's built-in
  `sklearn.cluster.HDBSCAN` and passes it to BERTopic.
