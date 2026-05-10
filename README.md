# AQIE ChatDB : Adaptive Query Interpretation Engine

**AQIE ChatDB** is a small web application that lets you **upload spreadsheet databases** (CSV or Excel), then **ask questions in plain English** and get **answers backed by real SQL** running on your data — similar in spirit to a ChatGPT-style assistant, but **grounded only on files you provide**.

It implements ideas from an **Adaptive Query Interpretation Engine (AQIE)** pipeline: automatic schema profiling, lightweight semantic typing, natural-language → SQL planning, and recovery logic when the first query returns no rows.

---

## What problem does this solve?

Typical “talk to your database” tools often assume a fixed schema or heavy upfront setup. This project targets **ad hoc analysis**:

- You drop in **one or more** `.csv`, `.xlsx`, or `.xls` files.
- The app **profiles columns**, infers useful metadata, and registers tables **in memory**.
- You type everyday prompts (filters, counts, date ranges, “contains …”) and see **the generated SQL**, **warnings**, and **result tables** (and optional charts).

No separate database server is required; queries execute locally via **DuckDB**.

---

## Features

| Area | Description |
|------|-------------|
| **Upload & multi-table** | Multiple files supported; multi-sheet Excel becomes multiple logical tables. |
| **Schema discovery** | Per-column types, null rates, cardinality, primary-key-like hints. |
| **Semantic signals** | Feature engineering + **Random Forest**–based semantic labels on columns (dissertation-style AQIE module). |
| **Natural language → SQL** | Intent detection (aggregate vs list vs filter), date/year constraints, “contains” / token recovery when results are empty. |
| **Optional LLM planner** | OpenAI-compatible **chat completions** API can propose DuckDB `SELECT` queries from your schema; falls back to rule-based SQL if the API fails or quota is exceeded. |
| **Chat UI** | **Streamlit** chat, SQL display, dataframes, simple Plotly charts for numeric results. |
| **Secrets** | API keys via `OPENAI_API_KEY` or project-root `.openai_api_key` (gitignored). |

---

## How it works (high level)

1. **Load files** → pandas reads CSV/Excel; tables are registered with DuckDB.
2. **Profile** → column statistics and optional semantic classification.
3. **Ask** → your prompt is interpreted into SQL (LLM and/or local planner).
4. **Execute** → DuckDB runs the query; optional recovery searches broaden filters if the first result is empty.

---

## Tech stack

- **Python 3.10+**
- **Streamlit** — web UI
- **pandas**, **openpyxl** — tabular load
- **DuckDB** — in-process SQL
- **scikit-learn** — semantic typing model
- **plotly** — charts
- **openai** (optional) — HTTPS API client for compatible LLM endpoints

---

## Project layout

```
DBEngine/
├── aqie/
│   ├── __init__.py
│   └── engine.py          # Core: schema, SQL generation, recovery
├── web/
│   └── app.py             # Streamlit UI (upload + chat)
├── scripts/
│   ├── with-gh.bat        # Run gh when not on PATH (Windows)
│   └── push-to-github.bat # PATH helper + gh repo create/push
├── app.py                 # Launcher → Streamlit runs web/app.py
├── run_ui.bat             # Quick start (Windows)
├── requirements.txt
├── pyproject.toml         # Optional: pip install -e .
├── .openai_api_key        # Optional; never commit (see .gitignore)
└── README.md
```

---

## Setup

```bash
cd DBEngine
pip install -r requirements.txt
```

---

## Run the UI

From the project root:

```bash
python -m streamlit run web/app.py
```

Alternative:

```bash
streamlit run app.py
```

On Windows you can double-click **`run_ui.bat`**.

Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## Usage

1. In the sidebar, upload one or more `.csv` / `.xlsx` / `.xls` files.
2. Click **Load into engine** (or equivalent) so tables are registered and profiled.
3. Optionally enable **LLM** mode and set model / base URL; put your API key in **`OPENAI_API_KEY`** or **`.openai_api_key`** in the repo root.
4. Ask questions in the chat; review SQL and tables below each reply.

---

## Example prompts

- `average total revenue`
- `show 10 rows`
- `patients whose blood type is B-`
- `data contains Blue Cross`
- `admission date between 2024`
- `count rows where insurance provider contains Medicare`

---

## Configuration & security

- **`OPENAI_API_KEY`** or **`.openai_api_key`** — used only if you enable LLM planning; keep keys private.
- **`.gitignore`** excludes `.openai_api_key`, `.env`, virtualenvs, and common IDE junk.
- LLM calls require a **valid API plan / quota**; otherwise the app continues with the **local planner**.

---

## Publish to GitHub (`mail2mbabar`)

This repo is intended to be pushed to **`https://github.com/mail2mbabar/DBEngine`** (adjust name if taken).

1. Install **Git** and **GitHub CLI** (`gh`), ensure both are on `PATH` (or use **`scripts\push-to-github.bat`** on Windows).

2. Log in:

```powershell
cd c:\projects\DBEngine
gh auth login
```

3. Create remote and push:

```powershell
gh repo create DBEngine --public --source=. --remote=origin --push
```

**Manual:** create an empty `DBEngine` repo on GitHub, then:

```powershell
git remote add origin https://github.com/mail2mbabar/DBEngine.git
git push -u origin main
```

For HTTPS pushes, use a [Personal Access Token](https://github.com/settings/tokens) when Git asks for a password.

---

## License & attribution

Dissertation / notebook materials bundled in this folder (if present) remain the author’s work; this README describes the **ChatDB application code** only.
