# AQIE ChatDB

Chat-style UI for querying **uploaded** Excel/CSV files with natural language. The engine profiles schema, builds semantic features, runs DuckDB SQL, and optionally calls an OpenAI-compatible API for text-to-SQL.

## Project layout

```
DBEngine/
├── aqie/                 # Core engine (schema, SQL, recovery logic)
│   ├── __init__.py
│   └── engine.py
├── web/
│   └── app.py            # Streamlit UI (upload + chat)
├── app.py                # Launcher → runs web/app.py
├── run_ui.bat            # Windows: double-click to start UI
├── requirements.txt
├── pyproject.toml        # Optional: pip install -e .
├── .openai_api_key       # Optional local key (gitignored)
└── README.md
```

## Setup

```bash
cd DBEngine
pip install -r requirements.txt
```

## Run the UI

From project root (`DBEngine`):

```bash
python -m streamlit run web/app.py
```

Or use the launcher (same result):

```bash
streamlit run app.py
```

Windows: run `run_ui.bat`.

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Usage

1. Sidebar → upload one or more `.xlsx`, `.xls`, or `.csv` files.
2. Click **Load into engine**.
3. Optional: enable LLM and set model / base URL (API key via `OPENAI_API_KEY` or `.openai_api_key` in project root).
4. Ask questions in the chat.

Multi-sheet Excel files become multiple tables. Queries run in-memory with DuckDB.

## Example prompts

- `average total revenue`
- `show 10 rows`
- `patients whose blood type is B-`
- `data contains Blue Cross`

## Notes

- LLM mode needs billing/quota on your API account; otherwise the local planner still answers.
- Keep `.openai_api_key` secret and out of version control (see `.gitignore`).

## Publish to GitHub (`mail2mbabar`)

Git is installed and this folder is already a repo with an initial commit on branch `main`.

1. Log in with GitHub CLI (browser or token):

```powershell
cd c:\projects\DBEngine
gh auth login
```

2. Create the remote repository and push (pick a **new** repo name if `DBEngine` is taken):

```powershell
gh repo create DBEngine --public --source=. --remote=origin --push
```

Your repo will be: `https://github.com/mail2mbabar/DBEngine`

**Manual alternative:** create an empty repository named `DBEngine` on GitHub, then:

```powershell
git remote add origin https://github.com/mail2mbabar/DBEngine.git
git push -u origin main
```

Use a [Personal Access Token](https://github.com/settings/tokens) as the password when Git prompts (HTTPS).
