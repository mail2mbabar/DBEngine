"""Backward-compatible launcher: `streamlit run app.py` from project root."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
ui = root / "web" / "app.py"
raise SystemExit(
    subprocess.call([sys.executable, "-m", "streamlit", "run", str(ui), *sys.argv[1:]])
)
