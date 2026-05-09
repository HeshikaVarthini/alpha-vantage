"""
Orchestrator entrypoint — launches the Streamlit dashboard.

Same idea as Job Notifier's main.py: run `python main.py` from the project root.
Alternatively: `streamlit run dashboard.py`
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    dashboard = root / "dashboard.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard),
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.run(cmd, cwd=str(root)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
