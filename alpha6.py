"""
Backward-compatible launcher.

The application lives in `dashboard.py`. Prefer `python main.py` or
`streamlit run dashboard.py`.
"""

from main import main

if __name__ == "__main__":
    raise SystemExit(main())
