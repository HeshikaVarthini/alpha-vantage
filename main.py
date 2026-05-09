from pathlib import Path
import subprocess
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parent
    dashboard_file = project_root / "dashboard.py"

    if not dashboard_file.exists():
        raise FileNotFoundError(f"Missing dashboard file: {dashboard_file}")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_file)],
        check=True,
    )


if __name__ == "__main__":
    main()
