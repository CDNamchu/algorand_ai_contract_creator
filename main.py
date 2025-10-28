"""
main.py — Entry point to run the AI-Powered Algorand Smart Contract Creator
"""

import sys
import os
from pathlib import Path
from streamlit.web import cli as stcli

# Ensure `src` is on sys.path so imports like `algorand_ai_contractor` work
# when running `python main.py` directly (behaves like PYTHONPATH=src).
ROOT = Path(__file__).resolve().parent
SRC_PATH = str((ROOT / "src").resolve())
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    # Also set PYTHONPATH in the environment for any child processes
    os.environ.setdefault("PYTHONPATH", SRC_PATH)

if __name__ == "__main__":
    # Define the script you want to run (your Streamlit file)
    app_file = "src/algorand_ai_contractor/ui/streamlit_app.py"  # Replace with actual filename

    # Equivalent to running: streamlit run app_file
    sys.argv = ["streamlit", "run", app_file]
    sys.exit(stcli.main())
