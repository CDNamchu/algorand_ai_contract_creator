"""
main.py — Entry point to run the AI-Powered Algorand Smart Contract Creator
"""

import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    # Define the script you want to run (your Streamlit file)
    app_file = "src/algorand_ai_contractor/ui/streamlit_app.py"  # Replace with actual filename

    # Equivalent to running: streamlit run app_file
    sys.argv = ["streamlit", "run", app_file]
    sys.exit(stcli.main())
