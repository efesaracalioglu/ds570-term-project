"""
Entry point — F1 Pit Stop Strategy Predictor.

Usage:
  python main.py          → launches the Streamlit dashboard on port 8501
"""
import os
import subprocess
import sys


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/dashboard.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
    ])
