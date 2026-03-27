"""
Windows UTF-8 wrapper for `modal serve`.
Fixes: 'charmap' codec can't encode character errors on Windows terminals.
"""
import os
import sys
import subprocess

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

result = subprocess.run(
    [sys.executable, "-m", "modal", "serve",
     "femscan-ai/modal_deploy.py"],
    env=os.environ,
)
sys.exit(result.returncode)
