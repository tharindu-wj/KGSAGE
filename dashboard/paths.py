"""Where the dashboard's files are. Paths only -- no logic.

The dashboard READS the inference component's run records and writes
outputs/dashboard/ -- nothing else. The React app in app/ serves
outputs/dashboard/ as its static root, so what export.py writes there is
what the browser fetches.
"""
from pathlib import Path

#: the repository root: the folder that contains kgsage/, dashboard/, data/
ROOT = Path(__file__).resolve().parents[1]

#: what the inference component made -- read-only here
INFERENCE = ROOT / "outputs" / "inference"
RUNS_MANIFEST = INFERENCE / "manifest.json"
RECORD = "corruptions.json"

#: this component's output folder; it writes nowhere else. DATA_OUT is
#: what the app fetches; the app's own build output stays inside app/.
DASHBOARD = ROOT / "outputs" / "dashboard"
DATA_OUT = DASHBOARD / "data"
APP = Path(__file__).resolve().parent / "app"
