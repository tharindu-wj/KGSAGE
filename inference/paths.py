"""Where the inference component's files are. Paths only -- no logic.

inference/ reads a trained checkpoint and a dataset, and writes
outputs/inference/ -- nothing else. The dashboard, the trainer and the
package's own eval tools are not named here because this component must
not write into their folders (check_boundaries.py holds it to that).

Run every script here from the repository root:

    python inference/corrupt.py ...
    python inference/explain.py ...
"""
from pathlib import Path

#: the repository root: the folder that contains kgsage/, inference/, data/
ROOT = Path(__file__).resolve().parents[1]

#: sources -- read-only for this component
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "outputs" / "checkpoints"

#: this component's output folder; it writes nowhere else
INFERENCE = ROOT / "outputs" / "inference"
RUNS_MANIFEST = INFERENCE / "manifest.json"

#: the file names inside one run folder (CONTRACTS.md)
RECORD = "corruptions.json"
CORRUPTIONS_TSV = "corruptions.tsv"
KG_CORRUPTED_TSV = "kg_corrupted.tsv"
LABELS_TSV = "labels.tsv"

#: where explain.py looks for the LLM key: beside the repository, then one
#: level up (a workspace that holds several sibling repositories)
ENV_FILES = (ROOT / ".env", ROOT.parent / ".env")
