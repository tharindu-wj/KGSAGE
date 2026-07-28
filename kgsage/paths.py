"""Where KGSAGE writes its run artifacts.

Every eval CLI writes under ``outputs/eval/<tool>/`` and the trainer writes under
``outputs/checkpoints/``. Both are resolved here, once, so that moving the
package does not silently relocate them -- which is exactly what happened when
each CLI carried its own ``Path(__file__).parents[N]`` expression.

Resolution order:

1. ``KGSAGE_OUTPUTS`` if set. Use this on a cluster to send artifacts to scratch,
   and after a NON-editable ``pip install`` -- otherwise the default below
   resolves into site-packages, which is read-only on most systems.
2. ``<repo root>/outputs`` otherwise, where the repo root is the directory that
   contains the ``kgsage`` package. This is the editable-install default and
   matches how the SLURM launchers and the notebooks expect to find artifacts.
"""
import os
from pathlib import Path

# parents[0] = kgsage/ (the package), parents[1] = the repo root beside it.
_REPO_ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_ROOT = Path(os.environ.get("KGSAGE_OUTPUTS") or (_REPO_ROOT / "outputs"))
EVAL_ROOT = OUTPUTS_ROOT / "eval"
CHECKPOINT_ROOT = OUTPUTS_ROOT / "checkpoints"

# Where the registry looks for `<name>/train.txt`. Override with KGSAGE_DATA when
# the benchmarks live on shared scratch rather than beside the checkout, which is
# the normal arrangement on a cluster. These are absolute, so a registry lookup
# gives the same answer from any working directory -- required now that the
# package is pip-installed and importable from anywhere.
DATA_ROOT = Path(os.environ.get("KGSAGE_DATA") or (_REPO_ROOT / "data"))

__all__ = ["OUTPUTS_ROOT", "EVAL_ROOT", "CHECKPOINT_ROOT", "DATA_ROOT"]
