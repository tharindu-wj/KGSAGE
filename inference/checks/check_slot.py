"""The generator's `slot=` option, checked on a real checkpoint. Offline.

    python inference/checks/check_slot.py
    python inference/checks/check_slot.py --ckpt outputs/checkpoints/run_codex-s_s0.pt --data codex-s

The option corrupt.py's --slot rests on (kgsage.corruption_generation,
added 9-10 Oct 2026). Properties, not a stored baseline, so it stays true
across retraining:
  - slot=None is exactly the call without the argument -- every recorded run
  - a forced slot corrupts only that slot, on every row that is not a null
  - a forced slot leaves the caller's rng where None leaves it (the 50/50
    coin is still drawn), so the same seed draws the same later sources
  - anything else is refused
Every line must read PASS. Run it in the environment with torch.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(ROOT))          # this checkout's kgsage
if sys.platform == "win32":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from kgsage import load_kg, resolve_dataset  # noqa: E402
from kgsage.corruption_generation import generate_negatives, load_checkpoint  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=str(ROOT / "outputs" / "checkpoints" / "run_codex-s_s0.pt"))
parser.add_argument("--data", default="codex-s")
parser.add_argument("--rows", type=int, default=400)
args = parser.parse_args()
torch.set_num_threads(1)

kg = load_kg(resolve_dataset(args.data)["path"])
payload = load_checkpoint(args.ckpt, device=torch.device("cpu"))
real = kg["triples_train"] + kg["triples_valid"] + kg["triples_test"]
rows = [real[i] for i in np.random.default_rng(123).choice(len(real), args.rows, replace=False)]
failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f"   [{detail}]"))
    if not ok:
        failures.append(label)


def run(**kw):
    rng = np.random.default_rng(0)
    corruptions, stats = generate_negatives(rows, payload, kg, rng=rng, **kw)
    return corruptions, stats, int(rng.integers(0, 2**31 - 1))


print(f"slot= on {Path(args.ckpt).name}, {len(rows)} rows of {args.data}")
plain = run()
none = run(slot=None)
check("slot=None is exactly the call without it", plain == none)
for forced, index in (("head", 0), ("tail", 2), (0, 0), (2, 2)):
    corruptions, stats, after = run(slot=forced)
    nulls = set(stats["null_indices"])
    other = 2 - index
    stray = [i for i, (source, made) in enumerate(zip(rows, corruptions))
             if i not in nulls and (source[other] != made[other] or source[index] == made[index])]
    counted = stats["slot_t"] if index == 0 else stats["slot_h"]
    check(f"slot={forced!r}: only that slot changes, and only it is counted",
          not stray and counted == 0, f"{len(stray)} rows, other-slot count {counted}")
    check(f"slot={forced!r}: the caller's rng ends where slot=None leaves it", after == plain[2])
for bad in ("relation", 1, "tails"):
    try:
        generate_negatives(rows[:2], payload, kg, rng=np.random.default_rng(0), slot=bad)
        check(f"slot={bad!r} is refused", False, "accepted")
    except ValueError:
        check(f"slot={bad!r} is refused", True)
print(f"\n{'ALL PASS' if not failures else str(len(failures)) + ' FAILED'}")
sys.exit(1 if failures else 0)
