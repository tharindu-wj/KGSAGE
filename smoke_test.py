"""Local smoke test for the STANDALONE KGSAGE package.

Verifies the package structure on a developer machine that has torch installed.
This test imports ONLY `kgsage.*` — proving the package is independently usable.
Keep it that way: any end-to-end check that needs a downstream detector belongs
in that detector's own repo, not here.

Run from the directory that CONTAINS `kgsage/` — the registry's dataset paths
are relative to it. Use -m: running the file by path would put `kgsage/` itself
on sys.path instead of its parent, and `import kgsage` would fail.
    python -m kgsage.smoke_test

Sections:
  1. Imports (kgsage.* only)
  2. Behaviour checks (resolve_dataset)
  3. Dual-discriminator architecture (candidate_v2) + corruption-generation
     imports
  4. load_kg works on dummy_kg
"""
import os
import sys


def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print("Python:", sys.version.split()[0])
    print("Platform:", sys.platform)

    # ---- SECTION 1: Imports ----
    section("SECTION 1: kgsage.* imports")

    import kgsage
    print(f"OK: import kgsage  (version={kgsage.__version__})")

    from kgsage import load_kg, resolve_dataset, KNOWN_DATASETS
    print("OK: eager public API (load_kg, resolve_dataset, KNOWN_DATASETS)")
    print(f"    KNOWN_DATASETS: {sorted(KNOWN_DATASETS.keys())}")

    from kgsage.preprocessing.loaders import load_kg as _load_kg2
    from kgsage.preprocessing.registry import resolve_dataset as _resolve2
    print("OK: kgsage.preprocessing.* sub-package imports")

    # ---- SECTION 2: Behaviour ----
    section("SECTION 2: Behaviour checks")

    print("Known dataset configs:")
    for name in sorted(KNOWN_DATASETS.keys()):
        cfg = resolve_dataset(name)
        print(f"  {name:10s}  path={cfg['path']}")

    try:
        resolve_dataset("not_a_real_dataset")
        print("FAIL: resolve_dataset should have raised")
        return 1
    except ValueError:
        print("OK: resolve_dataset rejects unknown name")

    # Exercise the path branch too, but source the path from the registry so
    # this stays correct if the dataset root ever moves again.
    dummy_dir = KNOWN_DATASETS["dummy_kg"]["default_path"]
    if os.path.isdir(dummy_dir):
        cfg = resolve_dataset(dummy_dir)
        print(f"OK: resolve_dataset(path) -> name={cfg['name']}, path={cfg['path']}")

    # ---- SECTION 3: dual-discriminator architecture imports (candidate_v2) ----
    section("SECTION 3: kgsage.gan.* (dual-discriminator) + corruption generation imports")

    # CandidateScoringGenerator (the generator) is the one and only
    # architecture; gumbel_softmax is the trainer's straight-through selection
    # helper.
    from kgsage.gan.generator import CandidateScoringGenerator, gumbel_softmax
    print("OK: kgsage.gan.generator.{CandidateScoringGenerator, gumbel_softmax}")

    # The rest of the adversarial training stack: the plausibility
    # discriminator judges "could this triple be real?", the neighbourhood
    # discriminator judges "does the filler fit this anchor's neighbourhood?".
    from kgsage.gan.plausibility_discriminator import PlausibilityDiscriminator
    from kgsage.gan.neighbourhood_discriminator import NeighbourhoodDiscriminator
    from kgsage.gan.membership_sketch import build_membership_sketches
    from kgsage.gan.candidate_sampler import CandidateSampler
    print("OK: kgsage.gan.{PlausibilityDiscriminator, NeighbourhoodDiscriminator, "
          "build_membership_sketches, CandidateSampler}")

    # Importing the trainer transitively verifies the whole candidate_v2
    # dependency graph.
    from kgsage.gan import train as _train
    print("OK: kgsage.gan.train (the dual-discriminator trainer)")

    from kgsage.corruption_generation import load_checkpoint, generate_negatives, render_stats as _rs
    print("OK: kgsage.corruption_generation.{load_checkpoint, generate_negatives, render_stats}")

    # ---- SECTION 4: load_kg on dummy_kg ----
    section("SECTION 4: load_kg on dummy_kg")

    if os.path.isdir(dummy_dir):
        kg = load_kg(dummy_dir)
        print(f"OK: load_kg({dummy_dir})")
        print(f"    {kg['n_ent']} entities, {kg['n_rel']} relations")
        print(f"    train={len(kg['triples_train'])}  "
              f"valid={len(kg['triples_valid'])}  test={len(kg['triples_test'])}")
    else:
        print(f"SKIP: {dummy_dir} not on disk (set KGSAGE_DATA if data lives elsewhere)")

    section("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
