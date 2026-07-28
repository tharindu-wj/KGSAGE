"""KGSAGE preprocessing layer — KG loading, conversion, and the dataset registry.

Everything that turns raw knowledge-graph files ON DISK into the in-memory
structures the rest of the package consumes. The dataset FILES themselves live
in `kgsage/data/<NAME>/`; this package is the CODE that reads them. Keep the
two straight:

    kgsage/preprocessing/   code  (this package)
    kgsage/data/            bytes (train.txt / valid.txt / test.txt per dataset)

Public API surfaces:
  load_kg              — load a KG from a TSV directory
  resolve_dataset      — look up known dataset defaults (or treat input as a path)
  KNOWN_DATASETS       — registry of pre-configured datasets

This is the layer every KGSAGE phase starts from: Phase 1 builds the context
table E' from the edge list load_kg returns, and Phase 3 (corruption
generation) reuses the same vocabulary maps.

Add a new dataset to KNOWN_DATASETS in registry.py; everything else just works.
yago_to_tsv.py is a standalone converter, not part of this import surface.
"""
from kgsage.preprocessing.loaders import load_kg
from kgsage.preprocessing.registry import resolve_dataset, KNOWN_DATASETS

__all__ = ["load_kg", "resolve_dataset", "KNOWN_DATASETS"]
