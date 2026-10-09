# KGSAGE

![kgsage — corrupting knowledge-graph triples into false-but-plausible negatives](assets/hero-image.svg)

**K**nowledge **G**raph **S**emantic **A**nomaly **GE**nerator — a standalone
Python package that turns real knowledge-graph triples into *corruptions*:
false-but-plausible triples in which exactly one entity slot (head or tail) has
been replaced. The relation is never changed.

Corruptions are useful as hard negatives for training, or as synthetic
anomalies for evaluating knowledge-graph anomaly detectors.

## How it works

An adversarial generator, trained without a link predictor:

1. **Neighbourhood context encoding** — an RGCN is warmed up on link
   prediction; its per-entity output table `E'` is then frozen, alongside Bloom
   membership sketches of each entity's 1–2 hop neighbours.
2. **Adversarial training** — a candidate-scoring generator picks one
   replacement entity per triple (straight-through Gumbel-Softmax) against two
   discriminators: a *plausibility* discriminator ("could this triple be
   real?") and a *neighbourhood* discriminator ("does this filler fit the
   anchor's neighbourhood?"). A PI controller balances the two.
3. **Corruption generation** — the trained checkpoint scores the relation's
   full entity pool and masks out every entity that would make the triple true
   (in any split), so emitted corruptions are guaranteed false.

## Install

```bash
# From this repo root. Editable, so the checkout stays authoritative.
pip install -e .

# Optional extra: only the LLM stage, inference/explain.py, needs it.
pip install -e ".[inference]"
```

The dashboard needs Node.js; `python dashboard/export.py --build` installs
its packages on first use.

On a cluster, install torch and torch-geometric **first**, matching the node's
CUDA build. The dependencies here are deliberately unpinned, so pip leaves an
existing install alone rather than replacing a CUDA wheel with a CPU one.

## Layout

```
<repo root>
  pyproject.toml             packaging; `pip install -e .`
  kgsage/                    THE PACKAGE
    paths.py                 where artifacts and datasets resolve to
    corruption_generation.py generation API (no PyG needed)
    gan/                     training stack: encoder, sketches, sampler,
                             generator, both discriminators, train.py
    cli/                     command-line tools (snapshot selection)
    preprocessing/           CODE: TSV loaders, dataset registry, converters
  inference/                 corrupt a dataset with a checkpoint; LLM reasoning
  dashboard/                 the case-by-case review app (React)
  data/                      DATA: the dataset directories themselves
  outputs/                   run artifacts: checkpoints, inference runs,
                             dashboard data (gitignored)
  slurm/                     HPC launchers (see slurm/README.md)
  CONTRACTS.md               the files between inference/ and dashboard/
  check_boundaries.py        proves the components stay separate
  smoke_test.py              import + structure check
```

`inference/` and `dashboard/` are applications over the package, not part
of it: `pip install` ships only `kgsage/`, and nothing in `kgsage/` imports
them. Each has its own README; [CONTRACTS.md](CONTRACTS.md) is the whole
interface between them.

`preprocessing/` is the code that reads knowledge graphs; `data/` holds the
graphs. (Before, both were `data/` — hence the split.)

Two environment variables relocate the two roots, which is how the SLURM
launchers keep large files off `$HOME`:

| Variable | Default | What moves |
|---|---|---|
| `KGSAGE_DATA` | `<repo root>/data` | where the registry looks for `<name>/train.txt` |
| `KGSAGE_OUTPUTS` | `<repo root>/outputs` | checkpoints and eval output |

Set `KGSAGE_OUTPUTS` after a **non-editable** install, or artifacts resolve into
site-packages.

## Usage

The package is installed, so these run from anywhere. Paths below assume you are
at the repo root.

```bash
# 0. verify the install (imports + loads the tiny data/dummy_kg fixture)
python smoke_test.py

# 1. train (saves one snapshot per epoch)
python -m kgsage.gan.train --data data/FB15K-237 \
    --out outputs/checkpoints/run.pt --epochs 8 --snapshot_every 1

# 2. pick the best snapshot: lowest mean knockout J@10
python -m kgsage.cli.knockout_eval --ckpt <snapshot>.pt --data data/FB15K-237

# 3. corrupt 5% of a dataset: the corrupted triple list + a record of every
#    corruption with its neighbourhood (outputs/inference/<run>/)
python inference/corrupt.py --ckpt <chosen>.pt --data codex-s --ratio 0.05 --seed 0

# 4. an LLM judges each corruption against KGSAGE's three goals
python inference/explain.py --run codex-s_r0.05_s0

# 5. review them, case by case
python dashboard/export.py --build --single
```

Steps 3 to 5 are built and tested on CoDEx-S, the one dataset with the
definition files (labels, descriptions, kinds) they read; see
[inference/README.md](inference/README.md) and
[dashboard/README.md](dashboard/README.md).

Registered datasets can be named instead of pathed: `--data fb15k237` resolves
through `kgsage/preprocessing/registry.py`.

### Handing a checkpoint to a detector

This package produces checkpoints; it never runs a detector. To use one
downstream, copy it across and point the detector at it:

```bash
cp outputs/checkpoints/run_wn18rr_s0.ep06.pt <detector-repo>/artifacts/kgsage/
```

[**ADKGD**](https://github.com/tharindu-wj/ADKGD) is the downstream anomaly
detector KGSAGE is tested against — it consumes these checkpoints from
`artifacts/kgsage/` and scores the emitted corruptions as anomalies. It is the
reference consumer of the contract described below, and a worked example of
what the other side of that `cp` looks like.

A detector that wants a frozen anomaly column instead of a checkpoint takes
`outputs/inference/<run>/corruptions.tsv`: three tab-separated ids per line,
the format KGMVAD's `inject --source frozen` and ADKGD's `--anomaly_file`
read.

## Use as a library

The package is import-safe and detector-agnostic: nothing in `kgsage/` knows
about your model. You give it triples, it gives you corruptions.

### Quick start

```python
import numpy as np
from kgsage import load_kg, load_checkpoint, generate_negatives

kg      = load_kg("kgsage/data/WN18RR")
payload = load_checkpoint("kgsage/outputs/checkpoints/run_wn18rr_s0.pt")

corruptions, stats = generate_negatives(
    kg["triples_train"][:1000],   # triples in YOUR id space
    payload,                      # the loaded checkpoint
    kg,                           # id_maps: supplies ent2id/rel2id/id2ent/id2rel
    rng=np.random.default_rng(7),
)

bad = set(stats["null_indices"])
usable = [c for i, c in enumerate(corruptions) if i not in bad]
```

### The stable API

Importing `kgsage` costs nothing — the torch-backed symbols are lazy, so
`import kgsage` and every registry lookup work on a machine with no torch
installed.

| symbol | needs torch | what it does |
|---|---|---|
| `load_kg(dir)` | no | read `train/valid/test.txt` → triples + vocab + edge list |
| `resolve_dataset(name_or_path)` | no | short name or path → `{name, path, n_relations}` |
| `KNOWN_DATASETS` | no | the registry dict |
| `load_checkpoint(path, device=None)` | yes | `.pt` → generator payload (no PyG needed) |
| `generate_negatives(triples, payload, id_maps, …)` | yes | → `(corruptions, stats)` |
| `CandidateScoringGenerator`, `NeighbourhoodContextEncoder` | yes | the models, if you are training your own |

`render_stats(stats)` is one import deeper — `from kgsage.corruption_generation
import render_stats` — and returns the frozen one-line log summary.

### What `load_kg` returns

A plain dict, no custom classes:

```
ent2id, rel2id        str -> int   (train-first, first-seen ordering)
id2ent, id2rel        int -> str
triples_train/valid/test   list[(h, r, t)] as ints
triple_set_train      set — collision checks
triple_set_all        set — the all-splits falseness filter
edge_index, edge_type directed train-graph edge list (plain lists, for the RGCN)
n_ent, n_rel          vocab sizes
```

`valid.txt` and `test.txt` are optional; missing splits come back empty.

### The `id_maps` contract

This is the one thing that bites. **Your** integer ids and the **checkpoint's**
integer ids are different numberings of the same entities — a checkpoint
trained elsewhere has its own vocabulary. `generate_negatives` translates
between them through entity *strings*, which is what `id_maps` is for: a dict
carrying your `ent2id`, `rel2id`, `id2ent`, `id2rel`.

Passing the `load_kg` dict straight in works, because it already has those four
keys. If your triples come from somewhere else, hand-build it:

```python
id_maps = {"ent2id": ..., "rel2id": ..., "id2ent": ..., "id2rel": ...}
```

Returned corruptions are in **your** id space, aligned index-for-index with the
triples you passed in. Every entity string in your vocabulary must also exist in
the checkpoint's, or the lookup raises `KeyError`.

### Guarantees, and the one you must handle

For every non-null row: the relation is unchanged, exactly one entity slot
differs, and the result is not a real triple in *any* split of the training KG.
Same `rng` seed → same corruptions.

The exception is **null corruptions**. When every candidate is masked out and
`max_resample` redraws all fail, KGSAGE emits the *original, true* triple rather
than inventing a random fallback, and records its position:

```python
corruptions, stats = generate_negatives(...)
for i in stats["null_indices"]:
    ...  # corruptions[i] is a REAL fact — drop or replace it
```

Training on those rows teaches your detector that a true triple is an anomaly.
There is no silent fallback by design — the failure is reported, not hidden.

Useful `stats` keys: `processed`, `used_original` (= `len(null_indices)`),
`resampled`, `type_valid`, `slot_h` / `slot_t` (head/tail split; `slot_r` is
always 0 — relations are never corrupted), `corroboration_lifted`.

### Options worth knowing

```python
generate_negatives(triples, payload, id_maps,
                   rng=np.random.default_rng(7),  # reproducibility
                   max_resample=8,                # redraws before a null
                   support_max=None,              # corroboration mask: off
                   guards=None,                   # eval-column guards: off
                   slot=None)                     # head or tail, 50/50
```

`slot="head"` or `slot="tail"` corrupts that slot on every row. The 50/50
coin is drawn either way, so the same seed draws the same sources and the
same generator noise; only the slot differs.

`support_max` is the hard one. Left at `None` the generator only guarantees
falseness. Set it to an integer `>= 0` and every candidate the anchor's
neighbourhood *corroborates* — direct neighbours, or entities sharing more than
`support_max` neighbours — becomes unpickable, so each corruption contradicts
the neighbourhood by construction. Use it when you want semantically hard
negatives rather than merely false ones. Rows where that leaves nothing
pickable lift the mask and are counted in `stats["corroboration_lifted"]`.

### Vocabulary

Inside this package the emitted object is always a **corruption**. Detectors
call the same object a *negative* while training and an *anomaly* while
evaluating — that seam is deliberate, and it is why the entry point is named
`generate_negatives` even though nothing here says "negative".

## Datasets

Drop `train.txt` / `valid.txt` / `test.txt` (tab-separated
`head<TAB>relation<TAB>tail`) into `kgsage/data/<NAME>/` and pass that path to
`--data`. Short names registered in
[preprocessing/registry.py](kgsage/preprocessing/registry.py): `fb15k237`, `wn18rr`,
`yago45`, `fb15k_mini`, `dummy_kg`, `codex-s`.

`data/codex-s/` also carries the CoDEx release's four definition files
(entity labels and descriptions, relation labels, entity kinds, kind
labels). The package never reads them; `inference/` does, and so does the
generator's kind guard through `--types`.

Registry paths are relative to the directory you run from — the one that
*contains* `kgsage/`. Datasets are gitignored; see [data/README.md](data/README.md).

Checkpoints and evaluation output land under `outputs/` — keep it out of git.
