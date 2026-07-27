# kgsage

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
pip install torch numpy
pip install torch_geometric        # step 1 only (RGCN encoder)
pip install matplotlib networkx    # optional, for the ego-graph figures
```

## Layout

```
kgsage/
  corruption_generation.py   generation API (no PyG needed)
  gan/                       training stack: encoder, sketches, sampler,
                             generator, both discriminators, train.py
  cli/                       command-line tools
  data/                      TSV loaders + dataset registry
  slurm/                     HPC launchers (see slurm/README.md)
  smoke_test.py              import + structure check
```

## Usage

Run all commands from the directory that **contains** `kgsage/`.

```bash
# 1. train (saves one snapshot per epoch)
python -m kgsage.gan.train --data data/FB15K-237 \
    --out kgsage/outputs/checkpoints/run.pt --epochs 8 --snapshot_every 1

# 2. pick the best snapshot: lowest mean knockout J@10
python -m kgsage.cli.knockout_eval --ckpt <snapshot>.pt --data data/FB15K-237

# 3. generate corruptions to CSV
python -m kgsage.cli.gen_corruptions_csv --ckpt <chosen>.pt \
    --data data/FB15K-237 --split test --per_rel 4 --seed 7
```

Helpers that read that CSV: `kgsage.cli.ego_from_csv` (ego-graph figures),
`kgsage.cli.format_for_llm` and `kgsage.cli.gen_neighbourhood_context`
(paste-ready blocks for LLM evaluation).

From Python:

```python
from kgsage import load_kg, load_checkpoint, generate_negatives

kg = load_kg("data/FB15K-237")
gen = load_checkpoint("kgsage/outputs/checkpoints/run.pt")
corruptions, stats = generate_negatives(gen, triples, ...)
```

Rows the generator could not corrupt are flagged in `stats["null_indices"]` —
skip them, never train on them.

## Datasets

Drop `train.txt` / `valid.txt` / `test.txt` (tab-separated
`head<TAB>relation<TAB>tail`) into `data/<NAME>/` and pass that path to
`--data`. Short names registered in [data/datasets.py](data/datasets.py):
`fb15k237`, `wn18rr`, `yago45`, `fb15k_mini`, `dummy_kg`.

Checkpoints and evaluation output land under `outputs/` — keep it out of git.
