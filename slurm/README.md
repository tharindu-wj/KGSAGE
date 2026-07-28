# KGSAGE SLURM scripts

HPC launchers for the generator side — they run Phase 1 (Neighbourhood Context
Encoding) and Phase 2 (Adversarial Generator Training) in one job. They live
inside `kgsage/` rather than in a host project's own `slurm/` directory because
KGSAGE is package-portable: copy the package anywhere and its launchers come
with it.

| script | what it does |
|---|---|
| `train.slurm` | trains the generator on a GPU node (`kgsage.gan.train`). `DATASET=fb15k237\|wn18rr\|yago45 SEED=n sbatch …` — derives the data dir from `DATASET`; saves a snapshot every adversarial epoch |
| `train_cpu.slurm` | CPU-node variant that skips the Phase-1 RGCN warm-up when a cached-E' donor exists (`--init_context_from`), and otherwise warms up on CPU; pass the partition on the `sbatch` command line |

## Before you submit

Both scripts assume a project directory that **contains** `kgsage/` — that
directory goes on `PYTHONPATH`, and every path in the scripts is relative to
it. Two env vars set it up:

| var | default | meaning |
|---|---|---|
| `PROJECT_DIR` | `$HOME/kgsage-project` | the directory holding `kgsage/` |
| `CONDA_ENV` | `$HOME/envs/kgsage` | conda env with torch (+ PyG for a fresh Phase-1 warm-up) |

```bash
PROJECT_DIR=$HOME/my-kg-work DATASET=wn18rr SEED=0 sbatch kgsage/slurm/train.slurm
```

Other knobs: `SEED`, `EPOCHS` (8), `SNAPSHOT_EVERY` (1), `CKPT_PATH`, and
`INIT_E` on the CPU variant. The locked training recipe is **not** here — it
lives as constants at the top of `kgsage/gan/train.py`.

## Frozen literals

The `DATASET` values (`fb15k237`/`wn18rr`/`yago45`) and the dataset directories
they select (`kgsage/data/FB15K-237`/`WN18RR`/`YAGO4.5`) are shared verbatim
with the dataset registry
([../preprocessing/registry.py](../preprocessing/registry.py)) — change the
`case` labels and lookups break. The scripts still pass `--data` explicitly, so
they never import the registry; the two just have to agree.

## After the job

Select the snapshot with the lowest mean knockout J@10 — the most
anchor-specific generator:

```bash
python -m kgsage.cli.knockout_eval --ckpt <each .epNN.pt> --data kgsage/data/<NAME>
```

Promote the winner to `generator_<dataset>.pt`, then feed it to whatever
consumes corruptions downstream.

Keep each job's `.out.txt`: the `corr-pick=` / `alpha=` / `D-acc=` /
`dm-online=` trajectories are the training-health record. Those log tokens are
frozen for comparability with every run already collected — `corr-pick=` is the
**corroborated**-pick fraction (not "corrupted"), `D-acc=` is the plausibility
discriminator's accuracy on real versus generated triples, and `dm-online=` /
`g_match=` belong to the neighbourhood discriminator (the BCE of its online
update, and its mean fit score for the candidates the generator picked).
