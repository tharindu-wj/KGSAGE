#!/bin/bash
# KGSAGE generator training on a GPU node: Phase 1 (Neighbourhood Context
# Encoding) + Phase 2 (Adversarial Generator Training, dual-discriminator
# architecture, LP-free). One job per dataset.
#
# Submit from the KGSAGE repo root:
#   DATASET=fb15k237 sbatch slurm/train.slurm
#   DATASET=wn18rr   sbatch slurm/train.slurm
#   DATASET=yago45   sbatch slurm/train.slurm   # build data/YAGO4.5 first:
#                                               # kgsage/preprocessing/yago_to_tsv.py
#   DATASET=codex-s  sbatch slurm/train.slurm   # CoDEx-S (Wikidata QIDs/PIDs)
#
# Snapshot selection on codex-s needs its own relations -- knockout_eval's
# defaults are Freebase paths that do not exist there:
#   python -m kgsage.cli.knockout_eval --ckpt <snap>.pt --data data/codex-s \
#       --relations P27 P106 P19 P136 P1412 P20
#
# REPO_DIR is the KGSAGE checkout itself (the directory holding kgsage/, data/,
# outputs/ and slurm/). The package is pip-installed, so no PYTHONPATH is set:
#   REPO_DIR=$HOME/kgsage DATASET=wn18rr sbatch slurm/train.slurm
#
# Datasets default to REPO_DIR/data/<NAME>. On a cluster it is usually better to
# keep them on scratch -- export KGSAGE_DATA=/scratch/$USER/kg and KGSAGE_OUTPUTS
# likewise, and both the registry and the eval CLIs follow.
#
# The locked recipe (curriculum, the PI controller on alpha, architecture)
# lives as constants at the top of kgsage/gan/train.py -- edit there to
# re-tune. Only operational knobs are env vars here: SEED, EPOCHS (8),
# SNAPSHOT_EVERY (1), CKPT_PATH. Runs stay SHORT on purpose: anchor-specificity
# peaks a few contradiction-pressure epochs after the alpha ramp then erodes,
# so snapshot every epoch and select the epoch by knockout J@10
# (cli/knockout_eval.py) after the run.

#SBATCH --job-name=kgsage_train
#SBATCH --output=%x-%j.out.txt
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --account=cse

set -eo pipefail

# The KGSAGE checkout -- override for your own path.
REPO_DIR="${REPO_DIR:-${PROJECT_DIR:-$HOME/kgsage}}"
CONDA_ENV="${CONDA_ENV:-$HOME/envs/kgsage}"

DATASET="${DATASET:-fb15k237}"
case "$DATASET" in
  fb15k237) DATA_DIR="${KGSAGE_DATA:-data}/FB15K-237"; TAG="fb15k237" ;;
  wn18rr)   DATA_DIR="${KGSAGE_DATA:-data}/WN18RR";    TAG="wn18rr"   ;;
  yago45)   DATA_DIR="${KGSAGE_DATA:-data}/YAGO4.5";   TAG="yago45"   ;;
  codex-s)  DATA_DIR="${KGSAGE_DATA:-data}/codex-s";   TAG="codex-s"  ;;
  *) echo "ERROR: DATASET must be fb15k237, wn18rr, yago45 or codex-s (got '$DATASET')" >&2; exit 1 ;;
esac

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-8}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1}"
CKPT_PATH="${CKPT_PATH:-${KGSAGE_OUTPUTS:-outputs}/checkpoints/run_${TAG}_s${SEED}.pt}"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Dataset : $DATASET ($DATA_DIR)  Seed: $SEED  Epochs: $EPOCHS  Snapshots: every $SNAPSHOT_EVERY"
echo "Recipe  : locked constants in kgsage/gan/train.py"
echo "Output  : $CKPT_PATH"
echo "----------------------------------------------------------------------"

nvidia-smi || echo "WARNING: nvidia-smi failed"
module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || echo "WARNING: conda activate failed; using absolute python."
PY="$CONDA_ENV/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY missing." >&2; exit 1; }

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
cd "$REPO_DIR"
# No PYTHONPATH: kgsage is pip-installed (`pip install -e .` from this repo).
$PY -c "import kgsage" 2>/dev/null || {
    echo "ERROR: kgsage is not importable in $CONDA_ENV." >&2
    echo "  Install it once:  cd $REPO_DIR && $PY -m pip install -e ." >&2
    exit 1; }

$PY -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('GPU:', torch.cuda.get_device_name(0))"
$PY -c "from torch_geometric.nn import RGCNConv; print('PyG OK')"
mkdir -p "$(dirname "$CKPT_PATH")"

$PY -m kgsage.gan.train \
    --data           "$DATA_DIR" \
    --out            "$CKPT_PATH" \
    --epochs         "$EPOCHS" \
    --snapshot_every "$SNAPSHOT_EVERY" \
    --seed           "$SEED" \
    --device         cuda

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
# corr-pick= is the CORROBORATED-pick fraction (not "corrupted"); alpha= is the
# weight on the neighbourhood penalty, set by the PI controller; dm-online=
# tracks the neighbourhood discriminator. All three are frozen log tokens
# -- keep them so old and new logs stay comparable.
echo "Bring back the checkpoint AND this .out.txt (the corr-pick/alpha/dm-online"
echo "trajectories are the training-health record). Then select the epoch:"
echo "  python -m kgsage.cli.knockout_eval --ckpt <each .epNN.pt> --data $DATA_DIR"
echo ""
echo "Hand the winning snapshot to the detector repo (separate checkout):"
echo "  cp $CKPT_PATH <adkgd>/artifacts/kgsage/"
