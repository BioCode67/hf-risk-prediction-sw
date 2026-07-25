#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# A6000 real-measurement runner for the early-warning GPU track.
#
# This laptop has no GPU, so the GRU/tuning paths were only validated on CPU
# (via automatic fallback). Run THIS on the NVIDIA A6000 dev server to get the
# real CUDA numbers and wall-times. It:
#   1. reports the GPU + torch/CUDA build,
#   2. trains the GRU sequence model at scale on the GPU (vitals_seq --gpu),
#   3. runs the tabular XGBoost with Optuna tuning on the GPU (vitals_train
#      --tune --gpu),
#   4. tees everything to a timestamped log under models/.
#
# Usage:
#   bash scripts/run_a6000.sh                 # defaults (3000 patients, 60 epochs, 60 trials)
#   PATIENTS=6000 EPOCHS=100 TRIALS=100 bash scripts/run_a6000.sh
#
# Optional SLURM submission (uncomment and `sbatch scripts/run_a6000.sh`):
# #SBATCH --job-name=ews-a6000
# #SBATCH --gres=gpu:a6000:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=32G
# #SBATCH --time=02:00:00
# #SBATCH --output=ews-a6000-%j.log
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve repo root from this script's location (works from any cwd).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PATIENTS="${PATIENTS:-3000}"
EPOCHS="${EPOCHS:-60}"
TRIALS="${TRIALS:-60}"
PY="${PYTHON:-python}"

mkdir -p models
STAMP="$("$PY" -c 'import datetime; print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))')"
LOG="models/a6000-run-${STAMP}.log"

{
  echo "=== A6000 early-warning run @ ${STAMP} ==="
  echo "patients=${PATIENTS} epochs=${EPOCHS} trials=${TRIALS}"
  echo

  echo "### GPU / environment"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "(nvidia-smi not found)"
  "$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "| CUDA build:", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PYEOF
  echo

  echo "### 1) GRU sequence model on GPU (scaled)"
  time "$PY" src/vitals_seq.py --gpu --patients "${PATIENTS}" --epochs "${EPOCHS}"
  echo

  echo "### 2) Tabular XGBoost + Optuna tuning on GPU"
  time "$PY" src/vitals_train.py --gpu --tune --trials "${TRIALS}"
  echo

  echo "=== done; artifacts in models/ (figures, vitals_ews_model.pkl, this log) ==="
} 2>&1 | tee "$LOG"

echo "Log saved to: ${LOG}"
