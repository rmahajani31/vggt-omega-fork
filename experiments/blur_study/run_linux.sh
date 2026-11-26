#!/usr/bin/env bash
# Train the four blur-study recipes on Linux (NCCL), one independent single-GPU run per recipe.
# With two or more GPUs, two recipes run at a time; with one GPU they run back to back.
#
# Usage: experiments/blur_study/run_linux.sh [recipe ...]   (default: all four)
# Env:   BLUR_STUDY_DATA, BLUR_STUDY_DIR, BLUR_STUDY_CHECKPOINT (see training/config/blur_study.yaml)
#        NUM_WORKERS (default 8), EXTRA_OVERRIDES (extra hydra overrides)
set -euo pipefail

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${STUDY_DIR}/../.." && pwd)"
export OPENCV_IO_ENABLE_OPENEXR=1
export BLUR_STUDY_DIR="${BLUR_STUDY_DIR:-${STUDY_DIR}}"
export BLUR_STUDY_DATA="${BLUR_STUDY_DATA:-${REPO}/data/unified}"
export BLUR_STUDY_CHECKPOINT="${BLUR_STUDY_CHECKPOINT:-${REPO}/checkpoint/vggt_omega_1b_416_reproduce.pt}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ ! -f "${BLUR_STUDY_DIR}/schedule.json" ]]; then
  echo "Missing ${BLUR_STUDY_DIR}/schedule.json; run experiments/blur_study/build_schedule.py first" >&2
  exit 1
fi

RECIPES=("$@")
if [[ ${#RECIPES[@]} -eq 0 ]]; then
  RECIPES=(clean fixed variable clean_or_heavy)
fi
NUM_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
PARALLEL=$(( NUM_GPUS >= 2 ? 2 : 1 ))
echo "GPUs: ${NUM_GPUS}; running ${PARALLEL} recipe(s) at a time: ${RECIPES[*]}"

run_recipe() {
  local recipe="$1" gpu="$2"
  local log="${REPO}/training/logs/blur_study_${recipe}"
  mkdir -p "${log}"
  echo "[$(date -Is)] start ${recipe} on GPU ${gpu}"
  (cd "${REPO}/training" && CUDA_VISIBLE_DEVICES="${gpu}" torchrun \
      --standalone --nproc_per_node=1 --master_port=$((29500 + gpu)) \
      launch.py --config blur_study recipe="${recipe}" \
      data.train.num_workers="${NUM_WORKERS}" ${EXTRA_OVERRIDES:-}) \
      > "${log}/stdout.log" 2>&1
  echo "[$(date -Is)] done ${recipe} (exit $?)"
}

i=0
while [[ ${i} -lt ${#RECIPES[@]} ]]; do
  pids=()
  for (( slot=0; slot<PARALLEL && i<${#RECIPES[@]}; slot++, i++ )); do
    run_recipe "${RECIPES[$i]}" "${slot}" &
    pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  if [[ ${status} -ne 0 ]]; then
    echo "A recipe failed (exit ${status}); see training/logs/blur_study_*/stdout.log" >&2
    exit "${status}"
  fi
done

python "${STUDY_DIR}/export_merged.py" --recipes "${RECIPES[@]}"
