#!/usr/bin/env bash
# Evaluate the original checkpoint and the four blur-study models on all three cohorts at the
# study severity (31 px at 846 px reference width), then write the summary tables.
#
# Usage: eval/run_study_eval.sh [model ...]   (default: original clean fixed variable clean_or_heavy)
# Env:   CHECKPOINT_DIR (default checkpoint/blur_study), ORIGINAL (reproduction checkpoint),
#        COHORTS (default "source_val sintel eth3d"), DEVICE (default cuda), CUDA_VISIBLE_DEVICES,
#        RESULTS_ROOT (default outputs/blur_study), PYTHON (default python),
#        SCENES_<cohort> (optional space-separated scene subset, e.g. SCENES_sintel="alley_1")
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${EVAL_DIR}/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO}/checkpoint/blur_study}"
ORIGINAL="${ORIGINAL:-${REPO}/checkpoint/vggt_omega_1b_416_reproduce.pt}"
COHORTS=(${COHORTS:-source_val sintel eth3d})
DEVICE="${DEVICE:-cuda}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO}/outputs/blur_study}"
PYTHON="${PYTHON:-python}"
MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(original clean fixed variable clean_or_heavy)
fi

for model in "${MODELS[@]}"; do
  if [[ "${model}" == original ]]; then ckpt="${ORIGINAL}"; else ckpt="${CHECKPOINT_DIR}/${model}.pt"; fi
  for cohort in "${COHORTS[@]}"; do
    data_root="${REPO}/data/${cohort}"
    subset_var="SCENES_${cohort}"
    scenes=()
    if [[ -n "${!subset_var:-}" ]]; then read -r -a scenes <<< "${!subset_var}"; scenes=(--scenes "${scenes[@]}"); fi
    echo "== ${model} on ${cohort}"
    "${PYTHON}" "${EVAL_DIR}/motion_blur.py" --dataset "${cohort}" --data-root "${data_root}" \
      --checkpoint "${ckpt}" --model-name "${model}" --blur-px 31 --reference-width 846 \
      --output-dir "${RESULTS_ROOT}/${cohort}/${model}" --device "${DEVICE}" --overwrite "${scenes[@]}"
  done
done

"${PYTHON}" "${EVAL_DIR}/summarize_study.py" --results-root "${RESULTS_ROOT}"
