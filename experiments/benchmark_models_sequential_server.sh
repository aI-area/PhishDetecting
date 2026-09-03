#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIPELINE=${PIPELINE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
BASELINES=${BASELINE_ROOT:-$PIPELINE/Baselines}
DATASET=${DATASET_PATH:-$PIPELINE/dataset/PhishTank.csv}
RESULTS=${RESULTS_ROOT:-$PIPELINE/results}
ROOT=${OUTPUT_ROOT:-$RESULTS/resource_benchmark/sequential}
RUNNER=$SCRIPT_DIR/benchmark_model_resources.py
CPU_CORE=${CPU_CORE:-0}
CONDA_COMMAND=${CONDA_COMMAND:-conda}
CONDA_ENVIRONMENT=${CONDA_ENVIRONMENT:-phishing}
mkdir -p "$ROOT"

artifact_for() {
  case "$1" in
    MUDS) echo "${MUDS_ARTIFACT_DIR:-$RESULTS/models/MUDS}" ;;
    E2Phish) echo "${E2PHISH_ARTIFACT_DIR:-$RESULTS/models/E2Phish}" ;;
    Ebbu) echo "${EBBU_ARTIFACT_DIR:-$RESULTS/models/Ebbu}" ;;
    TabNet) echo "${TABNET_ARTIFACT_DIR:-$RESULTS/models/TabNet}" ;;
    LitePhish) echo "${LITEPHISH_ARTIFACT_DIR:-$PIPELINE/artifacts}" ;;
  esac
}

run_one() {
  local model=$1 mode=$2 sample=$3 batch=$4 repeats=$5 warmup=$6
  local artifact stem rc
  artifact=$(artifact_for "$model")
  stem="${model}_${mode}"
  systemd-run --user --scope -p CPUQuota=100% -p MemoryMax=512M \
    /usr/bin/time -v -o "$ROOT/${stem}.time" \
    taskset -c "$CPU_CORE" env PYTHONHASHSEED=42 OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$CONDA_COMMAND" run --no-capture-output -n "$CONDA_ENVIRONMENT" \
    python -u "$RUNNER" "$model" "$artifact" "$DATASET" "$ROOT/${stem}.json" \
    --pipeline-root "$PIPELINE" --baseline-root "$BASELINES" \
    --sample-size "$sample" --batch-size "$batch" --repeats "$repeats" \
    --warmup-batches "$warmup" --profile-name "common_pinned_1cpu_512MiB_${mode}" \
    --declared-cpu-limit 1 --declared-memory-limit-mib 512 \
    > "$ROOT/${stem}.log" 2>&1
  rc=$?
  echo "$rc" > "$ROOT/${stem}.rc"
  printf '%s %s rc=%s %s\n' "$model" "$mode" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ROOT/progress.log"
  return "$rc"
}

# Interleave models so neither profile systematically benefits from run order.
for model in LitePhish MUDS E2Phish TabNet Ebbu; do
  run_one "$model" online 1000 1 3 20 || exit $?
done
for model in Ebbu TabNet E2Phish MUDS LitePhish; do
  run_one "$model" batched 5000 100 5 2 || exit $?
done

date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/launcher_complete_utc.txt"
