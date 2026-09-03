#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIPELINE=${PIPELINE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
BASELINES=${BASELINE_ROOT:-$PIPELINE/Baselines}
DATASET=${DATASET_PATH:-$PIPELINE/dataset/PhishTank.csv}
RESULTS=${RESULTS_ROOT:-$PIPELINE/results}
ROOT=${OUTPUT_ROOT:-$RESULTS/resource_benchmark/concurrent}
RUNNER=$SCRIPT_DIR/benchmark_model_resources.py
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

launch() {
  local model=$1
  local mode=$2
  local sample=$3
  local batch=$4
  local repeats=$5
  local warmup=$6
  local artifact
  artifact=$(artifact_for "$model")
  local stem="${model}_${mode}"
  (
    systemd-run --user --scope -p CPUQuota=100% -p MemoryMax=512M \
      /usr/bin/time -v -o "$ROOT/${stem}.time" \
      env PYTHONHASHSEED=42 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$CONDA_COMMAND" run --no-capture-output -n "$CONDA_ENVIRONMENT" \
      python -u "$RUNNER" "$model" "$artifact" "$DATASET" "$ROOT/${stem}.json" \
      --pipeline-root "$PIPELINE" --baseline-root "$BASELINES" \
      --sample-size "$sample" --batch-size "$batch" --repeats "$repeats" \
      --warmup-batches "$warmup" --profile-name "common_1cpu_512MiB_${mode}" \
      --declared-cpu-limit 1 --declared-memory-limit-mib 512 \
      > "$ROOT/${stem}.log" 2>&1
    echo $? > "$ROOT/${stem}.rc"
  ) &
  echo $! > "$ROOT/${stem}.pid"
}

for model in Ebbu E2Phish MUDS TabNet LitePhish; do
  launch "$model" online 1000 1 3 20
  launch "$model" batched 5000 100 5 2
done

wait
date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/launcher_complete_utc.txt"
