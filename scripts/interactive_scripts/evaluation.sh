#!/usr/bin/env bash


ROOT_DIR="/"
NUM_WORKERS=22
PARALLEL_ARGS=()
#PARALLEL_ARGS=(--parallel)


# ---------------------------------------------------------
# Paths and evaluation settings
# ---------------------------------------------------------

EVAL_SCRIPT="src/evaluation.py"
DATASET="mp20"
OUT_DIR="$ROOT_DIR/evaluation_results"


# Specify the run directories to evaluate. If empty, all runs in the logs directory will be evaluated.
RUN_DIRS=()

echo "Root dir: $ROOT_DIR"
echo "Output dir: $OUT_DIR"
echo "CPU workers: $NUM_WORKERS"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"


# ---------------------------------------------------------
# Offline evaluation (CPU)
# ---------------------------------------------------------

python3 "$EVAL_SCRIPT" \
  --global_path "$ROOT_DIR" \
  --out_dir "$OUT_DIR" \
  --run_dirs "${RUN_DIRS[@]}" \
  --dataset "$DATASET" \
  \
  --stol 0.3 \
  --angle_tol 10.0 \
  --ltol 0.2 \
  \
  "${PARALLEL_ARGS[@]}" \
  --num_workers "$NUM_WORKERS" \
  --chunksize 10 \
  \
  --novel_sample_k 1000 \
  --seed 0 \
  --cell_representation conventional \
  --skip_structure_reduction \
  \
  "$@"