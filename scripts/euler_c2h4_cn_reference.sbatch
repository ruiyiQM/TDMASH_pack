#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
set -eo pipefail
module load stack/2024-06 gcc/12.2.0 python/3.10.13
set -u
ROOT="${1:?Scratch study directory is required}"
MODE="${2:?Job mode is required}"
PY="${TDMASH_PYTHON:?Set TDMASH_PYTHON to a Python executable available on compute nodes}"
export PYTHONPATH="$ROOT/code/src:$ROOT/code/scripts:$ROOT/code/tests"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tdmash-c2cn-${SLURM_JOB_ID}"
cd "$ROOT"
sha256sum --check SHA256SUMS > /dev/null
SCRIPT="$ROOT/code/scripts/euler_c2h4_cn_reference.py"
if [ "$MODE" = gate ]; then
    srun --cpu-bind=cores "$PY" -B -m unittest discover -s "$ROOT/code/tests" -v
    srun --cpu-bind=cores "$PY" -B "$SCRIPT" self-check
    srun --cpu-bind=cores "$PY" -B "$ROOT/code/scripts/benchmark_c2h4_cn_pt.py" --self-check
    srun --cpu-bind=cores "$PY" -B "$ROOT/code/scripts/euler_c2h4_ehrenfest.py" initialize --root "$ROOT"
elif [ "$MODE" = run ]; then
    srun --cpu-bind=cores "$PY" -B -u "$SCRIPT" run --root "$ROOT" --case "$SLURM_ARRAY_TASK_ID"
elif [ "$MODE" = analyze ]; then
    srun --cpu-bind=cores "$PY" -B -u "$SCRIPT" analyze --root "$ROOT"
elif [ "$MODE" = timing ]; then
    srun --cpu-bind=cores "$PY" -B -u "$ROOT/code/scripts/benchmark_c2h4_cn_pt.py" --root "$ROOT"
else
    echo "Unknown job mode: $MODE" >&2
    exit 2
fi
