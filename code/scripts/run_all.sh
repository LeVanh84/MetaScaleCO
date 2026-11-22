#!/usr/bin/env bash
# Full pipeline
set -euo pipefail
cd "$(dirname "$0")/.."
RESDIR="results${METASCALE_TAG:+_$METASCALE_TAG}"
mkdir -p "$RESDIR"
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(uname -srm)"
  echo "config: ${METASCALE_CONFIG:-config.json}"
  uv run python -c "import sys, numpy, numba, torch; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| numba', numba.__version__, '| torch', torch.__version__)"
} > "$RESDIR/provenance.txt"
uv pip freeze > "$RESDIR/pip_freeze.txt" 2>/dev/null || true

step() { echo; echo "=== $1"; shift; uv run python -m "$@"; }
step "1/8 datasets, LKH references, C0"      metascale_co.data.datasets
step "2/8 trajectory bank (Sec. 6.2)"        metascale_co.training.collect_rollouts
step "3/8 VoC ensembles (Sec. 6.3)"          metascale_co.training.train_voc
step "4/8 tuning on validation"              metascale_co.training.tune
step "5/8 benchmark + anytime + stopping"    metascale_co.evaluation.budget_sweep
step "6/8 batch allocation (Sec. 5.7)"       metascale_co.evaluation.batch_eval
echo; echo "=== 7/8 statistics"
uv run python -m metascale_co.evaluation.stats | tee "$RESDIR/stats_summary.txt"
step "8/8 figures"                           metascale_co.evaluation.plots
