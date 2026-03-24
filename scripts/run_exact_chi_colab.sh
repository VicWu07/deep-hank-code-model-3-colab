#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_exact_chi_colab.sh --chi <float> [options]

Options:
  --chi <float>                  Required. MRS wage scale parameter.
  --train-epochs <int>           Default: 5000
  --checkpoint-every <int>       Default: 500
  --output-root <path>           Default: /content/drive/MyDrive/deep_hank_runs
  --label <text>                 Optional suffix for output folder/log file (e.g., wide).
  --python <executable>          Default: python3
  --no-log                       Disable tee to run_logs/*.log
  --dry-run                      Print resolved command and exit
  -h, --help                     Show this help

Example:
  ./scripts/run_exact_chi_colab.sh \
    --chi 1.5 \
    --train-epochs 5000 \
    --checkpoint-every 250 \
    --output-root /content/drive/MyDrive/deep_hank_runs \
    --label wide
EOF
}

CHI=""
TRAIN_EPOCHS=5000
CHECKPOINT_EVERY=500
OUTPUT_ROOT="/content/drive/MyDrive/deep_hank_runs"
LABEL=""
PYTHON_BIN="python3"
ENABLE_LOG=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chi)
      CHI="${2:-}"; shift 2 ;;
    --train-epochs)
      TRAIN_EPOCHS="${2:-}"; shift 2 ;;
    --checkpoint-every)
      CHECKPOINT_EVERY="${2:-}"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --label)
      LABEL="${2:-}"; shift 2 ;;
    --python)
      PYTHON_BIN="${2:-}"; shift 2 ;;
    --no-log)
      ENABLE_LOG=0; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "$CHI" ]]; then
  echo "Error: --chi is required." >&2
  usage
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CHI_TAG="$("$PYTHON_BIN" - "$CHI" <<'PY'
import sys
chi = float(sys.argv[1])
print(f"{chi:.3f}".replace("-", "m").replace(".", "p"))
PY
)"

RUN_NAME="exact_chi_${CHI_TAG}"
if [[ -n "$LABEL" ]]; then
  RUN_NAME="${RUN_NAME}_${LABEL}"
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}_outputs"
mkdir -p "$OUTPUT_DIR"
mkdir -p run_logs

CMD=(
  "$PYTHON_BIN" "run_hank_mrs_optimized_exact_chi.py"
  "--chi" "$CHI"
  "--train-epochs" "$TRAIN_EPOCHS"
  "--checkpoint-every" "$CHECKPOINT_EVERY"
  "--output-dir" "$OUTPUT_DIR"
)

echo "Project dir : $PROJECT_DIR"
echo "Run name    : $RUN_NAME"
echo "Output dir  : $OUTPUT_DIR"
echo "Command     : ${CMD[*]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run enabled; exiting before training."
  exit 0
fi

if [[ "$ENABLE_LOG" -eq 1 ]]; then
  LOG_FILE="run_logs/${RUN_NAME}.log"
  echo "Log file    : $LOG_FILE"
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
  "${CMD[@]}"
fi
