#!/usr/bin/env bash
# Train the baseline or IR-augmented YOLOv12 detector.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/train_detector.sh MODE DATA_YAML INITIAL_WEIGHTS [RUN_NAME]

MODE is one of: baseline, ir

Optional environment variables:
  PYTHON_BIN, DEVICE=0, EPOCHS=10, BATCH=32, NBS=$BATCH,
  IMGSZ=1280, WORKERS=8, LR=1e-4, PROJECT=runs/train,
  IR_SINE_P=0.1 (IR mode only), WANDB_OFFLINE=0
EOF
}

[[ $# -ge 3 && $# -le 4 ]] || { usage >&2; exit 2; }
MODE="$1"
DATA_YAML="$2"
INITIAL_WEIGHTS="$3"
RUN_NAME="${4:-$MODE}"

case "$MODE" in
  baseline) IR_PROBABILITY=0 ;;
  ir) IR_PROBABILITY="${IR_SINE_P:-0.1}" ;;
  *) echo "MODE must be 'baseline' or 'ir', got: $MODE" >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3 || command -v python)"
[[ -f "$DATA_YAML" ]] || { echo "Dataset YAML not found: $DATA_YAML" >&2; exit 1; }
[[ -f "$INITIAL_WEIGHTS" ]] || { echo "Checkpoint not found: $INITIAL_WEIGHTS" >&2; exit 1; }

DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-32}"
NBS="${NBS:-$BATCH}"
IMGSZ="${IMGSZ:-1280}"
WORKERS="${WORKERS:-8}"
LR="${LR:-0.0001}"
PROJECT="${PROJECT:-$ROOT/runs/train}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
command=(
  "$PYTHON_BIN" "$ROOT/train.py"
  --data "$DATA_YAML"
  --model "$ROOT/ultralytics/cfg/models/12/yolo12-UAV.yaml"
  --weights "$INITIAL_WEIGHTS"
  --epochs "$EPOCHS" --batch "$BATCH" --nbs "$NBS"
  --imgsz "$IMGSZ" --device "$DEVICE" --workers "$WORKERS"
  --optimizer AdamW --lr0 "$LR" --momentum 0.9
  --warmup-epochs 1 --warmup-bias-lr "$LR" --cos-lr
  --ir-sine-p "$IR_PROBABILITY" --no-multi-scale
  --project "$PROJECT" --name "$RUN_NAME"
)

if [[ "${WANDB_OFFLINE:-0}" == 1 ]]; then
  command+=(--wandb-offline --wandb-dir "${WANDB_DIR:-$ROOT/wandb-offline}")
fi

printf 'Training mode=%s, IR probability=%s, output=%s/%s\n' "$MODE" "$IR_PROBABILITY" "$PROJECT" "$RUN_NAME"
printf 'Command:'; printf ' %q' "${command[@]}"; printf '\n'
"${command[@]}"
