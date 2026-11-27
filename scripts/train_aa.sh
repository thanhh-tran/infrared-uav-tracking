#!/usr/bin/env bash
# Train the AA detector in two independent optimizer phases.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/train_aa.sh DATA_YAML BASE_DETECTOR_WEIGHTS [PROJECT]

Phase 1: base LR 1e-6, AA-head LR 1e-3, 10 epochs.
Phase 2: load phase-1 best weights into a fresh optimizer, LR 1e-4, 10 epochs.

Optional environment variables:
  PYTHON_BIN, DEVICE=0, BATCH=32, NBS=$BATCH, IMGSZ=1280,
  WORKERS=8, PHASE1_EPOCHS=10, PHASE2_EPOCHS=10,
  PHASE1_NAME=aa_warmup, PHASE2_NAME=aa, WANDB_OFFLINE=0
EOF
}

[[ $# -ge 2 && $# -le 3 ]] || { usage >&2; exit 2; }
DATA_YAML="$1"
BASE_WEIGHTS="$2"
PROJECT="${3:-$ROOT/runs/train}"
PHASE1_NAME="${PHASE1_NAME:-aa_warmup}"
PHASE2_NAME="${PHASE2_NAME:-aa}"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3 || command -v python)"
[[ -f "$DATA_YAML" ]] || { echo "Dataset YAML not found: $DATA_YAML" >&2; exit 1; }
[[ -f "$BASE_WEIGHTS" ]] || { echo "Checkpoint not found: $BASE_WEIGHTS" >&2; exit 1; }

DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-32}"
NBS="${NBS:-$BATCH}"
IMGSZ="${IMGSZ:-1280}"
WORKERS="${WORKERS:-8}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-10}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-10}"
PHASE1_BEST="$PROJECT/$PHASE1_NAME/weights/best.pt"

for run in "$PROJECT/$PHASE1_NAME" "$PROJECT/$PHASE2_NAME"; do
  [[ ! -e "$run/weights/last.pt" ]] || {
    echo "Refusing to overwrite existing run: $run" >&2
    echo "Choose PHASE1_NAME/PHASE2_NAME or move the old run." >&2
    exit 1
  }
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
common=(
  --data "$DATA_YAML"
  --model "$ROOT/ultralytics/cfg/models/12/yolo12-UAV-AA.yaml"
  --batch "$BATCH" --nbs "$NBS" --imgsz "$IMGSZ"
  --device "$DEVICE" --workers "$WORKERS"
  --optimizer AdamW --momentum 0.9
  --warmup-epochs 1 --cos-lr --ir-sine-p 0.1 --no-multi-scale
  --project "$PROJECT" --no-plots
)
logging=()
if [[ "${WANDB_OFFLINE:-0}" == 1 ]]; then
  logging=(--wandb-offline --wandb-dir "${WANDB_DIR:-$ROOT/wandb-offline}")
fi

echo "AA phase 1/2: initialize the AA branch while preserving the pretrained detector."
"$PYTHON_BIN" "$ROOT/train.py" "${common[@]}" "${logging[@]}" \
  --weights "$BASE_WEIGHTS" --epochs "$PHASE1_EPOCHS" \
  --lr0 0.000001 --aa-lr 0.001 --warmup-bias-lr 0.000001 \
  --name "$PHASE1_NAME"

[[ -f "$PHASE1_BEST" ]] || { echo "Phase-1 best checkpoint was not created: $PHASE1_BEST" >&2; exit 1; }

echo "AA phase 2/2: load phase-1 best weights and create a fresh optimizer."
# Intentionally no --resume: this transfers model/EMA weights but resets the
# optimizer and scheduler for the all-parameter fine-tuning phase.
"$PYTHON_BIN" "$ROOT/train.py" "${common[@]}" "${logging[@]}" \
  --weights "$PHASE1_BEST" --epochs "$PHASE2_EPOCHS" \
  --lr0 0.0001 --aa-lr 0 --warmup-bias-lr 0.0001 \
  --name "$PHASE2_NAME"

echo "Final checkpoint: $PROJECT/$PHASE2_NAME/weights/best.pt"
