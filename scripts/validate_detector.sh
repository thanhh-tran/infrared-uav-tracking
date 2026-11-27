#!/usr/bin/env bash
# Compute standard YOLO detection metrics and write JSON/CSV summaries.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/validate_detector.sh WEIGHTS DATA_YAML [RUN_NAME]" >&2
  exit 2
fi

WEIGHTS="$1"
DATA_YAML="$2"
RUN_NAME="${3:-validation}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3 || command -v python)"
[[ -f "$WEIGHTS" ]] || { echo "Checkpoint not found: $WEIGHTS" >&2; exit 1; }
[[ -f "$DATA_YAML" ]] || { echo "Dataset YAML not found: $DATA_YAML" >&2; exit 1; }

PROJECT="${PROJECT:-$ROOT/runs/validation}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" "$ROOT/validate.py" \
  --weights "$WEIGHTS" --data "$DATA_YAML" --split "${SPLIT:-test}" \
  --imgsz "${IMGSZ:-1280}" --batch "${BATCH:-16}" \
  --device "${DEVICE:-0}" --workers "${WORKERS:-8}" \
  --project "$PROJECT" --name "$RUN_NAME" \
  --output "$PROJECT/$RUN_NAME/metrics.json"
