#!/usr/bin/env bash
# Evaluate official per-video SA/AOA and optionally aggregate the 10 attributes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/evaluate_aoa.sh GROUND_TRUTH PREDICTIONS [ATTRIBUTE_ANN] [OUTPUT_DIR]

GROUND_TRUTH is the raw Anti-UAV410 test root containing video/IR_label.json.
PREDICTIONS contains video/predictions.jsonl from infer_variant.sh.
ATTRIBUTE_ANN is optional; when supplied, the script also reports all 10
official video-level attributes.
EOF
}

[[ $# -ge 2 && $# -le 4 ]] || { usage >&2; exit 2; }
GROUND_TRUTH="$1"
PREDICTIONS="$2"
ATTRIBUTE_ANN="${3:-}"
OUTPUT_DIR="${4:-$PREDICTIONS}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3 || command -v python)"

[[ -d "$GROUND_TRUTH" ]] || { echo "Ground-truth root not found: $GROUND_TRUTH" >&2; exit 1; }
[[ -d "$PREDICTIONS" ]] || { echo "Prediction root not found: $PREDICTIONS" >&2; exit 1; }
[[ -z "$ATTRIBUTE_ANN" || -d "$ATTRIBUTE_ANN" ]] || {
  echo "Attribute annotation directory not found: $ATTRIBUTE_ANN" >&2; exit 1;
}
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$ROOT/tools:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" "$ROOT/tools/evaluate_anti_uav_aoa.py" \
  --ground-truth "$GROUND_TRUTH" --predictions "$PREDICTIONS" \
  --conf-threshold 0.0 --output "$OUTPUT_DIR/aoa_report.json"

if [[ -n "$ATTRIBUTE_ANN" ]]; then
  "$PYTHON_BIN" "$ROOT/tools/summarize_anti_uav410_aoa_attributes.py" \
    --report "$OUTPUT_DIR/aoa_report.json" --annotations "$ATTRIBUTE_ANN" \
    --output "$OUTPUT_DIR/aoa_by_attribute.json"
fi

echo "Metrics written to: $OUTPUT_DIR"
