#!/usr/bin/env bash
# Run one of the five public detector/tracker variants on one video or frame directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/infer_variant.sh VARIANT WEIGHTS SOURCE [OUTPUT_ROOT]

VARIANT:
  baseline_botsort | ir_botsort | ir_tracktrack | aa_botsort | aa_tracktrack

The caller supplies the matching checkpoint. SOURCE may be one video/frame
directory or a dataset root whose immediate subdirectories are videos. Outputs
are written below OUTPUT_ROOT/VARIANT (default: runs/inference/VARIANT).

Optional environment variables:
  PYTHON_BIN, DEVICE=0, IMGSZ=1280, FPS=25, SAVE_FRAMES=0
EOF
}

[[ $# -ge 3 && $# -le 4 ]] || { usage >&2; exit 2; }
VARIANT="$1"
WEIGHTS="$2"
SOURCE="$3"
OUTPUT_ROOT="${4:-$ROOT/runs/inference}"

case "$VARIANT" in
  baseline_botsort|ir_botsort)
    TRACKER="$ROOT/ultralytics/cfg/trackers/botsort_uav.yaml"
    CONF=0.01
    ;;
  ir_tracktrack)
    TRACKER="$ROOT/ultralytics/cfg/trackers/tracktrack_uav.yaml"
    CONF=0.01
    ;;
  aa_botsort)
    TRACKER="$ROOT/ultralytics/cfg/trackers/botsort_aa.yaml"
    CONF=0.001
    ;;
  aa_tracktrack)
    TRACKER="$ROOT/ultralytics/cfg/trackers/tracktrack_aa.yaml"
    CONF=0.001
    ;;
  *) echo "Unknown variant: $VARIANT" >&2; usage >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3 || command -v python)"
[[ -f "$WEIGHTS" ]] || { echo "Checkpoint not found: $WEIGHTS" >&2; exit 1; }
[[ -e "$SOURCE" ]] || { echo "Source not found: $SOURCE" >&2; exit 1; }

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
printf 'Variant=%s tracker=%s detector_floor=%s\n' "$VARIANT" "$(basename "$TRACKER")" "$CONF"

run_one() {
  local source="$1"
  local name="$2"
  local command=(
    "$PYTHON_BIN" "$ROOT/infer.py"
    --weights "$WEIGHTS" --source "$source" --tracker "$TRACKER"
    --imgsz "${IMGSZ:-1280}" --conf "$CONF" --iou 0.7
    --device "${DEVICE:-0}" --project "$OUTPUT_ROOT" --name "$name"
    --fps "${FPS:-25}" --save-video --compact-labels
  )
  [[ "${SAVE_FRAMES:-0}" == 1 ]] && command+=(--save-frames)
  printf 'Command:'; printf ' %q' "${command[@]}"; printf '\n'
  "${command[@]}"
}

# A raw Anti-UAV test root contains one immediate directory per video. Process
# these sequentially on a workstation; cluster users can parallelize calls to
# this same script externally without changing the Python pipeline.
if [[ -d "$SOURCE" ]] && find "$SOURCE" -mindepth 1 -maxdepth 1 -type d -print -quit | grep -q .; then
  while IFS= read -r -d '' sequence; do
    run_one "$sequence" "$VARIANT/$(basename "$sequence")"
  done < <(find "$SOURCE" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
else
  run_one "$SOURCE" "$VARIANT"
fi
