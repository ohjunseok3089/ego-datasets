#!/usr/bin/env bash

# Run past-frame prediction JSON generation for each subdirectory in a base folder.
# For each subdirectory, this calls track_prediction_past_frame.py with that
# directory as input and output, producing one <group_id>_analysis.json per dir.

set -euo pipefail

# Resolve the directory of this script so we can locate the Python file reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/track_prediction_past_frame.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "Error: Python script not found at $PY_SCRIPT"
  exit 1
fi

# Usage
usage() {
  cat <<EOF
Usage: $(basename "$0") [BASE_DIR] [--mode advanced|default] [--fov DEG] [--fps FPS] [--force]

Arguments:
  BASE_DIR           Base directory that contains many subdirectories, each with video clips
                     Default: /mas/robots/prg-egocom/EGOCOM/720p/5min_parts/co-tracker

Options:
  --mode MODE        Filename parsing mode for the Python script (advanced|default). Default: advanced
  --fov DEG          Field of view degrees (float). Default: 104.0
  --fps FPS          Override FPS passed to the Python script (float). Optional
  --force            Reprocess even if output JSON already exists

The JSON will be saved inside each subdirectory.
EOF
}

# Defaults
BASE_DIR="/mas/robots/prg-egocom/EGOCOM/720p/5min_parts/co-tracker"
MODE="advanced"
FOV="104.0"
FPS_OVERRIDE=""
FORCE=0

# Parse args
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --mode)
      MODE="$2"; shift 2 ;;
    --fov)
      FOV="$2"; shift 2 ;;
    --fps)
      FPS_OVERRIDE="$2"; shift 2 ;;
    --force)
      FORCE=1; shift ;;
    --)
      shift; break ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  BASE_DIR="${POSITIONAL[0]}"
fi

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Error: Directory $BASE_DIR does not exist"
  exit 1
fi

echo "Base directory: $BASE_DIR"
echo "Mode: $MODE | FOV: $FOV | FPS override: ${FPS_OVERRIDE:-<none>} | Force: $FORCE"
echo "================================"

shopt -s nullglob

subdirs=("$BASE_DIR"/*/)
if [[ ${#subdirs[@]} -eq 0 ]]; then
  echo "No subdirectories found in $BASE_DIR"
  exit 0
fi

total=${#subdirs[@]}
done_count=0
skipped_count=0
failed_count=0

for dir_path in "${subdirs[@]}"; do
  dir_name="$(basename "$dir_path")"
  # Trim possible trailing slash from basename context
  dir_name="${dir_name%/}"

  # Determine expected JSON path. The Python script will name the output as
  # <group_id>_analysis.json, where group_id is parsed from filenames and
  # typically matches the directory name in this dataset layout.
  expected_json="$dir_path/${dir_name}_analysis.json"

  # Detect presence of video files to decide if this directory should be processed
  has_mp4=0
  shopt -s nullglob
  for f in "$dir_path"*.MP4 "$dir_path"*.mp4; do
    if [[ -f "$f" ]]; then has_mp4=1; break; fi
  done
  shopt -u nullglob

  if [[ $has_mp4 -eq 0 ]]; then
    echo "[skip] $dir_name -> no .mp4/.MP4 files found"
    ((skipped_count++))
    continue
  fi

  if [[ $FORCE -eq 0 && -f "$expected_json" ]]; then
    echo "[skip] $dir_name -> JSON already exists: $expected_json"
    ((skipped_count++))
    continue
  fi

  echo "[run ] $dir_name"
  set +e
  if [[ -n "$FPS_OVERRIDE" ]]; then
    python3 "$PY_SCRIPT" "$dir_path" --output_dir "$dir_path" --mode "$MODE" --fov "$FOV" --fps "$FPS_OVERRIDE"
  else
    python3 "$PY_SCRIPT" "$dir_path" --output_dir "$dir_path" --mode "$MODE" --fov "$FOV"
  fi
  exit_code=$?
  set -e

  if [[ $exit_code -eq 0 ]]; then
    echo "[done] $dir_name -> ${expected_json}"
    ((done_count++))
  else
    echo "[fail] $dir_name (exit code: $exit_code)"
    ((failed_count++))
  fi
  echo "--------------------------------"
done

echo "================================"
echo "Completed. Total: $total | Done: $done_count | Skipped: $skipped_count | Failed: $failed_count"


