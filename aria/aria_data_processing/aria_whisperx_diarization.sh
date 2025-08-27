#!/bin/bash
set -Eeuo pipefail

# === Config ===
RAW_DIR="/mas/robots/prg-aria/raw"          # Aria raw recording roots (contains per-session folders)
DATASET_DIR="/mas/robots/prg-aria/dataset"  # Flattened MP4 dataset
OUTPUT_BASE_DIR="/mas/robots/prg-aria"      # transcript/ lives here
NUM_GPUS=4
CONDA_ENV_NAME="ego-datasets"
CUDA_LIB_PATH="/usr/local/cuda-12/lib64"
MODEL_SIZE="large-v3"
LANGUAGE="en"
COMPUTE_TYPE="float16"
BATCH_SIZE=16
FPS=30.0

echo "Starting Aria WhisperX diarization..."
cd "$(dirname "$0")"

if [ ! -d "$RAW_DIR" ]; then echo "Error: RAW_DIR not found: $RAW_DIR"; exit 1; fi
if [ ! -d "$DATASET_DIR" ]; then echo "Error: DATASET_DIR not found: $DATASET_DIR"; exit 1; fi
mkdir -p "$OUTPUT_BASE_DIR/transcript"

# Collect media files:
# 1) All mp4 in DATASET_DIR
# 2) For each raw session dir, map to DATASET_DIR/<session_name>.mp4 if exists
tmp_list=$(mktemp)
find "$DATASET_DIR" -maxdepth 1 -type f -iname '*.mp4' -print >> "$tmp_list"
while IFS= read -r d; do
  base=$(basename "$d")
  [ -f "$DATASET_DIR/$base.mp4" ] && echo "$DATASET_DIR/$base.mp4" >> "$tmp_list"
done < <(find "$RAW_DIR" -mindepth 1 -maxdepth 1 -type d -print)
# De-duplicate and load into array
mapfile -t files < <(sort -u "$tmp_list")
rm -f "$tmp_list"
if [ ${#files[@]} -eq 0 ]; then echo "No input media found in $DATASET_DIR or mapped from $RAW_DIR"; exit 1; fi
echo "Found ${#files[@]} mp4 files. Launching $NUM_GPUS screens."

declare -a gpu_jobs
for ((i=0;i<NUM_GPUS;i++)); do gpu_jobs[$i]=""; done
for ((i=0;i<${#files[@]};i++)); do idx=$((i % NUM_GPUS)); gpu_jobs[$idx]+="${files[$i]}"$'\n'; done

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  sess="aria_whisperx_gpu${gpu}"
  log="${sess}.log"
  tmp="${sess}_run.sh"
  echo "#!/bin/bash" > "$tmp"
  echo "echo '[GPU $gpu] Aria WhisperX start. Log: $log'" >> "$tmp"
  echo "source \"\$(conda info --base)/etc/profile.d/conda.sh\"" >> "$tmp"
  echo "conda activate $CONDA_ENV_NAME" >> "$tmp"
  echo "export PYTHONNOUSERSITE=1; unset PYTHONPATH || true" >> "$tmp"
  echo 'PYVER=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")' >> "$tmp"
  # Discover PyTorch's bundled CUDA libs (includes cuDNN 8 for torch cu121)
  echo 'TORCH_LIB=$(python - <<PY
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
  )' >> "$tmp"
  # Do NOT export globally to avoid affecting other pipelines (e.g., face ORT expecting cuDNN 9)
  echo 'BASE_LDLP="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/curand/lib:'"$CUDA_LIB_PATH"':$LD_LIBRARY_PATH"' >> "$tmp"
  echo "export HF_TOKEN=\"\${HF_TOKEN:-}\"" >> "$tmp"

  cat >> "$tmp" <<'PY'
python - <<'PYIN'
import sys, ctypes
import torch
print('[Preflight] torch', torch.__version__, 'CUDA', torch.version.cuda)
print('cuDNN version (torch):', torch.backends.cudnn.version())
try:
    ctypes.CDLL('libcudnn.so.8')
    print('[Preflight] libcudnn.so.8 OK')
except OSError as e:
    print('[Preflight] Missing libcudnn.so.8:', e)
    sys.exit(1)
PYIN
PY

  while IFS= read -r f; do
    echo "echo \"[GPU $gpu] Processing: \$f\"" >> "$tmp"
    echo "LD_LIBRARY_PATH=\"\$TORCH_LIB:\$BASE_LDLP\" CUDA_VISIBLE_DEVICES=$gpu python -s whisperx_diarization_aria.py \\" >> "$tmp"
    echo "  --input_path \"\$f\" \\" >> "$tmp"
    echo "  --output_base_dir \"$OUTPUT_BASE_DIR\" \\" >> "$tmp"
    echo "  --model_size \"$MODEL_SIZE\" \\" >> "$tmp"
    echo "  --language \"$LANGUAGE\" \\" >> "$tmp"
    echo "  --compute_type \"$COMPUTE_TYPE\" \\" >> "$tmp"
    echo "  --batch_size $BATCH_SIZE \\" >> "$tmp"
    echo "  --fps $FPS" >> "$tmp"
  done < <(printf %s "${gpu_jobs[$gpu]}")

  chmod +x "$tmp"
  screen -dmS "$sess" bash -c "./$tmp &> $log"
  echo "Launched '$sess' for GPU $gpu. Log: $log"
done

echo "All Aria diarization jobs launched. Use 'screen -ls' to list." 
