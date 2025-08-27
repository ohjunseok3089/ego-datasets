#!/bin/bash
set -Eeuo pipefail

# === Config ===
BASE_DIR="/mas/robots/aria/raw"          # Set to your Aria media directory
OUTPUT_BASE_DIR="/mas/robots/aria/outputs" # Where transcript/ CSV will be written
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

if [ ! -d "$BASE_DIR" ]; then echo "Error: BASE_DIR not found: $BASE_DIR"; exit 1; fi
mkdir -p "$OUTPUT_BASE_DIR/transcript"

# Collect media files (mp4, wav)
mapfile -t files < <(ls -1 "$BASE_DIR"/*.mp4 "$BASE_DIR"/*.wav 2>/dev/null || true)
if [ ${#files[@]} -eq 0 ]; then echo "No input media in $BASE_DIR"; exit 1; fi
echo "Found ${#files[@]} files. Launching $NUM_GPUS screens."

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
  echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/curand/lib:'"$CUDA_LIB_PATH"':$LD_LIBRARY_PATH"' >> "$tmp"
  echo "export HF_TOKEN=\"\${HF_TOKEN:-}\"" >> "$tmp"

  cat >> "$tmp" <<'PY'
python - <<'PYIN'
import ctypes, onnxruntime as ort, sys
print('[Preflight] ORT:', ort.__version__, 'providers:', ort.get_available_providers())
if 'CUDAExecutionProvider' not in ort.get_available_providers():
    print('[Error] CUDAExecutionProvider missing'); sys.exit(1)
for so in ['libcublasLt.so.12','libcublas.so.12','libcudart.so.12','libcurand.so.10','libcufft.so.11','libnvrtc.so.12','libcudnn.so.9']:
    ctypes.CDLL(so)
print('[Preflight] CUDA deps OK')
PYIN
PY

  while IFS= read -r f; do
    echo "echo \"[GPU $gpu] Processing: \$f\"" >> "$tmp"
    echo "CUDA_VISIBLE_DEVICES=$gpu python -s whisperx_diarization_aria.py \\" >> "$tmp"
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

