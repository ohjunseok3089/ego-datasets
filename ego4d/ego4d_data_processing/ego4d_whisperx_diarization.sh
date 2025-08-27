#!/bin/bash
set -Eeuo pipefail

# Configuration
BASE_DIR="/mas/robots/prg-ego4d/raw/v2/full_scale.gaze/"
OUTPUT_BASE_DIR="/mas/robots/prg-ego4d"
NUM_GPUS=4
CONDA_ENV_NAME="ego-datasets"
CUDA_LIB_PATH="/usr/local/cuda-12/lib64"
MODEL_SIZE="large-v3"
LANGUAGE="en"
COMPUTE_TYPE="float16"
BATCH_SIZE=16

echo "Starting WhisperX Diarization Batch..."
cd "$(dirname "$0")"

if [ ! -d "$BASE_DIR" ]; then
  echo "Error: BASE_DIR $BASE_DIR not found"; exit 1
fi
mkdir -p "$OUTPUT_DIR"

# Gather input files (mp4 only here; extend if needed)
mapfile -t files < <(ls -1 "$BASE_DIR"/*.mp4 2>/dev/null || true)
if [ ${#files[@]} -eq 0 ]; then echo "No files found in $BASE_DIR"; exit 1; fi
echo "Found ${#files[@]} media files."

# Distribute across GPUs
declare -a gpu_jobs
for ((i=0; i<NUM_GPUS; i++)); do gpu_jobs[$i]=""; done
for ((i=0; i<${#files[@]}; i++)); do idx=$((i % NUM_GPUS)); gpu_jobs[$idx]+="${files[$i]}"$'\n'; done

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  session="whisperx_gpu${gpu}"
  log="${session}.log"
  tmp="${session}_run.sh"
  echo "#!/bin/bash" > "$tmp"
  echo "echo '[GPU $gpu] WhisperX start. Log: $log'" >> "$tmp"
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

  echo "mkdir -p \"$OUTPUT_BASE_DIR\"/transcript" >> "$tmp"
  # Iterate inputs for this GPU
  while IFS= read -r f; do
    echo "echo \"[GPU $gpu] Processing: \$f\"" >> "$tmp"
    echo "CUDA_VISIBLE_DEVICES=$gpu python -s whisperx_diarization.py \\" >> "$tmp"
    echo "  --input_path \"\$f\" \\" >> "$tmp"
    echo "  --output_base_dir \"$OUTPUT_BASE_DIR\" \\" >> "$tmp"
    echo "  --model_size \"$MODEL_SIZE\" \\" >> "$tmp"
    echo "  --language \"$LANGUAGE\" \\" >> "$tmp"
    echo "  --compute_type \"$COMPUTE_TYPE\" \\" >> "$tmp"
    echo "  --batch_size $BATCH_SIZE" >> "$tmp"
  done < <(printf %s "${gpu_jobs[$gpu]}")

  chmod +x "$tmp"
  screen -dmS "$session" bash -c "./$tmp &> $log"
  echo "Launched screen session '$session' for GPU $gpu. Log: $log"
done

echo "============================================="
echo "All WhisperX diarization jobs launched."
echo "Use 'screen -ls' then 'screen -r whisperx_gpu0' to attach."
