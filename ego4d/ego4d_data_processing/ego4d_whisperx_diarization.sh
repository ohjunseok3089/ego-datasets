#!/bin/bash
set -Eeuo pipefail

# =======================
# Config (WhisperX only)
# =======================
BASE_DIR="/mas/robots/prg-ego4d/raw/v2/full_scale.gaze/"
OUTPUT_BASE_DIR="/mas/robots/prg-ego4d"
NUM_GPUS=4
CONDA_ENV_NAME="whisperx" 
MODEL_SIZE="large-v3"
LANGUAGE="en"
COMPUTE_TYPE="float16"
BATCH_SIZE=16

echo "Starting WhisperX Diarization Batch..."
cd "$(dirname "$0")"

if [ ! -d "$BASE_DIR" ]; then
  echo "Error: BASE_DIR $BASE_DIR not found"; exit 1
fi
mkdir -p "$OUTPUT_BASE_DIR/transcript"

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
  {
    echo "#!/bin/bash"
    echo "echo '[GPU $gpu] WhisperX start. Log: $log'"
    echo 'source "$(conda info --base)/etc/profile.d/conda.sh"'
    echo "conda activate $CONDA_ENV_NAME"
    echo 'export PYTHONNOUSERSITE=1; unset PYTHONPATH || true'
    echo 'PYVER=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")'
    echo 'TORCH_LIB=$(python - <<PY'
    echo 'import sys, os'
    echo 'py=f"{sys.version_info.major}.{sys.version_info.minor}"'
    echo 'root=sys.prefix'
    echo 'cands=['
    echo '  f"{root}/lib/python{py}/site-packages/nvidia/cudnn/lib",'
    echo '  f"{root}/lib/python{py}/site-packages/torch/lib",'
    echo '  f"{root}/lib/python{py}/site-packages/nvidia/cublas/lib",'
    echo '  f"{root}/lib/python{py}/site-packages/nvidia/cufft/lib",'
    echo '  f"{root}/lib/python{py}/site-packages/nvidia/cuda_nvrtc/lib",'
    echo '  f"{root}/lib/python{py}/site-packages/nvidia/cusparse/lib",'
    echo '  f"{root}/lib/python{py}/site-packages/nvidia/curand/lib",'
    echo ']'
    echo 'paths=[p for p in cands if os.path.isdir(p)]'
    echo 'print(":".join(paths))'
    echo 'PY'
    echo ')'
    echo 'BASE_LDLP="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"'
    echo 'export HF_TOKEN="${HF_TOKEN:-}"'

    # ── Preflight: torch/cuDNN8/NumPy1.x 확인 (LD_LIBRARY_PATH 프로세스 한정)
    cat <<'PY'
LD_LIBRARY_PATH="$TORCH_LIB:$BASE_LDLP" python - <<'PYIN'
import sys, os
print("[LD_LIBRARY_PATH first]:", os.environ.get("LD_LIBRARY_PATH","").split(":")[0])
try:
    import numpy as np
    import torch
    print(f"[Preflight] torch {torch.__version__} CUDA {torch.version.cuda}")
    print(f"numpy {np.__version__}")
    cudnn_ver = torch.backends.cudnn.version()
    print("cuDNN version (torch):", cudnn_ver)
    if not (cudnn_ver and int(str(cudnn_ver))//1000 == 8):
        print("[WARN] Expected cuDNN 8.x for torch 2.2.2+cu121.")
    if not np.__version__.startswith("1."):
        print("[WARN] WhisperX stack prefers NumPy 1.x. Current:", np.__version__)
    print("[Preflight] torch.cuda.is_available:", torch.cuda.is_available())
except Exception as e:
    print("[Preflight ERROR]", e); sys.exit(1)
PYIN
PY

    echo "mkdir -p \"$OUTPUT_BASE_DIR/transcript\""
    # Inputs for this GPU
    while IFS= read -r f; do
      echo "echo \"[GPU $gpu] Processing: \$f\""
      echo "LD_LIBRARY_PATH=\"\$TORCH_LIB:\$BASE_LDLP\" CUDA_VISIBLE_DEVICES=$gpu \\"
      echo "  python -s whisperx_diarization.py \\"
      echo "    --input_path \"\$f\" \\"
      echo "    --output_base_dir \"$OUTPUT_BASE_DIR\" \\"
      echo "    --model_size \"$MODEL_SIZE\" \\"
      echo "    --language \"$LANGUAGE\" \\"
      echo "    --compute_type \"$COMPUTE_TYPE\" \\"
      echo "    --batch_size $BATCH_SIZE"
    done < <(printf %s "${gpu_jobs[$gpu]}")
  } > "$tmp"

  chmod +x "$tmp"
  screen -dmS "$session" bash -c "./$tmp &> $log"
  echo "Launched screen session '$session' for GPU $gpu. Log: $log"
done

echo "============================================="
echo "All WhisperX diarization jobs launched."
echo "Use 'screen -ls' then 'screen -r whisperx_gpu0' to attach."
