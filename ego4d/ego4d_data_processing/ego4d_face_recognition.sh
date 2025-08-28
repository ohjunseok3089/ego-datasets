#!/bin/bash
set -Eeuo pipefail

# --- Configuration ---
BASE_DIR="/mas/robots/prg-ego4d/raw/v2/full_scale.gaze/"
OUTPUT_DIR="/mas/robots/prg-ego4d/processed_face_recognition_videos/"
NUM_GPUS=4
CONDA_ENV_NAME="ego-datasets"
CUDA_LIB_PATH="/usr/local/lib/ollama/cuda_v12"
GROUND_TRUTH_DIR="/mas/robots/prg-ego4d/face_detection/"

# --- Script Start ---
echo "Starting Face Recognition Batch Processing..."
echo "============================================="

cd "$(dirname "$0")"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist!"
    exit 1
fi

if [ ! -d "$GROUND_TRUTH_DIR" ]; then
    echo "Warning: Ground truth directory $GROUND_TRUTH_DIR does not exist!"
    echo "Face recognition will proceed without ground truth matching."
    GROUND_TRUTH_DIR=""
fi

mkdir -p "$OUTPUT_DIR"

# Step 1: Find all video files
echo "Finding video files..."
video_files=()
for f in "$BASE_DIR"/*.mp4; do
    [ -f "$f" ] && video_files+=("$f")
done
num_videos=${#video_files[@]}
[ "$num_videos" -eq 0 ] && { echo "No valid video files found to process."; exit 1; }
echo "Found $num_videos video files to process."

# Step 2: Distribute across GPUs
declare -a gpu_jobs
for ((i=0; i<NUM_GPUS; i++)); do gpu_jobs[$i]=""; done
for ((i=0; i<num_videos; i++)); do
    gpu_index=$((i % NUM_GPUS))
    gpu_jobs[$gpu_index]+="${video_files[$i]};"
done

# Step 3: Launch parallel screen sessions
echo "Launching $NUM_GPUS parallel screen sessions..."
echo "============================================="

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    job_list=${gpu_jobs[$gpu]}
    if [ -z "$job_list" ]; then
        echo "[GPU $gpu] No jobs assigned. Skipping."
        continue
    fi

    session_name="face_rec_gpu${gpu}"
    log_file="${session_name}.log"
    tmp="${session_name}_run.sh"

    {
        echo "#!/bin/bash"
        echo "echo '[GPU $gpu] Starting processing... Log file: $log_file'"
        echo "echo '[GPU $gpu] Activating Conda environment: $CONDA_ENV_NAME'"
        echo 'source "$(conda info --base)/etc/profile.d/conda.sh"'
        echo "conda activate $CONDA_ENV_NAME"
        echo "export PYTHONNOUSERSITE=1"
        echo "unset PYTHONPATH || true"

        echo 'export INSIGHTFACE_HOME="${INSIGHTFACE_HOME:-$CONDA_PREFIX/.insightface}"'
        echo 'mkdir -p "$INSIGHTFACE_HOME/models"'

        echo 'PYVER=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")'
        # - cudnn9: $CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cudnn/lib
        # - cufft11: $CONDA_PREFIX/lib 또는 nvidia/cufft/lib
        # - nvrtc12: nvidia/cuda_nvrtc/lib
        # - curand10: /usr/local/lib/python3.10/dist-packages/nvidia/curand/lib  (머신에 존재)
        echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:'\
'/usr/local/lib/ollama/cuda_v12:'\
'$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cudnn/lib:'\
'$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cufft/lib:'\
'$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cuda_nvrtc/lib:'\
'/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:'\
'$LD_LIBRARY_PATH"'

        cat <<'PY'
python - <<'PYIN'
import sys, importlib
def ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, '__version__', 'unknown')
    except Exception as e:
        return f'ImportError: {e}'
print('Python', sys.version.split()[0],
      '| numpy', ver('numpy'),
      '| pandas', ver('pandas'),
      '| insightface', ver('insightface'),
      '| onnxruntime', ver('onnxruntime'))
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print('ONNX Runtime providers available:', providers)
    if 'CUDAExecutionProvider' not in providers:
        print('[Error] CUDAExecutionProvider not available.')
        sys.exit(1)
except Exception as e:
    print('onnxruntime import failed for provider check:', e)
    sys.exit(1)
PYIN
PY

        cat <<'PY'
python - <<'PYIN'
import ctypes, onnxruntime as ort, sys
print('[Preflight] ORT:', ort.__version__, 'providers:', ort.get_available_providers())
need = [
    'libcublasLt.so.12','libcublas.so.12','libcudart.so.12',
    'libcurand.so.10','libcufft.so.11','libnvrtc.so.12','libcudnn.so.9'
]
try:
    for so in need: ctypes.CDLL(so)
    print('[Preflight] CUDA runtime deps OK')
except OSError as e:
    print('[Preflight] Missing CUDA runtime dep:', e); sys.exit(1)
PYIN
PY

        cat <<'PY'
python - <<'PYIN'
import os
from face_recognition_global_gallery import create_face_analysis
home = os.environ.get('INSIGHTFACE_HOME')
try:
    app = create_face_analysis('auto', model_root=home, model_name='antelopev2')
    app.prepare(ctx_id=0, det_size=(640, 640))
    print('[Prefetch] InsightFace models prepared successfully.')
except Exception as e:
    print('[Prefetch] Error: model prefetch failed:', e); raise
PYIN
PY

        echo "IFS=';' read -ra videos_to_process <<< \"$job_list\""
        echo "for video_path in \"\${videos_to_process[@]}\"; do"
        echo "  [ -z \"\$video_path\" ] && continue"
        echo "  echo \"[GPU $gpu] --------------------------------------------------\""
        echo "  echo \"[GPU $gpu] Processing video: \$video_path\""
        echo "  CUDA_VISIBLE_DEVICES=$gpu python -s face_recognition_global_gallery.py \\"
        echo "    --video_path \"\$video_path\" \\"
        echo "    --execution_provider auto \\"
        echo "    --insightface_root \"\$INSIGHTFACE_HOME\" \\"
        echo "    --insightface_model antelopev2 \\"
        if [ -n "$GROUND_TRUTH_DIR" ]; then
            echo "    --output_dir \"$OUTPUT_DIR\" \\"
            echo "    --ground_truth_dir \"$GROUND_TRUTH_DIR\""
        else
            echo "    --output_dir \"$OUTPUT_DIR\""
        fi
        echo "done"
        echo "echo \"[GPU $gpu] All assigned jobs are complete.\""
    } > "$tmp"

    chmod +x "$tmp"
    screen -dmS "$session_name" bash -c "./$tmp &> $log_file"
    echo "Launched screen session '$session_name' for GPU $gpu. Log: $log_file"
done

echo "============================================="
echo "All processing jobs launched!"
echo "Use 'screen -ls' to see running sessions."
echo "Attach with 'screen -r face_rec_gpu0' to monitor a specific GPU."
echo "Output files will be saved in: $OUTPUT_DIR/"
