#!/bin/bash
set -Eeuo pipefail

# --- Configuration ---
BASE_DIR="/mas/robots/prg-ego4d/raw/v2/full_scale.gaze/"
OUTPUT_DIR="/mas/robots/prg-ego4d/processed_face_recognition_videos/"
NUM_GPUS=4
CONDA_ENV_NAME="ego-datasets"
CUDA_LIB_PATH="/usr/local/cuda-11.8/lib64"
GROUND_TRUTH_DIR="/mas/robots/prg-ego4d/face_detection/"

# --- Script Start ---
echo "Starting Face Recognition Batch Processing..."
echo "============================================="

# Always run from this script's directory so relative imports work
cd "$(dirname "$0")"

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist!"
    exit 1
fi

# Check if ground truth directory exists
if [ ! -d "$GROUND_TRUTH_DIR" ]; then
    echo "Warning: Ground truth directory $GROUND_TRUTH_DIR does not exist!"
    echo "Face recognition will proceed without ground truth matching."
    GROUND_TRUTH_DIR=""
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Step 1: Find all video files
echo "Finding video files..."

video_files=()
for f in "$BASE_DIR"/*.mp4; do
    if [ -f "$f" ]; then
        video_files+=("$f")
    fi
done

num_videos=${#video_files[@]}

if [ "$num_videos" -eq 0 ]; then
    echo "No valid video files found to process."
    exit 1
fi

echo "Found $num_videos video files to process."

# Step 2: Distribute the videos across the available GPUs
declare -a gpu_jobs
for ((i=0; i<NUM_GPUS; i++)); do
    gpu_jobs[$i]=""
done

for ((i=0; i<num_videos; i++)); do
    gpu_index=$((i % NUM_GPUS))
    gpu_jobs[$gpu_index]+="${video_files[$i]};"
done

# Step 3: Launch parallel screen sessions for each GPU
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
    
    temp_script="${session_name}_run.sh"
    echo "#!/bin/bash" > "$temp_script"
    echo "echo '[GPU $gpu] Starting processing... Log file: $log_file'" >> "$temp_script"
    
    echo "echo '[GPU $gpu] Activating Conda environment: $CONDA_ENV_NAME'" >> "$temp_script"
    echo "source \"\$(conda info --base)/etc/profile.d/conda.sh\"" >> "$temp_script"
    echo "conda activate $CONDA_ENV_NAME" >> "$temp_script"
    # Ensure user-site packages don't shadow the env (avoids numpy/pandas ABI mismatches)
    echo "export PYTHONNOUSERSITE=1" >> "$temp_script"
    echo "unset PYTHONPATH || true" >> "$temp_script"
    # Ensure insightface model cache is set; respect user-provided value if present
    echo 'export INSIGHTFACE_HOME="${INSIGHTFACE_HOME:-'"$OUTPUT_DIR/.insightface"'}"' >> "$temp_script"
    echo 'mkdir -p "$INSIGHTFACE_HOME"' >> "$temp_script"

    # Print runtime versions to the log for debugging
    cat >> "$temp_script" <<'PY'
python - <<'PYIN'
import sys, importlib
def ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, '__version__', 'unknown')
    except Exception as e:
        return f'ImportError: {e}'
print('Python', sys.version.split()[0], '| numpy', ver('numpy'), '| pandas', ver('pandas'), '| insightface', ver('insightface'), '| onnxruntime', ver('onnxruntime'))
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print('ONNX Runtime providers available:', providers)
    if 'CUDAExecutionProvider' not in providers:
        print('[Error] CUDAExecutionProvider not available. Install onnxruntime-gpu and ensure CUDA libs in LD_LIBRARY_PATH.')
        sys.exit(1)
except Exception as e:
    print('onnxruntime import failed for provider check:', e)
    sys.exit(1)
PYIN
PY

    # --- Set the CUDA library path ---
    echo "echo '[GPU $gpu] Setting CUDA library path...'" >> "$temp_script"
    echo 'PYVER=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")' >> "$temp_script"
    echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/curand/lib:$LD_LIBRARY_PATH"' >> "$temp_script"
    # Keep any additional system CUDA libs if specified
    echo "export LD_LIBRARY_PATH=$CUDA_LIB_PATH:\$LD_LIBRARY_PATH" >> "$temp_script"
    # -------------------------------------------

    # Preflight: check ORT providers and CUDA DLL deps are loadable
    cat >> "$temp_script" <<'PY'
python - <<'PYIN'
import ctypes, onnxruntime as ort, sys
print('[Preflight] ORT:', ort.__version__, 'providers:', ort.get_available_providers())
try:
    for so in [
        'libcublasLt.so.12','libcublas.so.12','libcudart.so.12',
        'libcurand.so.10','libcufft.so.11','libnvrtc.so.12','libcudnn.so.9']:
        ctypes.CDLL(so)
    print('[Preflight] CUDA runtime deps OK')
except OSError as e:
    print('[Preflight] Missing CUDA runtime dep:', e)
    sys.exit(1)
PYIN
PY

    # Prefetch InsightFace models once per session to avoid runtime assert
    cat >> "$temp_script" <<'PY'
python - <<'PYIN'
import os, sys
from face_recognition_global_gallery import create_face_analysis
try:
    app = create_face_analysis('auto', model_root=os.environ.get('INSIGHTFACE_HOME'), model_name='antelopev2')
    # CUDA-only mode; create_face_analysis enforces CUDA
    app.prepare(ctx_id=0, det_size=(640, 640))
    print('[Prefetch] InsightFace models prepared successfully.')
except Exception as e:
    # If models already exist, treat as success
    if isinstance(e, (FileExistsError, OSError)) and 'File exists' in str(e):
        print('[Prefetch] Models already present; continuing.')
    else:
        print('[Prefetch] Error: model prefetch failed:', e)
        sys.exit(1)
PYIN
PY

    echo "IFS=';' read -ra videos_to_process <<< \"$job_list\"" >> "$temp_script"
    echo "for video_path in \"\${videos_to_process[@]}\"; do" >> "$temp_script"
    echo "    if [ -n \"\$video_path\" ]; then" >> "$temp_script"
    echo "        echo \"[GPU $gpu] --------------------------------------------------\"" >> "$temp_script"
    echo "        echo \"[GPU $gpu] Processing video: \$video_path\"" >> "$temp_script"
    echo "        CUDA_VISIBLE_DEVICES=$gpu python -s face_recognition_global_gallery.py \\" >> "$temp_script"
        echo "            --video_path \"\$video_path\" \\" >> "$temp_script"
        echo "            --execution_provider auto \\" >> "$temp_script"
        echo "            --insightface_root \"\$INSIGHTFACE_HOME\" \\" >> "$temp_script"
        echo "            --insightface_model antelopev2 \\" >> "$temp_script"
    if [ -n "$GROUND_TRUTH_DIR" ]; then
        echo "            --output_dir \"$OUTPUT_DIR\" \\" >> "$temp_script"
        echo "            --ground_truth_dir \"$GROUND_TRUTH_DIR\"" >> "$temp_script"
    else
        echo "            --output_dir \"$OUTPUT_DIR\"" >> "$temp_script"
    fi
    echo "    fi" >> "$temp_script"
    echo "done" >> "$temp_script"
    echo "echo \"[GPU $gpu] All assigned jobs are complete.\"" >> "$temp_script"
    
    chmod +x "$temp_script"

    screen -dmS "$session_name" bash -c "./$temp_script &> $log_file"
    echo "Launched screen session '$session_name' for GPU $gpu. Log: $log_file"
done

echo "============================================="
echo "All processing jobs launched!"
echo "Use 'screen -ls' to see running sessions."
echo "Attach with 'screen -r face_rec_gpu0' to monitor a specific GPU."
echo "Output files will be saved in: $OUTPUT_DIR/"
