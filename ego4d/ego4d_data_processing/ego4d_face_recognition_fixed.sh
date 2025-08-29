#!/bin/bash
set -Eeuo pipefail

# --- Configuration ---
BASE_DIR="/mas/robots/prg-ego4d/raw/v2/full_scale.gaze/"
OUTPUT_DIR="/mas/robots/prg-ego4d/processed_face_recognition_videos_fixed/"
NUM_GPUS=4
CONDA_ENV_NAME="ego-datasets"
CUDA_LIB_PATH="/usr/local/lib/ollama/cuda_v12"
GROUND_TRUTH_DIR="/mas/robots/prg-ego4d/face_detection/"

# --- New Configuration for Fixed Implementation ---
VALIDATION_ENABLED=true
RECOGNITION_THRESHOLD=0.6  # Lenient threshold for video quality issues
SKIP_FACE_DETECTION_TEST=false  # Set to true to speed up validation

# --- Script Start ---
echo "Starting FIXED Face Recognition Batch Processing..."
echo "=================================================="
echo "This version prevents spurious person IDs (11, 12, etc.)"
echo "by strictly using ground truth person identities."
echo "=================================================="

cd "$(dirname "$0")"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist!"
    exit 1
fi

if [ ! -d "$GROUND_TRUTH_DIR" ]; then
    echo "ERROR: Ground truth directory $GROUND_TRUTH_DIR does not exist!"
    echo "Ground truth is REQUIRED for the fixed implementation."
    echo "Cannot proceed without ground truth to prevent person ID issues."
    exit 1
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

# Step 2: Pre-validation (optional but recommended)
if [ "$VALIDATION_ENABLED" = true ]; then
    echo ""
    echo "=== PRE-VALIDATION PHASE ==="
    echo "Validating a sample of videos to ensure ground truth is properly set up..."
    
    # Test validation on first few videos
    validation_sample=3
    validation_failures=0
    
    for ((i=0; i<validation_sample && i<num_videos; i++)); do
        video_path="${video_files[$i]}"
        video_basename=$(basename "$video_path" .mp4)
        echo "Validating: $video_basename"
        
        validation_cmd="python validate_ground_truth.py --video_path \"$video_path\" --ground_truth_dir \"$GROUND_TRUTH_DIR\""
        if [ "$SKIP_FACE_DETECTION_TEST" = true ]; then
            validation_cmd="$validation_cmd --skip_face_detection_test"
        fi
        
        if ! eval "$validation_cmd" >/dev/null 2>&1; then
            echo "  ❌ Validation failed for: $video_basename"
            ((validation_failures++))
        else
            echo "  ✅ Validation passed for: $video_basename"
        fi
    done
    
    if [ "$validation_failures" -gt 0 ]; then
        echo ""
        echo "❌ PRE-VALIDATION FAILED!"
        echo "   $validation_failures out of $validation_sample sample videos failed validation."
        echo "   Please fix ground truth issues before proceeding."
        echo "   Common issues:"
        echo "   - Missing ground truth CSV files"
        echo "   - Invalid ground truth format"
        echo "   - Face detection not working"
        echo ""
        echo "   To run validation manually:"
        echo "   python validate_ground_truth.py --video_path VIDEO_PATH --ground_truth_dir $GROUND_TRUTH_DIR"
        exit 1
    else
        echo ""
        echo "✅ PRE-VALIDATION PASSED!"
        echo "   All sample videos have valid ground truth. Proceeding with batch processing..."
    fi
else
    echo "⏭️  Skipping pre-validation (disabled)"
fi

# Step 3: Distribute across GPUs
declare -a gpu_jobs
for ((i=0; i<NUM_GPUS; i++)); do gpu_jobs[$i]=""; done
for ((i=0; i<num_videos; i++)); do
    gpu_index=$((i % NUM_GPUS))
    gpu_jobs[$gpu_index]+="${video_files[$i]};"
done

# Step 4: Launch parallel screen sessions
echo ""
echo "=== PROCESSING PHASE ==="
echo "Launching $NUM_GPUS parallel screen sessions..."
echo "============================================="

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    job_list=${gpu_jobs[$gpu]}
    if [ -z "$job_list" ]; then
        echo "[GPU $gpu] No jobs assigned. Skipping."
        continue
    fi

    session_name="face_rec_fixed_gpu${gpu}"
    log_file="${session_name}.log"
    tmp="${session_name}_run.sh"

    {
        echo "#!/bin/bash"
        echo "echo '[GPU $gpu] Starting FIXED face recognition processing... Log file: $log_file'"
        echo "echo '[GPU $gpu] Using recognition threshold: $RECOGNITION_THRESHOLD (lenient for video quality)'"
        echo "echo '[GPU $gpu] Activating Conda environment: $CONDA_ENV_NAME'"
        echo 'source "$(conda info --base)/etc/profile.d/conda.sh"'
        echo "conda activate $CONDA_ENV_NAME"
        echo "export PYTHONNOUSERSITE=1"
        echo "unset PYTHONPATH || true"

        echo 'export INSIGHTFACE_HOME="${INSIGHTFACE_HOME:-$CONDA_PREFIX/.insightface}"'
        echo 'mkdir -p "$INSIGHTFACE_HOME/models"'

        echo 'PYVER=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")'
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
print('[GPU $gpu] Python', sys.version.split()[0],
      '| numpy', ver('numpy'),
      '| pandas', ver('pandas'),
      '| insightface', ver('insightface'),
      '| onnxruntime', ver('onnxruntime'))
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print('[GPU $gpu] ONNX Runtime providers available:', providers)
    if 'CUDAExecutionProvider' not in providers:
        print('[GPU $gpu] [Error] CUDAExecutionProvider not available.')
        sys.exit(1)
except Exception as e:
    print('[GPU $gpu] onnxruntime import failed for provider check:', e)
    sys.exit(1)
PYIN
PY

        cat <<'PY'
python - <<'PYIN'
import ctypes, onnxruntime as ort, sys
print('[GPU $gpu] [Preflight] ORT:', ort.__version__, 'providers:', ort.get_available_providers())
need = [
    'libcublasLt.so.12','libcublas.so.12','libcudart.so.12',
    'libcurand.so.10','libcufft.so.11','libnvrtc.so.12','libcudnn.so.9'
]
try:
    for so in need: ctypes.CDLL(so)
    print('[GPU $gpu] [Preflight] CUDA runtime deps OK')
except OSError as e:
    print('[GPU $gpu] [Preflight] Missing CUDA runtime dep:', e); sys.exit(1)
PYIN
PY

        cat <<'PY'
python - <<'PYIN'
import os
from face_recognition_fixed import create_face_analysis
home = os.environ.get('INSIGHTFACE_HOME')
try:
    app = create_face_analysis('auto', model_root=home, model_name='antelopev2')
    app.prepare(ctx_id=0, det_size=(640, 640))
    print('[GPU $gpu] [Prefetch] InsightFace models prepared successfully.')
except Exception as e:
    print('[GPU $gpu] [Prefetch] Error: model prefetch failed:', e); raise
PYIN
PY

        echo "IFS=';' read -ra videos_to_process <<< \"$job_list\""
        echo "for video_path in \"\${videos_to_process[@]}\"; do"
        echo "  [ -z \"\$video_path\" ] && continue"
        echo "  video_basename=\$(basename \"\$video_path\" .mp4)"
        echo "  echo \"[GPU $gpu] ===========================================\""
        echo "  echo \"[GPU $gpu] Processing video: \$video_basename\""
        echo "  echo \"[GPU $gpu] ===========================================\""
        echo ""
        
        # Add individual validation step for each video
        echo "  echo \"[GPU $gpu] Step 1: Validating ground truth for \$video_basename...\""
        echo "  if ! CUDA_VISIBLE_DEVICES=$gpu python validate_ground_truth.py \\"
        echo "    --video_path \"\$video_path\" \\"
        echo "    --ground_truth_dir \"$GROUND_TRUTH_DIR\" \\"
        if [ "$SKIP_FACE_DETECTION_TEST" = true ]; then
            echo "    --skip_face_detection_test; then"
        else
            echo "    ; then"
        fi
        echo "    echo \"[GPU $gpu] ❌ Validation failed for \$video_basename - skipping\""
        echo "    continue"
        echo "  fi"
        echo ""
        
        echo "  echo \"[GPU $gpu] Step 2: Running fixed face recognition for \$video_basename...\""
        echo "  CUDA_VISIBLE_DEVICES=$gpu python face_recognition_fixed.py \\"
        echo "    --video_path \"\$video_path\" \\"
        echo "    --execution_provider auto \\"
        echo "    --insightface_root \"\$INSIGHTFACE_HOME\" \\"
        echo "    --insightface_model antelopev2 \\"
        echo "    --output_dir \"$OUTPUT_DIR\" \\"
        echo "    --ground_truth_dir \"$GROUND_TRUTH_DIR\" \\"
        echo "    --recognition_threshold $RECOGNITION_THRESHOLD \\"
        echo "    --max_frames 36000"
        echo ""
        echo "  if [ \$? -eq 0 ]; then"
        echo "    echo \"[GPU $gpu] ✅ Successfully processed \$video_basename\""
        echo "  else"
        echo "    echo \"[GPU $gpu] ❌ Failed to process \$video_basename\""
        echo "  fi"
        echo "  echo \"\""
        echo "done"
        echo "echo \"[GPU $gpu] All assigned jobs are complete.\""
    } > "$tmp"

    chmod +x "$tmp"
    screen -dmS "$session_name" bash -c "./$tmp &> $log_file"
    echo "Launched screen session '$session_name' for GPU $gpu. Log: $log_file"
done

echo "============================================="
echo "All FIXED face recognition jobs launched!"
echo ""
echo "🔧 FIXED IMPLEMENTATION FEATURES:"
echo "  ✅ Ground truth validation before processing"
echo "  ✅ Strict person ID constraints (no person 11, 12, etc.)"
echo "  ✅ Lenient recognition threshold ($RECOGNITION_THRESHOLD)"
echo "  ✅ Source tracking (GT vs recognized faces)"
echo "  ✅ Confidence scoring for recognized faces"
echo ""
echo "📊 MONITORING:"
echo "  Use 'screen -ls' to see running sessions."
echo "  Attach with 'screen -r face_rec_fixed_gpu0' to monitor a specific GPU."
echo "  Check logs: face_rec_fixed_gpu*.log"
echo ""
echo "📁 OUTPUT:"
echo "  Results will be saved in: $OUTPUT_DIR/"
echo "  Each video gets: *_fixed_face_recognition.csv and *_fixed_face_recognition.mp4"
echo ""
echo "🚨 VALIDATION NOTES:"
echo "  - Each video is validated individually before processing"
echo "  - Videos with validation failures are automatically skipped"
echo "  - Check logs for detailed validation results"
echo ""
echo "✅ The fixed implementation prevents spurious person IDs!"
echo "============================================="
