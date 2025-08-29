#!/bin/bash
set -Eeuo pipefail

# Single video runner for FIXED face recognition implementation
# Usage: ./run_single_video_fixed.sh VIDEO_PATH [OUTPUT_DIR] [GROUND_TRUTH_DIR]

# --- Default Configuration ---
DEFAULT_OUTPUT_DIR="./single_video_output_fixed"
DEFAULT_GROUND_TRUTH_DIR="/mas/robots/prg-ego4d/face_detection/"
CONDA_ENV_NAME="ego-datasets"
RECOGNITION_THRESHOLD=0.6
GPU_ID=0

# --- Parse Arguments ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 VIDEO_PATH [OUTPUT_DIR] [GROUND_TRUTH_DIR]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/video.mp4"
    echo "  $0 /path/to/video.mp4 ./my_output"
    echo "  $0 /path/to/video.mp4 ./my_output /path/to/ground_truth"
    echo ""
    echo "Default OUTPUT_DIR: $DEFAULT_OUTPUT_DIR"
    echo "Default GROUND_TRUTH_DIR: $DEFAULT_GROUND_TRUTH_DIR"
    exit 1
fi

VIDEO_PATH="$1"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"
GROUND_TRUTH_DIR="${3:-$DEFAULT_GROUND_TRUTH_DIR}"

# --- Validation ---
if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ Error: Video file does not exist: $VIDEO_PATH"
    exit 1
fi

if [ ! -d "$GROUND_TRUTH_DIR" ]; then
    echo "❌ Error: Ground truth directory does not exist: $GROUND_TRUTH_DIR"
    echo "   Ground truth is REQUIRED for the fixed implementation."
    exit 1
fi

VIDEO_BASENAME=$(basename "$VIDEO_PATH" .mp4)
echo "🎬 SINGLE VIDEO FIXED FACE RECOGNITION"
echo "======================================="
echo "Video: $VIDEO_PATH"
echo "Basename: $VIDEO_BASENAME"
echo "Output Dir: $OUTPUT_DIR"
echo "Ground Truth Dir: $GROUND_TRUTH_DIR"
echo "Recognition Threshold: $RECOGNITION_THRESHOLD (lenient)"
echo "GPU: $GPU_ID"
echo ""

mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")"

# --- Environment Setup ---
echo "🔧 Setting up environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV_NAME

# === 1) 환경 정리 + 모델 캐시 경로 고정 ===
echo "Setting up clean environment..."
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# InsightFace 모델 캐시 루트 (항상 이쪽을 보게 만듦)
export INSIGHTFACE_HOME="${INSIGHTFACE_HOME:-$CONDA_PREFIX/.insightface}"
mkdir -p "$INSIGHTFACE_HOME"

# === 2) ORT(CUDA 12 + cuDNN 9) 런타임 경로 구성 ===
echo "Configuring CUDA runtime paths..."
# 파이썬 부버전 확인 (3.10 등)
PYVER=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")

# 기본: conda 안에 설치된 CUDA 런타임 + cuDNN 9 + 기타 라이브러리
export LD_LIBRARY_PATH="\
$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cuda_runtime/lib:\
$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cublas/lib:\
$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cudnn/lib:\
$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cufft/lib:\
$CONDA_PREFIX/lib/python$PYVER/site-packages/nvidia/cuda_nvrtc/lib:\
$CONDA_PREFIX/lib:\
$LD_LIBRARY_PATH"

# (선택) 시스템에 있다면 추가 — 누락 시에만 보강용
if [ -d /usr/local/lib/ollama/cuda_v12 ]; then
  export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH"
fi

echo "✅ Environment setup complete"
echo ""

# === 3) 실행 전 프리플라이트 (ORT CUDA EP + 런타임 so 체크) ===
echo "🧪 Preflight Check: ORT CUDA EP + Runtime Libraries"
echo "====================================================="
python - <<'PYIN'
import ctypes, sys, os
import onnxruntime as ort

print('[Preflight] ORT:', ort.__version__, '| providers:', ort.get_available_providers())
assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'CUDA EP not available'

need = [
  'libcublasLt.so.12','libcublas.so.12','libcudart.so.12',
  'libcurand.so.10','libcufft.so.11','libnvrtc.so.12','libcudnn.so.9'
]
missing=[]
for so in need:
    try:
        ctypes.CDLL(so)
    except OSError as e:
        print('[Preflight] Missing:', so, '->', e); missing.append(so)
if missing:
    print('[Preflight] FAIL. Missing libs:', missing); sys.exit(2)
print('[Preflight] CUDA runtime chain OK')
PYIN

# === 4) InsightFace 모델 팩 체크 (기존 모델 보존) ===
echo ""
echo "🔧 InsightFace Model Pack Check"
echo "==============================="
python - <<'PYIN'
import os, sys, onnxruntime as ort
from insightface.app import FaceAnalysis

root = os.environ.get('INSIGHTFACE_HOME')
print(f'[Check] Testing existing antelopev2 models at {root}...')

try:
    app = FaceAnalysis(name='antelopev2', root=root, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640,640))
    print('[Check] ✅ Existing antelopev2 models work perfectly!')
    
    # Final dummy test
    import numpy as np
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = app.get(img)
    print('[OK] antelopev2 ready with dummy test passed')
    
except Exception as e:
    print('[Check] ❌ Existing models failed:', e)
    print('[Check] This might indicate missing or corrupted models.')
    print('[Check] Manual intervention required - check INSIGHTFACE_HOME.')
    print('[Check] You may need to manually reinstall InsightFace models.')
    sys.exit(1)
PYIN

echo "✅ All preflight checks PASSED - InsightFace is properly configured"
echo ""

# --- Step 1: Validation ---
echo "🧪 Step 1: Validating Ground Truth"
echo "===================================="
if CUDA_VISIBLE_DEVICES=$GPU_ID python validate_ground_truth.py \
    --video_path "$VIDEO_PATH" \
    --ground_truth_dir "$GROUND_TRUTH_DIR"; then
    echo ""
    echo "✅ Validation PASSED - proceeding with face recognition"
else
    echo ""
    echo "❌ Validation FAILED - cannot proceed"
    echo "   Please fix the ground truth issues before running face recognition."
    echo "   Common fixes:"
    echo "   - Ensure ground truth CSV exists: $GROUND_TRUTH_DIR/$VIDEO_BASENAME.csv"
    echo "   - Check CSV format and content"
    echo "   - Verify InsightFace installation"
    exit 1
fi

echo ""

# --- Step 2: Face Recognition ---
echo "🔧 Step 2: Running Fixed Face Recognition"
echo "=========================================="
if CUDA_VISIBLE_DEVICES=$GPU_ID python face_recognition_fixed.py \
    --video_path "$VIDEO_PATH" \
    --execution_provider auto \
    --insightface_root "$INSIGHTFACE_HOME" \
    --insightface_model antelopev2 \
    --output_dir "$OUTPUT_DIR" \
    --ground_truth_dir "$GROUND_TRUTH_DIR" \
    --recognition_threshold $RECOGNITION_THRESHOLD; then
    echo ""
    echo "✅ Face recognition COMPLETED successfully!"
else
    echo ""
    echo "❌ Face recognition FAILED"
    exit 1
fi

echo ""

# --- Step 3: Results Summary ---
echo "📊 Step 3: Results Summary"
echo "==========================="

OUTPUT_CSV="$OUTPUT_DIR/${VIDEO_BASENAME}_fixed_face_recognition.csv"
OUTPUT_VIDEO="$OUTPUT_DIR/${VIDEO_BASENAME}_fixed_face_recognition.mp4"

if [ -f "$OUTPUT_CSV" ]; then
    echo "✅ Results CSV: $OUTPUT_CSV"
    
    # Analyze results using Python
    python - <<PYIN
import pandas as pd
import sys

try:
    df = pd.read_csv("$OUTPUT_CSV")
    print(f"   📈 Total detections: {len(df)}")
    
    if 'person_id' in df.columns:
        unique_persons = sorted(df['person_id'].unique())
        print(f"   👥 Unique person IDs: {unique_persons}")
        
        # Check for problematic high person IDs
        problematic_ids = [pid for pid in unique_persons if str(pid).isdigit() and int(pid) > 10]
        if problematic_ids:
            print(f"   ⚠️  WARNING: Potentially problematic person IDs: {problematic_ids}")
        else:
            print("   ✅ No problematic high person IDs found!")
    
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        print(f"   🔍 Source breakdown: {dict(source_counts)}")
    
    if 'confidence' in df.columns:
        recognized_df = df[df['source'] == 'recognized'] if 'source' in df.columns else df
        if len(recognized_df) > 0:
            avg_conf = recognized_df['confidence'].mean()
            min_conf = recognized_df['confidence'].min()
            max_conf = recognized_df['confidence'].max()
            print(f"   📊 Recognition confidence - avg: {avg_conf:.3f}, range: [{min_conf:.3f}, {max_conf:.3f}]")

except Exception as e:
    print(f"   ❌ Error analyzing results: {e}")
PYIN

else
    echo "❌ Results CSV not found: $OUTPUT_CSV"
fi

if [ -f "$OUTPUT_VIDEO" ]; then
    echo "✅ Results video: $OUTPUT_VIDEO"
else
    echo "❌ Results video not found: $OUTPUT_VIDEO"
fi

echo ""
echo "🎉 PROCESSING COMPLETE!"
echo "========================"
echo "The fixed implementation has completed processing."
echo "No spurious person IDs (11, 12, etc.) should appear in the results."
echo ""
echo "📁 Output files:"
echo "   CSV: $OUTPUT_CSV"
echo "   Video: $OUTPUT_VIDEO"
echo ""
echo "🔍 Next steps:"
echo "   1. Review the CSV results for person ID validation"
echo "   2. Watch the annotated video to verify recognition quality"
echo "   3. Adjust recognition threshold if needed (current: $RECOGNITION_THRESHOLD)"
