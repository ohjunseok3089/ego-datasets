#!/bin/bash

# Tracklet-based Constrained Face Recognition for Ego4D
# This script implements an improved face recognition pipeline using:
# - Tracklet-based face tracking
# - Constrained graph clustering with must-link/cannot-link constraints
# - Quality-weighted embeddings
# - kNN graph + community detection

set -e

# Configuration
VIDEO_DIR=""
OUTPUT_DIR="processed_videos_tracklet"
GROUND_TRUTH_DIR=""
EXECUTION_PROVIDER="CUDAExecutionProvider"  # Change to CPUExecutionProvider if no GPU

# Parameters for tracklet-based clustering
SIMILARITY_THRESHOLD=0.6
K_NEIGHBORS=10
RECOGNITION_THRESHOLD=0.7
MAX_FRAMES=36000  # 20 minutes at 30fps

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed"
        exit 1
    fi
    
    # Check required Python packages
    python3 -c "import insightface, cv2, numpy, pandas, sklearn, networkx, scipy, faiss" 2>/dev/null || {
        print_error "Required Python packages are missing. Please install:"
        echo "pip install insightface opencv-python numpy pandas scikit-learn networkx scipy faiss-cpu"
        exit 1
    }
    
    print_success "All dependencies are available"
}

# Function to process a single video
process_video() {
    local video_path="$1"
    local video_name=$(basename "$video_path")
    
    print_info "Processing video: $video_name"
    
    # Create output directory for this video
    local video_output_dir="$OUTPUT_DIR/$(basename "$video_path" .mp4)"
    mkdir -p "$video_output_dir"
    
    # Build command
    local cmd="python3 face_recognition_tracklet_constrained.py"
    cmd="$cmd --video_path \"$video_path\""
    cmd="$cmd --output_dir \"$video_output_dir\""
    cmd="$cmd --similarity_threshold $SIMILARITY_THRESHOLD"
    cmd="$cmd --k_neighbors $K_NEIGHBORS"
    cmd="$cmd --recognition_threshold $RECOGNITION_THRESHOLD"
    cmd="$cmd --execution_provider $EXECUTION_PROVIDER"
    cmd="$cmd --max_frames $MAX_FRAMES"
    
    # Add ground truth if available
    if [ -n "$GROUND_TRUTH_DIR" ] && [ -d "$GROUND_TRUTH_DIR" ]; then
        cmd="$cmd --ground_truth_dir \"$GROUND_TRUTH_DIR\""
    fi
    
    print_info "Running command: $cmd"
    
    # Execute the command
    if eval $cmd; then
        print_success "Successfully processed: $video_name"
    else
        print_error "Failed to process: $video_name"
        return 1
    fi
}

# Function to process all videos in a directory
process_directory() {
    local dir_path="$1"
    
    print_info "Processing all videos in directory: $dir_path"
    
    # Find all video files
    local video_files=($(find "$dir_path" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) | sort))
    
    if [ ${#video_files[@]} -eq 0 ]; then
        print_warning "No video files found in: $dir_path"
        return
    fi
    
    print_info "Found ${#video_files[@]} video files"
    
    # Process each video
    local success_count=0
    local total_count=${#video_files[@]}
    
    for video_file in "${video_files[@]}"; do
        if process_video "$video_file"; then
            ((success_count++))
        fi
        echo "---"
    done
    
    print_success "Completed processing: $success_count/$total_count videos successfully"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] <video_path_or_directory>"
    echo ""
    echo "Options:"
    echo "  -o, --output-dir DIR     Output directory (default: $OUTPUT_DIR)"
    echo "  -g, --ground-truth DIR   Ground truth directory"
    echo "  -p, --provider PROVIDER  Execution provider (default: $EXECUTION_PROVIDER)"
    echo "  -s, --similarity THRESH  Similarity threshold (default: $SIMILARITY_THRESHOLD)"
    echo "  -k, --k-neighbors K      Number of neighbors (default: $K_NEIGHBORS)"
    echo "  -r, --recognition THRESH Recognition threshold (default: $RECOGNITION_THRESHOLD)"
    echo "  -f, --max-frames N       Max frames to process (default: $MAX_FRAMES)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 video.mp4"
    echo "  $0 -o results -g ground_truth/ video_directory/"
    echo "  $0 -p CPUExecutionProvider -s 0.7 video.mp4"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--ground-truth)
            GROUND_TRUTH_DIR="$2"
            shift 2
            ;;
        -p|--provider)
            EXECUTION_PROVIDER="$2"
            shift 2
            ;;
        -s|--similarity)
            SIMILARITY_THRESHOLD="$2"
            shift 2
            ;;
        -k|--k-neighbors)
            K_NEIGHBORS="$2"
            shift 2
            ;;
        -r|--recognition)
            RECOGNITION_THRESHOLD="$2"
            shift 2
            ;;
        -f|--max-frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            VIDEO_DIR="$1"
            shift
            ;;
    esac
done

# Check if video path is provided
if [ -z "$VIDEO_DIR" ]; then
    print_error "No video path or directory provided"
    show_usage
    exit 1
fi

# Check if video path exists
if [ ! -e "$VIDEO_DIR" ]; then
    print_error "Video path does not exist: $VIDEO_DIR"
    exit 1
fi

# Main execution
print_info "Starting Tracklet-based Constrained Face Recognition"
print_info "Output directory: $OUTPUT_DIR"
print_info "Execution provider: $EXECUTION_PROVIDER"
print_info "Similarity threshold: $SIMILARITY_THRESHOLD"
print_info "k-neighbors: $K_NEIGHBORS"
print_info "Recognition threshold: $RECOGNITION_THRESHOLD"
print_info "Max frames: $MAX_FRAMES"

# Check dependencies
check_dependencies

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process video(s)
if [ -f "$VIDEO_DIR" ]; then
    # Single video file
    process_video "$VIDEO_DIR"
elif [ -d "$VIDEO_DIR" ]; then
    # Directory of videos
    process_directory "$VIDEO_DIR"
else
    print_error "Invalid video path: $VIDEO_DIR"
    exit 1
fi

print_success "All processing completed!"
print_info "Results saved in: $OUTPUT_DIR"
