#!/bin/bash

# Graph-based Face Recognition Processing Script for Ego4D
# This script processes videos using tracklet-based constrained clustering

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/face_recognition_tracklet_constrained.py"
OUTPUT_BASE_DIR="${SCRIPT_DIR}/results_tracklet_constrained"

# Default parameters
SIMILARITY_THRESHOLD=0.6
K_NEIGHBORS=10
RECOGNITION_THRESHOLD=0.65
MAX_FRAMES=36000  # 20 minutes at 30fps for gallery creation
SKIP_FRAMES=2     # Process every other frame for gallery
USE_MULTI_PROTOTYPE=""  # Set to "--use_multi_prototype" to enable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if script exists
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
    
    # Check Python packages
    python3 -c "import insightface" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "InsightFace not installed. Installing requirements..."
        pip install -r "${SCRIPT_DIR}/requirements_tracklet_constrained.txt"
    fi
    
    print_info "Prerequisites check complete"
}

# Function to process a single video
process_video() {
    local video_path="$1"
    local video_name=$(basename "$video_path" .mp4)
    local output_dir="${OUTPUT_BASE_DIR}/${video_name}"
    
    print_info "Processing video: $video_name"
    print_info "Output directory: $output_dir"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Detect execution provider
    local exec_provider="auto"
    
    # Run the processing
    python3 "$PYTHON_SCRIPT" \
        --video_path "$video_path" \
        --output_dir "$output_dir" \
        --similarity_threshold $SIMILARITY_THRESHOLD \
        --k_neighbors $K_NEIGHBORS \
        --recognition_threshold $RECOGNITION_THRESHOLD \
        --execution_provider "$exec_provider" \
        --max_frames $MAX_FRAMES \
        --skip_frames $SKIP_FRAMES \
        $USE_MULTI_PROTOTYPE
    
    if [ $? -eq 0 ]; then
        print_info "Successfully processed: $video_name"
        return 0
    else
        print_error "Failed to process: $video_name"
        return 1
    fi
}

# Function to process multiple videos
process_video_list() {
    local video_list_file="$1"
    
    if [ ! -f "$video_list_file" ]; then
        print_error "Video list file not found: $video_list_file"
        exit 1
    fi
    
    local total_videos=$(wc -l < "$video_list_file")
    local current=0
    local success=0
    local failed=0
    
    print_info "Processing $total_videos videos from list"
    
    while IFS= read -r video_path; do
        # Skip empty lines and comments
        [[ -z "$video_path" || "$video_path" == \#* ]] && continue
        
        current=$((current + 1))
        print_info "[$current/$total_videos] Processing..."
        
        if process_video "$video_path"; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
        fi
    done < "$video_list_file"
    
    print_info "Batch processing complete: $success succeeded, $failed failed"
}

# Function to process directory of videos
process_video_directory() {
    local video_dir="$1"
    local pattern="${2:-*.mp4}"
    
    if [ ! -d "$video_dir" ]; then
        print_error "Directory not found: $video_dir"
        exit 1
    fi
    
    local videos=("$video_dir"/$pattern)
    local total_videos=${#videos[@]}
    local current=0
    local success=0
    local failed=0
    
    print_info "Found $total_videos videos in directory"
    
    for video_path in "${videos[@]}"; do
        [ ! -f "$video_path" ] && continue
        
        current=$((current + 1))
        print_info "[$current/$total_videos] Processing..."
        
        if process_video "$video_path"; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
        fi
    done
    
    print_info "Batch processing complete: $success succeeded, $failed failed"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] <video_path|video_list|directory>

Process videos using graph-based face recognition with tracklets.

Options:
    -h, --help                    Show this help message
    -s, --similarity <float>      Similarity threshold for clustering (default: $SIMILARITY_THRESHOLD)
    -k, --kneighbors <int>       Number of neighbors for kNN graph (default: $K_NEIGHBORS)
    -r, --recognition <float>     Recognition threshold (default: $RECOGNITION_THRESHOLD)
    -m, --max-frames <int>       Max frames for gallery creation (default: $MAX_FRAMES)
    -f, --skip-frames <int>      Frame skip rate for gallery (default: $SKIP_FRAMES)
    -p, --multi-prototype        Use multiple prototypes per person
    -o, --output-dir <path>      Base output directory (default: $OUTPUT_BASE_DIR)
    
Arguments:
    video_path                    Path to a single video file
    video_list                    Path to text file containing video paths (one per line)
    directory                     Directory containing video files

Examples:
    # Process single video
    $0 /path/to/video.mp4
    
    # Process list of videos
    $0 video_list.txt
    
    # Process all videos in directory
    $0 /path/to/video/directory/
    
    # Process with custom parameters
    $0 -s 0.7 -k 15 -p video.mp4

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -s|--similarity)
            SIMILARITY_THRESHOLD="$2"
            shift 2
            ;;
        -k|--kneighbors)
            K_NEIGHBORS="$2"
            shift 2
            ;;
        -r|--recognition)
            RECOGNITION_THRESHOLD="$2"
            shift 2
            ;;
        -m|--max-frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        -f|--skip-frames)
            SKIP_FRAMES="$2"
            shift 2
            ;;
        -p|--multi-prototype)
            USE_MULTI_PROTOTYPE="--use_multi_prototype"
            shift
            ;;
        -o|--output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        *)
            INPUT_PATH="$1"
            shift
            ;;
    esac
done

# Main execution
main() {
    print_info "========================================="
    print_info "Graph-based Face Recognition Pipeline"
    print_info "========================================="
    
    # Check prerequisites
    check_prerequisites
    
    # Check input
    if [ -z "$INPUT_PATH" ]; then
        print_error "No input specified"
        show_usage
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_BASE_DIR"
    
    # Determine input type and process
    if [ -f "$INPUT_PATH" ]; then
        # Check if it's a video or a list
        if [[ "$INPUT_PATH" == *.mp4 || "$INPUT_PATH" == *.avi || "$INPUT_PATH" == *.mov ]]; then
            print_info "Processing single video file"
            process_video "$INPUT_PATH"
        else
            print_info "Processing video list file"
            process_video_list "$INPUT_PATH"
        fi
    elif [ -d "$INPUT_PATH" ]; then
        print_info "Processing video directory"
        process_video_directory "$INPUT_PATH"
    else
        print_error "Input not found: $INPUT_PATH"
        exit 1
    fi
    
    print_info "========================================="
    print_info "Pipeline execution complete"
    print_info "Results saved in: $OUTPUT_BASE_DIR"
    print_info "========================================="
}

# Run main function
main