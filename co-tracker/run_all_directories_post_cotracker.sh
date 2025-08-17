#!/bin/bash

# Track Prediction Past Frame - Directory Batch Processing Script
# Processes all subdirectories in a given path using track_prediction_past_frame.py

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
INPUT_DIR=""
OUTPUT_DIR=""
MODE="advanced"

show_help() {
    cat << EOF
Usage: $0 [OPTIONS] <input_directory> <output_directory>

Process all subdirectories using track_prediction_past_frame.py

ARGUMENTS:
    input_directory     Path to directory containing subdirectories to process
    output_directory    Path to output directory for results

OPTIONS:
    -m, --mode MODE     Processing mode (default: advanced)
    -h, --help          Show this help message

EXAMPLES:
    # Process all subdirectories in /path/to/videos/ 
    $0 /path/to/videos/ /path/to/output/

    # With custom mode
    $0 -m basic /path/to/videos/ /path/to/output/

REQUIREMENTS:
    - track_prediction_past_frame.py must be in the same directory as this script
    - Python environment with required dependencies
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$INPUT_DIR" ]; then
                INPUT_DIR="$1"
            elif [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            else
                echo "Error: Too many arguments"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if both directories were provided
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Both input and output directories are required"
    show_help
    exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if track_prediction_past_frame.py exists
PYTHON_SCRIPT="$SCRIPT_DIR/track_prediction_past_frame.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: track_prediction_past_frame.py not found in $SCRIPT_DIR"
    exit 1
fi

# Convert to absolute paths
INPUT_DIR=$(realpath "$INPUT_DIR")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

echo "Directory Batch Processing - Track Prediction Past Frame"
echo "========================================================"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Mode: $MODE"
echo ""

# Find all subdirectories and process them
total_dirs=0
processed_dirs=0
failed_dirs=0

echo "Finding subdirectories..."
while IFS= read -r -d '' subdir; do
    ((total_dirs++))
    
    subdir_name=$(basename "$subdir")
    echo ""
    echo "[$processed_dirs/$total_dirs] Processing: $subdir_name"
    echo "=============================================="
    
    # Run track_prediction_past_frame.py for this subdirectory
    echo "Executing: python3 \"$PYTHON_SCRIPT\" \"$subdir\" --output_dir \"$OUTPUT_DIR\" --mode $MODE"
    
    if python3 "$PYTHON_SCRIPT" "$subdir" --output_dir "$OUTPUT_DIR" --mode "$MODE"; then
        echo "✓ Successfully processed: $subdir_name"
        ((processed_dirs++))
    else
        echo "✗ Failed to process: $subdir_name"
        ((failed_dirs++))
    fi
    
done < <(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

echo ""
echo "========================================================"
echo "Batch Processing Complete!"
echo "========================================================"
echo "Total directories found: $total_dirs"
echo "Successfully processed: $processed_dirs"
echo "Failed: $failed_dirs"
echo ""

if [ $failed_dirs -gt 0 ]; then
    echo "⚠️  Some directories failed to process. Check the output above for details."
    exit 1
else
    echo "🎉 All directories processed successfully!"
    exit 0
fi
