#!/bin/bash

# Single video processing script for CoTracker
# Usage: ./run_single_video.sh <video_path> [gpu_id]

# Check if video path is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <video_path> [gpu_id]"
    echo "Example: $0 /path/to/video.mp4 0"
    exit 1
fi

VIDEO_PATH="$1"
GPU_ID="${2:-0}"  # Default to GPU 0 if not specified

# Validate video file exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file does not exist: $VIDEO_PATH"
    exit 1
fi

# Validate GPU ID is a number
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU ID must be a number (got: $GPU_ID)"
    exit 1
fi

# Get video basename for session name
VIDEO_BASENAME=$(basename "$VIDEO_PATH")
VIDEO_NAME="${VIDEO_BASENAME%.*}"

# Create shorter session name by truncating long names and removing UUIDs
SHORT_NAME=$(echo "$VIDEO_NAME" | sed 's/[0-9a-f]\{8\}-[0-9a-f]\{4\}-[0-9a-f]\{4\}-[0-9a-f]\{4\}-[0-9a-f]\{12\}//g' | sed 's/[()\-]/_/g' | cut -c1-40)
SESSION_NAME="ct_${SHORT_NAME}_g${GPU_ID}"

# Create output directory
OUTPUT_DIR="/mas/robots/prg-ego4d/co-tracker"
mkdir -p "$OUTPUT_DIR"

echo "Processing single video with CoTracker..."
echo "Video: $VIDEO_PATH"
echo "GPU ID: $GPU_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Screen session: $SESSION_NAME"
echo "================================"

# Create temporary script for screen session
TEMP_SCRIPT="ct_single_${SHORT_NAME}_g${GPU_ID}_run.sh"

cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash

echo "Starting CoTracker processing..."
echo "Video: $VIDEO_PATH"
echo "GPU: $GPU_ID"
echo "Output: $OUTPUT_DIR"
echo "================================"

# Run CoTracker
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=.. python main.py \\
    --video_path "$VIDEO_PATH" \\
    --grid_size 30 \\
    --grid_query_frame 0 \\
    --save_dir "$OUTPUT_DIR"

exit_code=\$?

if [ \$exit_code -eq 0 ]; then
    echo "SUCCESS: Successfully processed $VIDEO_NAME"
else
    echo "FAILED: Failed to process $VIDEO_NAME (exit code: \$exit_code)"
    if [ \$exit_code -eq 1 ]; then
        echo "WARNING: Video appears to be corrupted"
    fi
fi

echo "Processing completed for $VIDEO_NAME"
EOF

# Make script executable
chmod +x "$TEMP_SCRIPT"

# Launch screen session
LOG_FILE="${SESSION_NAME}.log"
screen -dmS "$SESSION_NAME" bash -c "./$TEMP_SCRIPT &> $LOG_FILE"

echo "Screen session '$SESSION_NAME' launched successfully!"
echo "Log file: $LOG_FILE"
echo ""
echo "To monitor progress:"
echo "  screen -r $SESSION_NAME"
echo "To view log:"
echo "  tail -f $LOG_FILE"
echo "To list all sessions:"
echo "  screen -ls"

# Clean up temp script after a delay (optional)
(sleep 5 && rm -f "$TEMP_SCRIPT") &
