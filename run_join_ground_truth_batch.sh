#!/bin/bash

# Batch script to run join_ground_truth.py for all videos in EGOCOM/parts directory
# Usage: ./run_join_ground_truth_batch.sh

set -e

# Configuration
EGOCOM_ROOT="/mas/robots/prg-egocom/EGOCOM"
PARTS_DIR="$EGOCOM_ROOT/parts"
FACE_DETECTION_DIR="$EGOCOM_ROOT/face_detection"
BODY_DETECTION_DIR="$EGOCOM_ROOT/body_detection"
CO_TRACKER_DIR="$EGOCOM_ROOT/co-tracker-ground-truth"
TRANSCRIPT_CSV="$EGOCOM_ROOT/transcript/ground_truth_transcriptions_with_frames.csv"
FPS=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting batch processing of join_ground_truth.py${NC}"
echo "EGOCOM Root: $EGOCOM_ROOT"
echo "Parts Directory: $PARTS_DIR"
echo "Transcript CSV: $TRANSCRIPT_CSV"
echo ""

# Check if required directories exist
if [ ! -d "$PARTS_DIR" ]; then
    echo -e "${RED}Error: Parts directory not found: $PARTS_DIR${NC}"
    exit 1
fi

if [ ! -f "$TRANSCRIPT_CSV" ]; then
    echo -e "${RED}Error: Transcript CSV not found: $TRANSCRIPT_CSV${NC}"
    exit 1
fi

# Find all MP4 video files in parts
video_files=$(find "$PARTS_DIR" -maxdepth 1 -type f -name "vid_*.MP4" | sort)

if [ -z "$video_files" ]; then
    echo -e "${RED}Error: No MP4 video files found in $PARTS_DIR${NC}"
    exit 1
fi

echo "Found video files:"
echo "$video_files" | while read -r file; do
    echo "  - $(basename "$file")"
done
echo ""

# Counter for processed videos
total_videos=$(echo "$video_files" | wc -l)
processed=0
successful=0
failed=0

# Process each video file
echo "$video_files" | while read -r video_file; do
    if [ -z "$video_file" ]; then
        continue
    fi
    
    processed=$((processed + 1))
    video_filename=$(basename "$video_file")
    video_name="${video_filename%.MP4}"  # Remove .MP4 extension
    
    echo -e "${YELLOW}[$processed/$total_videos] Processing: $video_filename${NC}"
    
    # Extract base name without the parentheses part for file matching
    # e.g., vid_001__day_1__con_1__person_1_part1(0_1920_social_interaction) -> vid_001__day_1__con_1__person_1_part1
    base_name=$(echo "$video_name" | sed 's/(.*//')
    
    # Define expected file paths
    face_csv="$FACE_DETECTION_DIR/${base_name}_global_gallery_with_speaker.csv"
    body_csv="$BODY_DETECTION_DIR/${base_name}_detections_with_speaker.csv"
    co_tracker_json="$CO_TRACKER_DIR/${video_name}_analysis.json"
    
    # Check if required files exist
    missing_files=""
    if [ ! -f "$face_csv" ]; then
        missing_files="$missing_files\n  - Face CSV: $face_csv"
    fi
    if [ ! -f "$body_csv" ]; then
        missing_files="$missing_files\n  - Body CSV: $body_csv"
    fi
    if [ ! -f "$co_tracker_json" ]; then
        missing_files="$missing_files\n  - Co-tracker JSON: $co_tracker_json"
    fi
    
    if [ -n "$missing_files" ]; then
        echo -e "${RED}  Error: Missing required files:$missing_files${NC}"
        failed=$((failed + 1))
        echo ""
        continue
    fi
    
    echo "  Files found:"
    echo "    Base video: $video_file"
    echo "    Face CSV: $face_csv"
    echo "    Body CSV: $body_csv"
    echo "    Co-tracker JSON: $co_tracker_json"
    echo "    Transcript CSV: $TRANSCRIPT_CSV"
    echo ""
    
    # Run join_ground_truth.py
    echo "  Running join_ground_truth.py..."
    if python join_ground_truth.py \
        --base_video "$video_file" \
        --face_csv "$face_csv" \
        --body_csv "$body_csv" \
        --co_tracker_json "$co_tracker_json" \
        --transcript_csv "$TRANSCRIPT_CSV" \
        --fps "$FPS"; then
        echo -e "${GREEN}  ✓ Successfully processed $video_filename${NC}"
        successful=$((successful + 1))
    else
        echo -e "${RED}  ✗ Failed to process $video_filename${NC}"
        failed=$((failed + 1))
    fi
    
    echo ""
done

# Print summary
echo -e "${GREEN}=== Batch Processing Summary ===${NC}"
echo "Total videos found: $total_videos"
echo "Successfully processed: $successful"
echo "Failed: $failed"

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}All videos processed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some videos failed to process. Check the logs above for details.${NC}"
    exit 1
fi
