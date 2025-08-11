#!/bin/bash

# Ego4D Download Script
# Downloads annotations once and then specific video UIDs

# Configuration

OUTPUT_DIR="/mas/robots/prg-ego4d/raw"
DATASETS="full_scale annotations"

# Video UIDs to download
VIDEO_UIDS=(
    "30294c41-c90d-438a-af19-c1c74787d06b"
    "566ad4e5-1ce4-4679-9d19-ef63072c848c"
    "9c5b7322-d1cc-4b56-ae9d-85831f28fac1"
    "9ca2dc18-2c57-44cb-8c91-4b8b5c7ca223"
    "a223fcb2-8ffa-4826-bd0c-91027cf1c11e"
    "b3937482-c973-4263-957d-1d5366329dad"
)

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=== EGO4D Download Script ==="
echo "Output directory: $OUTPUT_DIR"
echo "Total videos to download: ${#VIDEO_UIDS[@]}"
echo "================================"

# Step 1: Download annotations (only once)
echo "Step 1: Downloading annotations..."
ego4d --output_directory "$OUTPUT_DIR" --datasets annotations -y

if [ $? -eq 0 ]; then
    echo "✓ Annotations downloaded successfully"
else
    echo "✗ Failed to download annotations"
    exit 1
fi

echo ""
echo "Step 2: Downloading videos..."
echo "================================"

# Step 2: Download each video UID
success_count=0
failed_count=0
failed_uids=()

for uid in "${VIDEO_UIDS[@]}"; do
    echo ""
    echo "Downloading video: $uid"
    echo "Progress: $((success_count + failed_count + 1))/${#VIDEO_UIDS[@]}"
    
    ego4d --output_directory "$OUTPUT_DIR" --datasets full_scale --video_uids "$uid" -y
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded: $uid"
        ((success_count++))
    else
        echo "✗ Failed to download: $uid"
        failed_uids+=("$uid")
        ((failed_count++))
    fi
done

# Final summary
echo ""
echo "================================"
echo "DOWNLOAD SUMMARY"
echo "================================"
echo "Successfully downloaded: $success_count videos"
echo "Failed downloads: $failed_count videos"

if [ ${#failed_uids[@]} -gt 0 ]; then
    echo ""
    echo "Failed UIDs:"
    for failed_uid in "${failed_uids[@]}"; do
        echo "  - $failed_uid"
    done
    
    echo ""
    echo "To retry failed downloads, run:"
    for failed_uid in "${failed_uids[@]}"; do
        echo "ego4d --output_directory \"$OUTPUT_DIR\" --datasets full_scale --video_uids \"$failed_uid\" -y --aws_profile_name ego4d"
    done
fi

echo ""
echo "All files saved to: $OUTPUT_DIR"
echo "Download script completed."