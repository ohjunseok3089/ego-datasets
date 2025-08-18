#!/bin/bash

# Set the base directory path
BASE_DIR="/mas/robots/prg-aria/parts/"

# Batch configuration: Parse batch number (1 or 2)
BATCH_NUM=${BATCH_NUM:-1}
if [ "$BATCH_NUM" != "1" ] && [ "$BATCH_NUM" != "2" ]; then
    echo "Error: BATCH_NUM must be 1 or 2 (got: $BATCH_NUM)"
    echo "Usage: BATCH_NUM=1 $0  or  BATCH_NUM=2 $0"
    exit 1
fi

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p /mas/robots/prg-aria/co-tracker/

echo "Starting batch processing of videos in parallel (4 GPUs per batch)..."
echo "Base directory: $BASE_DIR"
echo "Batch number: $BATCH_NUM (using GPUs for batch $BATCH_NUM)"
echo "================================"

# Gather all video files into an array (both .mp4 and .MP4)
mapfile -t all_video_files < <(find "$BASE_DIR" -maxdepth 1 \( -name '*.mp4' -o -name '*.MP4' \) | sort)
total_videos=${#all_video_files[@]}

if [ "$total_videos" -eq 0 ]; then
    echo "No video files (.mp4 or .MP4) found in $BASE_DIR"
    exit 1
fi

echo "Total videos found: $total_videos"

# First level batching: Split into 2 batches (for 8 GPUs total)
half_point=$((total_videos / 2))

if [ "$BATCH_NUM" = "1" ]; then
    # Batch 1: First half (0 to half_point-1)
    video_files=("${all_video_files[@]:0:$half_point}")
    echo "Batch 1: Processing videos 1-$half_point (${#video_files[@]} videos)"
else
    # Batch 2: Second half (half_point to end)
    remaining=$((total_videos - half_point))
    video_files=("${all_video_files[@]:$half_point:$remaining}")
    echo "Batch 2: Processing videos $((half_point + 1))-$total_videos (${#video_files[@]} videos)"
fi

# Second level batching: Split this batch's files into 4 groups for 4 GPUs
num_videos=${#video_files[@]}
num_gpus=4

declare -a groups
for ((i=0; i<num_gpus; i++)); do
    groups[$i]=""
done

for ((i=0; i<num_videos; i++)); do
    gpu_index=$((i % num_gpus))
    groups[$gpu_index]="${groups[$gpu_index]} ${video_files[$i]}"
done

# Display detailed file assignment for each GPU
echo ""
echo "=== FILE ASSIGNMENT FOR BATCH $BATCH_NUM ==="
for ((gpu=0; gpu<num_gpus; gpu++)); do
    group_videos=(${groups[$gpu]})
    echo "[Batch $BATCH_NUM GPU $gpu] Assigned ${#group_videos[@]} videos:"
    for ((j=0; j<${#group_videos[@]}; j++)); do
        video_basename=$(basename "${group_videos[$j]}")
        echo "  $((j+1)). $video_basename"
    done
    echo ""
done

# Function to process a group of videos (to be run in each screen session)
process_group() {
    local gpu_id=$1
    shift
    local videos=("$@")
    local count=0
    local failed=0
    for video_file in "${videos[@]}"; do
        video_basename=$(basename "$video_file" .mp4)
        echo "[GPU $gpu_id] Processing video: $video_basename"
        echo "[GPU $gpu_id] Video path: $video_file"

        echo "[GPU $gpu_id] Running CoTracker on $video_basename..."
        CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=.. python main.py \
            --video_path "$video_file" \
            --grid_size 30 \
            --grid_query_frame 0 \
            --save_dir "/mas/robots/prg-aria/co-tracker"
        if [ $? -eq 0 ]; then
            echo "[GPU $gpu_id] SUCCESS: Successfully processed $video_basename"
            ((count++))
        else
            echo "[GPU $gpu_id] FAILED: Failed to process $video_basename"
            ((failed++))
        fi
        echo "--------------------------------"
    done
    echo "[GPU $gpu_id] Done. Successfully processed: $count, Failed: $failed"
}

# Launch 4 parallel screen sessions, one for each GPU, using temp scripts for robust argument passing and logging
for ((gpu=0; gpu<num_gpus; gpu++)); do
    group_videos=(${groups[$gpu]})
    if [ ${#group_videos[@]} -eq 0 ]; then
        continue
    fi
    temp_script="cotracker_batch${BATCH_NUM}_gpu${gpu}_run.sh"
    echo "#!/bin/bash" > $temp_script
    echo "count=0" >> $temp_script
    echo "failed=0" >> $temp_script
    echo "" >> $temp_script
    for video_file in "${group_videos[@]}"; do
        # Properly escape the video file path for the script
        escaped_video_file=$(printf '%q' "$video_file")
        echo "video_file=$escaped_video_file" >> $temp_script
        echo "video_basename=\$(basename \"\$video_file\")" >> $temp_script
        echo "video_basename=\${video_basename%.*}" >> $temp_script
        echo "escaped_basename=\$(printf '%q' \"\$video_basename\")" >> $temp_script
        echo "echo \"[GPU $gpu] Processing video: \$escaped_basename\"" >> $temp_script
        echo "echo \"[GPU $gpu] Video path: \$video_file\"" >> $temp_script
        echo "echo \"[GPU $gpu] Running CoTracker on \$escaped_basename...\"" >> $temp_script
        echo "CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=.. python main.py --video_path \"\$video_file\" --grid_size 30 --grid_query_frame 0 --save_dir \"/mas/robots/prg-aria/co-tracker\"" >> $temp_script
        echo "exit_code=\$?" >> $temp_script
        echo "if [ \$exit_code -eq 0 ]; then" >> $temp_script
        echo "  echo \"[GPU $gpu] SUCCESS: Successfully processed \$escaped_basename\"" >> $temp_script
        echo "  ((count++))" >> $temp_script
        echo "else" >> $temp_script
        echo "  echo \"[GPU $gpu] FAILED: Failed to process \$escaped_basename (exit code: \$exit_code)\"" >> $temp_script
        echo "  if [ \$exit_code -eq 1 ]; then" >> $temp_script
        echo "    echo \"[GPU $gpu] WARNING: Video appears to be corrupted, continuing with next video...\"" >> $temp_script
        echo "  fi" >> $temp_script
        echo "  ((failed++))" >> $temp_script
        echo "fi" >> $temp_script
        echo "echo \"--------------------------------\"" >> $temp_script
    done
    echo "echo \"[Batch $BATCH_NUM GPU $gpu] Done. Successfully processed: \$count, Failed: \$failed\"" >> $temp_script
    chmod +x $temp_script
    screen -dmS cotracker_batch${BATCH_NUM}_gpu$gpu bash -c "./$temp_script &> cotracker_batch${BATCH_NUM}_gpu${gpu}.log"
    echo "Launched screen session 'cotracker_batch${BATCH_NUM}_gpu$gpu' for Batch $BATCH_NUM GPU $gpu with ${#group_videos[@]} videos. Log: cotracker_batch${BATCH_NUM}_gpu${gpu}.log"
done

echo "================================"
echo "Batch $BATCH_NUM processing launched in 4 parallel screen sessions!"
echo "Use 'screen -ls' to see running sessions. Attach with 'screen -r cotracker_batch${BATCH_NUM}_gpuX' (X=0,1,2,3)."
echo "Output videos saved in: /mas/robots/prg-aria/co-tracker/" 