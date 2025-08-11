#!/bin/bash

# Set the base directory path
BASE_DIR="/mas/robots/prg-egocom/EGOCOM/720p/20min/parts/"

# Sub-batch control (split each GPU's list into TOTAL_BATCHES parts, run only BATCH_INDEX)
# Change these as needed or override via env: TOTAL_BATCHES=4 BATCH_INDEX=2 ./run_all_videos.sh
TOTAL_BATCHES=${TOTAL_BATCHES:-4}
# Allow BATCH as a shorthand alias for BATCH_INDEX (user convenience)
if [ -n "$BATCH" ] && [ -z "$BATCH_INDEX" ]; then
    BATCH_INDEX="$BATCH"
fi
BATCH_INDEX=${BATCH_INDEX:-1}

# Validate batch settings
if ! [[ "$TOTAL_BATCHES" =~ ^[0-9]+$ ]] || ! [[ "$BATCH_INDEX" =~ ^[0-9]+$ ]]; then
    echo "Error: TOTAL_BATCHES and BATCH_INDEX must be positive integers"
    exit 1
fi
if [ "$TOTAL_BATCHES" -lt 1 ]; then
    echo "Error: TOTAL_BATCHES must be >= 1"
    exit 1
fi
if [ "$BATCH_INDEX" -lt 1 ] || [ "$BATCH_INDEX" -gt "$TOTAL_BATCHES" ]; then
    echo "Error: BATCH_INDEX ($BATCH_INDEX) must be in [1, $TOTAL_BATCHES]"
    exit 1
fi

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p /mas/robots/prg-egocom/EGOCOM/720p/20min/co-tracker/

echo "Starting batch processing of videos in parallel (4 GPUs)..."
echo "Base directory: $BASE_DIR"
echo "Batch selection: $BATCH_INDEX / $TOTAL_BATCHES (per-GPU sub-batch)"
echo "================================"

# Gather all video files into an array (both .mp4 and .MP4)
mapfile -t video_files < <(find "$BASE_DIR" -maxdepth 1 \( -name '*.mp4' -o -name '*.MP4' \) | sort)
num_videos=${#video_files[@]}

if [ "$num_videos" -eq 0 ]; then
    echo "No video files (.mp4 or .MP4) found in $BASE_DIR"
    exit 1
fi

# Split video files into 4 groups for 4 GPUs
num_gpus=4

declare -a groups
for ((i=0; i<num_gpus; i++)); do
    groups[$i]=""
done

for ((i=0; i<num_videos; i++)); do
    gpu_index=$((i % num_gpus))
    groups[$gpu_index]="${groups[$gpu_index]} ${video_files[$i]}"
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
            --save_dir "/mas/robots/prg-egocom/EGOCOM/720p/5min_parts/co-tracker"
        if [ $? -eq 0 ]; then
            echo "[GPU $gpu_id] ✓ Successfully processed $video_basename"
            ((count++))
        else
            echo "[GPU $gpu_id] ✗ Failed to process $video_basename"
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

    # Split this GPU's videos into TOTAL_BATCHES sub-batches and pick BATCH_INDEX
    group_size=${#group_videos[@]}
    # Ceiling division for per-batch size
    per_batch=$(( (group_size + TOTAL_BATCHES - 1) / TOTAL_BATCHES ))
    offset=$(( (BATCH_INDEX - 1) * per_batch ))
    # Adjust length at tail
    length=$per_batch
    end=$(( offset + length ))
    if [ $end -gt $group_size ]; then
        length=$(( group_size - offset ))
    fi
    if [ $length -le 0 ]; then
        echo "[GPU $gpu] No videos in this batch (batch $BATCH_INDEX/$TOTAL_BATCHES). Skipping."
        continue
    fi
    group_videos=("${group_videos[@]:$offset:$length}")

    echo "[GPU $gpu] Selected sub-batch $BATCH_INDEX/$TOTAL_BATCHES: ${#group_videos[@]} videos (offset=$offset, size=$length)"
    temp_script="cotracker_b${BATCH_INDEX}of${TOTAL_BATCHES}_gpu${gpu}_run.sh"
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
        echo "echo \"[GPU $gpu] Processing video: \$video_basename\"" >> $temp_script
        echo "echo \"[GPU $gpu] Video path: \$video_file\"" >> $temp_script
        echo "echo \"[GPU $gpu] Running CoTracker on \$video_basename...\"" >> $temp_script
        echo "CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=.. python main.py --video_path \"\$video_file\" --grid_size 30 --grid_query_frame 0 --save_dir \"/mas/robots/prg-egocom/EGOCOM/720p/5min_parts/co-tracker\"" >> $temp_script
        echo "exit_code=\$?" >> $temp_script
        echo "if [ \$exit_code -eq 0 ]; then" >> $temp_script
        echo "  echo \"[GPU $gpu] SUCCESS: \$video_basename\"" >> $temp_script
        echo "  count=\$((count + 1))" >> $temp_script
        echo "else" >> $temp_script
        echo "  echo \"[GPU $gpu] FAILED: \$video_basename exit_code=\$exit_code\"" >> $temp_script
        echo "  if [ \$exit_code -eq 1 ]; then" >> $temp_script
        echo "    echo \"[GPU $gpu] Video appears corrupted; skipping\"" >> $temp_script
        echo "  fi" >> $temp_script
        echo "  failed=\$((failed + 1))" >> $temp_script
        echo "fi" >> $temp_script
        echo "echo \"--------------------------------\"" >> $temp_script
    done
    echo "echo \"[GPU $gpu] Done. Successfully processed: \$count, Failed: \$failed\"" >> $temp_script
    chmod +x $temp_script
    screen -dmS cotracker_b${BATCH_INDEX}of${TOTAL_BATCHES}_gpu$gpu bash -c "./$temp_script &> cotracker_b${BATCH_INDEX}of${TOTAL_BATCHES}_gpu${gpu}.log"
    echo "Launched screen session 'cotracker_b${BATCH_INDEX}of${TOTAL_BATCHES}_gpu$gpu' for GPU $gpu with ${#group_videos[@]} videos (batch $BATCH_INDEX/$TOTAL_BATCHES). Log: cotracker_b${BATCH_INDEX}of${TOTAL_BATCHES}_gpu${gpu}.log"
done

echo "================================"
echo "Batch processing launched in 4 parallel screen sessions!"
echo "Use 'screen -ls' to see running sessions. Attach with 'screen -r cotracker_gpuX' (X=0,1,2,3)."
echo "Output videos saved in: /mas/robots/prg-egocom/EGOCOM/720p/5min_parts/co-tracker/" 
