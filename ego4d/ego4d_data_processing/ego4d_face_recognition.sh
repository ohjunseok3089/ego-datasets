#!/bin/bash

# --- Configuration ---
BASE_DIR="/mas/robots/prg-ego4d/raw/v2/full_scale.gaze/"
OUTPUT_DIR="/mas/robots/prg-ego4d/processed_face_recognition_videos/"
NUM_GPUS=4
CONDA_ENV_NAME="ego-dataset"
CUDA_LIB_PATH="/usr/local/cuda-11.8/lib64"
GROUND_TRUTH_DIR="/mas/robots/prg-ego4d/face_detection/"

# --- Script Start ---
echo "Starting Face Recognition Batch Processing..."
echo "============================================="

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist!"
    exit 1
fi

# Check if ground truth directory exists
if [ ! -d "$GROUND_TRUTH_DIR" ]; then
    echo "Warning: Ground truth directory $GROUND_TRUTH_DIR does not exist!"
    echo "Face recognition will proceed without ground truth matching."
    GROUND_TRUTH_DIR=""
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Step 1: Find all video files
echo "Finding video files..."

video_files=()
for f in "$BASE_DIR"/*.mp4; do
    if [ -f "$f" ]; then
        video_files+=("$f")
    fi
done

num_videos=${#video_files[@]}

if [ "$num_videos" -eq 0 ]; then
    echo "No valid video files found to process."
    exit 1
fi

echo "Found $num_videos video files to process."

# Step 2: Distribute the videos across the available GPUs
declare -a gpu_jobs
for ((i=0; i<NUM_GPUS; i++)); do
    gpu_jobs[$i]=""
done

for ((i=0; i<num_videos; i++)); do
    gpu_index=$((i % NUM_GPUS))
    gpu_jobs[$gpu_index]+="${video_files[$i]};"
done

# Step 3: Launch parallel screen sessions for each GPU
echo "Launching $NUM_GPUS parallel screen sessions..."
echo "============================================="

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    job_list=${gpu_jobs[$gpu]}
    if [ -z "$job_list" ]; then
        echo "[GPU $gpu] No jobs assigned. Skipping."
        continue
    fi

    session_name="face_rec_gpu${gpu}"
    log_file="${session_name}.log"
    
    temp_script="${session_name}_run.sh"
    echo "#!/bin/bash" > "$temp_script"
    echo "echo '[GPU $gpu] Starting processing... Log file: $log_file'" >> "$temp_script"
    
    echo "echo '[GPU $gpu] Activating Conda environment: $CONDA_ENV_NAME'" >> "$temp_script"
    echo "source \"\$(conda info --base)/etc/profile.d/conda.sh\"" >> "$temp_script"
    echo "conda activate $CONDA_ENV_NAME" >> "$temp_script"

    # --- Set the CUDA library path ---
    echo "echo '[GPU $gpu] Setting CUDA library path...'" >> "$temp_script"
    echo "export LD_LIBRARY_PATH=$CUDA_LIB_PATH:\$LD_LIBRARY_PATH" >> "$temp_script"
    # -------------------------------------------

    echo "IFS=';' read -ra videos_to_process <<< \"$job_list\"" >> "$temp_script"
    echo "for video_path in \"\${videos_to_process[@]}\"; do" >> "$temp_script"
    echo "    if [ -n \"\$video_path\" ]; then" >> "$temp_script"
    echo "        echo \"[GPU $gpu] --------------------------------------------------\"" >> "$temp_script"
    echo "        echo \"[GPU $gpu] Processing video: \$video_path\"" >> "$temp_script"
    echo "        CUDA_VISIBLE_DEVICES=$gpu python face_recognition_global_gallery.py \\" >> "$temp_script"
    echo "            --video_path \"\$video_path\" \\" >> "$temp_script"
    if [ -n "$GROUND_TRUTH_DIR" ]; then
        echo "            --output_dir \"$OUTPUT_DIR\" \\" >> "$temp_script"
        echo "            --ground_truth_dir \"$GROUND_TRUTH_DIR\"" >> "$temp_script"
    else
        echo "            --output_dir \"$OUTPUT_DIR\"" >> "$temp_script"
    fi
    echo "    fi" >> "$temp_script"
    echo "done" >> "$temp_script"
    echo "echo \"[GPU $gpu] All assigned jobs are complete.\"" >> "$temp_script"
    
    chmod +x "$temp_script"

    screen -dmS "$session_name" bash -c "./$temp_script &> $log_file"
    echo "Launched screen session '$session_name' for GPU $gpu. Log: $log_file"
done

echo "============================================="
echo "All processing jobs launched!"
echo "Use 'screen -ls' to see running sessions."
echo "Attach with 'screen -r face_rec_gpu0' to monitor a specific GPU."
echo "Output files will be saved in: $OUTPUT_DIR/"