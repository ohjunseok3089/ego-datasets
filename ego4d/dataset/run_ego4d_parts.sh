#!/bin/bash

# Set the base directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/ego4d_into_parts.py"

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

echo "Starting EGO4D video processing in screen session..."
echo "Python script: $PYTHON_SCRIPT"
echo "================================"

# Create a temporary script for the screen session
temp_script="ego4d_parts_run.sh"
echo "#!/bin/bash" > $temp_script
echo "cd \"$SCRIPT_DIR\"" >> $temp_script
echo "echo \"Starting EGO4D video splitting process...\"" >> $temp_script
echo "echo \"Working directory: \$(pwd)\"" >> $temp_script
echo "echo \"Python script: ego4d_into_parts.py\"" >> $temp_script
echo "echo \"Log will be saved to: ego4d_parts.log\"" >> $temp_script
echo "echo \"================================\"" >> $temp_script
echo "python ego4d_into_parts.py 2>&1 | tee ego4d_parts.log" >> $temp_script
echo "exit_code=\$?" >> $temp_script
echo "echo \"================================\"" >> $temp_script
echo "if [ \$exit_code -eq 0 ]; then" >> $temp_script
echo "  echo \"SUCCESS: EGO4D video processing completed successfully!\"" >> $temp_script
echo "else" >> $temp_script
echo "  echo \"FAILED: EGO4D video processing failed with exit code: \$exit_code\"" >> $temp_script
echo "fi" >> $temp_script
echo "echo \"Check ego4d_parts.log for detailed output\"" >> $temp_script
echo "echo \"Press any key to exit...\"" >> $temp_script
echo "read -n 1" >> $temp_script

chmod +x $temp_script

# Launch screen session
screen -dmS ego4d_parts bash -c "./$temp_script"

echo "Launched screen session 'ego4d_parts' for EGO4D video processing!"
echo "Use 'screen -ls' to see running sessions."
echo "Attach with 'screen -r ego4d_parts' to monitor progress."
echo "Log file: $SCRIPT_DIR/ego4d_parts.log"
echo "Output videos will be saved in: /mas/robots/prg-ego4d/parts/"
echo ""
echo "To monitor progress from another terminal:"
echo "  tail -f $SCRIPT_DIR/ego4d_parts.log"