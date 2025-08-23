#!/bin/bash

# Simple Aria Diarization using pyannote.audio
# Much simpler than the complex voice analysis version

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARIA_RAW_DIR="/mas/robots/prg-aria/raw"
ARIA_MP4_DIR="/mas/robots/prg-aria/dataset"  # MP4 files directory
OUTPUT_DIR="/mas/robots/prg-aria/processed_with_speakers"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Simple Aria Diarization with pyannote.audio ===${NC}"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
VENV_DIR="${SCRIPT_DIR}/venv_simple"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Install requirements
echo -e "${YELLOW}Installing simple requirements...${NC}"
pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements_simple.txt"

# Check for HuggingFace token
HF_TOKEN_FILE="${HOME}/.huggingface_token"
if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo -e "${RED}Error: HuggingFace token file not found: $HF_TOKEN_FILE${NC}"
    echo -e "${YELLOW}Please save your HuggingFace access token to: $HF_TOKEN_FILE${NC}"
    echo -e "${YELLOW}Get your token from: https://hf.co/settings/tokens${NC}"
    exit 1
fi

HF_TOKEN=$(cat "$HF_TOKEN_FILE")
echo -e "${GREEN}Found HuggingFace token${NC}"

# Check input directories
if [ ! -d "$ARIA_RAW_DIR" ]; then
    echo -e "${RED}Error: Input directory does not exist: $ARIA_RAW_DIR${NC}"
    exit 1
fi

if [ ! -d "$ARIA_MP4_DIR" ]; then
    echo -e "${RED}Error: MP4 directory does not exist: $ARIA_MP4_DIR${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the simple diarization script
echo -e "${GREEN}Starting simple pyannote.audio diarization...${NC}"
echo -e "Input directory: $ARIA_RAW_DIR"
echo -e "MP4 directory: $ARIA_MP4_DIR"
echo -e "Output directory: $OUTPUT_DIR"

# Using CUDA for faster processing

python3 "${SCRIPT_DIR}/aria_simple_diarization.py" \
    --input_dir "$ARIA_RAW_DIR" \
    --mp4_dir "$ARIA_MP4_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --auth_token "$HF_TOKEN"

echo -e "${GREEN}=== Simple Diarization Complete ===${NC}"
echo -e "Results saved to: $OUTPUT_DIR"

# Function to process a single recording
process_single() {
    if [ -z "$1" ]; then
        echo -e "${RED}Usage: $0 single <recording_name>${NC}"
        exit 1
    fi
    
    RECORDING_NAME="$1"
    SINGLE_DIR="$ARIA_RAW_DIR/$RECORDING_NAME"
    
    if [ ! -d "$SINGLE_DIR" ]; then
        echo -e "${RED}Error: Recording directory does not exist: $SINGLE_DIR${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Processing single recording: $RECORDING_NAME${NC}"
    
    python3 "${SCRIPT_DIR}/aria_simple_diarization.py" \
        --input_dir "$ARIA_RAW_DIR" \
        --mp4_dir "$ARIA_MP4_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --auth_token "$HF_TOKEN" \
        --single "$RECORDING_NAME"
}

# Handle command line arguments
case "${1:-}" in
    "single")
        process_single "$2"
        ;;
    "help"|"-h"|"--help")
        echo "Usage:"
        echo "  $0                         # Process entire dataset"
        echo "  $0 single <recording_name> # Process single recording"
        echo "  $0 help                    # Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 single loc1_script2_seq1_rec1"
        ;;
    "")
        # Default: process entire dataset (already done above)
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
