#!/bin/bash
#
# Video Analysis CLI - Unified Interface
#
# A convenient wrapper for all video analysis functions.
# Supports full workflow or individual steps.
#

set -e

# Default values
TEMP_DIR="/tmp/video_analysis"
OUTPUT_DIR="./video_analysis_output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: video_analysis.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  extract      Extract frames from video"
    echo "  transcribe   Transcribe audio from video"
    echo "  analyze      Analyze frames with Claude Vision"
    echo "  report       Generate analysis report"
    echo "  full         Run complete analysis workflow"
    echo "  check        Check dependencies"
    echo ""
    echo "Options:"
    echo "  -i, --input          Input video file or YouTube URL"
    echo "  -o, --output         Output directory (default: ./video_analysis_output)"
    echo "  -m, --mode           Analysis mode (describe|objects|actions|academic|tutorial|demo|meeting)"
    echo "  --frames-dir        Directory with extracted frames"
    echo "  --transcript        Path to transcript JSON"
    echo "  --analysis          Path to analysis JSON"
    echo "  --fps               Frames per second for extraction"
    echo "  --language          Language for transcription"
    echo "  --model             Whisper model size"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Extract frames from a video"
    echo "  ./video_analysis.sh extract -i video.mp4 -o ./frames --fps 1"
    echo ""
    echo "  # Full workflow for academic video"
    echo "  ./video_analysis.sh full -i video.mp4 -m academic"
    echo ""
    echo "  # Analyze YouTube video"
    echo "  ./video_analysis.sh full -i 'https://youtube.com/watch?v=xxx' -m tutorial"
}

check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    local missing=()
    
    # Check ffmpeg
    if ! command -v ffmpeg &> /dev/null; then
        missing+=("ffmpeg")
    fi
    
    # Check ffprobe
    if ! command -v ffprobe &> /dev/null; then
        missing+=("ffprobe")
    fi
    
    # Check yt-dlp (optional)
    if ! command -v yt-dlp &> /dev/null; then
        echo -e "${YELLOW}Warning: yt-dlp not found (YouTube download disabled)${NC}"
    fi
    
    # Check Python packages
    if ! python3 -c "import faster_whisper" 2>/dev/null; then
        missing+=("faster-whisper")
    fi
    
    if ! python3 -c "import anthropic" 2>/dev/null; then
        missing+=("anthropic")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}Missing dependencies: ${missing[*]}${NC}"
        echo "Install with:"
        echo "  pip install faster-whisper anthropic"
        echo "  brew install ffmpeg yt-dlp"
        return 1
    fi
    
    echo -e "${GREEN}All dependencies available${NC}"
}

cmd_extract() {
    local input=""
    local output="$OUTPUT_DIR/frames"
    local mode="uniform"
    local fps=1
    local threshold=0.3
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--input) input="$2"; shift 2 ;;
            -o|--output) output="$2"; shift 2 ;;
            -m|--mode) mode="$2"; shift 2 ;;
            --fps) fps="$2"; shift 2 ;;
            --threshold) threshold="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [ -z "$input" ]; then
        echo -e "${RED}Error: --input is required${NC}"
        exit 1
    fi
    
    mkdir -p "$output"
    
    python3 "$(dirname "$0")/extract_frames.py" \
        --input "$input" \
        --output "$output" \
        --mode "$mode" \
        --fps "$fps" \
        --threshold "$threshold" \
        --temp-dir "$TEMP_DIR"
}

cmd_transcribe() {
    local input=""
    local output="$OUTPUT_DIR/transcript"
    local model="medium"
    local language="auto"
    local format="both"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--input) input="$2"; shift 2 ;;
            -o|--output) output="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --language) language="$2"; shift 2 ;;
            -f|--format) format="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [ -z "$input" ]; then
        echo -e "${RED}Error: --input is required${NC}"
        exit 1
    fi
    
    mkdir -p "$output"
    
    python3 "$(dirname "$0")/transcribe.py" \
        --input "$input" \
        --output "$output" \
        --model "$model" \
        --language "$language" \
        --format "$format" \
        --temp-dir "$TEMP_DIR"
}

cmd_analyze() {
    local frames_dir=""
    local transcript=""
    local mode="describe"
    local batch_size=5
    local output="$OUTPUT_DIR/analysis.json"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --frames-dir) frames_dir="$2"; shift 2 ;;
            --transcript) transcript="$2"; shift 2 ;;
            -m|--mode) mode="$2"; shift 2 ;;
            --batch-size) batch_size="$2"; shift 2 ;;
            -o|--output) output="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [ -z "$frames_dir" ]; then
        echo -e "${RED}Error: --frames-dir is required${NC}"
        exit 1
    fi
    
    local transcript_arg=""
    if [ -n "$transcript" ]; then
        transcript_arg="--transcript $transcript"
    fi
    
    python3 "$(dirname "$0")/analyze_frames.py" \
        --frames-dir "$frames_dir" \
        $transcript_arg \
        --mode "$mode" \
        --batch-size "$batch_size" \
        --output "$output"
}

cmd_report() {
    local analysis=""
    local transcript=""
    local frames_dir=""
    local mode=""
    local output="$OUTPUT_DIR/REPORT.md"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --analysis) analysis="$2"; shift 2 ;;
            --transcript) transcript="$2"; shift 2 ;;
            --frames-dir) frames_dir="$2"; shift 2 ;;
            -m|--mode) mode="$2"; shift 2 ;;
            -o|--output) output="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [ -z "$analysis" ]; then
        echo -e "${RED}Error: --analysis is required${NC}"
        exit 1
    fi
    
    local transcript_arg=""
    if [ -n "$transcript" ]; then
        transcript_arg="--transcript $transcript"
    fi
    
    local frames_arg=""
    if [ -n "$frames_dir" ]; then
        frames_arg="--frames-dir $frames_dir"
    fi
    
    local mode_arg=""
    if [ -n "$mode" ]; then
        mode_arg="--mode $mode"
    fi
    
    python3 "$(dirname "$0")/generate_report.py" \
        --analysis "$analysis" \
        $transcript_arg \
        $frames_arg \
        $mode_arg \
        --output "$output"
}

cmd_full() {
    local input=""
    local mode="describe"
    local language="auto"
    local fps=1
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--input) input="$2"; shift 2 ;;
            -m|--mode) mode="$2"; shift 2 ;;
            --language) language="$2"; shift 2 ;;
            --fps) fps="$2"; shift 2 ;;
            -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [ -z "$input" ]; then
        echo -e "${RED}Error: --input is required${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}=== Video Analysis Full Workflow ===${NC}"
    echo "Input: $input"
    echo "Mode: $mode"
    echo "Output: $OUTPUT_DIR"
    echo ""
    
    # Create output directories
    local frames_dir="$OUTPUT_DIR/frames"
    local transcript_dir="$OUTPUT_DIR/transcript"
    local analysis_file="$OUTPUT_DIR/analysis.json"
    local report_file="$OUTPUT_DIR/REPORT.md"
    
    mkdir -p "$OUTPUT_DIR" "$frames_dir" "$transcript_dir"
    
    # Step 1: Extract frames
    echo -e "${YELLOW}Step 1: Extracting frames...${NC}"
    cmd_extract "$input" "$frames_dir" "scene" "$fps"
    
    # Step 2: Transcribe
    echo ""
    echo -e "${YELLOW}Step 2: Transcribing audio...${NC}"
    cmd_transcribe "$input" "$transcript_dir" "medium" "$language"
    
    # Step 3: Analyze
    echo ""
    echo -e "${YELLOW}Step 3: Analyzing frames...${NC}"
    cmd_analyze "$frames_dir" "$transcript_dir/transcript.json" "$mode"
    
    # Step 4: Generate report
    echo ""
    echo -e "${YELLOW}Step 4: Generating report...${NC}"
    cmd_report "$analysis_file" "$transcript_dir/transcript.json" "$frames_dir" "$mode"
    
    echo ""
    echo -e "${GREEN}=== Analysis Complete ===${NC}"
    echo "Report: $report_file"
}

# Main command dispatcher
COMMAND="${1:-help}"
shift 2>/dev/null || true

case "$COMMAND" in
    extract)
        cmd_extract "$@"
        ;;
    transcribe)
        cmd_transcribe "$@"
        ;;
    analyze)
        cmd_analyze "$@"
        ;;
    report)
        cmd_report "$@"
        ;;
    full)
        cmd_full "$@"
        ;;
    check)
        check_dependencies
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        usage
        exit 1
        ;;
esac
