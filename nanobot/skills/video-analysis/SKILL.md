---
name: video-analysis
description: >
  Video content analysis for extracting frames, transcribing audio, analyzing visual
  content with Claude Vision, and generating structured reports. Use when analyzing
  video files or YouTube URLs, transcribing speech to text, extracting key frames,
  generating academic/tutorial/demo reports, or processing conference talks and lectures.
---

# Video Analysis Skill

Extract, transcribe, analyze, and report on video content.

## Quick Start

```bash
# Full workflow (recommended for most use cases)
./scripts/video_analysis.sh full -i video.mp4 -m academic -o ./output

# Individual steps
./scripts/video_analysis.sh extract -i video.mp4 --fps 1
./scripts/video_analysis.sh transcribe -i video.mp4 --language en
./scripts/video_analysis.sh analyze --frames-dir ./frames --mode academic
./scripts/video_analysis.sh report --analysis ./analysis.json --transcript ./transcript.json
```

## Core Capabilities

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `extract_frames.py` | Frame extraction with scene detection | `--mode uniform\|scene\|keyframe`, `--fps`, `--threshold` |
| `transcribe.py` | Audio transcription with Whisper | `--model tiny\|base\|small\|medium\|large`, `--language` |
| `analyze_frames.py` | Claude Vision frame analysis | `--mode describe\|objects\|actions\|academic\|tutorial\|demo\|meeting` |
| `generate_report.py` | Structured Markdown report | `--mode academic\|tutorial\|demo\|general` |

## Analysis Modes

- **academic**: Paper talks, lectures — extract slides, diagrams, equations
- **tutorial**: How-to guides — extract steps, commands, expected outcomes
- **demo**: Product showcases — extract features, interactions, highlights
- **describe**: General scene description
- **objects**: Object/entity detection
- **actions**: Human activity recognition
- **meeting**: Meeting recordings — extract decisions, action items

See [references/analysis_prompts.md](references/analysis_prompts.md) for detailed prompt templates.

## Typical Workflows

### Academic Lecture Deep-Analysis
```bash
./scripts/video_analysis.sh full -i lecture.mp4 -m academic -o ./lecture_analysis
```

### Technical Tutorial Extraction
```bash
./scripts/video_analysis.sh full -i tutorial.mp4 -m tutorial -o ./tutorial_out
```

### YouTube Video Analysis
```bash
./scripts/video_analysis.sh full -i "https://youtube.com/watch?v=VIDEO_ID" -m describe
```

## Direct Python Usage

```bash
# 1. Extract frames (scene change detection)
python3 scripts/extract_frames.py --input video.mp4 --output ./frames --mode scene

# 2. Transcribe audio
python3 scripts/transcribe.py --input video.mp4 --output ./transcript --language auto

# 3. Analyze frames
python3 scripts/analyze_frames.py --frames-dir ./frames \
  --transcript ./transcript/transcript.json --mode academic --out analysis.json

# 4. Generate report
python3 scripts/generate_report.py --analysis analysis.json \
  --transcript ./transcript/transcript.json --frames-dir ./frames --out REPORT.md
```

## Environment Setup

```bash
# Install dependencies
pip install faster-whisper anthropic opencv-python Pillow yt-dlp tqdm

# Set API key
export ANTHROPIC_API_KEY="your-api-key"

# Check dependencies
./scripts/video_analysis.sh check
```

## Output Structure

```
output/
├── frames/
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── frame_index.json
├── transcript/
│   ├── transcript.json
│   └── transcript.srt
├── analysis.json
└── REPORT.md
```

## Integration Points

This skill integrates with other Auto-Research-Skills:
- **paper-research**: Analyze video paper talks
- **academic-writing**: Polish video transcripts
- **pptx**: Generate presentations from analysis
- **google-docs**: Write reports to Google Docs

## Dependencies

- **System**: ffmpeg, ffprobe, yt-dlp (optional for YouTube)
- **Python**: faster-whisper, anthropic, opencv-python, Pillow, yt-dlp, tqdm

## Notes

- Frames are saved locally for re-analysis without re-decoding
- Transcripts use local Whisper for privacy
- Analysis uses Claude Sonnet 4 Vision
- Rate limiting built-in (1 second delay between API calls)
- Batch processing with configurable batch size
