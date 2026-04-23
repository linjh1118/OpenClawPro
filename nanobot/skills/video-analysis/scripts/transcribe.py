#!/usr/bin/env python3
"""
Video Audio Transcription Script

Transcribe audio from video files using Faster Whisper:
- Extract audio track from video
- Transcribe with Whisper (supports multiple languages)
- Output JSON with timestamps and SRT subtitles
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio track from video using ffmpeg."""
    # Convert to wav with 16kHz sample rate for optimal Whisper performance
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path
    ]
    print(f"Extracting audio from {video_path}")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Audio extracted to {output_path}")
    return output_path


def transcribe_audio(
    audio_path: str,
    model_size: str = "medium",
    language: Optional[str] = None,
    device: str = "auto"
) -> List[Dict]:
    """Transcribe audio using Faster Whisper."""
    if not FASTER_WHISPER_AVAILABLE:
        print("ERROR: faster-whisper not installed. Install with: pip install faster-whisper")
        sys.exit(1)
    
    print(f"Loading Whisper model: {model_size}")
    
    # Determine compute type based on device
    compute_type = "float16" if device == "cuda" else "int8"
    
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )
    
    print(f"Transcribing audio...")
    
    # Run transcription
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                    "probability": word.probability
                }
                for word in segment.words
            ] if segment.words else []
        })
    
    return results


def generate_srt(segments: List[Dict], output_path: str) -> None:
    """Generate SRT subtitle file from transcript segments."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start_time = format_srt_time(segment["start"])
            end_time = format_srt_time(segment["end"])
            text = segment["text"]
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")


def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def is_youtube_url(url: str) -> bool:
    """Check if the input is a YouTube URL."""
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
    return any(domain in url for domain in youtube_domains)


def download_youtube(url: str, output_dir: str) -> str:
    """Download YouTube video using yt-dlp."""
    output_path = os.path.join(output_dir, "input_video.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "-o", output_path,
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        url
    ]
    print(f"Downloading YouTube audio: {url}")
    subprocess.run(cmd, check=True)
    
    # Find the downloaded file
    for ext in ['wav', 'm4a', 'webm']:
        potential_path = output_path.replace('%(ext)s', ext)
        if os.path.exists(potential_path):
            return potential_path
    
    raise FileNotFoundError("Failed to download audio")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio from video files using Whisper"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video file path or YouTube URL"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for transcript files"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["tiny", "base", "small", "medium", "large"],
        default="medium",
        help="Whisper model size (default: medium)"
    )
    parser.add_argument(
        "--language", "-l",
        help="Language code (e.g., en, zh, auto for auto-detection)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "srt", "both"],
        default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--temp-dir",
        default="/tmp/video_analysis",
        help="Temporary directory for downloaded content"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Determine input type and get video/audio path
    input_path = args.input
    
    if is_youtube_url(args.input):
        input_path = download_youtube(args.input, args.temp_dir)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Check if it's a video file (needs audio extraction)
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv'}
    is_video = any(input_path.lower().endswith(ext) for ext in video_extensions)
    
    audio_path = input_path
    if is_video:
        audio_path = os.path.join(args.temp_dir, "audio.wav")
        extract_audio(input_path, audio_path)
    
    # Determine language parameter
    language = None if args.language == "auto" else args.language
    
    # Transcribe
    segments = transcribe_audio(
        audio_path,
        model_size=args.model,
        language=language
    )
    
    # Output JSON
    json_output = {
        "language": args.language or "auto",
        "model": args.model,
        "segments": segments,
        "full_text": " ".join(seg["text"] for seg in segments)
    }
    
    json_path = os.path.join(args.output, "transcript.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"Transcript saved to {json_path}")
    
    # Output SRT
    if args.format in ["srt", "both"]:
        srt_path = os.path.join(args.output, "transcript.srt")
        generate_srt(segments, srt_path)
        print(f"SRT subtitles saved to {srt_path}")
    
    print(f"\nTranscription complete. {len(segments)} segments processed.")


if __name__ == "__main__":
    main()
