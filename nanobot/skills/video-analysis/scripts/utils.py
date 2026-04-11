#!/usr/bin/env python3
"""
Video Analysis Utility Functions

Shared utilities for video analysis workflows:
- Video validation and format checking
- Timestamp formatting
- Batch processing helpers
- File path utilities
"""

import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# Supported video formats
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v'}

# Supported image formats for frames
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def is_video_file(path: str) -> bool:
    """Check if file is a supported video format."""
    return any(path.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)


def is_image_file(path: str) -> bool:
    """Check if file is a supported image format."""
    return any(path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)


def is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube video."""
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
    return any(domain in url for domain in youtube_domains)


def get_video_info(video_path: str) -> Dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {}


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    info = get_video_info(video_path)
    
    # Try format duration first
    if "format" in info:
        duration = info["format"].get("duration")
        if duration:
            return float(duration)
    
    # Fallback to stream duration
    if "streams" in info:
        for stream in info["streams"]:
            if stream.get("codec_type") == "video":
                duration = stream.get("duration")
                if duration:
                    return float(duration)
    
    return 0.0


def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """Get video resolution as (width, height)."""
    info = get_video_info(video_path)
    
    if "streams" in info:
        for stream in info["streams"]:
            if stream.get("codec_type") == "video":
                width = stream.get("width", 0)
                height = stream.get("height", 0)
                return (width, height)
    
    return (0, 0)


def format_timestamp(seconds: float, include_hours: bool = False) -> str:
    """Format seconds to timestamp string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    if include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT subtitle timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def ensure_dir(path: str) -> str:
    """Ensure directory exists, create if needed."""
    os.makedirs(path, exist_ok=True)
    return path


def get_output_path(base_dir: str, prefix: str, extension: str) -> str:
    """Generate unique output path with prefix and extension."""
    ensure_dir(base_dir)
    
    counter = 0
    while True:
        path = os.path.join(base_dir, f"{prefix}_{counter:04d}.{extension}")
        if not os.path.exists(path):
            return path
        counter += 1


def scan_frames_directory(frames_dir: str) -> List[str]:
    """Scan directory for frame images, sorted by filename."""
    if not os.path.exists(frames_dir):
        return []
    
    frames = []
    for filename in sorted(os.listdir(frames_dir)):
        if is_image_file(filename):
            frames.append(os.path.join(frames_dir, filename))
    
    return frames


def load_frame_index(frames_dir: str) -> List[Dict]:
    """Load frame index JSON if available."""
    index_path = os.path.join(frames_dir, "frame_index.json")
    
    if os.path.exists(index_path):
        with open(index_path) as f:
            data = json.load(f)
            return data.get("frames", [])
    
    return []


def save_frame_index(frames_dir: str, frames: List[Dict], metadata: Dict = None) -> None:
    """Save frame index to JSON."""
    index_path = os.path.join(frames_dir, "frame_index.json")
    
    data = {
        "frames": frames,
        "metadata": metadata or {}
    }
    
    with open(index_path, "w") as f:
        json.dump(data, f, indent=2)


def batch_list(items: List, batch_size: int) -> List[List]:
    """Split list into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def estimate_processing_time(frame_count: int, mode: str = "describe") -> float:
    """Estimate processing time in seconds based on frame count and mode."""
    # Base times per frame (seconds)
    base_times = {
        "describe": 2.0,
        "objects": 3.0,
        "actions": 3.0,
        "academic": 4.0,
        "tutorial": 3.5,
        "demo": 3.0,
        "meeting": 2.5
    }
    
    base_time = base_times.get(mode, 3.0)
    return frame_count * base_time


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available."""
    dependencies = {}
    
    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        dependencies["ffmpeg"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        dependencies["ffmpeg"] = False
    
    # Check ffprobe
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        dependencies["ffprobe"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        dependencies["ffprobe"] = False
    
    # Check yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        dependencies["yt-dlp"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        dependencies["yt-dlp"] = False
    
    # Check Python packages
    try:
        import faster_whisper
        dependencies["faster-whisper"] = True
    except ImportError:
        dependencies["faster-whisper"] = False
    
    try:
        import cv2
        dependencies["opencv-python"] = True
    except ImportError:
        dependencies["opencv-python"] = False
    
    try:
        import anthropic
        dependencies["anthropic"] = True
    except ImportError:
        dependencies["anthropic"] = False
    
    return dependencies
