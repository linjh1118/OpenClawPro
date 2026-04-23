#!/usr/bin/env python3
"""
Video Frame Extraction Script

Extract frames from video files or YouTube URLs with multiple sampling modes:
- uniform: Extract frames at fixed time intervals
- scene: Extract frames based on scene change detection
- keyframe: Extract keyframes with quality filtering
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def is_youtube_url(url: str) -> bool:
    """Check if the input is a YouTube URL."""
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
    return any(domain in url for domain in youtube_domains)


def download_youtube(url: str, output_dir: str) -> str:
    """Download YouTube video using yt-dlp."""
    output_path = os.path.join(output_dir, "input_video.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", output_path,
        "--merge-output-format", "mp4",
        url
    ]
    print(f"Downloading YouTube video: {url}")
    subprocess.run(cmd, check=True)
    
    # Find the downloaded file
    for ext in ['mp4', 'mkv', 'webm']:
        potential_path = output_path.replace('%(ext)s', ext)
        if os.path.exists(potential_path):
            return potential_path
    
    raise FileNotFoundError("Failed to download video")


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data['format']['duration'])


def extract_uniform_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0
) -> list:
    """Extract frames at uniform intervals."""
    os.makedirs(output_dir, exist_ok=True)
    
    duration = get_video_duration(video_path)
    total_frames = int(duration * fps)
    
    print(f"Extracting {total_frames} frames at {fps} fps from {duration:.1f}s video")
    
    frames_info = []
    for i in range(total_frames):
        timestamp = i / fps
        output_file = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        
        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            "-y",
            output_file
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        
        frames_info.append({
            "frame_id": i,
            "timestamp": timestamp,
            "filename": os.path.basename(output_file)
        })
        print(f"  Extracted frame {i+1}/{total_frames} at {timestamp:.2f}s")
    
    return frames_info


def extract_scene_frames(
    video_path: str,
    output_dir: str,
    threshold: float = 0.3
) -> list:
    """Extract frames based on scene change detection."""
    if not OPENCV_AVAILABLE:
        print("WARNING: OpenCV not available. Falling back to uniform extraction.")
        return extract_uniform_frames(video_path, output_dir, fps=1.0)
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Detecting scene changes in {total_frames} frames...")
    
    frames_info = []
    prev_frame = None
    frame_count = 0
    keyframe_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(gray, prev_frame)
            score = np.mean(diff) / 255.0
            
            if score > threshold:
                timestamp = frame_count / fps
                output_file = os.path.join(output_dir, f"frame_{keyframe_count:04d}.jpg")
                
                cv2.imwrite(output_file, frame)
                frames_info.append({
                    "frame_id": keyframe_count,
                    "timestamp": timestamp,
                    "filename": os.path.basename(output_file),
                    "change_score": round(score, 4)
                })
                keyframe_count += 1
                print(f"  Scene change detected at {timestamp:.2f}s (score: {score:.3f})")
        
        prev_frame = gray
        frame_count += 1
    
    cap.release()
    
    print(f"Extracted {keyframe_count} scene change frames")
    return frames_info


def extract_keyframes(
    video_path: str,
    output_dir: str,
    min_quality: float = 30.0
) -> list:
    """Extract keyframes with quality filtering."""
    if not OPENCV_AVAILABLE:
        print("WARNING: OpenCV not available. Falling back to uniform extraction.")
        return extract_uniform_frames(video_path, output_dir, fps=0.5)
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Extracting keyframes from {total_frames} frames...")
    
    frames_info = []
    keyframe_count = 0
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = frame_id / fps
        
        # Calculate Laplacian variance for blur detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check if this is a keyframe (not blurry and different from previous)
        is_keyframe = False
        
        if laplacian_var > min_quality:
            if prev_frame is None:
                is_keyframe = True
            else:
                diff = cv2.absdiff(gray, prev_frame)
                if np.mean(diff) > 10:
                    is_keyframe = True
        
        if is_keyframe:
            output_file = os.path.join(output_dir, f"frame_{keyframe_count:04d}.jpg")
            cv2.imwrite(output_file, frame)
            frames_info.append({
                "frame_id": keyframe_count,
                "timestamp": timestamp,
                "filename": os.path.basename(output_file),
                "quality_score": round(laplacian_var, 2)
            })
            keyframe_count += 1
            print(f"  Keyframe at {timestamp:.2f}s (quality: {laplacian_var:.1f})")
        
        prev_frame = gray
    
    cap.release()
    
    print(f"Extracted {keyframe_count} keyframes")
    return frames_info


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video files or YouTube URLs"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video file path or YouTube URL"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for extracted frames"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["uniform", "scene", "keyframe"],
        default="uniform",
        help="Frame extraction mode (default: uniform)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second for uniform mode (default: 1.0)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.3,
        help="Change threshold for scene mode (default: 0.3)"
    )
    parser.add_argument(
        "--temp-dir",
        default="/tmp/video_analysis",
        help="Temporary directory for downloaded videos"
    )
    
    args = parser.parse_args()
    
    # Determine input type and get video path
    video_path = args.input
    
    if is_youtube_url(args.input):
        os.makedirs(args.temp_dir, exist_ok=True)
        video_path = download_youtube(args.input, args.temp_dir)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Extract frames based on mode
    if args.mode == "uniform":
        frames_info = extract_uniform_frames(video_path, args.output, args.fps)
    elif args.mode == "scene":
        frames_info = extract_scene_frames(video_path, args.output, args.threshold)
    elif args.mode == "keyframe":
        frames_info = extract_keyframes(video_path, args.output)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)
    
    # Save frame index
    index_path = os.path.join(args.output, "frame_index.json")
    with open(index_path, "w") as f:
        json.dump({
            "video_path": video_path,
            "mode": args.mode,
            "frames": frames_info
        }, f, indent=2)
    
    print(f"\nFrame extraction complete. {len(frames_info)} frames saved to {args.output}")
    print(f"Index saved to {index_path}")


if __name__ == "__main__":
    main()
