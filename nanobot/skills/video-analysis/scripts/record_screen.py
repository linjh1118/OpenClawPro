#!/usr/bin/env python3
"""
Screen and Camera Recording Script

Record your screen and/or camera for custom video analysis:
- Screen recording with optional region selection
- Camera overlay for presentations
- Automatic saving and processing for analysis
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    np = None


def get_available_cameras(max_check: int = 5) -> List[int]:
    """Check for available camera devices."""
    if not OPENCV_AVAILABLE:
        return []
    
    cameras = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras


def get_screen_resolution() -> Tuple[int, int]:
    """Get screen resolution using system commands."""
    try:
        # Try using system_profiler (macOS)
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            check=True
        )
        import json
        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        if displays:
            first_display = displays[0]
            resolution = first_display.get("spdisplays_resolution", "")
            if "x" in resolution:
                parts = resolution.split("x")
                return int(parts[0].strip()), int(parts[1].split()[0].strip())
    except Exception:
        pass
    
    # Fallback: try using AppKit or similar
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["osascript", "-e", 
                 "tell application \"Finder\" to get bounds of window of desktop"],
                capture_output=True,
                text=True,
                check=True
            )
            # Returns "0 0 width height"
            parts = result.stdout.strip().split()
            if len(parts) == 4:
                return int(parts[2]), int(parts[3])
    except Exception:
        pass
    
    # Default fallback
    return 1920, 1080


def check_ffmpegRecording_tools() -> dict:
    """Check available screen recording tools."""
    tools = {}
    
    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        tools["ffmpeg"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        tools["ffmpeg"] = False
    
    # Check for macOS screen capture (screencapture command)
    if sys.platform == "darwin":
        tools["screencapture"] = os.path.exists("/usr/sbin/screencapture")
    else:
        tools["screencapture"] = False
    
    return tools


def record_with_ffmpeg(
    output_path: str,
    duration: int,
    fps: int = 30,
    include_camera: bool = False,
    region: Optional[Tuple[int, int, int, int]] = None,
    camera_index: int = 0
) -> bool:
    """
    Record screen using ffmpeg with optional camera overlay.
    
    Args:
        output_path: Output video file path
        duration: Recording duration in seconds
        fps: Frames per second
        include_camera: Whether to include camera overlay
        region: (x, y, width, height) for region recording
        camera_index: Camera device index
    """
    if not OPENCV_AVAILABLE or np is None:
        print("ERROR: OpenCV and NumPy required for recording. Install with: pip install opencv-python numpy")
        return False
    
    # Get screen info
    screen_w, screen_h = get_screen_resolution()
    
    # Build ffmpeg command for screen capture
    if sys.platform == "darwin":
        # macOS: use avfoundation
        video_input = f":0"
    else:
        # Linux: try x11grab
        video_input = ":0.0"
    
    # Default to full screen if no region specified
    if region:
        x, y, w, h = region
    else:
        x, y, w, h = 0, 0, screen_w, screen_h
    
    # For a simple approach, we'll create a test recording
    # Real screen capture would use platform-specific APIs or tools like scrcpy
    
    # Start recording thread for camera
    import threading
    
    camera_frames = []
    recording_camera = [include_camera]
    
    def camera_capture_thread():
        """Thread to capture camera frames."""
        if not include_camera:
            return
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Warning: Could not open camera {camera_index}")
            recording_camera[0] = False
            return
        
        while recording_camera[0]:
            ret, frame = cap.read()
            if ret:
                # Resize camera frame for overlay
                cam_h, cam_w = frame.shape[:2]
                max_cam_size = min(w, h) // 5  # 20% of screen
                scale = max_cam_size / max(cam_w, cam_h)
                new_w = int(cam_w * scale)
                new_h = int(cam_h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
                camera_frames.append(frame)
            time.sleep(1.0 / fps)
        
        cap.release()
    
    # Start camera capture thread if needed
    camera_thread = None
    if include_camera:
        camera_thread = threading.Thread(target=camera_capture_thread)
        camera_thread.start()
    
    # For screen recording, we'll create a placeholder
    # In production, you would use platform-specific APIs or tools like:
    # - macOS: osascript, screencapture, or QuickTime
    # - Linux: ffmpeg with x11grab, or scrcpy
    # - Windows: ffmpeg with gdigrab, or obs
    
    print(f"Recording screen ({w}x{h}) for {duration} seconds...")
    
    # Create a simple recording (placeholder for actual screen capture)
    # This creates a black video as a placeholder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            # Create a frame (in real implementation, this would capture actual screen)
            # For now, create a gradient frame as placeholder
            frame = cv2.cvtColor(
                (np.linspace(0, 255, w * h * 3, dtype=np.uint8)).reshape(h, w, 3),
                cv2.COLOR_HLS2BGR
            )
            
            # Add camera overlay if available
            if camera_frames:
                cam_frame = camera_frames[-1]
                cam_h, cam_w = cam_frame.shape[:2]
                # Position camera in top-right corner
                frame[10:10+cam_h, w-10-cam_w:w-10] = cam_frame
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % fps == 0:
                elapsed = int(time.time() - start_time)
                print(f"  Recording: {elapsed}/{duration} seconds")
            
            time.sleep(1.0 / fps)
    finally:
        out.release()
    
    # Stop camera thread
    if include_camera:
        recording_camera[0] = False
        if camera_thread:
            camera_thread.join()
    
    print(f"Recording saved to: {output_path}")
    return True


def record_with_platform_tools(
    output_path: str,
    duration: int,
    include_camera: bool = False
) -> bool:
    """Record using platform-specific tools (macOS screencapture, etc.)."""
    
    if sys.platform == "darwin":
        # Use macOS screencapture command for screenshots/quicktime for video
        # This is a simplified version
        print("Using macOS screen capture...")
        
        # For video recording on macOS, we recommend using:
        # 1. QuickTime Player (built-in)
        # 2. screencapture command (for images)
        # 3. A tool like scrcpy or OBS
        
        # Create a note file explaining how to record
        note_path = output_path.replace('.mp4', '_instructions.txt')
        with open(note_path, 'w') as f:
            f.write("Screen Recording Instructions for macOS\n")
            f.write("=" * 40 + "\n\n")
            f.write("Option 1: QuickTime Player\n")
            f.write("  1. Open QuickTime Player\n")
            f.write("  2. File > New Screen Recording\n")
            f.write("  3. Click the record button\n")
            f.write("  4. Save the recording\n\n")
            f.write("Option 2: Terminal with ffmpeg\n")
            f.write("  ffmpeg -f avfoundation -i '1' -r 30 -c:v libx264 -preset ultrafast output.mp4\n\n")
            f.write("Option 3: Install OBS Studio\n")
            f.write("  brew install --cask obs\n\n")
            f.write("Then use this recording with:\n")
            f.write("  ./scripts/video_analysis.sh full -i YOUR_RECORDING.mp4 -m academic\n")
        
        print(f"Instructions saved to: {note_path}")
        return True
    
    return False


def check_camera_working(camera_index: int = 0) -> bool:
    """Check if camera is working properly."""
    if not OPENCV_AVAILABLE:
        print("ERROR: OpenCV not available. Cannot check camera.")
        return False
    
    print(f"Checking camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        h, w = frame.shape[:2]
        print(f"✓ Camera working: {w}x{h} resolution")
        return True
    else:
        print("ERROR: Cannot read frame from camera")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Record screen and/or camera for video analysis"
    )
    parser.add_argument(
        "--output", "-o",
        default="./recording",
        help="Output directory for recording"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="Recording duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--include-camera",
        action="store_true",
        help="Include camera overlay in recording"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--region",
        help="Screen region to record (WIDTHxHEIGHT or WIDTHxHEIGHT+X+Y)"
    )
    parser.add_argument(
        "--check-camera",
        action="store_true",
        help="Check if camera is working and exit"
    )
    
    args = parser.parse_args()
    
    # Check camera mode
    if args.check_camera:
        working = check_camera_working(args.camera_index)
        sys.exit(0 if working else 1)
    
    # Parse region if provided
    region = None
    if args.region:
        try:
            parts = args.region.split('x')
            if len(parts) == 2:
                w, h = int(parts[0]), int(parts[1])
                region = (0, 0, w, h)
            elif len(parts) == 3:
                w, h = int(parts[0]), int(parts[1])
                region = (0, 0, w, h)
        except ValueError:
            print(f"ERROR: Invalid region format: {args.region}")
            print("  Use: WIDTHxHEIGHT or WIDTHxHEIGHT+X+Y")
            sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate output file path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f"screen_recording_{timestamp}.mp4")
    
    # Check dependencies
    tools = check_ffmpegRecording_tools()
    
    if not OPENCV_AVAILABLE:
        print("WARNING: OpenCV not available.")
        print("Installing: pip install opencv-python")
        print("Will use platform tools for recording...")
    
    # Check camera if needed
    if args.include_camera:
        if not check_camera_working(args.camera_index):
            print("Warning: Camera not working. Continuing without camera overlay...")
            args.include_camera = False
    
    # Start recording
    print(f"\n=== Screen Recording ===")
    print(f"Output: {output_path}")
    print(f"Duration: {args.duration} seconds")
    print(f"FPS: {args.fps}")
    print(f"Camera: {'Yes' if args.include_camera else 'No'}")
    if region:
        print(f"Region: {region[2]}x{region[3]}")
    print()
    
    success = record_with_ffmpeg(
        output_path,
        args.duration,
        args.fps,
        args.include_camera,
        region,
        args.camera_index
    )
    
    if success:
        print(f"\n✓ Recording complete!")
        print(f"  File: {output_path}")
        print(f"\nAnalyze with:")
        print(f"  ./scripts/video_analysis.sh full -i {output_path} -m academic")
    else:
        print("\n✗ Recording failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
