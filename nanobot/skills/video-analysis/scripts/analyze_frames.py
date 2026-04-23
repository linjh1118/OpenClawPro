#!/usr/bin/env python3
"""
Video Frame Analysis Script

Analyze video frames using Claude Vision API:
- Batch frame analysis with multiple modes
- Joint text+visual analysis with transcripts
- Academic, tutorial, demo, and general description modes
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Analysis mode prompts
ANALYSIS_PROMPTS = {
    "describe": """Analyze this video frame. Describe:
1. What is shown in the scene
2. Key visual elements and objects
3. Text or UI elements visible
4. Overall context and setting

Be concise and objective.""",

    "objects": """Analyze this video frame for object detection. Identify:
1. Main entities (people, animals, objects)
2. UI elements and interface components
3. Any text visible
4. Spatial relationships between elements

List each item with its position and description.""",

    "actions": """Analyze this video frame for action recognition. Identify:
1. Human activities or movements
2. Operations being performed
3. Interactions between entities
4. Sequence context from the scene

Focus on understanding what actions are occurring.""",

    "academic": """Analyze this academic video frame (lecture, paper talk, etc.). Extract:
1. Slide content (titles, bullet points, main ideas)
2. Diagrams, charts, or visual illustrations
3. Mathematical equations or formulas
4. Key concepts being discussed
5. Any code snippets or pseudocode

Be thorough and capture all informational content.""",

    "tutorial": """Analyze this tutorial video frame. Extract:
1. Current step being demonstrated
2. UI elements and interface being used
3. Any code or commands shown
4. Key instructions or tips
5. Expected outcomes or results

Focus on actionable information for reproducibility.""",

    "demo": """Analyze this demo video frame. Identify:
1. Product or feature being demonstrated
2. User interactions and workflows
3. Key benefits or value propositions
4. Interface design elements
5. Notable moments or highlights

Capture the essence of what makes this demo effective.""",

    "meeting": """Analyze this meeting video frame. Identify:
1. Participants visible
2. Content on screens or shared displays
3. Key discussion points
4. Decisions or action items mentioned
5. Meeting context and setting

Focus on extracting actionable information."""
}


def load_api_key() -> str:
    """Load Anthropic API key from environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Set it with: export ANTHROPIC_API_KEY=your-key")
        sys.exit(1)
    return api_key


def encode_image(image_path: str) -> str:
    """Encode image to base64 for API upload."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_frames(frames_dir: str) -> List[Dict]:
    """Load frame information from directory."""
    # Check for frame index
    index_path = os.path.join(frames_dir, "frame_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            data = json.load(f)
            return [
                {
                    "timestamp": frame["timestamp"],
                    "filename": frame["filename"],
                    "path": os.path.join(frames_dir, frame["filename"])
                }
                for frame in data.get("frames", [])
            ]
    
    # Fallback: scan directory for images
    frames = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    for filename in sorted(os.listdir(frames_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            frames.append({
                "filename": filename,
                "path": os.path.join(frames_dir, filename),
                "timestamp": None
            })
    return frames


def load_transcript(transcript_path: str) -> Optional[Dict]:
    """Load transcript JSON if provided."""
    if transcript_path and os.path.exists(transcript_path):
        with open(transcript_path) as f:
            return json.load(f)
    return None


def get_transcript_context(transcript: Dict, timestamp: float, window: float = 30.0) -> str:
    """Get transcript context around a specific timestamp."""
    if not transcript or "segments" not in transcript:
        return ""
    
    context_segments = []
    for segment in transcript["segments"]:
        seg_start = segment.get("start", 0)
        seg_end = segment.get("end", 0)
        
        # Check if segment overlaps with time window
        if abs(seg_start - timestamp) <= window or abs(seg_end - timestamp) <= window:
            context_segments.append(segment.get("text", ""))
        elif seg_start > timestamp + window:
            break
    
    if context_segments:
        return "Audio context: " + " ".join(context_segments)
    return ""


def analyze_frame(
    client,
    image_path: str,
    mode: str,
    transcript_context: str = ""
) -> str:
    """Analyze a single frame using Claude Vision API."""
    image_data = encode_image(image_path)
    
    prompt = ANALYSIS_PROMPTS.get(mode, ANALYSIS_PROMPTS["describe"])
    if transcript_context:
        prompt = f"{transcript_context}\n\n{prompt}"
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    
    return response.content[0].text


def analyze_frames_batch(
    client,
    frames: List[Dict],
    mode: str,
    transcript: Optional[Dict] = None,
    batch_size: int = 5,
    rate_limit_delay: float = 1.0
) -> List[Dict]:
    """Analyze frames in batches with rate limiting."""
    results = []
    
    total = len(frames)
    for i in range(0, total, batch_size):
        batch = frames[i:i + batch_size]
        
        print(f"Processing frames {i+1}-{min(i+batch_size, total)} of {total}...")
        
        for frame in batch:
            timestamp = frame.get("timestamp", 0)
            transcript_context = get_transcript_context(transcript, timestamp) if transcript else ""
            
            try:
                analysis = analyze_frame(
                    client,
                    frame["path"],
                    mode,
                    transcript_context
                )
                
                results.append({
                    "filename": frame["filename"],
                    "timestamp": timestamp,
                    "analysis": analysis
                })
                
            except Exception as e:
                print(f"  Error analyzing {frame['filename']}: {e}")
                results.append({
                    "filename": frame["filename"],
                    "timestamp": timestamp,
                    "analysis": "",
                    "error": str(e)
                })
            
            # Rate limiting
            time.sleep(rate_limit_delay)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze video frames using Claude Vision API"
    )
    parser.add_argument(
        "--frames-dir", "-f",
        required=True,
        help="Directory containing extracted frames"
    )
    parser.add_argument(
        "--transcript", "-t",
        help="Path to transcript JSON (optional, for joint analysis)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["describe", "objects", "actions", "academic", "tutorial", "demo", "meeting"],
        default="describe",
        help="Analysis mode (default: describe)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=5,
        help="Number of frames per batch (default: 5)"
    )
    parser.add_argument(
        "--output", "-o",
        default="analysis.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    if not ANTHROPIC_AVAILABLE:
        print("ERROR: anthropic package not installed. Install with: pip install anthropic")
        sys.exit(1)
    
    # Load frames
    frames = load_frames(args.frames_dir)
    if not frames:
        print(f"No frames found in {args.frames_dir}")
        sys.exit(1)
    
    print(f"Loaded {len(frames)} frames from {args.frames_dir}")
    print(f"Analysis mode: {args.mode}")
    
    # Load transcript if provided
    transcript = None
    if args.transcript:
        transcript = load_transcript(args.transcript)
        if transcript:
            print(f"Loaded transcript with {len(transcript.get('segments', []))} segments")
    
    # Initialize Anthropic client
    api_key = load_api_key()
    client = Anthropic(api_key=api_key)
    
    # Analyze frames
    print("\nStarting frame analysis...")
    results = analyze_frames_batch(
        client,
        frames,
        args.mode,
        transcript,
        args.batch_size
    )
    
    # Save results
    output = {
        "mode": args.mode,
        "frames_dir": args.frames_dir,
        "transcript": args.transcript,
        "frame_count": len(frames),
        "results": results
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalysis complete. Results saved to {args.output}")
    successful = sum(1 for r in results if r.get("analysis"))
    print(f"Successfully analyzed {successful}/{len(results)} frames")


if __name__ == "__main__":
    main()
