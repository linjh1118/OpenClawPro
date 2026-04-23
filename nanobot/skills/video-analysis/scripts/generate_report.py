#!/usr/bin/env python3
"""
Video Analysis Report Generator

Generate structured Markdown reports from video analysis results:
- Combine frame analysis with transcripts
- Support multiple report modes (academic, tutorial, general)
- Include key frame gallery and chapter structure
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime


def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def load_analysis(analysis_path: str) -> Dict:
    """Load frame analysis JSON."""
    with open(analysis_path) as f:
        return json.load(f)


def load_transcript(transcript_path: str) -> Optional[Dict]:
    """Load transcript JSON."""
    if transcript_path and os.path.exists(transcript_path):
        with open(transcript_path) as f:
            return json.load(f)
    return None


def get_segment_at_time(transcript: Dict, timestamp: float) -> Optional[str]:
    """Get transcript segment at or near a specific timestamp."""
    if not transcript or "segments" not in transcript:
        return None
    
    for segment in transcript["segments"]:
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        
        if start <= timestamp <= end:
            return segment.get("text", "")
        elif abs(start - timestamp) < 5:  # Within 5 seconds
            return segment.get("text", "")
    
    return None


def detect_chapters(transcript: Dict, analysis: Dict) -> List[Dict]:
    """Detect chapter structure from transcript and analysis."""
    chapters = []
    
    if not transcript or "segments" not in transcript:
        return chapters
    
    segments = transcript["segments"]
    if not segments:
        return chapters
    
    # Simple chapter detection based on analysis changes
    results = analysis.get("results", [])
    prev_content_length = 0
    
    chapter_num = 1
    for i, result in enumerate(results):
        content = result.get("analysis", "")
        
        # Detect chapter change based on content length variation
        if i > 0:
            length_diff = abs(len(content) - prev_content_length)
            if length_diff > 200:  # Significant content change
                chapters.append({
                    "number": chapter_num,
                    "start": result.get("timestamp", 0),
                    "title": f"Chapter {chapter_num}"
                })
                chapter_num += 1
        
        prev_content_length = len(content)
    
    return chapters


def generate_basic_info_section(
    analysis: Dict,
    transcript: Optional[Dict] = None,
    video_path: Optional[str] = None
) -> str:
    """Generate basic information section."""
    lines = ["## Basic Information\n"]
    
    # Frame count
    frame_count = analysis.get("frame_count", 0)
    lines.append(f"- **Total Frames Analyzed**: {frame_count}")
    
    # Language
    if transcript:
        language = transcript.get("language", "Unknown")
        lines.append(f"- **Language**: {language}")
        
        # Duration estimate based on transcript
        segments = transcript.get("segments", [])
        if segments:
            duration = segments[-1].get("end", 0)
            lines.append(f"- **Duration**: ~{format_timestamp(duration)}")
    
    # Mode
    mode = analysis.get("mode", "unknown")
    lines.append(f"- **Analysis Mode**: {mode}")
    
    # Timestamp
    lines.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    lines.append("")
    return "\n".join(lines)


def generate_content_summary(
    analysis: Dict,
    transcript: Optional[Dict] = None
) -> str:
    """Generate overall content summary."""
    lines = ["## Content Summary\n"]
    
    results = analysis.get("results", [])
    if not results:
        lines.append("*No frame analysis available.*\n")
        return "\n".join(lines)
    
    # Generate summary from frame analyses
    summaries = []
    for result in results:
        analysis_text = result.get("analysis", "")
        if analysis_text:
            # Take first sentence or first 200 chars
            summary = analysis_text[:200]
            if len(analysis_text) > 200:
                summary = summary.rsplit('.', 1)[0] + '.'
            summaries.append(summary)
    
    if summaries:
        lines.append("### Key Moments\n")
        for i, summary in enumerate(summaries[:5], 1):  # Top 5 summaries
            timestamp = results[i-1].get("timestamp", 0)
            lines.append(f"{i}. **[{format_timestamp(timestamp)}]** {summary}\n")
    
    lines.append("")
    return "\n".join(lines)


def generate_chapter_structure(
    analysis: Dict,
    transcript: Optional[Dict] = None
) -> str:
    """Generate chapter/segment structure."""
    lines = ["## Chapter Structure\n"]
    
    chapters = detect_chapters(transcript, analysis)
    
    if not chapters:
        # Generate basic timeline from analysis results
        results = analysis.get("results", [])
        if results:
            lines.append("| Timestamp | Description |\n")
            lines.append("|-----------|-------------|\n")
            
            for result in results:
                timestamp = result.get("timestamp", 0)
                analysis_text = result.get("analysis", "")
                
                # Truncate long descriptions
                if len(analysis_text) > 80:
                    description = analysis_text[:77] + "..."
                else:
                    description = analysis_text or "*No description*"
                
                lines.append(f"| {format_timestamp(timestamp)} | {description} |\n")
    else:
        # Use detected chapters
        for chapter in chapters:
            lines.append(f"### {chapter['title']}\n")
            lines.append(f"- **Start**: {format_timestamp(chapter['start'])}\n")
            lines.append("")
    
    lines.append("")
    return "\n".join(lines)


def generate_key_frame_gallery(
    analysis: Dict,
    frames_dir: str
) -> str:
    """Generate key frame gallery section."""
    lines = ["## Key Frame Gallery\n"]
    
    results = analysis.get("results", [])
    
    # Limit gallery to avoid very long reports
    gallery_results = results[::max(1, len(results) // 10)][:10]  # Max 10 frames
    
    for result in gallery_results:
        timestamp = result.get("timestamp", 0)
        analysis_text = result.get("analysis", "")
        filename = result.get("filename", "")
        
        lines.append(f"### Frame at {format_timestamp(timestamp)}\n")
        
        # Include image if available
        if frames_dir and os.path.exists(os.path.join(frames_dir, filename)):
            lines.append(f"![Frame {filename}](../{frames_dir}/{filename})\n")
        
        lines.append(f"**Analysis**: {analysis_text}\n")
        
        if timestamp > 0:
            lines.append("")
    
    return "\n".join(lines)


def generate_transcript_section(
    transcript: Optional[Dict],
    analysis: Dict
) -> str:
    """Generate full transcript section."""
    lines = ["## Full Transcript\n"]
    
    if not transcript or "segments" not in transcript:
        lines.append("*No transcript available.*\n")
        return "\n".join(lines)
    
    # Include transcript with timestamps and corresponding analysis
    results = analysis.get("results", [])
    result_index = 0
    
    for segment in transcript["segments"]:
        start = segment.get("start", 0)
        text = segment.get("text", "")
        
        # Find corresponding analysis
        analysis_text = ""
        while result_index < len(results):
            result = results[result_index]
            if result.get("timestamp", 0) <= start:
                analysis_text = result.get("analysis", "")
                result_index += 1
            else:
                break
        
        lines.append(f"**[{format_timestamp(start)}]** {text}\n")
        
        if analysis_text:
            lines.append(f"> *Visual: {analysis_text[:150]}...*\n")
        
        lines.append("")
    
    return "\n".join(lines)


def generate_extracted_information(
    analysis: Dict,
    mode: str
) -> str:
    """Generate mode-specific extracted information section."""
    lines = ["## Extracted Information\n"]
    
    results = analysis.get("results", [])
    
    if mode == "academic":
        lines.append("### Academic Content\n")
        lines.append("Key academic content extracted from frames:\n\n")
        
        for result in results:
            analysis_text = result.get("analysis", "")
            if any(keyword in analysis_text.lower() for keyword in ['equation', 'formula', 'theorem', 'method', 'approach', 'result']):
                timestamp = result.get("timestamp", 0)
                lines.append(f"**At {format_timestamp(timestamp)}:**\n")
                lines.append(f"{analysis_text}\n\n")
    
    elif mode == "tutorial":
        lines.append("### Tutorial Steps\n")
        lines.append("Steps extracted from the tutorial:\n\n")
        
        step_num = 1
        for result in results:
            analysis_text = result.get("analysis", "")
            if any(keyword in analysis_text.lower() for keyword in ['step', 'click', 'select', 'enter', 'type', 'run', 'execute']):
                timestamp = result.get("timestamp", 0)
                lines.append(f"{step_num}. {analysis_text[:300]}\n")
                lines.append(f"   *(Timestamp: {format_timestamp(timestamp)})*\n\n")
                step_num += 1
    
    elif mode == "demo":
        lines.append("### Demo Features\n")
        lines.append("Features and highlights from the demo:\n\n")
        
        for result in results:
            analysis_text = result.get("analysis", "")
            timestamp = result.get("timestamp", 0)
            lines.append(f"**Feature at {format_timestamp(timestamp)}:**\n")
            lines.append(f"{analysis_text}\n\n")
    
    else:
        lines.append("### Summary\n")
        for result in results[:5]:
            analysis_text = result.get("analysis", "")
            timestamp = result.get("timestamp", 0)
            lines.append(f"- **[{format_timestamp(timestamp)}]** {analysis_text[:200]}\n")
    
    lines.append("")
    return "\n".join(lines)


def generate_report(
    analysis: Dict,
    output_path: str,
    transcript: Optional[Dict] = None,
    frames_dir: Optional[str] = None
) -> None:
    """Generate complete Markdown report."""
    mode = analysis.get("mode", "describe")
    
    # Report title
    lines = [f"# Video Analysis Report\n"]
    lines.append(f"*Analysis Mode: {mode.upper()}*\n\n")
    
    # Sections
    lines.append(generate_basic_info_section(analysis, transcript))
    lines.append(generate_content_summary(analysis, transcript))
    lines.append(generate_chapter_structure(analysis, transcript))
    lines.append(generate_extracted_information(analysis, mode))
    
    if frames_dir and os.path.exists(frames_dir):
        lines.append(generate_key_frame_gallery(analysis, frames_dir))
    
    if transcript:
        lines.append(generate_transcript_section(transcript, analysis))
    
    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate structured Markdown reports from video analysis"
    )
    parser.add_argument(
        "--analysis", "-a",
        required=True,
        help="Path to analysis JSON file"
    )
    parser.add_argument(
        "--transcript", "-t",
        help="Path to transcript JSON file"
    )
    parser.add_argument(
        "--frames-dir", "-f",
        help="Directory containing extracted frames"
    )
    parser.add_argument(
        "--output", "-o",
        default="REPORT.md",
        help="Output Markdown file path"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["academic", "tutorial", "demo", "general", "describe"],
        help="Report mode (overrides analysis mode)"
    )
    
    args = parser.parse_args()
    
    # Load analysis
    analysis = load_analysis(args.analysis)
    
    # Override mode if specified
    if args.mode:
        analysis["mode"] = args.mode
    
    # Load transcript
    transcript = None
    if args.transcript:
        transcript = load_transcript(args.transcript)
    
    # Generate report
    generate_report(
        analysis,
        args.output,
        transcript,
        args.frames_dir
    )


if __name__ == "__main__":
    main()
