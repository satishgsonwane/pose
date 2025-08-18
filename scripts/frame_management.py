#!/usr/bin/env python3
"""
Frame Management Utility

This script helps manage frames in the organized structure:
- assets/frames/processed/ - Production frames
- assets/frames/test/ - Test/development frames
"""

import argparse
import os
import shutil
from pathlib import Path


def list_frames():
    """List all frames in the organized structure."""
    frames_dir = Path("assets/frames")
    
    if not frames_dir.exists():
        print("No frames directory found.")
        return
    
    print("📁 Frame Structure:")
    print("=" * 50)
    
    # List processed frames
    processed_dir = frames_dir / "processed"
    if processed_dir.exists():
        print("\n🔄 Processed Frames:")
        for video_dir in processed_dir.iterdir():
            if video_dir.is_dir():
                frame_count = len(list(video_dir.glob("*.jpg")))
                print(f"  📹 {video_dir.name}/ ({frame_count} frames)")
    
    # List test frames
    test_dir = frames_dir / "test"
    if test_dir.exists():
        print("\n🧪 Test Frames:")
        for video_dir in test_dir.iterdir():
            if video_dir.is_dir():
                frame_count = len(list(video_dir.glob("*.jpg")))
                print(f"  📹 {video_dir.name}/ ({frame_count} frames)")


def cleanup_frames(frame_type="all", video_name=None):
    """Clean up frames based on type and video name."""
    frames_dir = Path("assets/frames")
    
    if frame_type == "all":
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
            print("🗑️  Removed all frames")
        return
    
    if frame_type == "processed":
        target_dir = frames_dir / "processed"
        if video_name:
            target_dir = target_dir / video_name
    elif frame_type == "test":
        target_dir = frames_dir / "test"
        if video_name:
            target_dir = target_dir / video_name
    else:
        print(f"❌ Unknown frame type: {frame_type}")
        return
    
    if target_dir.exists():
        if target_dir.is_file():
            target_dir.unlink()
        else:
            shutil.rmtree(target_dir)
        print(f"🗑️  Removed: {target_dir}")
    else:
        print(f"⚠️  Not found: {target_dir}")


def move_frames(source_video, target_type="processed", video_id=None):
    """Move frames from one location to another."""
    if video_id is None:
        video_id = Path(source_video).stem
    
    source_dir = Path("assets/frames") / source_video
    target_dir = Path("assets/frames") / target_type / video_id
    
    if not source_dir.exists():
        print(f"❌ Source not found: {source_dir}")
        return
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Move frames
    for frame_file in source_dir.glob("*.jpg"):
        shutil.move(str(frame_file), str(target_dir / frame_file.name))
    
    # Remove empty source directory
    if source_dir.exists() and not any(source_dir.iterdir()):
        source_dir.rmdir()
    
    print(f"📦 Moved {source_video} → {target_type}/{video_id}")


def main():
    parser = argparse.ArgumentParser(description="Frame Management Utility")
    parser.add_argument("action", choices=["list", "cleanup", "move"], 
                       help="Action to perform")
    parser.add_argument("--type", choices=["all", "processed", "test"], 
                       default="all", help="Frame type for cleanup")
    parser.add_argument("--video", help="Video name for cleanup or move")
    parser.add_argument("--source", help="Source directory for move")
    parser.add_argument("--target", choices=["processed", "test"], 
                       default="processed", help="Target type for move")
    parser.add_argument("--video-id", help="Video ID for move")
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_frames()
    elif args.action == "cleanup":
        cleanup_frames(args.type, args.video)
    elif args.action == "move":
        if not args.source:
            print("❌ Source directory required for move action")
            return
        move_frames(args.source, args.target, args.video_id)


if __name__ == "__main__":
    main()
