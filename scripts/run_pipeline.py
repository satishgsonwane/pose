#!/usr/bin/env python3
"""
Pipeline runner script for pose detection.

This script provides a convenient way to run pose detection on a video file.
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path


def run_pose_detection(video_path: str, model_path: str, nats_topic: str, **kwargs):
    """Run pose detection on the video."""
    cmd = [
        sys.executable, "src/pose_detection/pose_video_to_nats.py",
        "--video", video_path,
        "--model", model_path,
        "--nats-topic", nats_topic,
    ]
    
    # Add optional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"Running pose detection: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Pose detection failed: {result.stderr}")
        return False
    
    print("Pose detection completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run pose detection pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--pose-model", default="assets/models/yolo11x-pose.pt", help="Pose detection model")
    parser.add_argument("--height", type=int, default=640, help="Frame height")
    parser.add_argument("--pose-only", action="store_true", help="Run only pose detection")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    if not Path(args.pose_model).exists():
        print(f"Error: Pose model not found: {args.pose_model}")
        sys.exit(1)
    
    # Run pipeline
    success = True
    
    if not args.pose_only:
        print("=" * 50)
        print("Running Pose Detection")
        print("=" * 50)
        success &= run_pose_detection(
            video_path=args.video,
            model_path=args.pose_model,
            nats_topic="pose.detections",
            height=args.height
        )
    
    if success:
        print("=" * 50)
        print("Pipeline completed successfully!")
        print("=" * 50)
    else:
        print("=" * 50)
        print("Pipeline failed!")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
