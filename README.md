# Pose Detection Pipeline

A comprehensive computer vision pipeline for pose detection in sports videos using YOLO models and NATS messaging.

## Project Structure

```
Pose/
├── assets/
│   ├── frames/
│   │   ├── processed/           # Production frames (per video)
│   │   └── test/                # Test/development frames
│   ├── models/                  # Pre-trained pose models (yolo11x-pose.pt, ...)
│   └── videos/                  # Example video files
├── config/
│   └── pose_detection.yaml      # Pipeline configuration
├── docs/
│   ├── README_pose_detection.md
│   ├── pose_analysis_architecture.mermaid
│   └── ...                      # Additional docs and research papers
├── results/
│   ├── detections.ndjson        # Output detections
│   ├── OZ_Football.npz          # Joints
│   └── OZ_Football_with_bones.npz # Joints + bones
├── runs/
│   └── pose/                    # Tracking and experiment outputs
├── scripts/
│   ├── frame_management.py      # Frame management utility
│   └── run_pipeline.py          # Pipeline runner
├── src/
│   ├── __init__.py
│   ├── pose_detection/
│   │   ├── __init__.py
│   │   ├── main.py              # Main pose detection pipeline
│   │   └── testcode/
│   │       ├── pose_video_to_nats_realtime.py # Real-time pose detection script
│   │       └── pose.py          # Simple pose detection example
│   └── utils/
├── requirements.txt
├── setup.py
└── README.md
```

## Features

- **Multi-person tracking** with YOLO pose models
- **17 keypoints** per person (COCO format)
- **Real-time processing** with configurable FPS
- **NATS integration** for message publishing
- **Frame management** utilities for organizing and cleaning up extracted frames
- **Output to NDJSON and NPZ** for downstream analysis

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Pose Detection

```bash
python3 src/pose_detection/testcode/pose_video_to_nats_realtime.py \
    --video assets/videos/closeup.mp4 \
    --model assets/models/yolo11x-pose.pt \
    --nats-topic pose.detections
```

### Advanced Pipeline (with NDJSON/NPZ output)

```bash
python3 src/pose_detection/main.py \
    --video assets/videos/closeup.mp4 \
    --model assets/models/yolo11x-pose.pt \
    --nats-url nats://127.0.0.1:4222 \
    --nats-topic pose.detections \
    --out-joints results/OZ_Football.npz \
    --out-joint-bones results/OZ_Football_with_bones.npz
```

### Frame Management

List, clean up, or move frames:

```bash
python3 scripts/frame_management.py list
python3 scripts/frame_management.py cleanup --type processed
python3 scripts/frame_management.py move --source test_closeup --target processed --video-id closeup
```

## Configuration

Edit `config/pose_detection.yaml` to set model path, device, thresholds, NATS settings, and output locations.

## Documentation

- [Pose Detection Guide](docs/README_pose_detection.md)
- [Frame Organization](docs/frame_organization.md)
- [Architecture Overview](docs/pose_analysis_architecture.mermaid)

## Models

- `yolo11x-pose.pt` - High accuracy, slower inference
- `yolo11m-pose.pt` - Balanced accuracy/speed
- `yolo11s-pose.pt` - Fast inference, lower accuracy
- `yolo11n-pose.pt` - Lightweight, fastest inference


