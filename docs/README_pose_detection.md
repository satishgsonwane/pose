# Pose Detection to NATS

This script extracts pose data from video frames using YOLO pose detection and publishes the keypoints over NATS messaging.

## Overview

The `pose_video_to_nats.py` script focuses on pose detection:

1. **Frame Extraction**: Extracts frames from input video at specified FPS and resolution
2. **Pose Detection**: Runs YOLO pose detection on each frame to get keypoints
3. **NATS Publishing**: Publishes pose data as NATS messages with keypoints, confidence scores, and tracking info

## Features

- **Multi-person tracking**: Supports tracking multiple people in the same frame
- **Keypoint extraction**: Extracts 17 keypoints per person (nose, eyes, shoulders, elbows, wrists, hips, knees, ankles)
- **Confidence scores**: Includes confidence scores for each keypoint and bounding box
- **Normalized coordinates**: Keypoints are normalized to [0,1] range for consistent processing
- **Configurable parameters**: Adjustable confidence thresholds, IoU thresholds, and publishing rates

## Requirements

```bash
pip install ultralytics opencv-python tqdm nats-py numpy
```

## Usage

### Basic Usage

```bash
python3 pose_video_to_nats.py --video closeup.mp4 --nats-topic pose.detections
```

### Advanced Usage

```bash
python3 pose_video_to_nats.py \
    --video CAM01_08.mp4 \
    --model yolo11x-pose.pt \
    --target-fps 30.0 \
    --height 480 \
    --conf-threshold 0.6 \
    --iou-threshold 0.5 \
    --nats-topic soccer.pose.detections \
    --nats-url nats://127.0.0.1:4222 \
    --publish-interval 0.05
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | **required** | Path to input video file |
| `--model` | `yolo11x-pose.pt` | Path to YOLO pose model |
| `--work-dir` | `temp dir` | Directory to store extracted frames |
| `--target-fps` | `25.0` | Target FPS for frame extraction |
| `--height` | `640` | Frame height (maintains aspect ratio) |
| `--video-id` | `video filename` | Custom video ID for output |
| `--conf-threshold` | `0.5` | YOLO confidence threshold |
| `--iou-threshold` | `0.7` | YOLO IoU threshold |
| `--nats-url` | `nats://127.0.0.1:4222` | NATS server URL |
| `--nats-topic` | `pose.detections` | NATS topic for publishing |
| `--publish-interval` | `0.1` | Seconds between message publishes |

## Message Format

Each NATS message contains pose detection data in the following format:

```json
{
  "video": "closeup",
  "frame": 42,
  "timestamp": 1.68,
  "track_id": 1,
  "keypoints": [
    [0.5, 0.3, 0.9],   // [x, y, confidence] for nose
    [0.48, 0.28, 0.8], // left eye
    [0.52, 0.28, 0.8], // right eye
    // ... 17 keypoints total
  ],
  "keypoint_confidences": [0.9, 0.8, 0.8, ...],
  "bbox": [0.2, 0.1, 0.8, 0.9, 0.95], // [x1, y1, x2, y2, confidence]
  "bbox_confidence": 0.95,
  "source": "YOLO-Pose",
  "emitted_at": 1703123456.789
}
```

### Keypoint Indices

The 17 keypoints follow the COCO format:
- 0: nose
- 1: left eye
- 2: right eye
- 3: left ear
- 4: right ear
- 5: left shoulder
- 6: right shoulder
- 7: left elbow
- 8: right elbow
- 9: left wrist
- 10: right wrist
- 11: left hip
- 12: right hip
- 13: left knee
- 14: right knee
- 15: left ankle
- 16: right ankle

## Example Usage Script

Run the example script to see different usage patterns:

```bash
python3 example_pose_usage.py
```

This will:
1. Check dependencies and required files
2. Show different example commands
3. Allow you to run any example interactively

## Performance Considerations

- **Frame rate**: Higher target FPS increases processing time
- **Resolution**: Lower frame height reduces memory usage and speeds up inference
- **Confidence threshold**: Higher thresholds reduce false positives but may miss detections
- **Publish interval**: Lower intervals increase message frequency but may overwhelm NATS

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the YOLO model file exists at the specified path
2. **NATS connection failed**: Check if NATS server is running at the specified URL
3. **No detections**: Try lowering the confidence threshold
4. **Memory issues**: Reduce frame height or target FPS

### Debug Mode

Enable debug output by setting the publish interval to a higher value and checking the first few messages that are printed to console.

## Integration with Existing Pipeline

This script can be integrated with your existing pose detection pipeline:

1. Use pose data as input for pose classification models
2. Store pose data in databases for offline analysis
3. Real-time pose tracking for live video streams
