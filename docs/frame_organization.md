# Frame Organization

## Overview

All extracted video frames are now consolidated in the `assets/frames/` directory for better organization and consistency.

## Structure

```
assets/frames/
├── processed/           # Production/processed frames
│   ├── closeup/         # Frames from closeup.mp4
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── ...
│   ├── CAM01_08/        # Frames from CAM01_08.mp4
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── ...
│   └── [video_name]/    # Frames from other videos
│       ├── 000000.jpg
│       └── ...
└── test/                # Test/development frames
    ├── test_video_001/  # Test frames
    │   ├── 000000.jpg
    │   └── ...
    └── [test_name]/     # Other test frames
        ├── 000000.jpg
        └── ...
```

## Usage

### Extracting Frames

When you run pose detection, frames are automatically extracted to `assets/frames/processed/[video_id]/`:

```bash
# This will extract frames to assets/frames/processed/closeup/
python3 src/pose_detection/pose_video_to_nats.py \
    --video assets/videos/closeup.mp4 \
    --video-id closeup

# This will extract frames to assets/frames/processed/soccer_match/
python3 src/pose_detection/pose_video_to_nats.py \
    --video assets/videos/CAM01_08.mp4 \
    --video-id soccer_match
```

### Custom Frame Directory

You can specify a custom directory for frame extraction:

```bash
# For test frames
python3 src/pose_detection/pose_video_to_nats.py \
    --video assets/videos/closeup.mp4 \
    --work-dir assets/frames/test \
    --video-id test_closeup

# For custom location
python3 src/pose_detection/pose_video_to_nats.py \
    --video assets/videos/closeup.mp4 \
    --work-dir /path/to/custom/frames \
    --video-id closeup
```

## Benefits

1. **Consistency**: All frames are in one location
2. **Organization**: Clear separation by video source
3. **Reusability**: Frames can be reused across different analyses
4. **Cleanup**: Easy to manage and clean up frame data
5. **Version Control**: Frames are excluded from git (see .gitignore)

## File Naming Convention

- Frames are named with 6-digit zero-padded numbers: `000000.jpg`, `000001.jpg`, etc.
- This ensures proper sorting and maintains frame order
- Each video gets its own subdirectory to avoid conflicts

## Cleanup

To clean up extracted frames:

```bash
# Remove all processed frames
rm -rf assets/frames/processed/*

# Remove all test frames
rm -rf assets/frames/test/*

# Remove frames for a specific video
rm -rf assets/frames/processed/closeup/

# Remove specific test frames
rm -rf assets/frames/test/test_video_001/
```

## Integration with Pipeline

The frame extraction is integrated into the pose detection pipeline:

1. **Frame Extraction**: Videos are processed and frames are saved
2. **Analysis**: Pose detection runs on the frames
3. **Cleanup**: Optional cleanup of temporary frame data

This organization makes the pipeline more efficient and easier to manage.
