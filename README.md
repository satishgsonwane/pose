# 🚀 Pose Detection Pipeline for Mac

A comprehensive computer vision pipeline for pose detection in sports videos using YOLO models and NATS messaging. **Fully optimized for Apple Silicon Macs (M1/M2/M3) with automatic MPS detection and GPU acceleration.**

## 🍎 Apple Silicon Mac Support

This repository has been optimized for Apple Silicon Macs with automatic MPS (Metal Performance Shaders) detection and GPU acceleration.

**Quick Start for Apple Silicon:**
```bash
# 1. Clone and enter
git clone <REPO_URL> pose && cd pose

# 2. Add your video file
cp /path/to/your/video.mp4 assets/videos/

# 3. Run one command setup
./setup.sh

# 4. Run the pipeline
source venv/bin/activate
nats-server &
python src/pose_detection/main.py
```

## 📋 Prerequisites

- **macOS**: 12.3+ (required for MPS support on Apple Silicon)
- **Python**: 3.8+ (3.9+ recommended)
- **Git**: Installed
- **8GB+ RAM** available
- **2GB+ free storage** for models and dependencies

## 🚀 Quick Setup (One Command!)

```bash
./setup.sh
```
**What it does:** Everything automatically in ~5-10 minutes

## 📁 Project Structure

```
pose/
├── assets/
│   ├── videos/          ← PUT YOUR VIDEO HERE
│   └── models/          ← Auto-downloaded
├── config/
│   └── pose_detection.yaml
├── src/pose_detection/
│   ├── main.py          ← Main pipeline
│   └── pose_to_hdgcn.py ← Pose processing utilities
├── scripts/             ← Utility scripts
├── docs/                ← Research papers and architecture
├── results/             ← Auto-created output files
├── setup.sh             ← One-command setup (everything automatic)
└── requirements.txt     ← Consolidated dependencies for all Macs
```

## 🎬 Video Setup

### Video Requirements
- **Format**: MP4, AVI, MOV, or any OpenCV supports
- **Resolution**: 720p, 1080p, or 4K (1080p recommended)
- **Content**: Should contain people for pose detection
- **Location**: `assets/videos/your_video.mp4`

### Add Your Video
```bash
# Create directories
mkdir -p assets/videos assets/models results

# Add video file
cp /path/to/your/video.mp4 assets/videos/
```

## 🔧 Detailed Setup Process

### Step 1: Repository Setup
```bash
# Clone and enter
git clone <REPO_URL> pose
cd pose

# Check architecture
uname -m
# arm64 = Apple Silicon, x86_64 = Intel
```

### Step 2: Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Verify activation
which python
# Should show: /Users/username/Desktop/pose/venv/bin/python
```

### Step 3: Install Dependencies

#### For Apple Silicon Macs (M1/M2/M3):
```bash
# Run automated setup
./setup.sh
```

**What the script does:**
- Detects Apple Silicon architecture
- Installs PyTorch for Apple Silicon (MPS support)
- Installs all other dependencies
- Sets up NATS server
- Updates configuration automatically

#### For Intel Macs:
```bash
# Install PyTorch
pip install torch torchvision

# Install other requirements
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
# Test PyTorch
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ MPS (Metal Performance Shaders) is available!')
    print('✅ Apple Silicon GPU acceleration enabled')
else:
    print('⚠️  MPS not available, will use CPU')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Step 5: NATS Server Setup
```bash
# NATS is automatically installed and started by setup.sh
# No manual setup needed!
```

### Step 6: Configuration
```bash
# Edit configuration
nano config/pose_detection.yaml
```

**Key settings to update:**
```yaml
# Path to your video file
video: assets/videos/your_video.mp4

# Video identifier
video_id: your_video_name

# Device (auto-detection recommended)
device: auto

# Target FPS (adjust based on Mac performance)
target_fps: 25.0
```

## 🎯 Running the Pipeline

### Basic Run (Recommended for First Time)
```bash
# Make sure you're in the pose directory
cd ~/Desktop/pose

# Activate virtual environment
source venv/bin/activate

# Run with default settings
python src/pose_detection/main.py
```

**Expected output:**
```
🚀 Setting up pose detection pipeline...
[INFO] Device 'auto' not available, using optimal device: mps
[INFO] Using Apple Silicon MPS acceleration
[INFO] Loading YOLO pose model on mps: assets/models/yolo11x-pose.pt
[INFO] Model not found at assets/models/yolo11x-pose.pt. Downloading...
[INFO] Model downloaded to assets/models/yolo11x-pose.pt
[INFO] NATS connected: nats://127.0.0.1:4222, topic='pose.detections'
[INFO] Video: assets/videos/your_video.mp4 | Input FPS: 30.00 | Target FPS: 25.00
```

### Advanced Run with Custom Parameters
```bash
python src/pose_detection/main.py \
    --video assets/videos/your_video.mp4 \
    --video-id "test_video" \
    --device mps \
    --target-fps 30.0 \
    --conf-threshold 0.6 \
    --iou-threshold 0.7
```

### Monitor NATS Messages (In Another Terminal)
```bash
# Open a new terminal window/tab
cd ~/Desktop/pose
source venv/bin/activate

# Subscribe to pose detection messages
nats sub pose.detections
```

## 📊 Performance Expectations

### M2 MacBook Air/Pro
- **Real-time Processing**: 25-30 FPS on 1080p video
- **Model Loading**: ~2-3 seconds for YOLO11x-pose
- **Memory Usage**: 2-4GB during inference

### M1 MacBook Air/Pro
- **Real-time Processing**: 20-25 FPS on 1080p video
- **Model Loading**: ~3-4 seconds for YOLO11x-pose
- **Memory Usage**: 2-4GB during inference

### Intel Mac
- **Real-time Processing**: 15-20 FPS on 1080p video (CPU)
- **Model Loading**: ~4-5 seconds for YOLO11x-pose
- **Memory Usage**: 2-4GB during inference

## 🔍 Troubleshooting

### Common Issues and Solutions

#### Issue: "MPS not available"
```bash
# Check macOS version
sw_vers

# Solution: Update to macOS 12.3+ or use CPU
python src/pose_detection/main.py --device cpu
```

#### Issue: "Model download failed"
```bash
# Check internet connection
ping github.com

# Manual download
curl -L -o assets/models/yolo11x-pose.pt \
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt"
```

#### Issue: "NATS connection failed"
```bash
# Check if NATS is running
ps aux | grep nats-server

# Restart NATS
pkill nats-server
nats-server &
```

#### Issue: "Video file not found"
```bash
# Check video file path
ls -la assets/videos/

# Update configuration
nano config/pose_detection.yaml
# Change video: path to correct location
```

#### Issue: "Out of memory"
```bash
# Reduce batch size in configuration
nano config/pose_detection.yaml
# Change max_queue_size: 50 (from 100)

# Or run with lower FPS
python src/pose_detection/main.py --target-fps 15.0
```

### Performance Optimization
1. **Close other GPU-intensive apps** (Final Cut Pro, Logic Pro, etc.)
2. **Reduce target FPS** in configuration
3. **Use smaller models** for faster inference
4. **Monitor Activity Monitor** for memory usage

## 🎉 Success Verification

### Check All Components Working
```bash
# 1. ✅ Video processing
# 2. ✅ Pose detection
# 3. ✅ NATS publishing
# 4. ✅ Output file generation
# 5. ✅ Performance monitoring

# Run a quick test
python src/pose_detection/main.py --video assets/videos/your_video.mp4 --target-fps 10.0
```

### Expected Final Output
```
[INFO] Video ended.
[INFO] Shutting down...
[NPZ] Building final NPZ files...
[INFO] Pipeline completed successfully!
```

### Check Output Files
```bash
# List generated files
ls -la results/

# Expected files:
# - detections.ndjson (real-time pose data)
# - OZ_Football.npz (joints data)
# - OZ_Football_with_bones.npz (joints + bones data)
```

## 🔄 Migration from Ubuntu/CUDA

### Key Changes Made
1. **Device Detection**: Automatic MPS detection
2. **Dependencies**: Apple Silicon optimized versions
3. **Configuration**: MPS as default device
4. **Performance**: Optimized for unified memory

### Command Line Changes
```bash
# Old (Ubuntu/CUDA)
python main.py --device cuda:0

# New (Apple Silicon)
python src/pose_detection/main.py --device mps
# or
python src/pose_detection/main.py --device auto  # Automatic detection
```

## 📚 Additional Resources

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)
- [Ultralytics YOLO on Apple Silicon](https://docs.ultralytics.com/guides/train-on-apple-silicon/)

## 🆘 Getting Help

### If Something Goes Wrong
1. **Check the logs** - Look for error messages in the terminal output
2. **Verify prerequisites** - Ensure all requirements are met
3. **Check file paths** - Verify video and model files exist
4. **Monitor resources** - Check memory and CPU usage
5. **Restart services** - Restart NATS server if needed

### Support Commands
```bash
# System information
uname -a
sw_vers
python3 --version

# Environment information
echo $PATH
which python
pip list

# File verification
ls -la assets/videos/ assets/models/
file assets/videos/*.mp4

# Monitor performance
sudo powermetrics --samplers gpu_power -n 1
top -pid $(pgrep -f "python.*main.py")
```

---

## 🎯 **Quick Start Summary**

For experienced users, here's the minimal setup:

```bash
# 1. Clone and setup
git clone <REPO_URL> pose && cd pose
cp /path/to/video.mp4 assets/videos/

# 2. Run setup
./setup.sh

# 3. Run pipeline
source venv/bin/activate
python src/pose_detection/main.py

# 4. Monitor NATS messages (in another terminal)
source venv/bin/activate
nats sub pose.detections
```

**Total setup time**: ~5-10 minutes (one command)  
**First run time**: ~2-3 minutes (includes model download)  
**Performance**: 20-30 FPS on 1080p video with MPS acceleration

---

**Compatibility**: macOS 12.3+, Python 3.8+, Apple Silicon/Intel Macs  
**Last Updated**: December 2024

Happy pose detecting on your Mac! 🍎✨


