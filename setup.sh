#!/bin/bash

# 🚀 One-Command Pose Detection Pipeline Setup
# Just run this script and it does everything automatically!
# Only requirement: Place your video file in assets/videos/ before running

echo "🚀 Setting up pose detection pipeline in one command..."

# Create directories
mkdir -p assets/videos assets/models results

# Check for video file
if [ ! "$(ls -A assets/videos)" ]; then
    echo "⚠️  No video file found in assets/videos/"
    echo "   Please add your video file before running the pipeline:"
    echo "   cp /path/to/your/video.mp4 assets/videos/"
    echo ""
fi

# Setup Python environment
python3 -m venv venv 2>/dev/null || echo "Virtual environment already exists"
source venv/bin/activate

# Install dependencies based on architecture
if [[ "$(uname -m)" == "arm64" ]]; then
    echo "🍎 Apple Silicon detected - Installing MPS-optimized PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements-apple-silicon.txt
else
    echo "🖥️  Intel Mac detected - Installing standard PyTorch..."
    pip install torch torchvision
    pip install -r requirements.txt
fi

# Install NATS if Homebrew available
if command -v brew >/dev/null 2>&1; then
    brew install nats-io/nats-tools/nats 2>/dev/null || echo "NATS already installed"
else
    echo "⚠️  Homebrew not found. Install NATS manually from:"
    echo "   https://github.com/nats-io/nats-server/releases"
fi

# Update configuration
if [[ "$(uname -m)" == "arm64" ]]; then
    sed -i '' 's/device: cuda:0/device: mps/' config/pose_detection.yaml 2>/dev/null || echo "Configuration updated"
else
    sed -i '' 's/device: cuda:0/device: auto/' config/pose_detection.yaml 2>/dev/null || echo "Configuration updated"
fi

# Start NATS server in background
echo "🚀 Starting NATS server..."
nats-server > /dev/null 2>&1 &
NATS_PID=$!
sleep 2

# Verify NATS is running
if ps -p $NATS_PID > /dev/null; then
    echo "✅ NATS server started successfully (PID: $NATS_PID)"
else
    echo "⚠️  NATS server failed to start. You may need to start it manually:"
    echo "   nats-server &"
fi

echo ""
echo "🎉 Setup Complete! 🎉"
echo ""
echo "🚀 To run the pose detection pipeline:"
echo "1. source venv/bin/activate"
echo "2. python src/pose_detection/main.py"
echo ""
echo "📱 TO LISTEN TO NATS MESSAGES (in another terminal):"
echo "   cd $(pwd)"
echo "   source venv/bin/activate"
echo "   nats sub pose.detections"
echo ""
echo "📁 TO FIND RESULTS:"
echo "   - Real-time data: results/detections.ndjson"
echo "   - Joint data: results/OZ_Football.npz"
echo "   - Joint+bones data: results/OZ_Football_with_bones.npz"
echo ""
echo "🔍 Monitor pipeline progress in the main terminal"
echo "🔍 Monitor NATS messages in the second terminal"
echo ""
echo "Happy pose detecting! 🍎✨"
