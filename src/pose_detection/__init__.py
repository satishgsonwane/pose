"""
Pose Detection Module

This module contains pose detection functionality using YOLO models.
Optimized for Apple Silicon Macs with automatic MPS detection.
"""

# Import main components
from .main import RealtimePoseDetector, get_optimal_device
from .pose_to_hdgcn import PoseBuffer, MultiPersonPoseBuffer, yolo_to_hdgcn

__all__ = [
    'RealtimePoseDetector',
    'get_optimal_device', 
    'PoseBuffer',
    'MultiPersonPoseBuffer',
    'yolo_to_hdgcn'
]
