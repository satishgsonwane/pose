"""
Pose Detection Module

This module contains pose detection functionality using YOLO models.
"""

# from .pose_video_to_nats import main as run_pose_detection  # Removed, file does not exist
from .pose import model as pose_model

__all__ = ['pose_model']
