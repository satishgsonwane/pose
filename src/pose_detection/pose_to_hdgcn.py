import numpy as np
from typing import List, Dict, Optional

class PoseBuffer:
    """
    Maintains a temporal buffer of poses for a single track_id.
    Each pose is expected to be a (V, 3) array: [x, y, conf] for each joint.
    """
    def __init__(self, window_size: int = 64, num_joints: int = 17):
        self.buffer: List[np.ndarray] = []
        self.window_size = window_size
        self.num_joints = num_joints

    def add_frame(self, pose: np.ndarray):
        """Add a pose for the current frame (expects shape (V, 3))."""
        if pose.shape != (self.num_joints, 3):
            raise ValueError(f"Pose must have shape ({self.num_joints}, 3), got {pose.shape}")
        self.buffer.append(pose)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def is_ready(self) -> bool:
        """Returns True if buffer is full (ready for HD-GCN input)."""
        return len(self.buffer) >= self.window_size

    def get_window(self) -> np.ndarray:
        """Returns a (window_size, V, 3) array, padding with zeros if needed."""
        buf = self.buffer[-self.window_size:]
        arr = np.zeros((self.window_size, self.num_joints, 3), dtype=np.float32)
        arr[:len(buf)] = np.stack(buf, axis=0)
        return arr


def remap_joints(yolo_pose: np.ndarray, mapping: Optional[List[int]] = None) -> np.ndarray:
    """
    Remap joints from YOLO (COCO-17) order to target order if needed.
    Args:
        yolo_pose: (V, 3) array
        mapping: list of indices for remapping (len = target joints)
    Returns:
        (V', 3) array in target order
    """
    if mapping is None:
        return yolo_pose
    return yolo_pose[mapping]


def normalize_keypoints(keypoints: np.ndarray, image_width: int, image_height: int, mode: str = "zero_one") -> np.ndarray:
    """
    Normalize keypoints to [0, 1] or [-1, 1] range.
    Args:
        keypoints: (V, 3) array, x/y in pixel coords
        image_width: width of image
        image_height: height of image
        mode: "zero_one" or "minus_one_one"
    Returns:
        (V, 3) array with normalized x/y
    """
    kp = keypoints.copy()
    if mode == "zero_one":
        kp[:, 0] = kp[:, 0] / image_width
        kp[:, 1] = kp[:, 1] / image_height
    elif mode == "minus_one_one":
        kp[:, 0] = (kp[:, 0] - image_width / 2) / (image_width / 2)
        kp[:, 1] = (kp[:, 1] - image_height / 2) / (image_height / 2)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    return kp


def yolo_to_hdgcn(tracked_poses: List[np.ndarray], window_size: int = 64, num_joints: int = 17, channels: int = 2) -> np.ndarray:
    """
    Convert a list of YOLO pose arrays (per frame) to HD-GCN input format.
    Args:
        tracked_poses: list of (V, 3) arrays for each frame
        window_size: T (temporal window)
        num_joints: V (number of joints)
        channels: 2 (x, y) or 3 (x, y, conf)
    Returns:
        np.ndarray of shape (channels, window_size, num_joints)
    """
    hdgcn_input = np.zeros((channels, window_size, num_joints), dtype=np.float32)
    for frame_idx in range(window_size):
        if frame_idx < len(tracked_poses):
            pose = tracked_poses[frame_idx]  # (V, 3)
            hdgcn_input[0, frame_idx, :] = pose[:, 0]  # X
            hdgcn_input[1, frame_idx, :] = pose[:, 1]  # Y
            if channels == 3:
                hdgcn_input[2, frame_idx, :] = pose[:, 2]  # Conf
    return hdgcn_input

class MultiPersonPoseBuffer:
    """
    Manages PoseBuffers for multiple people (track_ids), supporting any tracker (YOLO, ByteTrack, BoT-SORT, etc).
    Usage:
        mpb = MultiPersonPoseBuffer(window_size=64)
        for frame in video:
            for track_id, pose in detections.items():
                mpb.add_pose(track_id, pose)
        ready = mpb.get_ready_buffers()  # {track_id: buffer}
    """
    def __init__(self, window_size: int = 64, num_joints: int = 17):
        self.window_size = window_size
        self.num_joints = num_joints
        self.buffers: Dict[int, PoseBuffer] = {}

    def add_pose(self, track_id: int, pose: np.ndarray):
        """Add a pose for a given track_id (person)."""
        if track_id not in self.buffers:
            self.buffers[track_id] = PoseBuffer(self.window_size, self.num_joints)
        self.buffers[track_id].add_frame(pose)

    def is_ready(self, track_id: int) -> bool:
        """Check if a buffer for a track_id is ready (full window)."""
        return track_id in self.buffers and self.buffers[track_id].is_ready()

    def get_ready_buffers(self) -> Dict[int, PoseBuffer]:
        """Return all track_ids with ready buffers (full window)."""
        return {tid: buf for tid, buf in self.buffers.items() if buf.is_ready()}

    def get_hdgcn_inputs(self, channels: int = 2) -> Dict[int, np.ndarray]:
        """Return HD-GCN input arrays for all ready track_ids."""
        ready = self.get_ready_buffers()
        return {tid: yolo_to_hdgcn(buf.buffer, self.window_size, self.num_joints, channels) for tid, buf in ready.items()}

# Example usage for multi-person, multi-tracker scenario:
# mpb = MultiPersonPoseBuffer(window_size=64)
# for frame in video:
#     detections = tracker.update(frame)  # detections: {track_id: pose}
#     for track_id, pose in detections.items():
#         mpb.add_pose(track_id, pose)
#     for track_id, hdgcn_input in mpb.get_hdgcn_inputs().items():
#         # Feed hdgcn_input to your model
