#!/usr/bin/env python3
"""
Real-time pose detection on video stream → publish NATS messages with keypoint data.

What this does:
1) Opens video stream (file, webcam, or RTSP)
2) Runs YOLO pose detection on each frame in real-time using optimal device (MPS/CUDA/CPU)
3) Publishes pose data as NATS messages with keypoints, confidence scores, and tracking info.

Requirements:
- ultralytics (YOLO), opencv-python, nats-py
- GPU acceleration: Apple Silicon MPS, CUDA, or CPU fallback
- A YOLO pose model (default: yolo11x-pose.pt)

Message format:
{
  "video": <video_id>,
  "frame": <frame_number>,
  "timestamp": <time_sec>,
  "track_id": <track_id>,
  "keypoints": [[x, y, conf], ...],  # normalized coordinates
  "bbox": [x1, y1, x2, y2, conf],   # bounding box
  "source": "YOLO-Pose-Realtime"
}
"""

import argparse, json, os, sys, time
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import asyncio
from nats.aio.client import Client as NATS
from ultralytics import YOLO
import numpy as np
import threading
from queue import Queue, Empty
import torch


def get_optimal_device(device_preference: str = "auto") -> str:
    """
    Automatically detect and return the best available device for inference.
    Priority: MPS (Apple Silicon) > CUDA > CPU
    """
    if device_preference == "auto":
        # Check for Apple Silicon MPS first
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        # Check for CUDA
        elif torch.cuda.is_available():
            return "cuda:0"
        # Fall back to CPU
        else:
            return "cpu"
    elif device_preference == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    elif device_preference.startswith("cuda") and torch.cuda.is_available():
        return device_preference
    else:
        return "cpu"


class RealtimePoseDetector:
    def __init__(
        self,
        model_path: str,
        nats_url: str,
        nats_topic: str,
        video_id: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.7,
        device: str = "auto",
        max_queue_size: int = 100
    ):
        self.model_path = model_path
        self.nats_url = nats_url
        self.nats_topic = nats_topic
        self.video_id = video_id
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = get_optimal_device(device)
        self.max_queue_size = max_queue_size
        
        # Initialize YOLO model on optimal device
        print(f"[INFO] Loading YOLO pose model on {self.device}: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Threading components
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.running = False
        self.nats_client = None
        self.video_finished = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        
    async def connect_nats(self):
        """Connect to NATS server"""
        self.nats_client = NATS()
        await self.nats_client.connect(self.nats_url)
        print(f"[INFO] NATS connected: {self.nats_url}, topic='{self.nats_topic}'")
    
    def preprocess_frame(self, frame: np.ndarray, target_height: int = 640) -> np.ndarray:
        """Resize frame while maintaining aspect ratio"""
        h, w = frame.shape[:2]
        scale = target_height / float(h)
        new_width = int(round(w * scale))
        resized = cv2.resize(frame, (new_width, target_height), interpolation=cv2.INTER_AREA)
        return resized
    
    def process_frame(self, frame: np.ndarray, frame_num: int, timestamp: float) -> Optional[Dict[str, Any]]:
        """Run pose detection on a single frame"""
        try:
            # Run inference
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
            
            if results[0].keypoints is not None and results[0].keypoints.conf is not None:
                keypoints = results[0].keypoints.xyn.cpu().numpy()  # normalized coordinates
                confidences = results[0].keypoints.conf.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
                box_conf = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
                track_ids = results[0].boxes.id.cpu().numpy() if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None else []
                
                detections = []
                for i in range(len(keypoints)):
                    detection = {
                        "track_id": int(track_ids[i]) if i < len(track_ids) else i,
                        "keypoints": keypoints[i].tolist(),
                        "keypoint_confidences": confidences[i].tolist(),
                        "bbox": boxes[i].tolist() if i < len(boxes) else [],
                        "bbox_confidence": float(box_conf[i]) if i < len(box_conf) else 0.0,
                    }
                    detections.append(detection)
                
                return {
                    "frame": frame_num,
                    "timestamp": timestamp,
                    "detections": detections
                }
        except Exception as e:
            print(f"[ERROR] Frame processing error: {e}")
        
        return None
    
    async def publish_pose_data(self, pose_result: Dict[str, Any]):
        """Publish pose detection results to NATS"""
        if not self.nats_client:
            return
            
        for detection in pose_result["detections"]:
            msg = {
                "video": self.video_id,
                "frame": pose_result["frame"],
                "timestamp": round(pose_result["timestamp"], 3),
                "track_id": detection["track_id"],
                "keypoints": detection["keypoints"],
                "keypoint_confidences": detection["keypoint_confidences"],
                "bbox": detection["bbox"],
                "bbox_confidence": detection["bbox_confidence"],
                "source": "YOLO-Pose-Realtime",
                "emitted_at": round(time.time(), 3),
            }
            
            try:
                await self.nats_client.publish(self.nats_topic, json.dumps(msg).encode("utf-8"))
                print(f"[NATS] Published frame {pose_result['frame']}, track {detection['track_id']}")
            except Exception as e:
                print(f"[ERROR] NATS publish error: {e}")
    
    def process_video_stream(self, video_source: str, target_fps: float = 25.0):
        """Process video stream in a separate thread"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_source}")
        
        # Get video properties
        in_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        frame_interval = 1.0 / target_fps
        
        print(f"[INFO] Video source: {video_source}")
        print(f"[INFO] Input FPS: {in_fps:.2f}, Target FPS: {target_fps:.2f}")
        print(f"[INFO] Frame interval: {frame_interval:.3f}s")
        
        self.start_time = time.time()
        last_frame_time = 0
        
        try:
            print(f"[INFO] Starting video processing loop")
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] Video stream ended")
                    self.video_finished = True
                    break
                
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Control frame rate
                if current_time - last_frame_time < frame_interval:
                    continue
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                
                # Add to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait((processed_frame, self.frame_count, elapsed))
                    self.frame_count += 1
                    last_frame_time = current_time
                except:
                    # Queue full, skip frame
                    pass
                
        finally:
            cap.release()
            print(f"[INFO] Video stream processing stopped")
    
    async def process_detection_queue(self):
        """Process frames from queue and publish results"""
        while self.running:
            try:
                # Get frame from queue with timeout
                frame, frame_num, timestamp = self.frame_queue.get(timeout=0.1)
                
                # Process frame
                pose_result = self.process_frame(frame, frame_num, timestamp)
                
                if pose_result and pose_result["detections"]:
                    await self.publish_pose_data(pose_result)
                    
                    # Print first few detections for debugging
                    if frame_num < 3:
                        print(f"[POSE] Frame {frame_num}: {len(pose_result['detections'])} detections")
                
            except Empty:
                # Queue timeout, continue
                continue
            except Exception as e:
                print(f"[ERROR] Processing error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    async def run(self, video_source: str, target_fps: float = 25.0):
        """Main run method"""
        self.running = True
        
        # Connect to NATS
        await self.connect_nats()
        
        # Start video processing thread
        video_thread = threading.Thread(
            target=self.process_video_stream,
            args=(video_source, target_fps)
        )
        video_thread.start()
        print(f"[INFO] Video processing thread started")
        
        print(f"[INFO] Starting real-time pose detection...")
        print(f"[INFO] Press Ctrl+C to stop")
        
        try:
            # Process detection queue
            await self.process_detection_queue()
        except KeyboardInterrupt:
            print(f"\n[INFO] Stopping...")
        finally:
            self.running = False
            video_thread.join()
            
            if self.nats_client:
                try:
                    if self.nats_client.is_connected:
                        await self.nats_client.drain()
                        await self.nats_client.close()
                except asyncio.exceptions.InvalidStateError as e:
                    print(f"[WARN] NATS already closed: {e}")
            
            # Print statistics
            if self.start_time:
                total_time = time.time() - self.start_time
                fps = self.frame_count / total_time if total_time > 0 else 0
                print(f"[INFO] Processed {self.frame_count} frames in {total_time:.2f}s ({fps:.2f} FPS)")


def parse_args():
    ap = argparse.ArgumentParser("Real-time video → YOLO Pose → NATS")
    ap.add_argument("--video", required=True, help="Path to input video, webcam index (0,1,2...), or RTSP URL")
    ap.add_argument("--model", default="assets/models/yolo11x-pose.pt", help="Path to YOLO pose model")
    ap.add_argument("--video-id", default="realtime", help="Video ID for NATS messages")
    ap.add_argument("--conf-threshold", type=float, default=0.5, help="YOLO confidence threshold")
    ap.add_argument("--iou-threshold", type=float, default=0.7, help="YOLO IoU threshold")
    ap.add_argument("--device", default="auto", help="Device to run inference on (auto, mps, cuda:0, cuda:1, cpu)")
    ap.add_argument("--target-fps", type=float, default=25.0, help="Target processing FPS")
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    ap.add_argument("--nats-topic", default="pose.detections")
    ap.add_argument("--max-queue-size", type=int, default=100, help="Maximum frame queue size")
    return ap.parse_args()


def main():
    args = parse_args()
    
    # Device detection and validation
    optimal_device = get_optimal_device(args.device)
    if optimal_device != args.device:
        print(f"[INFO] Device '{args.device}' not available, using optimal device: {optimal_device}")
        args.device = optimal_device
    
    if args.device == "mps":
        print(f"[INFO] Using Apple Silicon MPS acceleration")
    elif args.device.startswith("cuda"):
        print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[INFO] Using CPU for inference")
    
    # Create detector
    detector = RealtimePoseDetector(
        model_path=args.model,
        nats_url=args.nats_url,
        nats_topic=args.nats_topic,
        video_id=args.video_id,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
        max_queue_size=args.max_queue_size
    )
    
    # Run
    asyncio.run(detector.run(args.video, args.target_fps))


if __name__ == "__main__":
    main()
