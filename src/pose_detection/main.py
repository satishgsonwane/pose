#!/usr/bin/env python3
"""
Phase-1 Combo:
- YOLO-Pose → NATS publisher (pose.detections)
- NATS sink → results/detections.ndjson
- Build NPZ (joints & bones) in results/ at exit or on interval

Run example:
python phase1_combo.py \
  --video assets/videos/closeup.mp4 \
  --model assets/models/yolo11x-pose.pt \
  --nats-url nats://127.0.0.1:4222 \
  --nats-topic pose.detections \
  --device mps \
  --T 64 --stride 32
# Output files will be written to the results/ directory by default
"""

import argparse, asyncio, json, os, signal, time, threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from nats.aio.client import Client as NATS
from queue import Queue, Empty
import yaml
import urllib.request
from pose_to_hdgcn import *

# Default model download URL (YOLOv8n-pose as example)
DEFAULT_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt"

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

def download_model_if_missing(model_path: str, url: str = DEFAULT_MODEL_URL):
    if not os.path.exists(model_path):
        print(f"[INFO] Model not found at {model_path}. Downloading from {url}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"[INFO] Model downloaded to {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download model: {e}")
            raise

# ---------- Shared config / constants ----------
V, C, M = 17, 3, 1  # COCO-17, channels=(x,y,conf), single person per sequence
COCO_EDGES: List[Tuple[int,int]] = [
    (5,7),(7,9),(6,8),(8,10),
    (11,13),(13,15),(12,14),(14,16),
    (5,6),(11,12),(5,11),(6,12),
    (0,5),(0,6),(0,1),(0,2),(1,3),(2,4),
]

# ---------- Publisher (your code, minimally adapted) ----------
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
        max_queue_size: int = 100,
        tracker: str = "bytetrack.yaml"
    ):
        self.model_path = model_path
        self.nats_url = nats_url
        self.nats_topic = nats_topic
        self.video_id = video_id
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = get_optimal_device(device)
        self.max_queue_size = max_queue_size
        self.tracker = tracker

        # Download model if missing
        download_model_if_missing(model_path)

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
        self.nats_client = NATS()
        await self.nats_client.connect(self.nats_url)
        print(f"[INFO] NATS connected: {self.nats_url}, topic='{self.nats_topic}'")

    def preprocess_frame(self, frame: np.ndarray, target_height: int = 640) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = target_height / float(h)
        new_width = int(round(w * scale))
        return cv2.resize(frame, (new_width, target_height), interpolation=cv2.INTER_AREA)

    def process_frame(self, frame: np.ndarray, frame_num: int, timestamp: float) -> Optional[Dict[str, Any]]:
        try:
            # Use tracker from instance variable
            results = self.model.track(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False, persist=True, tracker=self.tracker)
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
        if not self.nats_client:
            return
        for det in pose_result["detections"]:
            msg = {
                "video": self.video_id,
                "frame": pose_result["frame"],
                "timestamp": round(pose_result["timestamp"], 3),
                "track_id": det["track_id"],
                "keypoints": det["keypoints"],
                "keypoint_confidences": det["keypoint_confidences"],
                "bbox": det["bbox"],
                "bbox_confidence": det["bbox_confidence"],
                "source": "YOLO-Pose-Realtime",
                "emitted_at": round(time.time(), 3),
            }
            try:
                await self.nats_client.publish(self.nats_topic, json.dumps(msg).encode("utf-8"))
                await self.nats_client.flush()  # Force real-time delivery
            except Exception as e:
                print(f"[ERROR] NATS publish error: {e}")

    def process_video_stream(self, video_source: str, target_fps: float):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_source}")

        in_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        frame_interval = 1.0 / target_fps
        print(f"[INFO] Video: {video_source} | Input FPS: {in_fps:.2f} | Target FPS: {target_fps:.2f}")
        self.start_time = time.time()
        last_frame_time = 0.0

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.video_finished = True
                    break
                now = time.time()
                if (now - last_frame_time) < frame_interval:
                    continue
                processed = self.preprocess_frame(frame)
                try:
                    self.frame_queue.put_nowait((processed, self.frame_count, now - self.start_time))
                    self.frame_count += 1
                    last_frame_time = now
                except:
                    pass
        finally:
            cap.release()

    async def serve(self, video_source: str, target_fps: float):
        self.running = True
        await self.connect_nats()
        t = threading.Thread(target=self.process_video_stream, args=(video_source, target_fps))
        t.start()
        try:
            while self.running:
                try:
                    frame, frame_num, ts = self.frame_queue.get(timeout=0.1)
                except Empty:
                    await asyncio.sleep(0.001)
                    continue
                res = self.process_frame(frame, frame_num, ts)
                if res and res["detections"]:
                    await self.publish_pose_data(res)
        finally:
            self.running = False
            t.join()
            if self.nats_client:
                await self.nats_client.drain()
                await self.nats_client.close()
            if self.start_time:
                total = time.time() - self.start_time
                fps = self.frame_count / total if total > 0 else 0
                print(f"[INFO] Processed {self.frame_count} frames in {total:.2f}s ({fps:.2f} FPS)")

# ---------- Sink: subscribe → write NDJSON & keep in-memory buffer ----------
class PoseSink:
    def __init__(self, nats_url: str, topic: str, ndjson_path: str, flush_every: int = 50,
                 mem_buffer_max: int = 1_000_000):
        self.nats_url = nats_url
        self.topic = topic
        self.ndjson_path = ndjson_path
        self.flush_every = flush_every
        self.mem_buffer_max = mem_buffer_max
        self.nc: Optional[NATS] = None
        self.file = None
        self.count = 0
        self.messages: List[Dict[str, Any]] = []  # compact memory buffer for NPZ build

    async def start(self):
        Path(self.ndjson_path).parent.mkdir(parents=True, exist_ok=True)
        # append mode, line-buffered
        self.file = open(self.ndjson_path, "a", buffering=1, encoding="utf-8")
        self.nc = NATS()
        await self.nc.connect(self.nats_url)
        await self.nc.subscribe(self.topic, cb=self._on_msg)
        print(f"[SINK] writing → {self.ndjson_path}")

    async def stop(self):
        if self.nc:
            await self.nc.drain()
            await self.nc.close()
        if self.file:
            self.file.flush()
            self.file.close()

    async def _on_msg(self, msg):
        try:
            payload = json.loads(msg.data.decode("utf-8"))
            # normalize: accept [x,y] + confidences, or [x,y,conf] tuples
            k = payload.get("keypoints", [])
            kc = payload.get("keypoint_confidences", [])
            if k and len(k[0]) == 3:
                kc = [p[2] for p in k]
                k = [[p[0], p[1]] for p in k]
                payload["keypoints"] = k
                payload["keypoint_confidences"] = kc

            line = json.dumps(payload, ensure_ascii=False)
            self.file.write(line + "\n")
            self.count += 1
            if self.count % self.flush_every == 0:
                self.file.flush()

            # Keep a memory copy for faster NPZ build
            if len(self.messages) < self.mem_buffer_max:
                self.messages.append(payload)
        except Exception as e:
            print(f"[SINK] bad message skipped: {e}")

# ---------- NPZ Builders (joints + bones) ----------
# NOTE: See pose_to_hdgcn.py for reusable pose-to-HDG-CN conversion utilities (buffer, normalization, remapping, etc.)
def make_windows(n: int, T: int, stride: int) -> List[Tuple[int,int]]:
    if n == 0: return []
    if n <= T: return [(0, n)]
    wins, s = [], 0
    while s + T <= n:
        wins.append((s, s + T))
        s += stride
    if wins and wins[-1][1] < n:
        wins.append((n - T, n))
    elif not wins:
        wins = [(0, T)]
    return wins

def pad_to_T(seq: np.ndarray, T: int) -> np.ndarray:
    t = seq.shape[0]
    if t == T: return seq
    pad = np.repeat(seq[-1:], T - t, axis=0)
    return np.concatenate([seq, pad], axis=0)

def build_tracks(msgs: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    tracks: Dict[int, List[Dict[str, Any]]] = {}
    for m in msgs:
        tid = int(m["track_id"]); frm = int(m["frame"])
        kxy = np.array(m["keypoints"], np.float32)  # (V,2)
        kc  = np.array(m["keypoint_confidences"], np.float32).reshape(-1)[:V]
        if kxy.shape != (V,2): continue
        tracks.setdefault(tid, []).append({"frame": frm, "kxy": kxy, "kc": kc})
    for tid in tracks:
        tracks[tid].sort(key=lambda d: d["frame"])
    return tracks

def build_npz_arrays(tracks: Dict[int, List[Dict[str, Any]]], T: int, stride: int):
    seqs, index = [], []
    for tid, items in tracks.items():
        wins = make_windows(len(items), T, stride)
        for s, e in wins:
            chunk = items[s:e]
            t = len(chunk)
            xy = np.stack([c["kxy"] for c in chunk], axis=0)  # (t,V,2)
            kc = np.stack([c["kc"]  for c in chunk], axis=0)  # (t,V)
            xyc = np.concatenate([xy, kc[...,None]], axis=-1) # (t,V,3)
            xyc = pad_to_T(xyc, T)                            # (T,V,3)
            xyc = xyc[..., None]                              # (T,V,3,1)
            seqs.append(xyc)
            frames = [c["frame"] for c in chunk]
            index.append({"track_id": tid, "start_frame": frames[0], "end_frame": frames[-1], "num_frames": t})
    x_train = np.stack(seqs, axis=0).astype(np.float32) if seqs else np.zeros((0,T,V,C,M), np.float32)
    meta = {"T": T, "V": V, "C": C, "M": M, "joint_format": "COCO-17", "index": index}
    return x_train, meta

def joints_to_bones(x_train: np.ndarray) -> np.ndarray:
    N,T,V,C,M = x_train.shape
    bone = np.zeros_like(x_train, np.float32)
    x = x_train[..., 0, :]  # (N,T,V,M)
    y = x_train[..., 1, :]
    c = x_train[..., 2, :] if C >= 3 else np.ones((N,T,V,M), np.float32)
    for p, ch in COCO_EDGES:
        dx = x[:,:,ch,:] - x[:,:,p,:]
        dy = y[:,:,ch,:] - y[:,:,p,:]
        bc = np.minimum(c[:,:,p,:], c[:,:,ch,:])
        bone[:,:,ch,0,:] = dx
        bone[:,:,ch,1,:] = dy
        if C >= 3: bone[:,:,ch,2,:] = bc
    return bone

def build_and_save_npz(memory_msgs: List[Dict[str, Any]], ndjson_path: str,
                        out_joints: str, out_joint_bones: str, T: int, stride: int):
    # Prefer in-memory buffer; if empty, fall back to reading NDJSON
    msgs = memory_msgs
    if not msgs and Path(ndjson_path).exists():
        with open(ndjson_path, "r", encoding="utf-8") as f:
            msgs = [json.loads(line) for line in f if line.strip()]
    tracks = build_tracks(msgs)
    x_train, meta = build_npz_arrays(tracks, T=T, stride=stride)
    Path(out_joints).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_joints, x_train=x_train, meta=json.dumps(meta))
    bone = joints_to_bones(x_train)
    np.savez_compressed(out_joint_bones, x_train_joints=x_train, x_train_bones=bone, meta=json.dumps(meta))
    print(f"[NPZ] Saved:\n  - {out_joints}  (x_train {x_train.shape})\n  - {out_joint_bones}  (joints+bones)")

# ---------- Orchestration ----------
async def main():
    # Load config if present
    config_path = "config/pose_detection.yaml"
    config = {}
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    ap = argparse.ArgumentParser()
    # publisher args (same as yours)
    # ap.add_argument("--video",required=True)
    ap.add_argument("--video", default=config.get("video", "assets/videos/long.mp4"))
    ap.add_argument("--model", default=config.get("model_path", "assets/models/yolo11x-pose.pt"))
    ap.add_argument("--video-id", default=config.get("video_id", "realtime"))
    ap.add_argument("--conf-threshold", type=float, default=config.get("conf_threshold", 0.5))
    ap.add_argument("--iou-threshold", type=float, default=config.get("iou_threshold", 0.7))
    ap.add_argument("--device", default=config.get("device", "auto"))
    ap.add_argument("--target-fps", type=float, default=config.get("target_fps", 30.0))
    ap.add_argument("--nats-url", default=config.get("nats_url", "nats://127.0.0.1:4222"))
    ap.add_argument("--nats-topic", default=config.get("nats_topic", "pose.detections"))
    ap.add_argument("--max-queue-size", type=int, default=config.get("max_queue_size", 100))
    ap.add_argument("--tracker", default=config.get("tracker", "bytetrack.yaml"), help="Tracker YAML file (default: bytetrack.yaml)")

    # sink + builder args
    ap.add_argument("--ndjson", default=config.get("output_ndjson", "results/detections.ndjson"), help="Output NDJSON file (default: results/detections.ndjson)")
    ap.add_argument("--out-joints", default=config.get("output_npz", "results/OZ_Football.npz"), help="Output NPZ file for joints (default: results/OZ_Football.npz)")
    ap.add_argument("--out-joint-bones", default=config.get("output_npz_bones", "results/OZ_Football_with_bones.npz"), help="Output NPZ file for joints+bones (default: results/OZ_Football_with_bones.npz)")
    ap.add_argument("--T", type=int, default=config.get("T", 64))
    ap.add_argument("--stride", type=int, default=config.get("stride", 32))
    ap.add_argument("--snapshot-sec", type=int, default=config.get("snapshot_sec", 0),
                    help="If >0, build NPZ snapshots every N seconds (optional).")
    ap.add_argument("--flush-every", type=int, default=config.get("flush_every", 50),
                    help="Flush NDJSON file every N messages (set 1 for real-time)")

    args = ap.parse_args()

    # Device detection and validation
    optimal_device = get_optimal_device(args.device)
    if optimal_device != args.device:
        print(f"[INFO] Device '{args.device}' not available, using optimal device: {optimal_device}")
        args.device = optimal_device
    
    if args.device == "mps":
        print(f"[INFO] Using Apple Silicon MPS acceleration")
    elif args.device.startswith("cuda"):
        print(f"[INFO] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[INFO] Using CPU for inference")

    # components
    detector = RealtimePoseDetector(
        model_path=args.model, nats_url=args.nats_url, nats_topic=args.nats_topic,
        video_id=args.video_id, conf_threshold=args.conf_threshold, iou_threshold=args.iou_threshold,
        device=args.device, max_queue_size=args.max_queue_size, tracker=args.tracker if isinstance(args.tracker, str) else (args.tracker.get('config') if isinstance(args.tracker, dict) else 'bytetrack.yaml')
    )
    sink = PoseSink(nats_url=args.nats_url, topic=args.nats_topic, ndjson_path=args.ndjson, flush_every=args.flush_every)

    # Graceful shutdown
    stop_event = asyncio.Event()
    def _stop(*_):
        if not stop_event.is_set():
            print("\n[CTRL-C] Shutting down…")
            stop_event.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _stop)

    # start sink first (so it doesn’t miss early messages), then publisher
    await sink.start()
    pub_task = asyncio.create_task(detector.serve(args.video, args.target_fps))

    # optional periodic snapshots
    async def snapshot_loop():
        if args.snapshot_sec <= 0: return
        while not stop_event.is_set():
            await asyncio.sleep(args.snapshot_sec)
            try:
                build_and_save_npz(sink.messages, args.ndjson, args.out_joints, args.out_joint_bones,
                                   T=args.T, stride=args.stride)
            except Exception as e:
                print(f"[NPZ] snapshot error: {e}")
    snap_task = asyncio.create_task(snapshot_loop())

    # wait for video to finish or CTRL-C
    while not stop_event.is_set():
        await asyncio.sleep(0.2)
        if detector.video_finished:
            print("[INFO] Video ended.")
            stop_event.set()

    # tear down
    await sink.stop()
    await asyncio.sleep(0.05)
    pub_task.cancel()
    snap_task.cancel()
    try: await pub_task
    except: pass
    try: await snap_task
    except: pass

    # final NPZ build
    try:
        build_and_save_npz(sink.messages, args.ndjson, args.out_joints, args.out_joint_bones,
                           T=args.T, stride=args.stride)
    except Exception as e:
        print(f"[NPZ] build error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
