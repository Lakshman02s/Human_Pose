from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class FramePacket:
    frame_id: int
    timestamp_ms: float
    frame_bgr: any


class VideoSource:
    def __init__(
        self,
        source: str | int,
        source_type: str = "video",
        rtsp_mode: str = "smooth",
        rtsp_queue_size: int = 512,
    ):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.source = source
        self.source_type = source_type
        self.rtsp_mode = rtsp_mode
        self._packet_queue: queue.Queue[FramePacket] | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_packet: FramePacket | None = None
        if source_type == "rtsp":
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if rtsp_mode == "live":
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self._packet_queue = queue.Queue(maxsize=max(32, rtsp_queue_size))
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open source: {source}")
        self.frame_id = 0
        if self.source_type == "rtsp" and self.rtsp_mode == "smooth":
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            packet = FramePacket(
                frame_id=self.frame_id,
                timestamp_ms=time.time() * 1000.0,
                frame_bgr=frame,
            )
            self.frame_id += 1
            try:
                self._packet_queue.put(packet, timeout=0.1)
            except queue.Full:
                # In smooth mode, wait for the consumer instead of discarding frames aggressively.
                continue

    def read(self) -> FramePacket | None:
        if self.source_type == "rtsp" and self.rtsp_mode == "smooth" and self._packet_queue is not None:
            while not self._stop_event.is_set():
                try:
                    packet = self._packet_queue.get(timeout=0.2)
                    self._last_packet = packet
                    return packet
                except queue.Empty:
                    if self._reader_thread is not None and not self._reader_thread.is_alive():
                        return None
            return None

        ok, frame = self.cap.read()
        if not ok:
            return None
        packet = FramePacket(
            frame_id=self.frame_id,
            timestamp_ms=self.cap.get(cv2.CAP_PROP_POS_MSEC),
            frame_bgr=frame,
        )
        self.frame_id += 1
        return packet

    def fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 1e-6 else 25.0

    def frame_size(self) -> tuple[int, int]:
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def release(self) -> None:
        self._stop_event.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self.cap.release()


class VideoSink:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Unable to open writer: {path}")

    def write(self, frame_bgr) -> None:
        self.writer.write(frame_bgr)

    def release(self) -> None:
        self.writer.release()
