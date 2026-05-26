import time
import os
import cv2
import queue

import numpy as np
from PyQt5.QtCore import QThread, QObject, pyqtSignal
import subprocess


class CameraStream(QObject):
    frame_ready = pyqtSignal(str, float, object)

    status = pyqtSignal(str)

    def __init__(
        self, camera_code: str, rtsp_url: str, polygons=None, frame_queue=None, skip = 2
    ):
        super().__init__()
        self.camera_code = camera_code
        self.rtsp_url = rtsp_url
        self.polygons = polygons or []
        self.frame_queue = frame_queue
        self.running = True
        self.process = None

        self.width = 640
        self.height = 360

        self.frame_size = self.width * self.height * 3
        self.skip = skip
        self.frame_count = 0
    def run(self):
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "cuda",
            "-rtsp_transport",
            "tcp",
            "-fflags",
            "nobuffer+discardcorrupt",
            "-flags",
            "low_delay",
            "-max_delay",
            "0",
            "-i",
            self.rtsp_url,
            "-vf",
            "scale=640:360",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        self.start_ffmpeg(command)
        self.status.emit(f"Camera {self.camera_code} stream started")
        while self.running:
            try:
                raw = self.process.stdout.read(self.frame_size)
                if len(raw) != self.frame_size:
                    self.status.emit(f"Camera {self.camera_code} stream ended")
                    self.restart_ffmpeg(command)
                    continue
                self.frame_count += 1
                if self.frame_count % self.skip != 0:
                    continue
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (self.height, self.width, 3)
                )
                timestamp = time.time()

                h, w = frame.shape[:2]
                offset_x, offset_y = 0, 0
                orig_w, orig_h = w, h

                if self.polygons:
                    min_x, min_y = w, h
                    max_x, max_y = 0, 0
                    has_points = False
                    for zone in self.polygons:
                        points = zone.get("polygon")
                        if not points:
                            continue
                        has_points = True
                        pts = np.array(points, np.float32)
                        pts_x = pts[:, 0] * w
                        pts_y = pts[:, 1] * h
                        min_x = min(min_x, np.min(pts_x))
                        min_y = min(min_y, np.min(pts_y))
                        max_x = max(max_x, np.max(pts_x))
                        max_y = max(max_y, np.max(pts_y))

                    if has_points:
                        min_x = max(0, int(min_x))
                        min_y = max(0, int(min_y))
                        max_x = min(w, int(max_x))
                        max_y = min(h, int(max_y))
                        if max_x > min_x and max_y > min_y:
                            frame = frame[min_y:max_y, min_x:max_x]
                            offset_x, offset_y = min_x, min_y

                item = {
                    "camera_code": self.camera_code,
                    "timestamp": timestamp,
                    "frame": frame,
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                    "orig_w": orig_w,
                    "orig_h": orig_h,
                }

                if self.frame_queue is not None:
                    try:
                        self.frame_queue.put_nowait(item)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.frame_queue.put_nowait(item)
                        except queue.Full:
                            pass

            except Exception as e:
                self.status.emit(f"Error in camera {self.camera_code} stream: {str(e)}")
                self.restart_ffmpeg(command)

    def start_ffmpeg(self, command):
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, bufsize=self.frame_size * 3
        )

    def restart_ffmpeg(self, command):
        try:
            if self.process:
                self.process.terminate()
                self.process.wait(timeout=5)
        except Exception:
            pass
        time.sleep(1)
        self.start_ffmpeg(command)

    def stop(self):
        self.running = False
        try:
            if self.process:
                self.process.terminate()
                self.process.wait(timeout=5)
        except Exception:
            pass