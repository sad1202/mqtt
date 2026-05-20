import time
import os
import cv2

import numpy as np
from PyQt5.QtCore import QThread, QObject, pyqtSignal
import subprocess


class CameraStream(QObject):
    frame_ready = pyqtSignal(str, float, object)

    status = pyqtSignal(str)

    def __init__(self, camera_code: str, rtsp_url: str):
        self.camera_code = camera_code
        self.rtsp_url = rtsp_url
        self.running = True
        self.process = None

        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.cap.release()

        if self.width <= 0 or self.height <= 0:
            self.status.emit(f"Failed to open camera {self.camera_code}")
            self.running = False

        self.frame_size = self.width * self.height * 3

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
            "nobuffer",
            "-flags",
            "low_delay",
            "-i",
            self.rtsp_url,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        self.start_ffmpeg(command)
        self.status.emit(f"Camera {self.camera_code} stream ended")
        while self.running:
            try:
                raw = self.process.stdout.read(self.frame_size)
                if len(raw) != self.frame_size:
                    self.status.emit(f"Camera {self.camera_code} stream ended")
                    self.restart_ffmpeg(command)
                    continue
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (self.height, self.width, 3)
                )
                timestamp = time.time()
                frame = frame // 255.0
                self.frame_ready.emit(self.camera_code, timestamp, frame)
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