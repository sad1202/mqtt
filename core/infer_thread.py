import time
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from ultralytics import YOLO
import queue
import numpy as np


class InferThread(QThread):
    results_ready = pyqtSignal(object)

    def __init__(self, model_path: str, device="cuda", cam_queues=None):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.cam_queues = cam_queues or {}

    def run(self):

        self.model = YOLO(self.model_path, task="detect")

        while True:
            if not self.cam_queues:
                time.sleep(1)
                continue

            batch = []
            for cam_code, q in self.cam_queues.items():
                latest_item = None
                while True:
                    try:
                        latest_item = q.get_nowait()
                    except queue.Empty:
                        break
                if latest_item is not None:
                    batch.append(latest_item)

            if not batch:
                time.sleep(0.005)
                continue

            frame = []
            meta = []
            for item in batch:
                frame.append(item["frame"])
                meta.append(
                    {
                        "camera_code": item["camera_code"],
                        "timestamp": item["timestamp"],
                        "offset_x": item.get("offset_x", 0),
                        "offset_y": item.get("offset_y", 0),
                        "orig_w": item.get("orig_w", 1),
                        "orig_h": item.get("orig_h", 1),
                    }
                )

            t1 = time.time()

            results = self.model.predict(
                source=frame,
                device=self.device,
                imgsz=640,
                verbose=False,
                conf=0.25,
                classes=[2, 3],
                batch=len(frame),
            )

            t2 = time.time()

            output = []
            for i, result in enumerate(results):
                output.append(
                    {
                        "camera_code": meta[i]["camera_code"],
                        "timestamp": meta[i]["timestamp"],
                        "offset_x": meta[i]["offset_x"],
                        "offset_y": meta[i]["offset_y"],
                        "orig_w": meta[i]["orig_w"],
                        "orig_h": meta[i]["orig_h"],
                        "result": result,
                    }
                )

            latency_ms = (t2 - t1) * 1000
            print(f"[AI] Batch Size: {len(frame)} | Latency: {latency_ms:.2f} ms")
            self.results_ready.emit(output)
