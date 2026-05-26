import time
import queue
import torch

from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO


class InferThread(QThread):
    results_ready = pyqtSignal(object)

    def __init__(self, model_path: str, device="cuda", cam_queues=None):
        super().__init__()

        self.model_path = model_path
        self.device = device
        self.cam_queues = cam_queues or {}

        self.running = True
        self.model = None

    def sync_cuda(self):
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def run(self):
        print("[AI] Loading YOLO model...", flush=True)

        self.model = YOLO(self.model_path, task="detect")

        print("[AI] YOLO model loaded", flush=True)

        while self.running:
            if not self.cam_queues:
                time.sleep(1)
                continue

            batch = []

            for cam_code, q in self.cam_queues.items():
                try:
                    item = q.get_nowait()
                    batch.append(item)
                except queue.Empty:
                    pass

            if not batch:
                time.sleep(0.005)
                continue

            frames = []
            meta = []

            for item in batch:
                frames.append(item["frame"])
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

            try:
                self.sync_cuda()
                t1 = time.perf_counter()

                results = self.model.predict(
                    source=frames,
                    device=self.device,
                    imgsz=640,
                    verbose=False,
                    conf=0.25,
                    classes=[2, 3, 5, 7],
                    batch=len(frames),
                )

                self.sync_cuda()
                t2 = time.perf_counter()

            except Exception as e:
                print(f"[AI] Predict error: {e}", flush=True)
                time.sleep(0.1)
                continue

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

            latency = t2 - t1
            batch_size = len(frames)

            if latency > 0:
                batch_fps = 1.0 / latency
                image_fps = batch_size / latency
            else:
                batch_fps = 0.0
                image_fps = 0.0

            # print(
            #     f"[AI] batch={batch_size} | "
            #     f"latency={latency * 1000:.2f} ms | "
            #     f"batch_fps={batch_fps:.2f} | ",
            #     flush=True,
            # )

            self.results_ready.emit(output)

    def stop(self):
        self.running = False