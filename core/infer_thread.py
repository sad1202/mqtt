import time
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from ultralytics import YOLO


class InferThread(QThread):
    results_ready = pyqtSignal(object)

    def __init__(self, model_path: str, device="cuda"):
        super().__init__()
        self.model = YOLO(model_path, task="detect")
        self.device = device

    @pyqtSlot(object)
    def infer(self, batch):
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
        results = self.model(
            source=frame, device=self.device, imgsz=640, conf=0.25, classes=[2, 3]
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
        print(f"Inference time: {t2 - t1:.2f} seconds")
        self.results_ready.emit(output)
