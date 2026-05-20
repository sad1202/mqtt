from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import numpy as np
import queue


class BatchManager(QObject):
    def __init__(self, batch_size=3, polygons=None, frame_queue=None):
        super().__init__()
        self.batch_size = batch_size
        self.current_batch = {}
        self.polygons = polygons or {}
        self.frame_queue = frame_queue

    @pyqtSlot(str, float, object)
    def add_frame(self, camera_code, timestamp, frame):
        cam_polygons = self.polygons.get(camera_code, [])
        h, w = frame.shape[:2]
        
        offset_x, offset_y = 0, 0
        orig_w, orig_h = w, h
        
        if cam_polygons:
            min_x, min_y = w, h
            max_x, max_y = 0, 0
            has_points = False
            for zone in cam_polygons:
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

        self.current_batch[camera_code] = {
            "timestamp": timestamp, 
            "frame": frame,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "orig_w": orig_w,
            "orig_h": orig_h
        }
        
        if len(self.current_batch) >= self.batch_size:
            batch = []
            for cam_code, item in self.current_batch.items():
                batch.append(
                    {
                        "camera_code": cam_code,
                        "timestamp": item["timestamp"],
                        "frame": item["frame"],
                        "offset_x": item["offset_x"],
                        "offset_y": item["offset_y"],
                        "orig_w": item["orig_w"],
                        "orig_h": item["orig_h"],
                    }
                )

            self.current_batch.clear()
            
            if self.frame_queue is not None:
                try:
                    self.frame_queue.put_nowait(batch)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    
                    try:
                        self.frame_queue.put_nowait(batch)
                    except queue.Full:
                        pass
