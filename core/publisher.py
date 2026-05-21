import json
import cv2
import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class MQTTPublisher(QObject):

    publish_signal = pyqtSignal(object)

    def __init__(self, client, topic_template, ai_module="PLATE", polygons=None):
        super().__init__()

        self.client = client
        self.topic_template = topic_template
        self.ai_module = ai_module
        self.polygons = polygons or {}

        self.publish_signal.connect(self.publish)

    def is_in_polygon(self, x, y, zones):
        if not zones:
            return True
        for zone in zones:
            points = zone.get("polygon")
            if not points:
                continue
            pts = np.array(points, np.float32).reshape((-1, 1, 2))
            if cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0:
                return True
        return False

    @pyqtSlot(object)
    def publish(self, payload):

        for item in payload:
            camera_code = item["camera_code"]
            topic = self.topic_template.format(camera_code=camera_code)

            result = item["result"]
            offset_x = item.get("offset_x", 0)
            offset_y = item.get("offset_y", 0)
            orig_w = item.get("orig_w", 1)
            orig_h = item.get("orig_h", 1)

            detections = []
            cam_polygons = self.polygons.get(camera_code, [])

            for i, box in enumerate(result.boxes):

                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                abs_x1, abs_y1, abs_x2, abs_y2 = box.xyxy[0].tolist()

                real_x1 = abs_x1 + offset_x
                real_y1 = abs_y1 + offset_y
                real_x2 = abs_x2 + offset_x
                real_y2 = abs_y2 + offset_y

                nx1 = real_x1 / orig_w
                ny1 = real_y1 / orig_h
                nx2 = real_x2 / orig_w
                ny2 = real_y2 / orig_h

                cx, cy = (nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0
                if not self.is_in_polygon(cx, cy, cam_polygons):
                    continue

                label = (
                    result.names[cls_id]
                    if hasattr(result, "names") and cls_id in result.names
                    else str(cls_id)
                )

                colors = [
                    "#FF3838",
                    "#FF9D97",
                    "#FF701F",
                    "#FFB21D",
                    "#CFD231",
                    "#48F90A",
                    "#92CC17",
                    "#3DDB86",
                    "#1A9334",
                    "#00D4BB",
                ]
                color = colors[cls_id % len(colors)]

                detections.append(
                    {
                        "id": f"obj_{item['camera_code']}_{i}",
                        "cls": label,
                        "class": label,
                        "label": label,
                        "class_id": cls_id,
                        "confidence": round(conf, 2),
                        "bbox": [
                            round(nx1, 2),
                            round(ny1, 2),
                            round(nx2, 2),
                            round(ny2, 2),
                        ],
                        "color": color,
                    }
                )

            mqtt_payload = {
                "camera_code": item["camera_code"],
                "ai_module": self.ai_module,
                "ai_modules": [self.ai_module],
                "timestamp": item["timestamp"],
                "detections": detections,
            }

            self.client.publish(
                topic, json.dumps(mqtt_payload, ensure_ascii=False), qos=0
            )
