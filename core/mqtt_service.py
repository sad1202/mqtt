import json
import time
import paho.mqtt.client as mqtt
from ultralytics import cfg


class MQTTService:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=f"pyqt5_ai_{int(time.time())}"
        )
        self.client.username_pw_set(self.cfg.get("username"), self.cfg.get("password"))

    def start(self):
        self.client.connect(self.cfg.get("broker"), self.cfg.get("port", 1883))
        self.client.loop_start()
        print("MQTT client started")

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
        print("MQTT client stopped")

    def publish(self, camera_code, payload):
        topic = self.cfg.get(
            "bbox_topic_template", "cameras/{camera_code}/detections"
        ).format(camera_code=camera_code)
        self.client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=0)
