import sys
import queue
import argparse
from PyQt5.QtCore import QCoreApplication, QThread
import signal

from core.camera_stream import CameraStream
from core.infer_thread import InferThread
from core.mqtt_service import MQTTService
from core.publisher import MQTTPublisher

from mock_bbox_publisher import MQTT_CONFIG, load_cameras, load_polygons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-code", default="")
    args = parser.parse_args()

    app = QCoreApplication(sys.argv)

    cfg = MQTT_CONFIG
    print("Loading cameras from MQTT...")

    cameras = load_cameras(cfg)
    print("Loading polygons from MQTT (this may take up to 5 seconds)...")
    polygons = load_polygons(cfg)

    if args.camera_code.strip():
        camera_codes = [args.camera_code.strip()]
    else:
        camera_codes = [camera["code"] for camera in cameras if camera.get("code")]

    if not camera_codes:
        print("No ONLINE camera found. Exiting.")
        sys.exit(1)

    active_cameras = [c for c in cameras if c.get("code") in camera_codes]
    print(f"Found {len(active_cameras)} active cameras.")

    if len(active_cameras) == 0:
        sys.exit(1)

    mqtt_service = MQTTService(cfg)
    mqtt_service.start()

    publisher = MQTTPublisher(
        mqtt_service.client,
        cfg["bbox_topic_template"],
        ai_module=cfg.get("ai_module", "PLATE"),
        polygons=polygons,
    )

    import os
    if os.path.exists("yolo26n.engine"):
        model_path = "yolo26n.engine"
        print("TensorRT model found! Using: yolo26n.engine")
    else:
        model_path = "yolo26n.pt"
        print("TensorRT model not found. Using fallback PyTorch model: yolo26n.pt")

    cam_queues = { camera["code"]: queue.Queue(maxsize=3) for camera in active_cameras }

    infer_thread = InferThread(
        model_path=model_path, device="cuda", cam_queues=cam_queues
    )
    infer_thread.start()

    infer_thread.results_ready.connect(publisher.publish_signal)
    camera_threads = []
    camera_streams = []

    for camera in active_cameras:
        camera_code = camera["code"]
        rtsp_url = camera["rtsp"]

        print(f"Initializing stream for camera {camera_code}: {rtsp_url}")

        thread = QThread()
        stream = CameraStream(
            camera_code=camera_code, 
            rtsp_url=rtsp_url, 
            polygons=polygons.get(camera_code, []), 
            frame_queue=cam_queues[camera_code]
        )
        stream.status.connect(lambda msg, code=camera_code: print(f"[{code}] {msg}"))

        stream.moveToThread(thread)
        thread.started.connect(stream.run)

        camera_threads.append(thread)
        camera_streams.append(stream)

        thread.start()

    print("Pipeline started successfully. Waiting for frames... (Press Ctrl+C to stop)")

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
