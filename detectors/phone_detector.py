from ultralytics import YOLO

class PhoneDetector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.25):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_phone(self, frame):
        results = self.model(frame, verbose=False)

        phone_found = False
        best_box = None
        best_conf = 0.0

        for result in results:
            boxes = result.boxes
            names = result.names

            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = names[cls_id]

                # sometimes class label can vary slightly, so safer check:
                if "phone" in label.lower() and conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    if conf > best_conf:
                        best_conf = conf
                        best_box = (x1, y1, x2, y2, conf, label)
                        phone_found = True

        return phone_found, best_box