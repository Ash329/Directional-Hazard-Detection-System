from ultralytics import YOLO
import torch


class ObjectDetector:


    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.25):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame, classes=None):
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        result = results[0]

        if result.boxes is not None:
            boxes = result.boxes
            names = result.names

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())

                if classes is not None and names[cls_id] not in classes:
                    continue

                xyxy = boxes.xyxy[i].tolist()
                detections.append({
                    "label": names[cls_id],
                    "confidence": conf,
                    "box": [int(v) for v in xyxy],
                    "class_id": cls_id
                })

        return detections, result

    def draw_detections(self, result):
        return result.plot()

    def detect_and_draw(self, frame, classes=None):
        detections, result = self.detect(frame, classes=classes)
        annotated = self.draw_detections(result)
        return detections, annotated