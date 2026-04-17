import torch
import cv2
import numpy as np



class ObjectDetector:
    def __init__(self, model_name='yolov5s', conf_threshold=0.25, classes=None):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print("Loading YOLOv5 model...")
        self.model = torch.hub.load('ultralytics/yolov5', model_name)

        self.model.conf = conf_threshold

        if classes is not None:
            self.model.classes = classes

        self.model.to(self.device)
        self.model.eval()

    def detect(self, frame):

        with torch.no_grad():
            results = self.model(frame)

        detections = []

        df = results.pandas().xyxy[0]

        for _, row in df.iterrows():
            detections.append({
                "label": row["name"],
                "confidence": float(row["confidence"]),
                "box": [
                    int(row["xmin"]),
                    int(row["ymin"]),
                    int(row["xmax"]),
                    int(row["ymax"])
                ],
                "class_id": int(row["class"])
            })

        return detections, results

    def draw_detections(self, results):

        rendered = results.render()[0]
        return rendered

    def detect_and_draw(self, frame):

        detections, results = self.detect(frame)
        annotated_frame = self.draw_detections(results)
        return detections, annotated_frame