from ultralytics import YOLO
import torch
import cv2
import numpy as np


class GroundHazardSegmenter:
    def __init__(self, model_path="best-seg.pt", conf_threshold=0.25):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def segment(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        result = results[0]
        hazards = []

        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes
            names = result.names

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()

                hazards.append({
                    "label": names[cls_id],
                    "confidence": conf,
                    "box": [int(v) for v in xyxy],
                    "mask": masks[i],
                    "class_id": cls_id
                })

        return hazards, result

    def draw_segments(self, result):
        return result.plot()