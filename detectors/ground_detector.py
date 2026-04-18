from ultralytics import YOLO
import torch


class PotholeDetector:
    def __init__(self, model_path="runs/detect/pothole_detector/weights/best.pt", conf=0.3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            device=self.device,
            verbose=False
        )

        return results[0]

    def draw(self, result):
        return result.plot()