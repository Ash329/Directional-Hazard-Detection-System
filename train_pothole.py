from pathlib import Path

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    model = YOLO(str(BASE_DIR / "yolov8n.pt"))

    model.train(
        data=str(BASE_DIR / "data/pothole_yolo/data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        name="pothole_detector",
        # Absolute path: Ultralytics prefixes relative projects with runs/detect/
        project=str(BASE_DIR / "runs/detect"),
        device=device,
        workers=4,
    )


if __name__ == "__main__":
    main()
