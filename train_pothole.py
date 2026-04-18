from ultralytics import YOLO
import torch


def main():
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO("yolov8n.pt")

    model.train(
        data="data/pothole_yolo/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="pothole_detector",
        project="runs/detect",
        device=device,
        workers=4
    )


if __name__ == "__main__":
    main()