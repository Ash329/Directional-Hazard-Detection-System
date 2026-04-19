from ultralytics import YOLOWorld
import torch


class OpenVocabDetector:
    def __init__(self, model_path="yolov8s-world.pt", conf_threshold=0.2):
        # Force CPU for YOLO-World because set_classes() can crash with CPU/CUDA mismatch
        self.device = "cpu"
        self.model = YOLOWorld(model_path)
        self.model.to("cpu")

        self.conf_threshold = conf_threshold
        self.current_prompts = None

    def detect(self, frame, prompts):
        prompts = list(prompts)

        # Only reset classes if prompts changed
        if self.current_prompts != prompts:
            self.model.set_classes(prompts)
            self.current_prompts = prompts.copy()

        with torch.inference_mode():
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False
            )

        detections = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            names = result.names

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
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