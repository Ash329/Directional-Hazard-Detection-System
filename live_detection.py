from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from importlib import util
from pathlib import Path
import threading

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DETECTOR_CANDIDATES = ("detectory.py", "detector.py")
LEFT_BOUNDARY = 1 / 3
RIGHT_BOUNDARY = 2 / 3
HAZARD_PRIORITY = {
    "person": 6,
    "bicycle": 6,
    "motorcycle": 6,
    "car": 6,
    "bus": 6,
    "truck": 6,
    "train": 5,
    "dog": 4,
    "cat": 4,
    "chair": 4,
    "bench": 4,
    "stop sign": 5,
    "traffic light": 5,
    "fire hydrant": 4,
    "skateboard": 4,
    "suitcase": 4,
}

_detector_lock = threading.Lock()
_detector_instance = None
_detector_source = "Pretrained YOLOv5"
_detector_error = ""
_detector_attempted = False


@dataclass
class DetectionBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class LiveDetection:
    label: str
    confidence: float
    direction: str
    box: DetectionBox
    area_ratio: float


@dataclass
class LiveDetectionResult:
    summary_text: str
    detections: list[dict]
    primary_direction: str | None
    source: str
    has_hazard: bool

    def to_payload(self) -> dict:
        return asdict(self)


def analyze_live_frame_data_url(data_url: str) -> dict:
    frame = _decode_data_url(data_url)
    return analyze_live_frame(frame).to_payload()


def analyze_live_frame(frame: np.ndarray) -> LiveDetectionResult:
    height, width = frame.shape[:2]
    raw_detections = _run_detector(frame)
    detections = _normalize_detections(raw_detections, width, height)
    prioritized = _prioritize_detections(detections)

    top_detections = prioritized[:3]
    summary_text = _build_summary(top_detections)
    primary_direction = top_detections[0].direction if top_detections else None

    return LiveDetectionResult(
        summary_text=summary_text,
        detections=[asdict(detection) for detection in top_detections],
        primary_direction=primary_direction,
        source=_detector_source,
        has_hazard=bool(top_detections),
    )


def _decode_data_url(data_url: str) -> np.ndarray:
    if "," not in data_url:
        raise ValueError("Invalid frame payload.")

    _, encoded_frame = data_url.split(",", 1)

    try:
        frame_bytes = base64.b64decode(encoded_frame)
    except Exception as exc:
        raise ValueError("Frame could not be decoded.") from exc

    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Frame image is invalid.")

    return frame


def _run_detector(frame: np.ndarray) -> list[dict]:
    detector = _get_or_load_detector()
    if detector is None:
        return []

    try:
        detections, _ = detector.detect(frame)
    except Exception:
        return []

    return detections


def _get_or_load_detector():
    global _detector_attempted, _detector_error, _detector_instance, _detector_source

    if _detector_attempted:
        return _detector_instance

    with _detector_lock:
        if _detector_attempted:
            return _detector_instance

        _detector_attempted = True

        for candidate_name in DETECTOR_CANDIDATES:
            candidate_path = BASE_DIR / candidate_name
            if not candidate_path.exists():
                continue

            try:
                module = _load_module(candidate_path)
            except Exception as exc:
                _detector_error = f"{candidate_name}: {exc.__class__.__name__}"
                continue

            detector_class = getattr(module, "ObjectDetector", None)
            if detector_class is None:
                continue

            try:
                _detector_instance = detector_class(model_name="yolov5s", conf_threshold=0.35)
                _detector_source = f"Pretrained YOLOv5 ({candidate_name})"
                return _detector_instance
            except Exception as exc:
                _detector_error = f"{candidate_name}: {exc.__class__.__name__}"
                _detector_instance = None

    return _detector_instance


def _load_module(module_path: Path):
    spec = util.spec_from_file_location(f"live_detector_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path.name}")

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_detections(raw_detections: list[dict], width: int, height: int) -> list[LiveDetection]:
    normalized: list[LiveDetection] = []

    for detection in raw_detections:
        box = detection.get("box") or []
        if len(box) != 4:
            continue

        x1, y1, x2, y2 = [int(value) for value in box]
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue

        center_ratio = ((x1 + x2) / 2) / width
        direction = _direction_from_ratio(center_ratio)
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, width * height)

        normalized.append(
            LiveDetection(
                label=str(detection.get("label", "object")).replace("_", " "),
                confidence=float(detection.get("confidence", 0.0)),
                direction=direction,
                box=DetectionBox(
                    x1=x1 / width,
                    y1=y1 / height,
                    x2=x2 / width,
                    y2=y2 / height,
                ),
                area_ratio=area_ratio,
            )
        )

    return normalized


def _prioritize_detections(detections: list[LiveDetection]) -> list[LiveDetection]:
    if not detections:
        return []

    prioritized = sorted(
        detections,
        key=lambda detection: (
            HAZARD_PRIORITY.get(detection.label.lower(), 1),
            detection.area_ratio,
            detection.confidence,
        ),
        reverse=True,
    )

    filtered = [detection for detection in prioritized if detection.confidence >= 0.2]
    return filtered or prioritized[:1]


def _direction_from_ratio(center_ratio: float) -> str:
    if center_ratio < LEFT_BOUNDARY:
        return "left"
    if center_ratio > RIGHT_BOUNDARY:
        return "right"
    return "center"


def _build_summary(detections: list[LiveDetection]) -> str:
    if not detections:
        return "No immediate hazards detected."

    phrases = [_detection_phrase(detection) for detection in detections[:2]]
    if len(detections) == 1:
        return phrases[0]

    return "Multiple hazards detected. " + " ".join(phrases)


def _detection_phrase(detection: LiveDetection) -> str:
    direction_phrases = {
        "left": "on the left",
        "center": "ahead",
        "right": "on the right",
    }
    direction = direction_phrases.get(detection.direction, detection.direction)
    label = detection.label.replace("-", " ")
    return f"{label.title()} {direction}."
