from __future__ import annotations

import base64
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
import threading

import cv2
import numpy as np

from hazard_logic import HazardAnalyzer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(BASE_DIR / "yolov8n.pt")
DEFAULT_POTHOLE_MODEL_PATH = str(BASE_DIR / "runs/detect/runs/detect/pothole_detector/weights/best.pt")

LEFT_BOUNDARY = 1 / 3
RIGHT_BOUNDARY = 2 / 3

HAZARD_PRIORITY = {
    "hazard": 3,
    "caution": 2,
    "ignore": 1,
}

OPEN_VOCAB_PROMPTS = ["cone", "barrier", "trash bag", "branch"]

_detector_lock = threading.Lock()
_object_detector_instance = None
_open_vocab_detector_instance = None
_pothole_detector_instance = None
_hazard_analyzer_instance = None

_object_detector_attempted = False
_open_vocab_attempted = False
_pothole_attempted = False

_detector_source = "ObjectDetector + OpenVocabDetector + PotholeDetector"


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
    proximity: str
    motion: str
    severity: str
    is_hazard: bool
    track_id: int | None


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
    analyzed_detections = _analyze_hazards(raw_detections, width, height)
    detections = _normalize_detections(analyzed_detections, width, height)
    prioritized = _prioritize_detections(detections)

    top_detections = prioritized[:3]
    summary_text = _build_summary(top_detections)
    primary_direction = top_detections[0].direction if top_detections else None

    return LiveDetectionResult(
        summary_text=summary_text,
        detections=[asdict(detection) for detection in top_detections],
        primary_direction=primary_direction,
        source=_detector_source,
        has_hazard=any(detection.is_hazard for detection in top_detections),
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
    detections: list[dict] = []

    object_detector = _get_or_load_object_detector()
    if object_detector is not None:
        try:
            object_detections, _ = object_detector.detect(frame)
            detections.extend(object_detections)
        except Exception:
            logger.exception("ObjectDetector failed")

    open_vocab_detector = _get_or_load_open_vocab_detector()
    if open_vocab_detector is not None:
        try:
            open_vocab_detections, _ = open_vocab_detector.detect(frame, OPEN_VOCAB_PROMPTS)
            detections.extend(open_vocab_detections)
        except Exception:
            logger.exception("OpenVocabDetector failed")

    pothole_detector = _get_or_load_pothole_detector()
    if pothole_detector is not None:
        try:
            pothole_result = pothole_detector.detect(frame)
            pothole_detections = _parse_pothole_result(pothole_result)
            detections.extend(pothole_detections)
        except Exception:
            logger.exception("PotholeDetector failed")

    return detections


def _get_or_load_object_detector():
    global _object_detector_attempted, _object_detector_instance

    if _object_detector_attempted:
        return _object_detector_instance

    with _detector_lock:
        if _object_detector_attempted:
            return _object_detector_instance

        _object_detector_attempted = True

        try:
            from detectors.object_detector import ObjectDetector

            _object_detector_instance = ObjectDetector(
                model_path=DEFAULT_MODEL_PATH,
                conf_threshold=0.35,
            )
            logger.info("Loaded ObjectDetector")
        except Exception:
            logger.exception("Failed to load ObjectDetector")
            _object_detector_instance = None

    return _object_detector_instance


def _get_or_load_open_vocab_detector():
    global _open_vocab_attempted, _open_vocab_detector_instance

    if _open_vocab_attempted:
        return _open_vocab_detector_instance

    with _detector_lock:
        if _open_vocab_attempted:
            return _open_vocab_detector_instance

        _open_vocab_attempted = True

        try:
            from detectors.open_vocab import OpenVocabDetector

            _open_vocab_detector_instance = OpenVocabDetector(conf_threshold=0.2)
            logger.info("Loaded OpenVocabDetector")
        except Exception:
            logger.exception("Failed to load OpenVocabDetector")
            _open_vocab_detector_instance = None

    return _open_vocab_detector_instance


def _get_or_load_pothole_detector():
    global _pothole_attempted, _pothole_detector_instance

    if _pothole_attempted:
        return _pothole_detector_instance

    with _detector_lock:
        if _pothole_attempted:
            return _pothole_detector_instance

        _pothole_attempted = True

        try:
            from detectors.ground_detector import PotholeDetector

            _pothole_detector_instance = PotholeDetector(
                model_path=DEFAULT_POTHOLE_MODEL_PATH,
                conf=0.3,
            )
            logger.info("Loaded PotholeDetector")
        except Exception:
            logger.exception("Failed to load PotholeDetector")
            _pothole_detector_instance = None

    return _pothole_detector_instance


def _get_hazard_analyzer() -> HazardAnalyzer:
    global _hazard_analyzer_instance

    with _detector_lock:
        if _hazard_analyzer_instance is None:
            _hazard_analyzer_instance = HazardAnalyzer()
        return _hazard_analyzer_instance


def _parse_pothole_result(result) -> list[dict]:
    detections: list[dict] = []

    if result is None or result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes

    for i in range(len(boxes)):
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].tolist()

        detections.append({
            "label": "pothole",
            "confidence": conf,
            "box": [int(v) for v in xyxy],
            "class_id": 0,
        })

    return detections


def _analyze_hazards(raw_detections: list[dict], width: int, height: int) -> list[dict]:
    if not raw_detections:
        return []

    cleaned = []

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

        cleaned.append({
            "label": str(detection.get("label", "object")).replace("_", " "),
            "confidence": float(detection.get("confidence", 0.0)),
            "box": [x1, y1, x2, y2],
        })

    analyzer = _get_hazard_analyzer()
    return analyzer.analyze(cleaned, width, height)


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

        normalized.append(
            LiveDetection(
                label=str(detection.get("label", "object")).replace("_", " "),
                confidence=float(detection.get("confidence", 0.0)),
                direction=str(detection.get("position", _direction_from_ratio(((x1 + x2) / 2) / width))),
                box=DetectionBox(
                    x1=x1 / width,
                    y1=y1 / height,
                    x2=x2 / width,
                    y2=y2 / height,
                ),
                area_ratio=float(
                    detection.get(
                        "relative_area",
                        ((x2 - x1) * (y2 - y1)) / max(1, width * height),
                    )
                ),
                proximity=str(detection.get("proximity", "far")),
                motion=str(detection.get("motion", "unknown")),
                severity=str(detection.get("severity", "ignore")),
                is_hazard=bool(detection.get("is_hazard", False)),
                track_id=detection.get("track_id"),
            )
        )

    return normalized


def _prioritize_detections(detections: list[LiveDetection]) -> list[LiveDetection]:
    if not detections:
        return []

    filtered = [d for d in detections if d.confidence >= 0.2 and d.severity in {"hazard", "caution"}]
    pool = filtered if filtered else [d for d in detections if d.confidence >= 0.2]
    if not pool:
        pool = detections[:]

    return sorted(
        pool,
        key=lambda detection: (
            HAZARD_PRIORITY.get(detection.severity, 0),
            detection.area_ratio,
            detection.confidence,
        ),
        reverse=True,
    )


def _direction_from_ratio(center_ratio: float) -> str:
    if center_ratio < LEFT_BOUNDARY:
        return "left"
    if center_ratio > RIGHT_BOUNDARY:
        return "right"
    return "center"


def _build_summary(detections: list[LiveDetection]) -> str:
    if not detections or not any(d.is_hazard for d in detections):
        return "No immediate hazards detected."

    phrases = [_detection_phrase(detection) for detection in detections[:2]]
    if len(phrases) == 1:
        return phrases[0]

    return "Multiple hazards detected. " + " ".join(phrases)


def _detection_phrase(detection: LiveDetection) -> str:
    direction_phrases = {
        "left": "on the left",
        "center": "ahead",
        "right": "on the right",
    }
    severity_phrases = {
        "hazard": "Hazard",
        "caution": "Caution",
        "ignore": "Object",
    }

    direction = direction_phrases.get(detection.direction, detection.direction)
    label = detection.label.replace("-", " ")
    prefix = severity_phrases.get(detection.severity, "Object")

    if detection.motion == "approaching":
        return f"{prefix}: {label.title()} {direction}, approaching."
    if detection.motion == "moving_away":
        return f"{prefix}: {label.title()} {direction}, moving away."

    return f"{prefix}: {label.title()} {direction}."
