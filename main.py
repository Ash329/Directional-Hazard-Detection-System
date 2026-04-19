import os
import cv2
import random

from detectors.object_detector import ObjectDetector
from detectors.open_vocab import OpenVocabDetector
from hazard_logic import HazardAnalyzer


INPUT_VIDEO = "data/test.mp4"
OUTPUT_VIDEO = "data/output_hazard_10s.mp4"

USE_OBJECT_DETECTOR = True
USE_OPEN_VOCAB = True

OBJECT_CLASSES = ["person", "car", "bicycle"]
OPEN_VOCAB_PROMPTS = ["cone", "barrier", "trash bag", "branch"]


def get_random_clip(video_path, seconds=10):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        raise RuntimeError("Invalid FPS from video")

    clip_frames = int(fps * seconds)
    max_start = max(0, total_frames - clip_frames)

    start = random.randint(0, max_start) if max_start > 0 else 0
    end = min(start + clip_frames, total_frames)

    return fps, start, end


def draw_hazards(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]

        label = det.get("label", "unknown")
        severity = det.get("severity", "ignore")
        motion = det.get("motion", "unknown")
        position = det.get("position", "unknown")
        proximity = det.get("proximity", "unknown")

        if severity == "hazard":
            color = (0, 0, 255)
        elif severity == "caution":
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        text = f"{label} | {severity} | {position} | {proximity} | {motion}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return frame


def main():
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"Video not found: {INPUT_VIDEO}")

    fps, start_frame, end_frame = get_random_clip(INPUT_VIDEO, seconds=10)

    print(f"Start frame: {start_frame}")
    print(f"End frame: {end_frame}")
    print(f"FPS: {fps}")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    width = 640
    height = 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps / 2, (width, height))

    object_detector = ObjectDetector("yolov8n.pt", 0.25) if USE_OBJECT_DETECTOR else None
    open_vocab_detector = OpenVocabDetector("yolov8s-world.pt", 0.2) if USE_OPEN_VOCAB else None
    hazard_analyzer = HazardAnalyzer()

    if object_detector:
        print(f"Object detector device: {object_detector.device}")
    if open_vocab_detector:
        print("Open vocab device: cpu")

    frame_idx = start_frame
    processed = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        if frame_idx % 2 == 0:
            all_detections = []

            if object_detector:
                detections, _ = object_detector.detect(frame, classes=OBJECT_CLASSES)
                all_detections.extend(detections)

            if open_vocab_detector:
                detections, _ = open_vocab_detector.detect(frame, prompts=OPEN_VOCAB_PROMPTS)
                all_detections.extend(detections)

            analyzed = hazard_analyzer.analyze(all_detections, width, height)

            # only keep caution or hazard
            analyzed = [det for det in analyzed if det.get("is_hazard", False)]

            annotated = draw_hazards(frame.copy(), analyzed)
            writer.write(annotated)

            processed += 1

            if processed % 30 == 0:
                print(f"Processed {processed} frames...")

        frame_idx += 1

    cap.release()
    writer.release()

    print("Done.")
    print(f"Saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()