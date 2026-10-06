The website is available at: https://mohabs3-directional-hazard-detection.hf.space

# Directional Hazard Detection System

An accessibility-focused computer-vision web app that helps visually impaired users detect hazards in real time. The user opens the site on a phone, grants camera access, and the app speaks directional alerts — *"Car ahead"*, *"Person on the left"* — as objects appear in the camera feed.

The app is a Flask backend, a mobile-first vanilla-JS frontend, and three Ultralytics detectors: pretrained **YOLOv8** for everyday objects, **YOLO-World** for open-vocabulary obstacles (cones, barriers, trash bags, branches), and a custom-trained **pothole** model. It is packaged as an installable Progressive Web App (PWA) so it can be added to a phone's home screen and
launched like a native app.

## How it works

Every ~1 second while the camera is live:

1. **Browser** — captures a frame from the video element into a hidden canvas and encodes it as a JPEG data URL.
2. **Backend** — `POST /api/live-detect` decodes the frame and runs all three
   detectors (`ObjectDetector`, `OpenVocabDetector`, `PotholeDetector`).
3. **Hazard logic** — `HazardAnalyzer` buckets each detection into
   *left / center / right* thirds, estimates proximity from box area, tracks
   objects between frames to tell whether they are approaching, and assigns a
   severity (`hazard`, `caution`, `ignore`). Detections are prioritized by
   severity, then box area, then confidence.
4. **Response** — JSON payload with `summary_text`, `primary_direction`,
   `has_hazard`, and the top detections.
5. **Frontend** — draws colored bounding boxes over the live video and, when
   `has_hazard` is true, speaks the summary via the browser's
   `SpeechSynthesis` API. When the path is clear the app stays silent.

## Tech stack

- **Backend:** Python 3, Flask, OpenCV, Ultralytics (YOLOv8, YOLO-World, custom pothole model)
- **Frontend:** HTML, Tailwind (via CDN), vanilla JS, Canvas 2D overlay
- **Audio:** browser-native `SpeechSynthesis` (no cloud TTS)
- **PWA:** `manifest.webmanifest`, service worker with offline app-shell cache,
  maskable icons, iOS meta tags

## Project structure

```
Directional-Hazard-Detection-System/
├── app.py                       # Flask entrypoint + routes
├── live_detection.py            # Frame decode, runs detectors, builds summary
├── hazard_logic.py              # Direction, proximity, tracking, severity
├── detectors/
│   ├── object_detector.py       # YOLOv8 wrapper (COCO classes)
│   ├── open_vocab.py            # YOLO-World wrapper (text-prompted obstacles)
│   └── ground_detector.py       # Custom pothole model wrapper
├── templates/
│   ├── base.html                # PWA meta tags + service worker registration
│   └── index.html               # Camera stage + control dock
├── static/
│   ├── app.js                   # Camera capture, detection loop, TTS
│   ├── styles.css
│   ├── manifest.webmanifest
│   ├── service-worker.js        # App-shell cache (API is never cached)
│   └── icons/                   # PWA icons (192, 512, maskable, apple-touch)
├── train_pothole.py             # Trains the pothole model
├── data/pothole_yolo/           # Pothole dataset in YOLO format
├── runs/detect/pothole_detector/weights/best.pt   # Trained pothole weights
├── yolov8n.pt                   # Pretrained YOLOv8 weights
├── yolov8s-world.pt             # YOLO-World weights (auto-downloaded if missing)
└── requirements.txt
```

## Retraining the pothole model

```bash
python train_pothole.py
```

Results are written to `runs/detect/pothole_detector/`. If that folder already
exists, Ultralytics writes to `pothole_detector2/` instead, so copy the new
`weights/best.pt` over the old one to use it in the app.

## Endpoints

| Method | Path                      | Purpose                                         |
|--------|---------------------------|-------------------------------------------------|
| GET    | `/`                       | PWA shell (camera UI)                           |
| POST   | `/api/live-detect`        | Body: `{image: <jpeg data URL>}` → JSON result  |
| GET    | `/health`                 | Liveness probe (`{"status": "ok"}`)             |
| GET    | `/manifest.webmanifest`   | PWA manifest (also served from `/static/`)      |
| GET    | `/service-worker.js`      | Service worker (scoped to `/`)                  |

## Deployment

The app is deployed as a **Docker Space on Hugging Face Spaces** (CPU Basic tier, free, 16 GB RAM). Everything the Space needs lives in this repo. The link is https://mohabs3-directional-hazard-detection.hf.space. 
