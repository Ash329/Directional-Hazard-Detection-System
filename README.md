---
title: Directional Hazard Detection
emoji: 🦺
colorFrom: teal
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Real-time directional hazard detection with spoken alerts.
---

# Directional Hazard Detection System

An accessibility-focused computer-vision web app that helps visually impaired users
detect hazards in real time. The user opens the site on a phone, grants camera
access, and the app speaks directional alerts — *"Car ahead"*, *"Person on the
left"* — as objects appear in the camera feed.

The app is a Flask backend, a mobile-first vanilla-JS frontend, and a
pretrained Ultralytics **YOLOv8** detector. It is packaged as an installable
Progressive Web App (PWA) so it can be added to a phone's home screen and
launched like a native app.

## How it works

Every ~1 second while the camera is live:

1. **Browser** — captures a frame from the video element into a hidden canvas
   and encodes it as a JPEG data URL.
2. **Backend** — `POST /api/live-detect` decodes the frame and runs YOLOv8
   (`yolov8n.pt`) via the `ObjectDetector` class.
3. **Direction logic** — each detection's bounding-box center is bucketed into
   *left / center / right* thirds of the frame. Detections are prioritized by
   hazard class (person, car, bicycle, …) then by bounding-box area.
4. **Response** — JSON payload with `summary_text`, `primary_direction`,
   `has_hazard`, and the top detections.
5. **Frontend** — draws colored bounding boxes over the live video and, when
   `has_hazard` is true, speaks the summary via the browser's
   `SpeechSynthesis` API. When the path is clear the app stays silent.

## Tech stack

- **Backend:** Python 3, Flask, OpenCV, Ultralytics YOLOv8 (`yolov8n.pt`)
- **Frontend:** HTML, Tailwind (via CDN), vanilla JS, Canvas 2D overlay
- **Audio:** browser-native `SpeechSynthesis` (no cloud TTS)
- **PWA:** `manifest.webmanifest`, service worker with offline app-shell cache,
  maskable icons, iOS meta tags

## Project structure

```
Directional-Hazard-Detection-System/
├── app.py                       # Flask entrypoint + routes
├── live_detection.py            # Frame decode, YOLO call, direction logic
├── detectors/
│   └── object_detector.py       # Ultralytics YOLOv8 wrapper
├── templates/
│   ├── base.html                # PWA meta tags + service worker registration
│   └── index.html               # Camera stage + control dock
├── static/
│   ├── app.js                   # Camera capture, detection loop, TTS
│   ├── styles.css
│   ├── manifest.webmanifest
│   ├── service-worker.js        # App-shell cache (API is never cached)
│   └── icons/                   # PWA icons (192, 512, maskable, apple-touch)
├── yolov8n.pt                   # Pretrained YOLOv8 weights
└── requirements.txt
```

## Quick start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the Flask app
python app.py
```

Open `http://localhost:5000/`, click the camera icon, and point the camera
at the surroundings. Watch the terminal for `Loaded detector: Pretrained
YOLOv8 …` on the first request — that means the model warmed up successfully.

## PWA install

With the site open in Chrome / Edge / Safari:

- **Desktop:** use the browser's *Install app* icon in the address bar.
- **Android:** tap the *Install* / *Add to Home Screen* prompt.
- **iOS Safari:** Share → *Add to Home Screen*.

Once installed, the shell (HTML, CSS, JS, icons) loads offline. Detection
itself still requires the backend — frames are never cached.

## HTTPS requirement

`getUserMedia` (camera) and service workers only work on **HTTPS** or
**localhost**. Plain `http://` over LAN will not get camera access.

For local testing, `http://localhost:5000` works. For phone access on the
same network, use a tunnel (e.g. `cloudflared`, `ngrok`) or deploy behind a
TLS-terminating reverse proxy such as Caddy.

## Deployment

The app is a standard Flask WSGI application. For production:

- **WSGI server:** Gunicorn with **one worker, multiple threads** — each
  worker loads its own YOLO model into RAM, so running multiple workers
  multiplies memory use:
  ```bash
  gunicorn -w 1 --threads 4 -b 127.0.0.1:8000 --timeout 60 app:app
  ```
- **TLS:** [Caddy](https://caddyserver.com/) in front of Gunicorn handles
  Let's Encrypt automatically. Minimal `Caddyfile`:
  ```
  yourname.duckdns.org {
      reverse_proxy 127.0.0.1:8000
  }
  ```
- **Dynamic DNS:** [DuckDNS](https://www.duckdns.org/) gives a free
  `*.duckdns.org` hostname; point it at the server's public IP and refresh
  every few minutes via cron.
- **Process supervision:** systemd (Linux) or launchd (macOS) to auto-restart
  on crashes and reboots.

## Endpoints

| Method | Path                      | Purpose                                         |
|--------|---------------------------|-------------------------------------------------|
| GET    | `/`                       | PWA shell (camera UI)                           |
| POST   | `/api/live-detect`        | Body: `{image: <jpeg data URL>}` → JSON result  |
| GET    | `/health`                 | Liveness probe (`{"status": "ok"}`)             |
| GET    | `/manifest.webmanifest`   | PWA manifest (also served from `/static/`)      |
| GET    | `/service-worker.js`      | Service worker (scoped to `/`)                  |

## Configuration

Defaults live at the top of `live_detection.py`:

| Constant              | Meaning                                                    |
|-----------------------|------------------------------------------------------------|
| `DEFAULT_MODEL_PATH`  | Path to the YOLO weights file (`yolov8n.pt`)               |
| `LEFT_BOUNDARY`       | Frame-width fraction below which a box counts as *left*    |
| `RIGHT_BOUNDARY`      | Frame-width fraction above which a box counts as *right*   |
| `HAZARD_PRIORITY`     | Per-class priority weights used for sorting detections     |

The detection interval (1000 ms) and frame capture width (640 px) are set
in `static/app.js` inside `startDetectionLoop` and `analyzeFrame`.

## Troubleshooting

- **"No module named 'ultralytics'"** — reinstall with
  `pip install -r requirements.txt`. YOLOv8 requires the `ultralytics` package.
- **Always silent / no boxes drawn** — check the server terminal. If you see
  `Failed to load ObjectDetector`, the full traceback there will point at the
  real cause (missing weights, torch/CUDA mismatch, etc.). Prior to this fix
  the app failed silently; now it logs the load result explicitly.
- **First request is slow** — expected. YOLOv8 loads on the first frame
  (several seconds on CPU). Subsequent frames are fast.
- **Camera does not start on phone** — confirm the URL is `https://` or
  `localhost`. Camera access is blocked on plain HTTP.
- **TTS never speaks** — check that the speaker button in the control dock is
  active (not muted), and that at least one hazard has been detected (the app
  is silent when the path is clear).

## Notes and limitations

- **CPU inference only by default.** YOLOv8n on a modern x86 CPU gives
  ~5–10 FPS, which is plenty for a 1 Hz detection loop. A CUDA GPU is used
  automatically if available.
- **2D direction only.** Direction is derived from horizontal bounding-box
  position. There is no depth estimation — a close car and a distant car
  both say "ahead".
- **Browser TTS voice varies by device.** The spoken voice depends on the
  OS and browser. Android/Chrome and iOS/Safari both have reliable built-in
  voices.
- **HAZARD_PRIORITY covers common pedestrian-relevant COCO classes.** Add or
  adjust entries in `live_detection.py` to tune what gets announced first.
