# Directional Hazard Detection System

Directional Hazard Detection System is an accessibility-focused computer vision project built to help visually impaired individuals detect and respond to hazards in real time. Using live camera input, the system identifies nearby objects such as vehicles, bicycles, and obstacles, determines whether they are on the left, center, or right side of the user’s path, and delivers directional audio alerts that support safer navigation.

The project is presented as a mobile-first Flask web application powered by a pretrained YOLOv5 model. A user opens the site on a phone, grants camera access, and immediately receives live hazard analysis through browser-based text-to-speech while the backend processes frames from the camera feed.

## What This Project Showcases

- Real-time hazard awareness for visually impaired users
- A Python Flask backend that receives live camera frames and runs detection
- A mobile-first frontend built with HTML, Jinja, CSS, and Tailwind CSS
- Browser-based text-to-speech for directional audio alerts
- A pretrained YOLOv5 detection workflow integrated into an accessible web interface

## How To Use

1. Create and activate a Python virtual environment.
2. Install the project dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Flask application:

```bash
python app.py
```

4. Open the site in a browser.
5. Allow camera access when prompted.
6. Point the camera toward the surrounding environment.
7. Listen for spoken directional hazard alerts as the backend analyzes the live feed.

## Important Notes

- For live camera access on a phone, the site should be served over HTTPS or opened from `localhost`.
- The backend exposes a live frame analysis endpoint at `/api/live-detect`.
- The frontend is designed for mobile use first, with a full-screen camera view and large icon controls.
- Audio guidance is delivered through the browser’s speech synthesis support, so the final voice depends on the device and browser.
