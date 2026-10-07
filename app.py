from __future__ import annotations

import os

from flask import Flask, jsonify, make_response, render_template, request, send_from_directory, url_for

from live_detection import analyze_live_frame_data_url, warm_up_detectors


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

    @app.context_processor
    def asset_helpers():
        # Append the file's mtime so the service worker's cache-first lookup misses
        # whenever a CSS/JS file changes, instead of serving a stale copy.
        def asset_url(filename: str) -> str:
            path = os.path.join(app.static_folder, filename)
            version = int(os.path.getmtime(path)) if os.path.exists(path) else 0
            return url_for("static", filename=filename, v=version)

        return {"asset_url": asset_url}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/service-worker.js")
    def service_worker():
        response = make_response(
            send_from_directory(app.static_folder, "service-worker.js", mimetype="application/javascript")
        )
        response.headers["Service-Worker-Allowed"] = "/"
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/manifest.webmanifest")
    def manifest():
        return send_from_directory(
            app.static_folder, "manifest.webmanifest", mimetype="application/manifest+json"
        )

    @app.post("/api/live-detect")
    def live_detect():
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")

        if not isinstance(image_data, str) or not image_data:
            return jsonify({"error": "A base64-encoded frame is required."}), 400

        try:
            result = analyze_live_frame_data_url(image_data)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            return jsonify({"error": "Frame analysis failed."}), 500

        return jsonify(result)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    warm_up_detectors()

    return app


app = create_app()


if __name__ == "__main__":
    # The reloader would import this module in a second process and load the models twice.
    app.run(debug=True, use_reloader=False)
