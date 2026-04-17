from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from live_detection import analyze_live_frame_data_url


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

    @app.get("/")
    def index():
        return render_template("index.html")

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

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
