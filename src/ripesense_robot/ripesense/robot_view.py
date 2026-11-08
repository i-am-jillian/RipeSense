"""Live view of the Hiwonder AiNex fruit-picking robot inside the RipeSense app.

Everything is proxied through this Flask app, so the browser only talks to
localhost. The robot side is robot/ainex_fruit_pick (fruit_detector_node.py):
  http://<robot>:8080/stream?topic=/fruit_detection/image_result   annotated MJPEG
  http://<robot>:8090/status                                        detections + pick phase
  http://<robot>:8090/target.jpg                                    crop of the target fruit
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from flask import Response, jsonify, render_template, stream_with_context

DEFAULT_ROBOT = os.environ.get("RIPESENSE_ROBOT", "10.42.0.151")
TOPICS = {"annotated": "/fruit_detection/image_result", "raw": "/camera/image_raw"}


def register(app, robot_host: str = DEFAULT_ROBOT):
    base_video = f"http://{robot_host}:8080"
    base_status = f"http://{robot_host}:8090"

    @app.get("/robot")
    def robot_page():
        return render_template("robot.html", robot_host=robot_host)

    @app.get("/robot/stream")
    @app.get("/robot/stream/<kind>")
    def robot_stream(kind: str = "annotated"):
        topic = TOPICS.get(kind, TOPICS["annotated"])
        url = f"{base_video}/stream?topic={topic}&quality=70"
        try:
            upstream = urllib.request.urlopen(url, timeout=5)
        except (urllib.error.URLError, OSError) as exc:
            return jsonify(error=f"Robot video unavailable at {robot_host}: {exc}"), 502

        def gen():
            try:
                while True:
                    chunk = upstream.read(16384)
                    if not chunk:
                        break
                    yield chunk
            finally:
                upstream.close()

        return Response(stream_with_context(gen()), content_type=upstream.headers.get("Content-Type"),
                        headers={"Cache-Control": "no-store"})

    @app.get("/robot/status")
    def robot_status():
        try:
            with urllib.request.urlopen(f"{base_status}/status", timeout=2) as r:
                data = json.loads(r.read())
            data["online"] = True
        except (urllib.error.URLError, OSError, ValueError) as exc:
            data = {"online": False, "error": str(exc)}
        data["robot"] = robot_host
        return jsonify(data)

    @app.get("/robot/target.jpg")
    def robot_target():
        try:
            with urllib.request.urlopen(f"{base_status}/target.jpg", timeout=2) as r:
                return Response(r.read(), content_type="image/jpeg", headers={"Cache-Control": "no-store"})
        except (urllib.error.URLError, OSError) as exc:
            return jsonify(error=f"No target crop: {exc}"), 404

    return app
