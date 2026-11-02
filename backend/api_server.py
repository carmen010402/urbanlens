"""
api_server.py

A small, separate web server that exposes your segment_manager.py data as
JSON, so your teammate's dashboard (index.html) can fetch it -- alongside
his existing camera feed, not replacing it.

WHY A SEPARATE SERVER:
Your teammate's backend already runs on port 8000, serving /video_feed and
/api/results (the live camera + YOLO detections). This runs on a DIFFERENT
port (8001) so both can run at the same time with zero conflict. His
frontend just needs a few more fetch() calls added, pointing at this
server instead of his, for the segment/mission data specifically.

ENDPOINTS (matching dashboard_layout_spec.md exactly):
  GET /api/coverage         -> list of {segment_id, status}
  GET /api/findings         -> list of finding rows (the Findings Table)
  GET /api/segment/<id>     -> full evidence for one segment (Current Segment
                                panel + the Evidence popup, same data)
  GET /api/mission_report   -> the full Mission Report (Generate Report button)

HOW TO RUN (separate terminal, alongside your other ROS2/Gazebo terminals):
  cd ~/urban-ground-truth-rover/backend
  python3 api_server.py

Then test it works by opening in a browser, or:
  curl http://localhost:8001/api/coverage
"""

from flask import Flask, jsonify
from flask_cors import CORS

from segment_manager import (
    get_coverage,
    get_findings_table,
    get_segment_evidence,
    generate_mission_report,
)

app = Flask(__name__)
CORS(app)  # allows index.html (opened as a local file or from port 8000) to fetch this


@app.route("/api/coverage")
def api_coverage():
    return jsonify(get_coverage())


@app.route("/api/findings")
def api_findings():
    return jsonify(get_findings_table())


@app.route("/api/segment/<segment_id>")
def api_segment(segment_id):
    evidence = get_segment_evidence(segment_id)
    if evidence is None:
        return jsonify({"error": f"No findings yet for segment {segment_id}"}), 404
    return jsonify(evidence)


@app.route("/api/mission_report")
def api_mission_report():
    return jsonify(generate_mission_report())


if __name__ == "__main__":
    print("Serving segment/mission data on http://localhost:8001")
    print("Endpoints: /api/coverage  /api/findings  /api/segment/<id>  /api/mission_report")
    app.run(host="0.0.0.0", port=8001, debug=False)
