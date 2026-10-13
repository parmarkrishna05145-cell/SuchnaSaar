"""SuchnaSaar web app.

Two routes do the work:

    GET  /            the single-page interface
    POST /analyze     takes {"text": "...", "lang": "en"|"hi"} and returns the
                      full analysis produced by agent.analyse
    GET  /health      reports whether the model is loaded and what it is

The app holds one fitted model in memory, loaded once at import. It does not
write the submitted text to disk, to a database, or to a log. That is stated
in the interface as well as here, because a citizen pasting a notice that
carries their name and address is entitled to know.

There is no API key, no paid service and no outbound network call in this
file. The app runs completely offline once the model is trained, which is
what makes it deployable on a free tier.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import agent
import data
import features

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model.joblib"
METRICS_PATH = BASE / "metrics.json"

app = Flask(__name__)

# --------------------------------------------------------------------------
# Model loading. The app must still start if the artefact is missing, so a
# missing model is reported through /health rather than raised at import.
# --------------------------------------------------------------------------
MODEL = None
MODEL_ERROR = None
try:
    import joblib

    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
    else:
        MODEL_ERROR = f"model artefact not found at {MODEL_PATH.name}; run train.py first"
except Exception as exc:  # pragma: no cover
    MODEL_ERROR = f"could not load the model: {exc}"


def _load_metrics() -> dict:
    try:
        with open(METRICS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


@app.route("/")
def index():
    metrics = _load_metrics()
    return render_template(
        "index.html",
        samples=[{"id": n["id"], "title": n["title"], "issuer": n["issuer"]} for n in data.NOTICES],
        metrics=metrics,
        portals=data.PORTALS,
        helplines=data.HELPLINES,
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    if MODEL is None:
        return jsonify({"error": MODEL_ERROR or "the model is not loaded"}), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with a 'text' field."}), 400

    raw = payload.get("text")
    text = raw.strip() if isinstance(raw, str) else ""
    if len(text) < 20:
        return jsonify({"error": "Paste the text of a notice -- at least a couple of sentences."}), 400

    result = agent.analyse(text, MODEL)
    result["char_count"] = len(text)
    return jsonify(result)


@app.route("/sample/<notice_id>")
def sample(notice_id: str):
    for n in data.NOTICES:
        if n["id"].lower() == notice_id.lower():
            return jsonify({"id": n["id"], "title": n["title"], "issuer": n["issuer"], "text": n["body"]})
    return jsonify({"error": "no such sample notice"}), 404


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok" if MODEL is not None else "degraded",
            "project": "SuchnaSaar",
            "model": "sentence-level notice triage (reference / action / deadline)",
            "model_error": MODEL_ERROR,
            "engineered_signals": features.N_ENGINEERED,
            "samples": len(data.NOTICES),
            "stores_submitted_text": False,
            "outbound_network_calls": False,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
