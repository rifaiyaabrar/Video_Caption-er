"""
Video Captioner demo
--------------------
Upload a video in the browser -> it plays locally on the left,
while Gemini generates timed captions that stream in on the right
and stay in sync with playback.

Run:
    pip install -r requirements.txt
    export GEMINI_API_KEY="your-key"          # Windows (PowerShell): $env:GEMINI_API_KEY="your-key"
    python app.py
    open http://127.0.0.1:5000
"""

import os
import tempfile
import time

from flask import Flask, Response, jsonify, render_template, request
from google import genai

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
# Tried in order if the main model is overloaded (503) or rate-limited (429)
FALLBACK_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_FALLBACKS", "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite"
    ).split(",")
    if m.strip()
]
RETRIES_PER_MODEL = 2

# Models offered in the dropdown (strict mode). Edit freely, or set
# GEMINI_MODELS="model-a,model-b" in the environment.
AVAILABLE_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_MODELS",
        "gemini-4-argon,gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite",
    ).split(",")
    if m.strip()
]
for _m in [MODEL, *FALLBACK_MODELS]:
    if _m not in AVAILABLE_MODELS:
        AVAILABLE_MODELS.append(_m)

client = genai.Client()  # reads GEMINI_API_KEY from the environment
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024**3  # 2 GB upload limit

PROMPT = """You are a video captioner. Watch the entire video (visuals and audio)
and write timed captions describing what happens.

Rules:
- Output ONLY caption lines, one per line, in exactly this format:
  MM:SS-MM:SS | caption text
  (use H:MM:SS if the video is longer than one hour)
- Lines must be in chronological order, cover the whole video from 00:00
  to the end with no gaps, and must not overlap.
- Each segment should be 2-8 seconds long; start a new segment whenever
  the action or scene changes.
- Each caption is one short sentence (max ~20 words) describing what is
  visible, plus any important speech or sound.
- No headings, numbering, markdown, or extra commentary."""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/models")
def models():
    """What the dropdown shows: the auto chain plus every model for strict mode."""
    return jsonify(
        auto_chain=[MODEL] + [m for m in FALLBACK_MODELS if m != MODEL],
        models=AVAILABLE_MODELS,
    )


@app.route("/upload", methods=["POST"])
def upload():
    """Receive the video, push it to the Gemini Files API, wait until it's ready."""
    video = request.files.get("video")
    if video is None:
        return jsonify(error="No video received"), 400

    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        video.save(path)
        uploaded = client.files.upload(
            file=path, config={"mime_type": video.mimetype or "video/mp4"}
        )
    except Exception as e:
        return jsonify(error=f"Upload to Gemini failed: {e}"), 500
    finally:
        os.remove(path)

    # Gemini needs a moment to process video before it can be used
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)

    if uploaded.state.name != "ACTIVE":
        return jsonify(error=f"Gemini could not process the video ({uploaded.state.name})"), 500

    return jsonify(file_name=uploaded.name)


@app.route("/captions")
def captions():
    """Stream caption lines back to the browser as Gemini generates them."""
    file_name = request.args.get("file")
    if not file_name:
        return jsonify(error="Missing file"), 400

    # mode=auto -> main model + fallbacks; mode=strict&model=X -> only X
    mode = request.args.get("mode", "auto")
    if mode == "strict":
        chosen = request.args.get("model", "")
        if chosen not in AVAILABLE_MODELS:
            return jsonify(error=f"Unknown model: {chosen}"), 400
        models = [chosen]
    else:
        models = [MODEL] + [m for m in FALLBACK_MODELS if m != MODEL]

    def is_busy(err):
        msg = str(err)
        return any(s in msg for s in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded"))

    def generate():
        try:
            video_file = client.files.get(name=file_name)
        except Exception as e:
            yield f"\nERROR: {e}\n"
            return

        last_err = None
        for model in models:
            for attempt in range(RETRIES_PER_MODEL):
                started = False
                try:
                    stream = client.models.generate_content_stream(
                        model=model, contents=[video_file, PROMPT]
                    )
                    for chunk in stream:
                        if chunk.text:
                            if not started:
                                yield f"MODEL: {model}\n"   # tells the page who is writing
                            started = True
                            yield chunk.text
                    app.logger.info("Captions generated with %s", model)
                    return
                except Exception as e:
                    last_err = e
                    # Once captions have started arriving we can't restart cleanly
                    if started or not is_busy(e):
                        yield f"\nERROR ({model}): {e}\n"
                        return
                    app.logger.warning("%s busy (attempt %d): %s", model, attempt + 1, e)
                    yield f"BUSY: {model}\n"
                    time.sleep(3 * (attempt + 1))
        if len(models) == 1:
            yield (f"\nERROR: {models[0]} is busy right now (strict mode, no fallback). "
                   f"Try again in a minute or pick Auto. Last error: {last_err}\n")
        else:
            yield f"\nERROR: all models are busy right now, try again in a minute. Last error: {last_err}\n"

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)