"""Visualize cached full-frame detections with a close-up, without gap filling."""
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import yaml

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SID = "session_20260917013532709_2"
SOURCE = ROOT / "dataset" / SID / "DJI_20260917013501_0005_D.MP4"
cfg = yaml.safe_load((OUT / "config.yaml").read_text())
REVIEW = ROOT / "outputs" / cfg["source_review"]
style, mode = cfg["annotation"], cfg["detection_mode"]
VIDEO = OUT / "thumb_marked.mp4"
for name in ["thumb_marked.mp4", "manifest.json", "detections.csv", "tracking_config.yaml"]:
    if (OUT / name).exists():
        raise FileExistsError(OUT / name)
shutil.copy2(REVIEW / "frame_diagnostics.csv", OUT / "detections.csv")
shutil.copy2(REVIEW / "config.yaml", OUT / "tracking_config.yaml")
with (OUT / "detections.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
assert [int(r["frame_index"]) for r in rows] == list(range(len(rows)))
summary = json.loads((REVIEW / "summary.json").read_text())
cap = cv2.VideoCapture(str(SOURCE))
assert cap.isOpened()
cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
fps = cap.get(cv2.CAP_PROP_FPS)
sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = style["output_width_px"]
height = round(sh * width / sw)
assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(rows)
pw, ph = summary["preview_size_px"]
zx1, zy1, zx2, zy2 = style["zoom_source_preview_px"]
dx1, dy1, dx2, dy2 = style["zoom_destination_px"]
command = ["/opt/homebrew/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}",
           "-framerate", str(fps), "-i", "pipe:0", "-i", str(SOURCE),
           "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", style["video_preset"],
           "-crf", str(style["video_crf"]), "-pix_fmt", "yuv420p", "-c:a", "copy",
           "-movflags", "+faststart", str(VIDEO)]
encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
index = marked = 0

def marker(image, point, color):
    radius, thickness = style["marker_radius_px"], style["marker_thickness_px"]
    cv2.circle(image, point, radius + 2, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.circle(image, point, radius, color, thickness, cv2.LINE_AA)
    cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)

try:
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        assert index < len(rows)
        row = rows[index]
        found = row[mode + "_detected"] == "1"
        color = (0, 255, 0) if found else (0, 180, 255)
        frame = cv2.resize(raw, (width, height))
        zoom = raw[round(zy1 * sh / ph):round(zy2 * sh / ph), round(zx1 * sw / pw):round(zx2 * sw / pw)]
        zoom = cv2.resize(zoom, (dx2 - dx1, dy2 - dy1))
        if found:
            x, y = float(row[mode + "_tip_x_native_px"]), float(row[mode + "_tip_y_native_px"])
            assert math.isfinite(x) and math.isfinite(y)
            marker(frame, (round(x * width / sw), round(y * height / sh)), color)
            zoom_point = (round((x * pw / sw - zx1) * (dx2 - dx1) / (zx2 - zx1)),
                          round((y * ph / sh - zy1) * (dy2 - dy1) / (zy2 - zy1)))
            marker(zoom, zoom_point, color)
            marked += 1
        frame[dy1:dy2, dx1:dx2] = zoom
        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (230, 230, 230), 2)
        cv2.putText(frame, "CLOSE-UP (same full-frame detection)", (dx1, dy1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, style["text_scale"], (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (12, 12), (740, 117), (15, 15, 15), -1)
        labels = [("MediaPipe thumb tip (full frame; model 0)", (235, 235, 235)),
                  (f"Frame {index + 1}/{len(rows)} | {float(row['video_time_ms']) / 1000:.2f}s", (235, 235, 235)),
                  ("DETECTED" if found else "NO DETECTION", color)]
        for line, (label, label_color) in enumerate(labels):
            cv2.putText(frame, label, (24, 40 + 32 * line), cv2.FONT_HERSHEY_SIMPLEX,
                        style["text_scale"], label_color, 2, cv2.LINE_AA)
        encoder.stdin.write(frame.tobytes())
        index += 1
        if index % 300 == 0:
            print(f"Rendered {index}/{len(rows)} frames", flush=True)
finally:
    cap.release()
    encoder.stdin.close()
error = encoder.stderr.read().decode()
assert encoder.wait() == 0, error
assert index == len(rows)
assert marked == sum(r[mode + "_detected"] == "1" for r in rows)
probe = json.loads(subprocess.check_output(["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,codec_name,width,height,nb_frames,duration:format=duration", "-of", "json", str(VIDEO)]))
stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
assert int(stream["nb_frames"]) == len(rows)
assert [stream["width"], stream["height"]] == [width, height]
assert abs(float(stream["duration"]) - summary["video"]["duration_s"]) < 1 / fps
assert any(s["codec_type"] == "audio" for s in probe["streams"])

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

source_sha = sha(SOURCE)
assert source_sha == summary["inputs"][SOURCE.name]["sha256"]
manifest = {"session_id": SID, "purpose": "visualize cached full-frame thumb detections; no prediction or gap filling",
            "source_video": str(SOURCE), "source_sha256": source_sha, "source_review": str(REVIEW),
            "detection_mode": mode, "frame_count": index, "marked_frames": marked, "no_detection_frames": index - marked,
            "source_fps": fps, "source_size_px": [sw, sh], "output_size_px": [width, height],
            "timestamps": "source is constant 25 FPS; all frames preserved in order",
            "audio": "original AAC stream copied", "source_verified_unchanged": True, "verification": probe,
            "command": "conda run -n visual_pretouch python " + str(Path(__file__).resolve()),
            "environment_prefix": sys.prefix, "output": VIDEO.name,
            "sha256": {p.name: sha(p) for p in [OUT / "render.py", OUT / "config.yaml", OUT / "tracking_config.yaml", OUT / "detections.csv", VIDEO]}}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(f"Saved {VIDEO}; {index} frames, {marked} marked, {index - marked} missing, audio preserved.", flush=True)
