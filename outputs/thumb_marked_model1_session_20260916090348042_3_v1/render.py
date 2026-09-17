#!/usr/bin/env python3
"""Overlay cached per-frame thumb detections without interpolation."""
import csv, hashlib, json, math, shutil, subprocess, sys
from pathlib import Path
import cv2, yaml

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SID = "session_20260916090348042_3"
SOURCE = ROOT / "dataset" / SID / "VID_20260916_170352.mp4"
REVIEW = OUT
VIDEO = OUT / "thumb_marked.mp4"
for name in ["thumb_marked.mp4", "video_manifest.json", "detections.csv"]:
    if (OUT / name).exists():
        raise FileExistsError(OUT / name)
cfg = yaml.safe_load((OUT / "config.yaml").read_text())
style = cfg["annotation"]
shutil.copy2(REVIEW / "cropped_hand_diagnostics.csv", OUT / "detections.csv")
with (OUT / "detections.csv").open(newline="") as file:
    rows = list(csv.DictReader(file))
assert [int(r["frame_index"]) for r in rows] == list(range(len(rows)))
cap = cv2.VideoCapture(str(SOURCE))
assert cap.isOpened()
cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(rows)
command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}",
           "-framerate", str(fps), "-i", "pipe:0", "-i", str(SOURCE),
           "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264",
           "-preset", style["video_preset"], "-crf", str(style["video_crf"]),
           "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(VIDEO)]
encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
index = marked = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert index < len(rows)
        row = rows[index]
        found = row["hand_detected"] == "1"
        color = (0,255,0) if found else (0,180,255)
        if found:
            x, y = float(row["thumb_tip_x_px"]), float(row["thumb_tip_y_px"])
            assert math.isfinite(x) and math.isfinite(y)
            point = (round(x), round(y))
            radius, thickness = style["marker_radius_px"], style["marker_thickness_px"]
            cv2.circle(frame, point, radius+2, (0,0,0), thickness+3, cv2.LINE_AA)
            cv2.circle(frame, point, radius, color, thickness, cv2.LINE_AA)
            cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)
            cv2.putText(frame, "thumb tip", (point[0]+18,point[1]-14),
                        cv2.FONT_HERSHEY_SIMPLEX, style["text_scale"], (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, "thumb tip", (point[0]+18,point[1]-14),
                        cv2.FONT_HERSHEY_SIMPLEX, style["text_scale"], color, 2, cv2.LINE_AA)
            marked += 1
        cv2.rectangle(frame,(12,12),(700,117),(15,15,15),-1)
        labels = [(f"MediaPipe thumb tip (crop; model {cfg['tracking']['model_complexity']})",(235,235,235)),
                  (f"Frame {index+1}/{len(rows)} | {float(row['video_time_ms'])/1000:.2f}s",(235,235,235)),
                  ("DETECTED" if found else "NO DETECTION",color)]
        for line, (text, text_color) in enumerate(labels):
            cv2.putText(frame,text,(24,40+32*line),cv2.FONT_HERSHEY_SIMPLEX,
                        style["text_scale"],text_color,2,cv2.LINE_AA)
        encoder.stdin.write(frame.tobytes())
        index += 1
        if index % 500 == 0:
            print(f"Rendered {index}/{len(rows)} frames",flush=True)
finally:
    cap.release()
    encoder.stdin.close()
error = encoder.stderr.read().decode()
assert encoder.wait() == 0, error
assert index == len(rows)
assert marked == sum(r["hand_detected"]=="1" for r in rows)
probe = json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries",
        "stream=codec_type,codec_name,width,height,nb_frames,duration:format=duration","-of","json",str(VIDEO)]))
video_stream = next(s for s in probe["streams"] if s["codec_type"]=="video")
assert int(video_stream["nb_frames"])==len(rows)
assert [video_stream["width"],video_stream["height"]]==[width,height]
assert video_stream["codec_name"]=="h264"
assert any(s["codec_type"]=="audio" for s in probe["streams"])
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda:file.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
source_sha = sha(SOURCE)
expected_sha = json.loads((REVIEW/"summary.json").read_text())["inputs"][SOURCE.name]["sha256"]
assert source_sha==expected_sha
manifest = {"session_id":SID,"purpose":"visualize cached per-frame detections; no prediction or gap filling",
            "source_video":str(SOURCE),"source_sha256":source_sha,
            "source_review":str(REVIEW),"detections":"detections.csv","detections_sha256":sha(OUT/"detections.csv"),
            "config":"config.yaml","config_sha256":sha(OUT/"config.yaml"),
            "code":"render.py","code_sha256":sha(Path(__file__)),
            "command":"conda run -n visual_pretouch python "+str(Path(__file__).resolve()),
            "frame_count":index,"marked_frames":marked,"no_detection_frames":index-marked,
            "source_fps":fps,"timestamps":"frame order preserved; encoded at source average FPS",
            "audio":"original AAC stream copied","verification":probe,"source_verified_unchanged":True,
            "output":"thumb_marked.mp4","output_sha256":sha(VIDEO)}
(OUT/"video_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
print(f"Saved {VIDEO}; {index} frames, {marked} marked, {index-marked} unmarked, audio preserved.",flush=True)


