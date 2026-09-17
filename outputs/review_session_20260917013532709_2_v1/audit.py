"""Data quality review only; preserve raw inputs and refuse existing results."""
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SID = "session_20260917013532709_2"
RAW = ROOT / "dataset" / SID
VIDEO = RAW / "DJI_20260917013501_0005_D.MP4"
CFG = yaml.safe_load((OUT / "config.yaml").read_text())
for name in ["summary.json", "contact_sheet.jpg", "video_probe.json", "pipeline_snapshot.py"]:
    if (OUT / name).exists():
        raise FileExistsError(OUT / name)
shutil.copy2(ROOT / "scripts/visual_pretouch.py", OUT / "pipeline_snapshot.py")
spec = importlib.util.spec_from_file_location("pipeline", OUT / "pipeline_snapshot.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
session = json.loads((RAW / (SID + "_session.json")).read_text())
rows = pipeline.read_csv(RAW / (SID + "_touches.csv"))
uv = np.array([[float(r["touch_u"]), float(r["touch_v"])] for r in rows])
targets = np.array([[float(r["target_u"]), float(r["target_v"])] for r in rows])
times = np.array([float(r["touch_time_browser_ms"]) for r in rows])
residuals = [[float(r["touch_u"]) - (float(r["client_x"]) - float(r["region_left"])) / float(r["region_width_css_px"]),
              float(r["touch_v"]) - (float(r["client_y"]) - float(r["region_top"])) / float(r["region_height_css_px"])] for r in rows]
cap, orientation = pipeline.open_video(VIDEO)
fps, width, height = pipeline.video_meta(cap)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
small_h = CFG["review"]["preview_height_px"]
small_w = round(width * small_h / height)
samples, tiles = [], []
for index in np.linspace(0, count - 1, CFG["review"]["sample_count"]).round().astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot decode sample {index}")
    t = pipeline.frame_time_ms(cap, int(index), fps)
    small = cv2.resize(frame, (small_w, small_h))
    name = f"frame_{index:05d}.jpg"
    cv2.imwrite(str(OUT / name), small)
    tile = cv2.copyMakeBorder(small, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(tile, f"{t / 1000:.2f}s f{index}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(tile)
    samples.append({"frame_index": int(index), "video_time_ms": t, "path": name})
cap.release()
cv2.imwrite(str(OUT / "contact_sheet.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
inputs = {}
for path in sorted(RAW.iterdir()):
    if path.is_file():
        sha = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(block)
        inputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha.hexdigest()}
probe = json.loads(subprocess.check_output(["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(VIDEO)]))
pipeline.write_json(OUT / "video_probe.json", probe)
intervals = np.diff(times) / 1000
summary = {
    "session_id": SID, "device": session["device"], "schema_version": session["schema_version"],
    "touch_count": len(rows), "declared_trial_count": session["trial_count"],
    "trial_ids_unique": len({r["trial_id"] for r in rows}) == len(rows),
    "trial_ids_complete": [r["trial_id"] for r in rows] == [f"trial_{i:04d}" for i in range(session["trial_count"])],
    "session_ids": sorted({r["session_id"] for r in rows}), "device_ids": sorted({r["device_id"] for r in rows}),
    "touch_indices": sorted({r["touch_index"] for r in rows}), "time_monotonic": bool(np.all(np.diff(times) > 0)),
    "touch_span_s": float((times[-1] - times[0]) / 1000),
    "inter_touch_interval_s": {"min": float(intervals.min()), "median": float(np.median(intervals)), "max": float(intervals.max())},
    "touch_uv_min": uv.min(axis=0).tolist(), "touch_uv_max": uv.max(axis=0).tolist(),
    "target_uv_min": targets.min(axis=0).tolist(), "target_uv_max": targets.max(axis=0).tolist(),
    "all_touches_right": bool(np.all(uv[:, 0] > .5)), "all_touches_inside": bool(np.all((uv >= 0) & (uv <= 1))),
    "collection_config": session["collection_config"],
    "target_circle_top_v_min": float(np.min(targets[:, 1] - 24 / np.array([float(r["region_height_css_px"]) for r in rows]))),
    "normalization_max_absolute_residual": float(np.abs(residuals).max()),
    "region_width_css_px_values": sorted({r["region_width_css_px"] for r in rows}),
    "region_height_css_px_values": sorted({r["region_height_css_px"] for r in rows}),
    "screen_width_css_px_values": sorted({r["screen_width_css_px"] for r in rows}),
    "screen_height_css_px_values": sorted({r["screen_height_css_px"] for r in rows}),
    "fullscreen_values": sorted({r["fullscreen"] for r in rows}),
    "video": {"fps": fps, "frame_count": count, "oriented_width_px": width, "oriented_height_px": height,
              "orientation_degrees": orientation, "duration_s": float(probe["format"]["duration"]),
              "encoder": probe["format"].get("tags", {}).get("encoder")},
    "preview_size_px": [small_w, small_h], "samples": samples, "inputs": inputs,
}
pipeline.write_json(OUT / "summary.json", summary)
print(json.dumps({k: v for k, v in summary.items() if k not in ("samples", "inputs")}, ensure_ascii=False, indent=2))
