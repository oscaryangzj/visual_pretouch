"""Render cached detections around selected flashes; no new inference."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SID = "session_20260917013532709_2"
CFG = yaml.safe_load((OUT / "config.yaml").read_text())
review = CFG["review"]
for name in ["action_samples.jpg", "action_frames.json"]:
    if (OUT / name).exists():
        raise FileExistsError(OUT / name)
spec = importlib.util.spec_from_file_location("pipeline", OUT / "pipeline_snapshot.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
diag = json.loads((OUT / "diagnostics.json").read_text())
summary = json.loads((OUT / "summary.json").read_text())
rows = pipeline.read_csv(OUT / "frame_diagnostics.csv")
times = np.array([float(r["video_time_ms"]) for r in rows])
matched = next(c for c in diag["flash_checks"] if c["count"] == summary["touch_count"])
cap, _ = pipeline.open_video(ROOT / "dataset" / SID / "DJI_20260917013501_0005_D.MP4")
fps, width, height = pipeline.video_meta(cap)
pw, ph = summary["preview_size_px"]
x1, y1, x2, y2 = review["action_view_preview_px"]
scale = review["action_view_scale"]
grids, selected = [], []
for trial in review["action_trial_indices"]:
    flash = matched["events"][trial]["video_time_ms"]
    tiles = []
    for offset in review["action_offsets_before_flash_ms"]:
        index = int(np.searchsorted(times, flash - offset, side="right") - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(index)
        crop = frame[round(y1 * height / ph):round(y2 * height / ph), round(x1 * width / pw):round(x2 * width / pw)]
        crop = cv2.resize(crop, ((x2 - x1) * scale, (y2 - y1) * scale))
        row = rows[index]
        for mode, color in [("full", (0, 200, 255)), ("crop", (0, 255, 0))]:
            if int(row[mode + "_detected"]):
                x = (float(row[mode + "_tip_x_native_px"]) * pw / width - x1) * scale
                y = (float(row[mode + "_tip_y_native_px"]) * ph / height - y1) * scale
                cv2.circle(crop, (round(x), round(y)), 8, color, 2)
        tile = cv2.copyMakeBorder(crop, 54, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        label = f"trial {trial:02d} flash-{flash - times[index]:.0f}ms f{index}"
        status = "full: " + ("OK" if int(row["full_detected"]) else "MISS") + "  crop: " + ("OK" if int(row["crop_detected"]) else "MISS")
        cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, status, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
        selected.append({"trial_index": trial, "flash_time_video_ms": flash, "frame_index": index,
                         "video_time_ms": float(times[index]), "actual_before_flash_ms": float(flash - times[index]),
                         "full_detected": int(row["full_detected"]), "crop_detected": int(row["crop_detected"])})
    grids.append(np.hstack(tiles))
cap.release()
cv2.imwrite(str(OUT / "action_samples.jpg"), np.vstack(grids))
pipeline.write_json(OUT / "action_frames.json", selected)
manifest = json.loads((OUT / "manifest.json").read_text())
verified = {}
for name, recorded in summary["inputs"].items():
    sha = hashlib.sha256()
    with (ROOT / "dataset" / SID / name).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    verified[name] = sha.hexdigest() == recorded["sha256"]
manifest.update({"raw_input_hashes_unchanged": verified,
                 "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.py")},
                 "config_sha256": hashlib.sha256((OUT / "config.yaml").read_bytes()).hexdigest(),
                 "outputs": sorted(p.name for p in OUT.iterdir() if p.is_file())})
manifest["commands"].append(f"conda run -n visual_pretouch python {OUT / 'inspect_actions.py'}")
pipeline.write_json(OUT / "manifest.json", manifest)
print(json.dumps({"selected_frames": selected, "raw_input_hashes_unchanged": verified}, indent=2))
