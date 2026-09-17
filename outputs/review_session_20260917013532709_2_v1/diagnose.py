"""Sequential full/cropped MediaPipe and flash diagnostics, without prediction."""
import hashlib
import importlib.util
import json
import os
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
os.environ["MPLCONFIGDIR"] = str(OUT / "matplotlib_cache")
CFG = yaml.safe_load((OUT / "config.yaml").read_text())
review, tracking = CFG["review"], CFG["tracking"]
for name in ["frame_diagnostics.csv", "diagnostics.json", "full_thumb_contact_sheet.jpg", "cropped_thumb_contact_sheet.jpg", "flash_timing.png", "manifest.json"]:
    if (OUT / name).exists():
        raise FileExistsError(OUT / name)
spec = importlib.util.spec_from_file_location("pipeline", OUT / "pipeline_snapshot.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
summary = json.loads((OUT / "summary.json").read_text())
touches = pipeline.read_csv(RAW / (SID + "_touches.csv"))
import mediapipe as mp
models = {mode: mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=tracking["max_num_hands"],
    model_complexity=tracking["model_complexity"],
    min_detection_confidence=tracking["min_detection_confidence"],
    min_tracking_confidence=tracking["min_tracking_confidence"])
    for mode in ("full", "crop")}
cap, orientation = pipeline.open_video(RAW / "DJI_20260917013501_0005_D.MP4")
fps, width, height = pipeline.video_meta(cap)
small_h = review["preview_height_px"]
small_w = round(width * small_h / height)
cx1, cy1, cx2, cy2 = review["operating_hand_crop_preview_px"]
fx1, fy1, fx2, fy2 = review["flash_roi_preview_px"]
sample_indices = {s["frame_index"] for s in summary["samples"]}
rows, overlays = [], {mode: [] for mode in models}
index = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = pipeline.frame_time_ms(cap, index, fps)
        small = cv2.resize(frame, (small_w, small_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        row = {"frame_index": index, "video_time_ms": timestamp,
               "flash_roi_mean_gray": float(gray[fy1:fy2, fx1:fx2].mean())}
        for mode, model in models.items():
            image = small if mode == "full" else small[cy1:cy2, cx1:cx2]
            dx, dy = (0, 0) if mode == "full" else (cx1, cy1)
            result = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            points = None
            row.update({mode + "_detected": 0, mode + "_tip_x_native_px": "", mode + "_tip_y_native_px": "",
                        mode + "_wrist_x_native_px": "", mode + "_wrist_y_native_px": "",
                        mode + "_raw_handedness": "", mode + "_raw_handedness_score": ""})
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                label = result.multi_handedness[0].classification[0]
                points = [(p.x * image.shape[1] + dx, p.y * image.shape[0] + dy) for p in hand.landmark]
                tip, wrist = points[4], points[0]
                row.update({mode + "_detected": 1,
                            mode + "_tip_x_native_px": tip[0] * width / small_w,
                            mode + "_tip_y_native_px": tip[1] * height / small_h,
                            mode + "_wrist_x_native_px": wrist[0] * width / small_w,
                            mode + "_wrist_y_native_px": wrist[1] * height / small_h,
                            mode + "_raw_handedness": label.label, mode + "_raw_handedness_score": label.score})
            if index in sample_indices:
                image_out = small.copy()
                if mode == "crop":
                    cv2.rectangle(image_out, (cx1, cy1), (cx2, cy2), (255, 150, 0), 1)
                if points:
                    cv2.circle(image_out, tuple(round(x) for x in points[4]), 7, (0, 255, 0), 2)
                    cv2.circle(image_out, tuple(round(x) for x in points[0]), 5, (0, 255, 255), 2)
                    for a, b in [(1, 2), (2, 3), (3, 4)]:
                        cv2.line(image_out, tuple(round(x) for x in points[a]), tuple(round(x) for x in points[b]), (0, 255, 0), 2)
                tile = cv2.copyMakeBorder(image_out, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
                text = f"{timestamp / 1000:.2f}s {mode}: " + (row[mode + "_raw_handedness"] or "NO HAND")
                cv2.putText(tile, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
                overlays[mode].append(tile)
        rows.append(row)
        index += 1
        if index % 300 == 0:
            print(f"Reviewed {index} frames", flush=True)
finally:
    cap.release()
    for model in models.values():
        model.close()
pipeline.write_csv(OUT / "frame_diagnostics.csv", rows, list(rows[0]))
for mode, tiles in overlays.items():
    filename = "full_thumb_contact_sheet.jpg" if mode == "full" else "cropped_thumb_contact_sheet.jpg"
    cv2.imwrite(str(OUT / filename), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
scores = np.array([r["flash_roi_mean_gray"] for r in rows])
vt = np.array([r["video_time_ms"] for r in rows])
bt = np.array([float(r["touch_time_browser_ms"]) for r in touches])
flash_checks = []
for threshold in review["flash_thresholds_gray"]:
    active, previous, events = False, None, []
    for row in rows:
        score = row["flash_roi_mean_gray"]
        rise = 0 if previous is None else score - previous
        if not active and score >= threshold and rise >= CFG["alignment"]["flash_rise_threshold_gray"]:
            events.append({"frame_index": row["frame_index"], "video_time_ms": row["video_time_ms"], "mean_gray": score})
            active = True
        if active and score < threshold - review["flash_reset_hysteresis_gray"]:
            active = False
        previous = score
    check = {"threshold_gray": threshold, "count": len(events), "events": events}
    if len(events) == len(touches):
        et = np.array([e["video_time_ms"] for e in events])
        offset = float(np.median(et - bt))
        residual = et - bt - offset
        slope, _ = np.polyfit(bt - bt[0], et - et[0], 1)
        check.update({"median_video_minus_browser_ms": offset, "median_absolute_residual_ms": float(np.median(np.abs(residual))),
                      "p90_absolute_residual_ms": float(np.percentile(np.abs(residual), 90)),
                      "max_absolute_residual_ms": float(np.abs(residual).max()), "clock_scale": float(slope)})
    flash_checks.append(check)
matched = next((c for c in flash_checks if c["count"] == len(touches)), None)
model_checks = {}
for mode in models:
    valid = np.array([bool(r[mode + "_detected"]) for r in rows])
    windows = []
    if matched:
        for window in review["diagnostic_windows_ms"]:
            frame_count, detected, trials_any, trials_two = 0, 0, 0, 0
            for event in matched["events"]:
                mask = (vt >= event["video_time_ms"] - window) & (vt < event["video_time_ms"])
                n = int(valid[mask].sum())
                frame_count += int(mask.sum())
                detected += n
                trials_any += n > 0
                trials_two += n >= 2
            windows.append({"before_flash_ms": window, "total_frames": frame_count, "detected_frames": detected,
                            "coverage": detected / frame_count, "trials_with_any_detection": trials_any,
                            "trials_with_at_least_two_detections": trials_two, "total_trials": len(touches)})
    model_checks[mode] = {"detected_frames": int(valid.sum()), "total_frames": len(rows), "coverage": float(valid.mean()),
                          "raw_handedness_counts": {label: sum(r[mode + "_raw_handedness"] == label for r in rows) for label in ("Left", "Right")},
                          "pre_flash_windows": windows, "not_verified_tip_accuracy": True}
initial = np.array(review["screen_boundary_samples"][0]["corners_preview_px"], dtype=np.float32)
unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
H = cv2.getPerspectiveTransform(initial, unit)
motion = []
for sample in review["screen_boundary_samples"]:
    corners = np.array(sample["corners_preview_px"], dtype=np.float32)
    motion.append({"frame_index": sample["frame_index"], "center_shift_preview_px": (corners.mean(axis=0) - initial.mean(axis=0)).tolist(),
                   "center_shift_native_px": ((corners.mean(axis=0) - initial.mean(axis=0)) * height / small_h).tolist(),
                   "area_ratio_to_first_frame": float(cv2.contourArea(corners) / cv2.contourArea(initial)),
                   "max_corner_displacement_under_first_mapping_normalized": float(np.linalg.norm(pipeline.project(H, corners) - unit, axis=1).max())})
diag = {"processed_frames": len(rows), "timestamps_strictly_increasing": bool(np.all(np.diff(vt) > 0)),
        "frame_interval_ms": {"min": float(np.diff(vt).min()), "median": float(np.median(np.diff(vt))), "max": float(np.diff(vt).max())},
        "mediapipe_version": mp.__version__, "model_checks": model_checks, "flash_checks": flash_checks,
        "flash_roi_mean_gray_range": [float(scores.min()), float(scores.max())], "screen_motion_samples": motion,
        "screen_corners_are_manual_approximate_visible_boundary_not_viewport_calibration": True,
        "windows_are_before_uncalibrated_flash_not_centerline_crossing": True, "no_prediction_accuracy_reported": True}
pipeline.write_json(OUT / "diagnostics.json", diag)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
axes[0].plot(vt / 1000, scores, lw=.8)
axes[0].set(xlabel="Video time (s)", ylabel="ROI mean grayscale", title="White flash diagnostic")
if matched:
    et = np.array([e["video_time_ms"] for e in matched["events"]])
    axes[0].scatter(et / 1000, [e["mean_gray"] for e in matched["events"]], s=12, color="red")
    axes[1].plot(np.arange(len(bt)), et - bt - matched["median_video_minus_browser_ms"], "o-", ms=3)
axes[1].set(xlabel="Trial index", ylabel="Residual after median offset (ms)")
for ax in axes:
    ax.grid(alpha=.2)
fig.savefig(OUT / "flash_timing.png", dpi=150)
plt.close(fig)
manifest = {"session_id": SID, "purpose": review["purpose"], "inputs": summary["inputs"], "config": "config.yaml",
            "commands": [f"conda run -n visual_pretouch python {OUT / name}" for name in ("audit.py", "diagnose.py")],
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "git_status": subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout,
            "environment": {"python_prefix": sys.prefix, "mediapipe": mp.__version__, "opencv": cv2.__version__, "numpy": np.__version__},
            "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.py")}}
pipeline.write_json(OUT / "manifest.json", manifest)
print(json.dumps({k: v for k, v in diag.items() if k != "flash_checks"}, ensure_ascii=False, indent=2), flush=True)
print("Flash checks:", json.dumps([{k: v for k, v in c.items() if k != "events"} for c in flash_checks], indent=2), flush=True)
