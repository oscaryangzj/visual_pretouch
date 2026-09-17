#!/usr/bin/env python3
"""Quality diagnostics only; no screen tracking or prediction evaluation."""
import os, sys, csv, json, hashlib, importlib.util, subprocess
from pathlib import Path
import cv2, numpy as np, yaml

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
os.environ["MPLCONFIGDIR"] = str(OUT / "matplotlib_cache")
os.environ["XDG_CACHE_HOME"] = str(OUT / "cache")
cfg = yaml.safe_load((OUT / "config.yaml").read_text())
review = cfg["review"]
spec = importlib.util.spec_from_file_location("pipeline", OUT / "pipeline_snapshot.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
sid = "session_20260916090348042_3"
raw = ROOT / "dataset" / sid
touches = pipeline.read_csv(raw / (sid + "_touches.csv"))
summary = json.loads((OUT / "summary.json").read_text())
for name in ["frame_diagnostics.csv", "diagnostics.json", "thumb_contact_sheet.jpg", "flash_timing.png", "manifest.json"]:
    if (OUT / name).exists():
        raise FileExistsError(OUT / name)

import mediapipe as mp
tracking = cfg["tracking"]
hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=tracking["max_num_hands"],
    model_complexity=tracking["model_complexity"],
    min_detection_confidence=tracking["min_detection_confidence"],
    min_tracking_confidence=tracking["min_tracking_confidence"])
cap, orientation = pipeline.open_video(raw / "VID_20260916_170352.mp4")
fps, width, height = pipeline.video_meta(cap)
small_h = review["preview_height_px"]
small_w = round(width * small_h / height)
x1, y1, x2, y2 = review["flash_roi_preview_px"]
sample_indices = {r["frame_index"] for r in summary["samples"]}
frame_rows, overlays = [], []
index = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = pipeline.frame_time_ms(cap, index, fps)
        small = cv2.resize(frame, (small_w, small_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        score = float(gray[y1:y2, x1:x2].mean())
        result = hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        row = {"frame_index": index, "video_time_ms": timestamp, "flash_roi_mean_gray": score,
               "hand_detected": 0, "thumb_tip_x_px": "", "thumb_tip_y_px": "",
               "wrist_x_px": "", "wrist_y_px": "", "raw_handedness": "", "raw_handedness_score": ""}
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            tip, wrist = hand.landmark[4], hand.landmark[0]
            raw_label = result.multi_handedness[0].classification[0]
            row.update({"hand_detected": 1, "thumb_tip_x_px": tip.x * width, "thumb_tip_y_px": tip.y * height,
                        "wrist_x_px": wrist.x * width, "wrist_y_px": wrist.y * height,
                        "raw_handedness": raw_label.label, "raw_handedness_score": raw_label.score})
        if index in sample_indices:
            if row["hand_detected"]:
                tip_xy = (round(tip.x * small_w), round(tip.y * small_h))
                wrist_xy = (round(wrist.x * small_w), round(wrist.y * small_h))
                cv2.circle(small, tip_xy, 9, (0,255,0), 2)
                cv2.circle(small, wrist_xy, 7, (0,255,255), 2)
                for a, b in [(1,2),(2,3),(3,4)]:
                    pa, pb = hand.landmark[a], hand.landmark[b]
                    cv2.line(small, (round(pa.x*small_w),round(pa.y*small_h)),
                             (round(pb.x*small_w),round(pb.y*small_h)), (0,255,0), 2)
            tile = cv2.copyMakeBorder(small,36,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
            label = row["raw_handedness"] or "no hand"
            cv2.putText(tile,f"{timestamp/1000:.2f}s  {label}",(8,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            overlays.append(tile)
        frame_rows.append(row)
        index += 1
        if index % 400 == 0:
            print(f"Reviewed {index} video frames", flush=True)
finally:
    hands.close()
    cap.release()

pipeline.write_csv(OUT / "frame_diagnostics.csv", frame_rows, list(frame_rows[0]))
cv2.imwrite(str(OUT / "thumb_contact_sheet.jpg"),
            np.vstack([np.hstack(overlays[i:i+4]) for i in range(0,len(overlays),4)]))
scores = np.asarray([r["flash_roi_mean_gray"] for r in frame_rows])
video_times = np.asarray([r["video_time_ms"] for r in frame_rows])
browser_times = np.asarray([float(r["touch_time_browser_ms"]) for r in touches])
flash_checks = []
for threshold in review["flash_thresholds_gray"]:
    active, previous, events = False, None, []
    for row in frame_rows:
        score = row["flash_roi_mean_gray"]
        rise = score - previous if previous is not None else 0
        if not active and score >= threshold and rise >= cfg["alignment"]["flash_rise_threshold_gray"]:
            events.append({"frame_index":row["frame_index"],"video_time_ms":row["video_time_ms"],"mean_gray":score})
            active = True
        if active and score < threshold - review["flash_reset_hysteresis_gray"]:
            active = False
        previous = score
    check = {"threshold_gray": threshold, "count": len(events), "events": events}
    if len(events) == len(touches):
        event_times = np.asarray([e["video_time_ms"] for e in events])
        offsets = event_times - browser_times
        residual = offsets - np.median(offsets)
        slope, intercept = np.polyfit(browser_times-browser_times[0], event_times-event_times[0], 1)
        check.update({"median_video_minus_browser_ms":float(np.median(offsets)),
                      "median_absolute_residual_ms":float(np.median(np.abs(residual))),
                      "p90_absolute_residual_ms":float(np.percentile(np.abs(residual),90)),
                      "max_absolute_residual_ms":float(np.abs(residual).max()),
                      "clock_scale":float(slope),
                      "inter_flash_inter_touch_max_difference_ms":float(np.max(np.abs(np.diff(event_times)-np.diff(browser_times))))})
    flash_checks.append(check)
valid = [r for r in frame_rows if r["hand_detected"]]
wrong_side = [r for r in valid if r["wrist_x_px"] > review["other_hand_wrist_min_fraction_x"] * width]
centers = [np.asarray(s["corners_preview_px"], dtype=np.float32).mean(axis=0) for s in review["screen_boundary_samples"]]
initial = np.asarray(review["screen_boundary_samples"][0]["corners_preview_px"],dtype=np.float32)
H = cv2.getPerspectiveTransform(initial, np.asarray([[0,0],[1,0],[1,1],[0,1]],dtype=np.float32))
motion = []
for sample, center in zip(review["screen_boundary_samples"], centers):
    corners = np.asarray(sample["corners_preview_px"], dtype=np.float32)
    projected = pipeline.project(H,corners)
    distances = np.linalg.norm(projected-np.asarray([[0,0],[1,0],[1,1],[0,1]]),axis=1)
    motion.append({"frame_index":sample["frame_index"],"center_shift_preview_px":(center-centers[0]).tolist(),
                   "center_shift_native_px":((center-centers[0])*height/small_h).tolist(),
                   "max_corner_displacement_under_first_frame_mapping_normalized":float(distances.max()),
                   "area_ratio_to_first_frame":float(cv2.contourArea(corners)/cv2.contourArea(initial))})
diag = {"processed_frames":len(frame_rows),"preview_size_px":[small_w,small_h],
        "mediapipe_version":mp.__version__,"hand_detected_frames":len(valid),
        "hand_detection_coverage":len(valid)/len(frame_rows),
        "raw_handedness_counts":{label:sum(r["raw_handedness"]==label for r in valid) for label in ("Left","Right")},
        "detections_with_wrist_on_other_hand_side":len(wrong_side),
        "hand_detection_is_not_verified_thumb_accuracy":True,
        "flash_roi_mean_gray_range":[float(scores.min()),float(scores.max())],
        "flash_checks":flash_checks,"screen_motion_samples":motion,
        "motion_corners_are_approximate_visible_screen_boundary_not_viewport_calibration":True,
        "operation_area_height_fraction_of_reported_screen":summary["session_region_height_css_px"]/summary["session_screen_height_css_px"],
        "height_mm_if_uniform_scale_and_full_screen_height":cfg["devices"][cfg["device"]]["region_height_mm"]*summary["session_region_height_css_px"]/summary["session_screen_height_css_px"],
        "no_prediction_accuracy_reported":True}
pipeline.write_json(OUT / "diagnostics.json",diag)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
axes[0].plot(video_times/1000,scores,lw=.8)
axes[0].set(xlabel="Video time (s)",ylabel="ROI mean grayscale",title="White flash diagnostic (fixed interior ROI)")
matched = next((c for c in flash_checks if c["count"]==len(touches)), None)
if matched:
    et = np.asarray([e["video_time_ms"] for e in matched["events"]])
    residual = et-browser_times-matched["median_video_minus_browser_ms"]
    axes[0].scatter(et/1000,[e["mean_gray"] for e in matched["events"]],s=14,color="red")
    axes[1].plot(np.arange(len(touches)),residual,"o-",ms=4)
    axes[1].set(xlabel="Trial index",ylabel="Timing residual (ms)",title="Video flash time minus browser touch time, median offset removed")
else:
    axes[1].text(.05,.5,"No tested threshold produced exactly 30 events",transform=axes[1].transAxes)
for ax in axes: ax.grid(alpha=.2)
fig.savefig(OUT / "flash_timing.png",dpi=160)
plt.close(fig)
manifest={"session_id":sid,"purpose":"quality diagnostics, not research performance evaluation",
          "inputs":summary["inputs"],"config":"config.yaml","command":"conda run -n visual_pretouch python "+str(Path(__file__).resolve()),
          "git_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True).stdout.strip(),
          "git_status":subprocess.run(["git","status","--short"],cwd=ROOT,capture_output=True,text=True).stdout,
          "code_snapshots":["sample_and_audit.py","diagnose.py","pipeline_snapshot.py"],
          "code_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.py")},
          "outputs":["summary.json","diagnostics.json","frame_diagnostics.csv","contact_sheet.jpg","thumb_contact_sheet.jpg","flash_timing.png"]}
pipeline.write_json(OUT / "manifest.json",manifest)
print(json.dumps({k:v for k,v in diag.items() if k!="flash_checks"},ensure_ascii=False,indent=2),flush=True)
print("Flash counts and timing:",json.dumps([{k:v for k,v in c.items() if k!="events"} for c in flash_checks],indent=2),flush=True)

