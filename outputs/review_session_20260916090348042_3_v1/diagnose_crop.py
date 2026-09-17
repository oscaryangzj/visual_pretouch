#!/usr/bin/env python3
"""Second diagnostic: exclude the other hand with a fixed, manually chosen image crop."""
import os, sys, json, importlib.util
from pathlib import Path
import cv2, numpy as np, yaml
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
os.environ["MPLCONFIGDIR"] = str(OUT / "matplotlib_cache")
os.environ["XDG_CACHE_HOME"] = str(OUT / "cache")
cfg = yaml.safe_load((OUT / "config.yaml").read_text())
review = cfg["review"]
summary = json.loads((OUT / "summary.json").read_text())
for name in ["cropped_hand_diagnostics.csv","cropped_hand_summary.json","cropped_thumb_contact_sheet.jpg"]:
    if (OUT / name).exists(): raise FileExistsError(OUT / name)
spec = importlib.util.spec_from_file_location("pipeline", OUT / "pipeline_snapshot.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
import mediapipe as mp
tracking = cfg["tracking"]
hands = mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=tracking["max_num_hands"],
                               model_complexity=tracking["model_complexity"],
                               min_detection_confidence=tracking["min_detection_confidence"],
                               min_tracking_confidence=tracking["min_tracking_confidence"])
cap, _ = pipeline.open_video(ROOT / "dataset" / summary["session_id"] / "VID_20260916_170352.mp4")
fps, width, height = pipeline.video_meta(cap)
small_h = review["preview_height_px"]; small_w=round(width*small_h/height)
x1,y1,x2,y2=review["operating_hand_crop_preview_px"]
sample_indices={r["frame_index"] for r in summary["samples"]}
rows, tiles=[],[]
index=0
try:
    while True:
        ok, frame=cap.read()
        if not ok: break
        timestamp=pipeline.frame_time_ms(cap,index,fps)
        small=cv2.resize(frame,(small_w,small_h))
        image=small[y1:y2,x1:x2].copy()
        result=hands.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        row={"frame_index":index,"video_time_ms":timestamp,"hand_detected":0,
             "thumb_tip_x_px":"","thumb_tip_y_px":"","wrist_x_px":"","wrist_y_px":"","raw_handedness":""}
        if result.multi_hand_landmarks:
            hand=result.multi_hand_landmarks[0]
            tip,wrist=hand.landmark[4],hand.landmark[0]
            label=result.multi_handedness[0].classification[0].label
            row.update({"hand_detected":1,"thumb_tip_x_px":(tip.x*(x2-x1)+x1)*width/small_w,
                        "thumb_tip_y_px":(tip.y*(y2-y1)+y1)*height/small_h,
                        "wrist_x_px":(wrist.x*(x2-x1)+x1)*width/small_w,
                        "wrist_y_px":(wrist.y*(y2-y1)+y1)*height/small_h,"raw_handedness":label})
        if index in sample_indices:
            if row["hand_detected"]:
                tipxy=(round(tip.x*(x2-x1)),round(tip.y*(y2-y1)))
                cv2.circle(image,tipxy,8,(0,255,0),2)
                for a,b in [(1,2),(2,3),(3,4)]:
                    pa,pb=hand.landmark[a],hand.landmark[b]
                    cv2.line(image,(round(pa.x*(x2-x1)),round(pa.y*(y2-y1))),
                                   (round(pb.x*(x2-x1)),round(pb.y*(y2-y1))),(0,255,0),2)
            tile=cv2.copyMakeBorder(image,36,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
            cv2.putText(tile,f"{timestamp/1000:.2f}s  "+(row["raw_handedness"] or "no hand"),(8,24),
                        cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            tiles.append(tile)
        rows.append(row);index+=1
        if index%400==0: print(f"Cropped review: {index} frames",flush=True)
finally:
    hands.close();cap.release()
pipeline.write_csv(OUT / "cropped_hand_diagnostics.csv",rows,list(rows[0]))
cv2.imwrite(str(OUT / "cropped_thumb_contact_sheet.jpg"),
            np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
valid=[r for r in rows if r["hand_detected"]]
events=json.loads((OUT/"diagnostics.json").read_text())["flash_checks"][0]["events"]
window_stats=[]
for window in cfg["evaluation"]["lead_times_ms"]:
    total=valid_count=0
    for event in events:
        start=event["video_time_ms"]-window
        interval=[r for r in rows if start<=r["video_time_ms"]<event["video_time_ms"]]
        total+=len(interval);valid_count+=sum(r["hand_detected"] for r in interval)
    window_stats.append({"before_flash_ms":window,"hand_detected_frames":valid_count,"frames":total,
                         "coverage":valid_count/total if total else None})
diag={"processed_frames":len(rows),"crop_preview_px":[x1,y1,x2,y2],
      "hand_detected_frames":len(valid),"hand_detection_coverage":len(valid)/len(rows),
      "raw_handedness_counts":{label:sum(r["raw_handedness"]==label for r in valid) for label in ("Left","Right")},
      "detections_with_wrist_on_other_hand_side":sum(r["wrist_x_px"]>review["other_hand_wrist_min_fraction_x"]*width for r in valid),
      "before_flash_window_coverage":window_stats,
      "crop_was_chosen_for_quality_diagnostics_not_registered_prediction_preprocessing":True,
      "detection_coverage_is_not_verified_thumb_accuracy":True,
      "flash_times_are_not_calibrated_absolute_touch_times":True}
pipeline.write_json(OUT/"cropped_hand_summary.json",diag)
manifest=json.loads((OUT/"manifest.json").read_text())
manifest["additional_diagnostic_command"]="conda run -n visual_pretouch python "+str(Path(__file__).resolve())
manifest["code_snapshots"].append("diagnose_crop.py")
manifest["outputs"]+=["cropped_hand_diagnostics.csv","cropped_hand_summary.json","cropped_thumb_contact_sheet.jpg"]
import hashlib
manifest["code_sha256"]["diagnose_crop.py"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
pipeline.write_json(OUT/"manifest.json",manifest)
print(json.dumps(diag,indent=2),flush=True)

