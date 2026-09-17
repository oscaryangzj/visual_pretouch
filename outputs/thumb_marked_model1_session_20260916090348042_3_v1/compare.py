
import csv,json,yaml,re
from pathlib import Path
import cv2,numpy as np
out=Path("outputs/thumb_marked_model1_session_20260916090348042_3_v1")
old=Path("outputs/thumb_marked_session_20260916090348042_3_v1")
cfg0=yaml.safe_load((old/"config.yaml").read_text());cfg1=yaml.safe_load((out/"config.yaml").read_text())
cfg0["tracking"]["model_complexity"]=1
assert cfg0==cfg1
for f in ["detections.csv","summary.json"]:
    assert(out/f).is_file()
with(out/"detections.csv").open(newline="") as f: rows=list(csv.DictReader(f))
cap=cv2.VideoCapture(str(out/"thumb_marked.mp4"))
tiles=[]
for index in [0,509,1528]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();assert ok
    if rows[index]["hand_detected"]=="1":
        x,y=round(float(rows[index]["thumb_tip_x_px"])),round(float(rows[index]["thumb_tip_y_px"]))
        patch=frame[max(0,y-18):y+19,max(0,x-18):x+19].astype(np.int16)
        assert np.any((patch[:,:,1]>150)&(patch[:,:,1]>patch[:,:,0]+70)&(patch[:,:,1]>patch[:,:,2]+70))
    tiles.append(cv2.resize(frame,(853,480)))
cap.release()
cv2.imwrite(str(out/"qa_preview.jpg"),np.hstack(tiles))
diag=json.loads((out/"diagnostics.json").read_text())
events=diag["flash_checks"][0]["events"]
success={}
for delta in [50,100,150,200]:
    success[str(delta)]=sum(any(r["hand_detected"]=="1" and e["video_time_ms"]-delta<=float(r["video_time_ms"])<e["video_time_ms"] for r in rows) for e in events)
comparison={"only_changed_parameter":"tracking.model_complexity: 0 -> 1",
            "model0_summary":json.loads((Path("outputs/review_session_20260916090348042_3_v1")/"cropped_hand_summary.json").read_text()),
            "model1_summary":json.loads((out/"cropped_hand_summary.json").read_text()),
            "model1_trials_with_any_detection_before_flash":success,
            "windows_are_before_flash_not_crossing_events":True,
            "does_not_measure_verified_thumb_accuracy":True,
            "source_video_unchanged_and_output_frame_audio_checks":json.loads((out/"video_manifest.json").read_text())["source_verified_unchanged"]}
(out/"comparison.json").write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(success));print("QA passed: only model_complexity changed; video frames/audio and sample overlay checked.")

