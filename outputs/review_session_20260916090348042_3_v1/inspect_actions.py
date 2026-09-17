
import sys,json,importlib.util
from pathlib import Path
import cv2,numpy as np,yaml
sys.dont_write_bytecode=True
out=Path("outputs/review_session_20260916090348042_3_v1").resolve()
cfg=yaml.safe_load((out/"config.yaml").read_text())
diag=json.loads((out/"diagnostics.json").read_text())
spec=importlib.util.spec_from_file_location("p",out/"pipeline_snapshot.py")
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
rows=p.read_csv(out/"cropped_hand_diagnostics.csv")
ts=np.asarray([float(r["video_time_ms"]) for r in rows])
events=diag["flash_checks"][0]["events"]
windows=[]
for delta in cfg["evaluation"]["lead_times_ms"]:
    stats=[]
    for i,event in enumerate(events):
        selected=[r for r in rows if event["video_time_ms"]-delta<=float(r["video_time_ms"])<event["video_time_ms"]]
        n=sum(r["hand_detected"]=="1" for r in selected)
        stats.append({"trial_id":f"trial_{i:04d}","window_before_flash_ms":delta,"detected_frames":n,"frames":len(selected)})
    windows+=stats
p.write_csv(out/"per_trial_hand_coverage.csv",windows,list(windows[0]))
cap,_=p.open_video(Path("dataset/session_20260916090348042_3/VID_20260916_170352.mp4"))
tiles=[]
x1,y1,x2,y2=cfg["review"]["operating_hand_crop_preview_px"]
for trial in [0,14,29]:
    event=events[trial]
    for delta in [200,100,33,0]:
        cutoff=event["video_time_ms"]-delta
        i=max(0,int(np.searchsorted(ts,cutoff,side="right")-1))
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read()
        if not ok: raise RuntimeError(i)
        small=cv2.resize(frame,(853,480))
        r=rows[i]
        if r["hand_detected"]=="1":
            xy=(round(float(r["thumb_tip_x_px"])*853/1920),round(float(r["thumb_tip_y_px"])*480/1080))
            cv2.circle(small,xy,8,(0,255,0),2)
        crop=small[y1:y2,x1:x2]
        tile=cv2.copyMakeBorder(crop,36,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
        cv2.putText(tile,f"T{trial:02d} flash-{delta}ms / "+("tip" if r["hand_detected"]=="1" else "no hand"),(6,24),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        tiles.append(tile)
cap.release()
cv2.imwrite(str(out/"action_samples.jpg"),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
for delta in [50,100,150,200]:
    selected=[r for r in windows if r["window_before_flash_ms"]==delta]
    print("Before flash",delta,"ms:",sum(r["detected_frames"]>0 for r in selected),"/ 30 trials with any detection;",
          sum(r["detected_frames"]>=2 for r in selected),"/ 30 with at least 2 frames")

