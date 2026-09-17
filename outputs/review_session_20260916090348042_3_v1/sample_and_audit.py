
import sys, json, csv, hashlib, shutil, importlib.util
from pathlib import Path
import cv2, numpy as np, yaml
sys.dont_write_bytecode = True
root = Path.cwd()
sid = "session_20260916090348042_3"
raw = root / "dataset" / sid
out = root / "outputs" / ("review_" + sid + "_v1")
if out.exists():
    raise FileExistsError(out)
out.mkdir()
spec = importlib.util.spec_from_file_location("pipeline", root / "scripts/visual_pretouch.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
video = raw / "VID_20260916_170352.mp4"
session = json.loads((raw / (sid + "_session.json")).read_text())
rows = pipeline.read_csv(raw / (sid + "_touches.csv"))
uv = np.asarray([[float(r["touch_u"]), float(r["touch_v"])] for r in rows])
targets = np.asarray([[float(r["target_u"]), float(r["target_v"])] for r in rows])
times = np.asarray([float(r["touch_time_browser_ms"]) for r in rows])
residuals = []
for r in rows:
    residuals.append([float(r["touch_u"]) - (float(r["client_x"]) - float(r["region_left"]))/float(r["region_width_css_px"]),
                      float(r["touch_v"]) - (float(r["client_y"]) - float(r["region_top"]))/float(r["region_height_css_px"])])
cap, orientation = pipeline.open_video(video)
fps, width, height = pipeline.video_meta(cap)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
indices = np.linspace(0, max(count-1, 0), 12).round().astype(int)
samples = []
tiles = []
for index in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok: continue
    t = pipeline.frame_time_ms(cap, int(index), fps)
    small = cv2.resize(frame, (round(width*480/height), 480))
    name = f"frame_{int(index):05d}.jpg"
    cv2.imwrite(str(out / name), small)
    tile = cv2.copyMakeBorder(small, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25,25,25))
    cv2.putText(tile, f"{t/1000:.2f}s  f{index}", (8,24), cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    tiles.append(tile)
    samples.append({"frame_index":int(index),"video_time_ms":t,"path":name,"resize_scale":480/height})
cap.release()
sheet = np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
cv2.imwrite(str(out / "contact_sheet.jpg"),sheet)
hashes = {}
for p in sorted(raw.iterdir()):
    if p.is_file():
        sha=hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024), b""):sha.update(block)
        hashes[p.name]={"sha256":sha.hexdigest(),"bytes":p.stat().st_size}
summary = {
 "session_id":sid,"device":session["device"],"schema_version":session["schema_version"],
 "touch_count":len(rows),"declared_trial_count":session["trial_count"],
 "trial_ids_unique":len({r["trial_id"] for r in rows})==len(rows),
 "trial_ids_complete":[r["trial_id"] for r in rows]==[f"trial_{i:04d}" for i in range(30)],
 "touch_indices":sorted({r["touch_index"] for r in rows}),
 "time_monotonic":bool(np.all(np.diff(times)>0)),
 "touch_span_s":float((times[-1]-times[0])/1000),
 "inter_touch_interval_s":{"min":float(np.diff(times).min()/1000),"median":float(np.median(np.diff(times))/1000),"max":float(np.diff(times).max()/1000)},
 "touch_uv_min":uv.min(axis=0).tolist(),"touch_uv_max":uv.max(axis=0).tolist(),
 "target_uv_min":targets.min(axis=0).tolist(),"target_uv_max":targets.max(axis=0).tolist(),
 "all_touches_right":bool(np.all(uv[:,0]>.5)),"all_touches_inside":bool(np.all((uv>=0)&(uv<=1))),
 "target_distance_normalized":{"mean":float(np.linalg.norm(uv-targets,axis=1).mean()),"p90":float(np.percentile(np.linalg.norm(uv-targets,axis=1),90))},
 "normalization_max_absolute_residual":float(np.abs(residuals).max()),
 "region_width_css_px_values":sorted({r["region_width_css_px"] for r in rows}),
 "region_height_css_px_values":sorted({r["region_height_css_px"] for r in rows}),
 "screen_height_css_px_values":sorted({r["screen_height_css_px"] for r in rows}),
 "fullscreen_values":sorted({r["fullscreen"] for r in rows}),
 "session_region_height_css_px":session["region_height_css_px"],
 "session_screen_height_css_px":session["screen_height_css_px"],
 "video":{"fps":fps,"frame_count":count,"oriented_width_px":width,"oriented_height_px":height,"orientation_degrees":orientation,"nominal_duration_s":count/fps},
 "samples":samples,"inputs":hashes
}
(out / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
cfg = yaml.safe_load((root / "config.yaml").read_text())
cfg["review"]={"sample_count":12,"preview_height_px":480,"purpose":"data quality review, no prediction evaluation"}
(out / "config.yaml").write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
shutil.copy2(root / "scripts/visual_pretouch.py",out / "pipeline_snapshot.py")
print(json.dumps({k:v for k,v in summary.items() if k not in ("inputs","samples")},ensure_ascii=False,indent=2))
print("Contact sheet:",out / "contact_sheet.jpg")

