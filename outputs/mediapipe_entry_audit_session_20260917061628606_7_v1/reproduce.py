import sys,json,cv2,numpy as np
from pathlib import Path
sys.path.insert(0,'/Users/oscaryang/PycharmProjects/visual_pretouch/scripts')
import visual_pretouch as p
import mediapipe as mp
import yaml
OUT=Path(__file__).resolve().parent
CFG=yaml.safe_load((OUT/'config.yaml').read_text())
root=OUT.parents[1]; session=root/CFG['session_dir']
c=json.loads((session/'.review_cache.json').read_text()); video=next(session.glob('*.MP4'))
modes=CFG['modes']
models={k:mp.solutions.hands.Hands(static_image_mode=s,max_num_hands=CFG['max_num_hands'],model_complexity=m,min_detection_confidence=CFG['min_detection_confidence'],min_tracking_confidence=CFG['min_tracking_confidence']) for k,(m,s,h) in modes.items()}
cap,_=p.open_video(video); captures={}; results={k:{} for k in modes}; diffs=[]
for i in range(CFG['last_frame']+1):
 ok,f=cap.read(); assert ok
 for name,(m,s,h) in modes.items():
  im=cv2.resize(f,(round(f.shape[1]*h/f.shape[0]),h)); r=models[name].process(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
  pts=None if not r.multi_hand_landmarks else [[v.x*f.shape[1],v.y*f.shape[0]] for v in r.multi_hand_landmarks[0].landmark]
  if i>=CFG['record_from_frame']:
   results[name][i]={'points':pts,'hand':r.multi_handedness[0].classification[0].label if pts else None}
   if name=='lite480' and pts and c['frames'][i][1]: diffs.append(float(np.linalg.norm(np.array(pts[4])-c['frames'][i][1])))
 if i in CFG['capture_frames']: captures[i]=f
 if i%200==0: print(i,flush=True)
cap.release()
for m in models.values():m.close()
print('cached reproduction max error',max(diffs),flush=True)
out=OUT/'rerun';out.mkdir(exist_ok=False)
(out/'results.json').write_text(json.dumps(results))
for i,f in captures.items():
 tiles=[]
 for name in modes:
  im=f.copy(); pts=results[name][i]['points']
  if pts:
   for a,b in mp.solutions.hands.HAND_CONNECTIONS:cv2.line(im,tuple(map(round,pts[a])),tuple(map(round,pts[b])),(160,160,160),2)
   for j,pt in enumerate(pts):
    xy=tuple(map(round,pt));color=(0,255,0) if j==4 else (255,70,200) if j==20 else (30,180,255)
    cv2.circle(im,xy,5,color,2);cv2.putText(im,str(j),(xy[0]+5,xy[1]),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
  x1,y1,x2,y2=CFG['display_crop_xyxy']; tile=im[y1:y2,x1:x2].copy(); cv2.rectangle(tile,(0,0),(600,40),(0,0,0),-1)
  cv2.putText(tile,f'{name} f{i} '+str(results[name][i]['hand']),(10,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1);tiles.append(tile)
 cv2.imwrite(str(out/f'frame_{i}.jpg'),np.hstack(tiles))
print(out,flush=True)
