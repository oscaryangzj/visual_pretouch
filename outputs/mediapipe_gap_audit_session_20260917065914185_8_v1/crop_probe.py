"""Selected-frame crop/rotation/lite diagnostics; ROI is diagnostic only."""
import json
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
OUT = Path(__file__).parent
selected = [4300, 4436, 4470, 4490, 4500, 4700, 4900]
variants = [('crop_full',1,.5,'crop',0), ('crop_low',1,.1,'crop',0),
            ('crop_lite',0,.3,'crop',0), ('lite',0,.3,'full',0),
            ('rot90',1,.3,'full',1), ('rot180',1,.3,'full',2), ('rot270',1,.3,'full',3)]
models = {n:mp.solutions.hands.Hands(static_image_mode=True,model_complexity=c,
            min_detection_confidence=d) for n,c,d,_,_ in variants}
results=[]
for index in selected:
    frame=cv2.imread(str(OUT/f'original_{index}.png'))
    row={'frame':index,'variants':{}}
    for name,_,_,kind,rotation in variants:
        inp=frame[300:1080,500:1260] if kind=='crop' else cv2.resize(frame,(853,480))
        if kind=='crop': inp=cv2.resize(inp,(468,480))
        inp=np.ascontiguousarray(np.rot90(inp,rotation))
        result=models[name].process(cv2.cvtColor(inp,cv2.COLOR_BGR2RGB))
        points=[[[p.x,p.y] for p in h.landmark] for h in result.multi_hand_landmarks or []]
        row['variants'][name]=points
    results.append(row)
    print(index,{k:len(v) for k,v in row['variants'].items()},flush=True)
for m in models.values():m.close()
(OUT/'crop_probes.json').write_text(json.dumps(results))
