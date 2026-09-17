"""Render unchanged-video samples for visual checking of candidate thresholds."""
from pathlib import Path
import cv2
import json
import numpy as np
OUT = Path(__file__).parent
rows=json.loads((OUT/'session_20260917065914185_8_low.json').read_text())
base=json.loads((OUT/'session_20260917065914185_8_sequence.json').read_text())['baseline']
panels=[]
for i in [4436,4470,4490,4500,4700,4900]:
    f=cv2.imread(str(OUT/f'original_{i}.png'))
    p=[]
    for name,seq in [('baseline',base),*rows.items()]:
        # Fixed diagnostic crop only; never used as model input.
        panel=f[280:1080,500:1300].copy()
        for point,color in [(seq[i][1],(0,255,0)),(seq[i][2],(255,100,0))]:
            if point is not None:
                pos=tuple(np.round(np.array(point)-[500,280]).astype(int))
                cv2.circle(panel,pos,12,color,3)
        cv2.putText(panel,f'{i} {name} {seq[i][3]}',(10,30),0,.7,(0,255,255),2)
        p.append(cv2.resize(panel,(400,400)))
    panels.append(np.hstack(p))
cv2.imwrite(str(OUT/'threshold_comparison.jpg'),np.vstack(panels))
