"""Review pre-flash samples across all trials, not just model coverage."""
from pathlib import Path
import json
import cv2
import numpy as np
import sys
OUT=Path(__file__).parent
ROOT=OUT.parents[1]
session=ROOT/'dataset'/sys.argv[1]
cache=json.loads((session/'.review_cache.json').read_text())
low=json.loads((OUT/f'{session.name}_low.json').read_text())['threshold02']
samples={e['frame_index']-12:trial for trial,e in enumerate(cache['events'])}
cap=cv2.VideoCapture(str(next(session.glob('*.MP4'))))
panels=[]
for i in range(max(samples)+1):
    ok,f=cap.read()
    if not ok:raise RuntimeError(i)
    if i not in samples:continue
    for point,color in [(low[i][1],(0,255,0)),(low[i][2],(255,100,0))]:
        if point is not None:cv2.circle(f,tuple(np.round(point).astype(int)),12,color,3)
    panel=cv2.resize(f[280:1080,500:1300],(320,320))
    cv2.putText(panel,f'{samples[i]+1} {low[i][3]}',(8,22),0,.55,(0,255,255),2)
    panels.append(panel)
cap.release()
for page in range(3):
    group=panels[page*10:(page+1)*10]
    cv2.imwrite(str(OUT/f'{session.name}_spotcheck_{page+1}.jpg'),
                np.vstack([np.hstack(group[i:i+5]) for i in (0,5)]))
