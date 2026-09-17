"""Independent static-image probes of cached misses; never alters session data."""
import json
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
selected = [0, 4226, 4300, 4436, 4470, 4490, 4500, 4700, 4900, 4969, 5289]
session = ROOT / 'dataset/session_20260917065914185_8'
cache = json.loads((session / '.review_cache.json').read_text())
variants = [('full480', 480, .5, False), ('full1080', 1080, .5, False),
            ('low480', 480, .2, False), ('mirror480', 480, .5, True)]
models = {name: mp.solutions.hands.Hands(static_image_mode=True, model_complexity=1,
          min_detection_confidence=confidence) for name, _, confidence, _ in variants}
cap = cv2.VideoCapture(str(next(session.glob('*.MP4'))))
results, panels = [], []
for i in range(max(selected)+1):
    ok, frame = cap.read()
    if not ok: raise RuntimeError(f'Decode stopped {i}')
    if i not in selected: continue
    row = {'frame': i, 'cached_raw': cache['raw_frames'][i], 'variants': {}}
    panel = cv2.resize(frame, (960, 540))
    cv2.putText(panel, f'frame {i}', (10, 25), 0, .7, (255,255,255), 2)
    for index, (name, height, _, mirror) in enumerate(variants):
        small = cv2.resize(frame, (round(frame.shape[1]*height/frame.shape[0]), height))
        if mirror: small = cv2.flip(small, 1)
        result = models[name].process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        hands = []
        for hand in result.multi_hand_landmarks or []:
            points = [[(1-p.x if mirror else p.x)*frame.shape[1], p.y*frame.shape[0]] for p in hand.landmark]
            hands.append(points)
            color = [(0,255,0),(255,100,0),(0,255,255),(255,0,255)][index]
            tip = tuple(np.round(np.array(points[4])*.5).astype(int))
            cv2.circle(panel, tip, 8+index*3, color, 2)
        row['variants'][name] = hands
        cv2.putText(panel, f'{name}: {len(hands)}', (10, 55+index*25), 0, .6,
                    [(0,255,0),(255,100,0),(0,255,255),(255,0,255)][index], 2)
    results.append(row)
    panels.append(panel)
    print(i, {k: len(v) for k,v in row['variants'].items()}, flush=True)
cap.release()
for model in models.values(): model.close()
(OUT / 'static_probes.json').write_text(json.dumps(results))
for i, panel in enumerate(panels): cv2.imwrite(str(OUT / f'frame_{selected[i]}.jpg'), panel)
