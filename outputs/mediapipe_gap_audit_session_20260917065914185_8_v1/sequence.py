"""Sequential baseline/mirror comparison with unchanged model and thresholds."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from thumb_tracking import ThumbTracker
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('session')
parser.add_argument('--low', action='store_true')
parser.add_argument('--max-num-hands', type=int, default=1)
args = parser.parse_args()
session = ROOT / 'dataset' / args.session
out = Path(__file__).parent
config = yaml.safe_load((ROOT / 'config.yaml').read_text())
cache = json.loads((session / '.review_cache.json').read_text())
names = ['baseline', 'mirror'] if args.session.endswith('_8') else ['mirror']
if args.low: names = ['threshold02', 'threshold01'] if args.session.endswith('_8') else ['threshold02']
models = {n: mp.solutions.hands.Hands(model_complexity=1, max_num_hands=args.max_num_hands,
          min_detection_confidence=(.2 if n=='threshold02' else .1 if n=='threshold01' else .5),
          min_tracking_confidence=(.2 if n=='threshold02' else .1 if n=='threshold01' else .5)) for n in names}
trackers = {n: ThumbTracker(config['tracking']['continuity']) for n in names}
rows = {n: [] for n in names}
cap = cv2.VideoCapture(str(next(session.glob('*.MP4'))))
i = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    small = cv2.resize(frame, (853, 480))
    for name in names:
        inp = cv2.flip(small, 1) if name == 'mirror' else small
        result = models[name].process(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))
        raw = None
        if result.multi_hand_landmarks:
            tip = result.multi_hand_landmarks[0].landmark[4]
            raw = [(1-tip.x if name == 'mirror' else tip.x)*frame.shape[1], tip.y*frame.shape[0]]
        time = cache['frames'][i][0]
        point, source = trackers[name].update(frame, raw, time)
        rows[name].append([time, raw, point, source])
    i += 1
    if i % 500 == 0: print(args.session, i, flush=True)
cap.release()
for m in models.values(): m.close()
assert i == len(cache['frames'])
path = out / f'{args.session}_{"low" if args.low else "sequence"}.json'
path.write_text(json.dumps(rows))
for name, data in rows.items():
    print(name, 'raw', sum(r[1] is not None for r in data), 'corrected', sum(r[2] is not None for r in data), flush=True)
