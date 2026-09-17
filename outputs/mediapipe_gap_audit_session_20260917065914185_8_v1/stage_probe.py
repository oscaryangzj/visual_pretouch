"""Inspect exposed palm/landmark graph outputs separately on loss frames."""
import json
from pathlib import Path
import cv2
from mediapipe.python.solution_base import SolutionBase
OUT=Path(__file__).parent
results=[]
for detection, presence in [(.5,.5),(.2,.2),(.1,.05)]:
    model=SolutionBase(binary_graph_path='mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb',
      side_inputs={'model_complexity':1,'num_hands':1,'use_prev_landmarks':False},
      calculator_params={'palmdetectioncpu__TensorsToDetectionsCalculator.min_score_thresh':detection,
                        'handlandmarkcpu__ThresholdingCalculator.threshold':presence},
      outputs=['palm_detections','multi_hand_landmarks','hand_rects_from_palm_detections',
               'handlandmarkcpu__hand_presence_score'])
    for i in ['old_trial21',0,4226,4300,4436,4470,4490,4500,4700,4900,4969,5289]:
        frame=cv2.imread(str(OUT / ('old_trial21.png' if isinstance(i,str) else f'original_{i}.png')))
        result=model.process({'image':cv2.cvtColor(cv2.resize(frame,(853,480)),cv2.COLOR_BGR2RGB)})
        row={'frame':i,'detection_threshold':detection,'presence_threshold':presence,
             'palm_scores':[list(p.score) for p in result.palm_detections or []],
             'hand_presence_score':result.handlandmarkcpu__hand_presence_score,
             'hands':[[[p.x,p.y] for p in h.landmark] for h in result.multi_hand_landmarks or []]}
        results.append(row)
        print(i,detection,presence,'palms',row['palm_scores'],'presence',row['hand_presence_score'],
              'hands',len(row['hands']),flush=True)
    model.close()
(OUT/'stage_probes.json').write_text(json.dumps(results))
