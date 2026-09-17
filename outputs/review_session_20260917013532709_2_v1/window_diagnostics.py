"""Recount exact window boundaries in integer microseconds, preserving first results."""
import csv
import json
from pathlib import Path

import numpy as np
import yaml

OUT = Path(__file__).resolve().parent
output = OUT / "window_diagnostics.json"
if output.exists():
    raise FileExistsError(output)
cfg = yaml.safe_load((OUT / "config.yaml").read_text())
summary = json.loads((OUT / "summary.json").read_text())
diag = json.loads((OUT / "diagnostics.json").read_text())
with (OUT / "frame_diagnostics.csv").open() as f:
    rows = list(csv.DictReader(f))
times_us = np.rint(np.array([float(r["video_time_ms"]) for r in rows]) * 1000).astype(np.int64)
matched = next(c for c in diag["flash_checks"] if c["count"] == summary["touch_count"])
result = {"source": "frame_diagnostics.csv", "flash_threshold_gray": matched["threshold_gray"],
          "interval": "[flash-window, flash)", "comparison_time_unit": "integer microseconds",
          "reason": "Avoid sub-microsecond floating-point exclusion at exact 200 ms boundaries; preserve diagnostics.json.",
          "windows_are_not_centerline_crossing": True, "model_checks": {}}
for mode in ("full", "crop"):
    valid = np.array([bool(int(r[mode + "_detected"])) for r in rows])
    windows = []
    for window in cfg["review"]["diagnostic_windows_ms"]:
        total, detected, trials_any, trials_two = 0, 0, 0, 0
        for event in matched["events"]:
            flash_us = int(round(event["video_time_ms"] * 1000))
            mask = (times_us >= flash_us - window * 1000) & (times_us < flash_us)
            n = int(valid[mask].sum())
            total += int(mask.sum())
            detected += n
            trials_any += n > 0
            trials_two += n >= 2
        windows.append({"before_flash_ms": window, "total_frames": total, "detected_frames": detected,
                        "coverage": detected / total, "trials_with_any_detection": trials_any,
                        "trials_with_at_least_two_detections": trials_two, "total_trials": summary["touch_count"]})
    missing_runs, current = [], []
    for i, ok in enumerate(valid):
        if not ok:
            current.append(i)
        elif current:
            missing_runs.append(current)
            current = []
    if current:
        missing_runs.append(current)
    longest = max(missing_runs, key=len) if missing_runs else []
    result["model_checks"][mode] = {"pre_flash_windows": windows,
        "longest_missing_run": {"frames": len(longest), "start_frame": longest[0] if longest else None,
                                "end_frame": longest[-1] if longest else None,
                                "nominal_duration_ms": len(longest) * 1000 / summary["video"]["fps"]}}
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
