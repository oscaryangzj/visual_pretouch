#!/usr/bin/env python3
"""Small, offline P0 pipeline for visual touch prediction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def validate_p0_task(task: dict) -> None:
    expected = {"hand": "left", "crossing_direction": "left_to_right", "target_side": "right"}
    for key, value in expected.items():
        if task.get(key) != value:
            raise ValueError(f"P0 requires task.{key}={value!r}; got {task.get(key)!r}")


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def as_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=-1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def ensure_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {path}")
    path.mkdir(parents=True, exist_ok=True)


def ensure_not_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {path}")


def open_video(path: Path) -> tuple[cv2.VideoCapture, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    orientation = float(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    auto_enabled = cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    if not auto_enabled and not math.isclose(orientation % 360.0, 0.0):
        cap.release()
        raise RuntimeError("video backend cannot apply orientation metadata")
    return cap, orientation


def video_meta(cap: cv2.VideoCapture) -> tuple[float, int, int]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, width, height


def frame_time_ms(cap: cv2.VideoCapture, index: int, fps: float) -> float:
    pos = cap.get(cv2.CAP_PROP_POS_MSEC)
    return float(pos) if pos > 0 else index * 1000.0 / fps


def validate_calibration_size(calibration: dict, width: int, height: int) -> None:
    expected = [width, height]
    if calibration.get("image_size_px") != expected:
        raise ValueError(
            f"calibration image_size_px={calibration.get('image_size_px')} "
            f"does not match oriented video size {expected}"
        )


def project(H: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    points = np.asarray(uvs, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, H).reshape(-1, 2)


def parse_points(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as f:
        points = np.asarray(json.load(f), dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("points file must contain four [x, y] points")
    return points


def cmd_calibrate(args: argparse.Namespace) -> None:
    cap, orientation = open_video(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {args.frame_index}")
    height, width = frame.shape[:2]
    if args.points_file:
        image_points = parse_points(args.points_file)
    else:
        image_points = []
        window = "click screen corners: TL, TR, BR, BL"
        display = frame.copy()

        def on_click(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN and len(image_points) < 4:
                image_points.append([x, y])
                cv2.circle(display, (x, y), 7, (0, 0, 255), -1)
                cv2.putText(display, str(len(image_points)), (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_click)
        while len(image_points) < 4:
            cv2.imshow(window, display)
            if cv2.waitKey(20) & 0xFF == 27:
                cv2.destroyAllWindows()
                raise RuntimeError("calibration cancelled")
        cv2.destroyAllWindows()
        image_points = np.asarray(image_points, dtype=np.float32)
    screen_points = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H, _ = cv2.findHomography(image_points, screen_points, method=0)
    if H is None:
        raise RuntimeError("homography estimation failed")
    result = {
        "schema_version": 1,
        "session_id": args.session_id,
        "reference_frame_index": args.frame_index,
        "image_size_px": [width, height],
        "video_orientation_meta_degrees": orientation,
        "image_points_px": image_points.tolist(),
        "screen_points_uv": screen_points.tolist(),
        "homography_image_to_screen": H.tolist(),
        "screen_region": "browser operation area",
    }
    ensure_not_exists(args.output)
    write_json(args.output, result)
    print(f"saved calibration: {args.output}")


def cmd_extract(args: argparse.Namespace) -> None:
    import mediapipe as mp

    cfg = load_yaml(args.config)
    tracking = cfg.get("tracking", {})
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        raise RuntimeError("P0 requires MediaPipe 0.10.x with mp.solutions.hands; reinstall requirements.txt")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    H = np.asarray(calibration["homography_image_to_screen"], dtype=np.float32)
    cap, _ = open_video(args.video)
    fps, width, height = video_meta(cap)
    validate_calibration_size(calibration, width, height)
    fields = ["schema_version", "frame_index", "video_time_ms", "thumb_tip_x_px", "thumb_tip_y_px",
              "thumb_u", "thumb_v", "valid", "invalid_reason"]
    rows = []
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=int(tracking.get("max_num_hands", 1)),
        model_complexity=int(tracking.get("model_complexity", 0)),
        min_detection_confidence=float(tracking.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(tracking.get("min_tracking_confidence", 0.5)),
    )
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_time_ms(cap, index, fps)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            row = {"schema_version": 1, "frame_index": index, "video_time_ms": f"{timestamp:.3f}",
                   "thumb_tip_x_px": "", "thumb_tip_y_px": "", "thumb_u": "", "thumb_v": "",
                   "valid": 0, "invalid_reason": "no_hand"}
            if result.multi_hand_landmarks:
                landmark = result.multi_hand_landmarks[0].landmark[4]
                x, y = landmark.x * width, landmark.y * height
                uv = project(H, np.asarray([[x, y]], dtype=np.float32))[0]
                row.update({"thumb_tip_x_px": f"{x:.3f}", "thumb_tip_y_px": f"{y:.3f}",
                            "thumb_u": f"{uv[0]:.6f}", "thumb_v": f"{uv[1]:.6f}",
                            "valid": 1, "invalid_reason": ""})
            rows.append(row)
            index += 1
    finally:
        hands.close()
        cap.release()
    ensure_not_exists(args.output)
    write_csv(args.output, rows, fields)
    print(f"saved {len(rows)} trajectory rows: {args.output}")


def screen_mask(calibration: dict, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    points = np.asarray(calibration["image_points_px"], dtype=np.int32)
    cv2.fillConvexPoly(mask, points, 255)
    return mask


def cmd_align(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    acfg = cfg.get("alignment", {})
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    touches = read_csv(args.touches)
    cap, _ = open_video(args.video)
    fps, width, height = video_meta(cap)
    validate_calibration_size(calibration, width, height)
    mask = None
    events = []
    active = False
    index = 0
    previous = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mask is None:
            mask = screen_mask(calibration, frame.shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(cv2.mean(gray, mask=mask)[0])
        rise = score - previous if previous is not None else 0
        if (not active and score >= float(acfg.get("flash_threshold_gray", 245))
                and rise >= float(acfg.get("flash_rise_threshold_gray", 20))):
            events.append((index, frame_time_ms(cap, index, fps), score))
            active = True
        if active and score < float(acfg.get("flash_threshold_gray", 245)) - 10:
            active = False
        previous = score
        index += 1
    cap.release()
    fields = ["schema_version", "trial_id", "touch_index", "flash_frame", "flash_time_video_ms",
              "touch_time_video_ms", "alignment_method", "uncertainty_ms", "status"]
    rows = []
    delay = float(acfg.get("flash_after_touch_delay_ms", 0))
    uncertainty = float(acfg.get("uncertainty_ms", 50))
    for i, touch in enumerate(touches):
        row = {"schema_version": 1, "trial_id": touch.get("trial_id", f"trial_{i:04d}"),
               "touch_index": touch.get("touch_index", 0), "flash_frame": "",
               "flash_time_video_ms": "", "touch_time_video_ms": "",
               "alignment_method": "screen_flash_sequence", "uncertainty_ms": f"{uncertainty:.3f}",
               "status": "missing_flash"}
        if i < len(events):
            flash_frame, flash_time, _ = events[i]
            row.update({"flash_frame": flash_frame, "flash_time_video_ms": f"{flash_time:.3f}",
                        "touch_time_video_ms": f"{flash_time - delay:.3f}", "status": "matched"})
        rows.append(row)
    ensure_not_exists(args.output)
    write_csv(args.output, rows, fields)
    print(f"detected {len(events)} flashes for {len(touches)} touches; saved {args.output}")


def trajectory_rows(path: Path) -> list[dict]:
    rows = read_csv(path)
    rows.sort(key=lambda row: as_int(row.get("frame_index")))
    return rows


def latest_valid_frame(rows: list[dict], start_ms: float, cutoff_ms: float) -> dict | None:
    latest = None
    for row in rows:
        timestamp = as_float(row.get("video_time_ms"))
        if timestamp < start_ms:
            continue
        if timestamp > cutoff_ms:
            break
        if row.get("valid") == "1" and math.isfinite(as_float(row.get("thumb_u"))):
            latest = row
    return latest


def find_crossing(rows: list[dict], start_ms: float, end_ms: float, centerline: float,
                  margin: float, max_gap_ms: float) -> dict | None:
    previous = None
    for row in rows:
        t = as_float(row.get("video_time_ms"))
        if not math.isfinite(t) or t < start_ms:
            continue
        if t > end_ms:
            break
        if str(row.get("valid", "0")) != "1":
            continue
        u = as_float(row.get("thumb_u"))
        if not math.isfinite(u):
            continue
        if (previous is not None and t - previous["time"] <= max_gap_ms
                and previous["u"] <= centerline - margin and u > centerline + margin):
            return row
        previous = {"time": t, "u": u}
    return None


def linear_prediction(history: list[dict], window_ms: float, horizon_ms: float, min_points: int):
    values = []
    if not history:
        return None
    end = as_float(history[-1].get("video_time_ms"))
    for row in history:
        t, u, v = as_float(row.get("video_time_ms")), as_float(row.get("thumb_u")), as_float(row.get("thumb_v"))
        if math.isfinite(t) and math.isfinite(u) and math.isfinite(v) and end - t <= window_ms:
            values.append((t, u, v))
    if len(values) < min_points:
        return None
    t = np.asarray([x[0] for x in values], dtype=np.float64)
    t -= t[-1]
    u = np.asarray([x[1] for x in values], dtype=np.float64)
    v = np.asarray([x[2] for x in values], dtype=np.float64)
    u_slope, u_intercept = np.polyfit(t, u, 1)
    v_slope, v_intercept = np.polyfit(t, v, 1)
    h = horizon_ms
    return float(u_slope * h + u_intercept), float(v_slope * h + v_intercept)


def baseline_mean(touches: list[dict]) -> tuple[float, float] | None:
    points = [(as_float(r.get("touch_u")), as_float(r.get("touch_v"))) for r in touches]
    points = [(u, v) for u, v in points if math.isfinite(u) and math.isfinite(v)]
    if not points:
        return None
    return float(np.mean([p[0] for p in points])), float(np.mean([p[1] for p in points]))


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    task, pcfg, tcfg = cfg.get("task", {}), cfg.get("prediction", {}), cfg.get("tracking", {})
    validate_p0_task(task)
    trajectories = trajectory_rows(args.trajectory)
    alignment = read_csv(args.alignment)
    alignment.sort(key=lambda row: as_int(row.get("touch_index")))
    prior_touches = read_csv(args.baseline_touches) if args.baseline_touches else []
    prior_touches = [r for r in prior_touches
                     if float(task.get("centerline_u", 0.5)) < as_float(r.get("touch_u")) <= 1
                     and 0 <= as_float(r.get("touch_v")) <= 1]
    mean_point = baseline_mean(prior_touches)
    fields = ["schema_version", "trial_id", "method", "prediction_mode", "prediction_frame",
              "prediction_time_ms", "pred_u", "pred_v", "status", "failure_reason"]
    output = []
    centerline = float(task.get("centerline_u", 0.5))
    margin = float(pcfg.get("crossing_margin_u", 0.0))
    max_gap = float(tcfg.get("max_gap_ms", 100))
    reset_gap = float(pcfg.get("reset_gap_ms", 300))
    previous_touch = -float("inf")
    methods = [("b0_mean", mean_point), ("b1_current", None), ("b2_linear", None)]
    for i, row in enumerate(alignment):
        trial_id = row.get("trial_id", f"trial_{i:04d}")
        touch_time = as_float(row.get("touch_time_video_ms"))
        start = 0.0 if i == 0 or not math.isfinite(previous_touch) else previous_touch + reset_gap
        previous_touch = touch_time if math.isfinite(touch_time) else previous_touch
        crossing = None
        if row.get("status") == "matched" and math.isfinite(touch_time):
            crossing = find_crossing(trajectories, start, touch_time, centerline, margin, max_gap)
        reason = ""
        if row.get("status") != "matched":
            reason = "alignment_failed"
        elif crossing is None:
            reason = "no_left_to_right_crossing_before_touch"
        modes = [("centerline_crossing", crossing, reason)]
        for lead_ms in cfg.get("evaluation", {}).get("lead_times_ms", []):
            mode = f"touch_minus_{int(lead_ms)}ms"
            event = None
            mode_reason = ""
            if row.get("status") != "matched" or not math.isfinite(touch_time):
                mode_reason = "alignment_failed"
            else:
                event = latest_valid_frame(trajectories, start, touch_time - float(lead_ms))
                if event is None:
                    mode_reason = "no_valid_frame_at_cutoff"
            modes.append((mode, event, mode_reason))
        for mode, event, mode_reason in modes:
            history = []
            if event:
                frame_index = as_int(event.get("frame_index"))
                history = [r for r in trajectories if as_int(r.get("frame_index")) <= frame_index]
            for method, fixed in methods:
                prediction = fixed
                failure = mode_reason
                if event and method == "b1_current":
                    prediction = (as_float(event.get("thumb_u")), as_float(event.get("thumb_v")))
                    failure = "" if all(math.isfinite(x) for x in prediction) else "invalid_current_projection"
                elif event and method == "b2_linear":
                    prediction = linear_prediction(history, float(pcfg.get("linear_history_ms", 100)),
                                                   float(pcfg.get("linear_horizon_ms", 100)),
                                                   int(pcfg.get("linear_min_points", 3)))
                    failure = "" if prediction is not None else "insufficient_history"
                elif method == "b0_mean" and fixed is None:
                    failure = "baseline_touches_required_for_b0"
                ok = prediction is not None and not failure
                output.append({
                    "schema_version": 1, "trial_id": trial_id, "method": method,
                    "prediction_mode": mode, "prediction_frame": event.get("frame_index", "") if event else "",
                    "prediction_time_ms": event.get("video_time_ms", "") if event else "",
                    "pred_u": f"{prediction[0]:.6f}" if ok else "",
                    "pred_v": f"{prediction[1]:.6f}" if ok else "",
                    "status": "ok" if ok else "failed", "failure_reason": failure,
                })
    ensure_not_exists(args.output)
    write_csv(args.output, output, fields)
    print(f"saved {len(output)} predictions: {args.output}")


def valid_row_status(row: dict) -> bool:
    return row.get("status") == "ok" and math.isfinite(as_float(row.get("pred_u")))


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q, method="linear")) if values else None


def evaluation_device(cfg: dict, touches: list[dict], override: str | None = None) -> tuple[str | None, dict]:
    recorded = {r["device_id"] for r in touches if r.get("device_id")}
    if len(recorded) > 1:
        raise ValueError("evaluate requires touch records from a single device")
    recorded_id = next(iter(recorded), None)
    if override and recorded_id and override != recorded_id:
        raise ValueError(f"--device={override!r} conflicts with recorded device_id={recorded_id!r}")
    profiles = cfg.get("devices")
    if profiles:
        device_id = override or recorded_id or cfg.get("device")
        if not isinstance(device_id, str) or device_id not in profiles:
            raise ValueError(f"unknown device {device_id!r}; choose a key from config.devices")
        profile = profiles[device_id]
    else:
        # Support config snapshots from the original single-device pipeline.
        if override or recorded_id:
            raise ValueError("device_id requires a config with per-device profiles")
        device_id = None
        profile = {**cfg.get("device", {}), **cfg.get("evaluation", {})}
    for key in ("region_width_mm", "region_height_mm"):
        value = profile.get(key)
        if value is not None and (not math.isfinite(as_float(value)) or as_float(value) <= 0):
            raise ValueError(f"{key} must be a positive number or null")
    return device_id, profile


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    task = cfg.get("task", {})
    validate_p0_task(task)
    ecfg = cfg.get("evaluation", {})
    predictions = read_csv(args.predictions)
    touch_rows = read_csv(args.touches)
    device_id, device_cfg = evaluation_device(cfg, touch_rows, args.device)
    touches = {r.get("trial_id"): r for r in touch_rows}
    alignment = {r.get("trial_id"): r for r in read_csv(args.alignment)}
    details = []
    groups = sorted({(r.get("method", ""), r.get("prediction_mode", "")) for r in predictions})
    width_mm = device_cfg.get("region_width_mm")
    height_mm = device_cfg.get("region_height_mm")
    physical = width_mm is not None and height_mm is not None
    ensure_new(args.output)
    centerline = float(task.get("centerline_u", 0.5))
    for row in predictions:
        gt = touches.get(row.get("trial_id"), {})
        al = alignment.get(row.get("trial_id"), {})
        gu, gv = as_float(gt.get("touch_u")), as_float(gt.get("touch_v"))
        pu, pv = as_float(row.get("pred_u")), as_float(row.get("pred_v"))
        du, dv = pu - gu, pv - gv
        normalized = math.hypot(du, dv) if all(math.isfinite(x) for x in (du, dv)) else float("nan")
        error_mm = math.hypot(du * float(width_mm), dv * float(height_mm)) if physical and math.isfinite(normalized) else float("nan")
        lead = as_float(al.get("touch_time_video_ms")) - as_float(row.get("prediction_time_ms"))
        valid_gt = math.isfinite(gu) and math.isfinite(gv) and centerline < gu <= 1 and 0 <= gv <= 1
        valid_eval = valid_row_status(row) and al.get("status") == "matched" and valid_gt and lead >= 0
        details.append({**row, "gt_u": f"{gu:.6f}" if math.isfinite(gu) else "", "gt_v": f"{gv:.6f}" if math.isfinite(gv) else "",
                        "error_normalized": f"{normalized:.6f}" if math.isfinite(normalized) else "",
                        "error_mm": f"{error_mm:.6f}" if math.isfinite(error_mm) else "",
                        "lead_time_ms": f"{lead:.3f}" if math.isfinite(lead) else "",
                        "valid_for_metrics": int(valid_eval)})
    detail_fields = list(details[0].keys()) if details else ["trial_id"]
    write_csv(args.output / "prediction_details.csv", details, detail_fields)
    protocol_trials = {tid for tid, row in touches.items()
                       if centerline < as_float(row.get("touch_u")) <= 1
                       and 0 <= as_float(row.get("touch_v")) <= 1
                       and alignment.get(tid, {}).get("status") == "matched"}
    quality = []
    for tid, touch in touches.items():
        aligned = alignment.get(tid, {})
        quality.append({"schema_version": 1, "trial_id": tid, "stage": "alignment",
                        "status": aligned.get("status", "missing"),
                        "reason": "" if aligned.get("status") == "matched" else "missing_flash",
                        "method": "", "prediction_mode": ""})
        gt_u, gt_v = as_float(touch.get("touch_u")), as_float(touch.get("touch_v"))
        gt_ok = (math.isfinite(gt_u) and math.isfinite(gt_v)
                 and centerline < gt_u <= 1 and 0 <= gt_v <= 1)
        quality.append({"schema_version": 1, "trial_id": tid, "stage": "ground_truth",
                        "status": "valid" if gt_ok else "invalid",
                        "reason": "" if gt_ok else "touch_not_in_right_half",
                        "method": "", "prediction_mode": ""})
    for row in details:
        quality.append({"schema_version": 1, "trial_id": row.get("trial_id", ""), "stage": "prediction",
                        "status": "valid" if row.get("valid_for_metrics") == 1 else "invalid",
                        "reason": "" if row.get("valid_for_metrics") == 1 else row.get("failure_reason", "invalid_for_metrics"),
                        "method": row.get("method", ""), "prediction_mode": row.get("prediction_mode", "")})
    write_csv(args.output / "quality.csv", quality,
              ["schema_version", "trial_id", "stage", "status", "reason", "method", "prediction_mode"])
    metrics = {"schema_version": 1, "physical_error_available": physical,
               "device_id": device_id, "device_model": device_cfg.get("model"),
               "region_width_mm": width_mm, "region_height_mm": height_mm,
               "protocol_valid_trials": len(protocol_trials), "by_prediction_mode": {}}
    radii = [float(x) for x in ecfg.get("hit_radii_mm", [10, 20, 30])]
    for method, mode in groups:
        method_rows = [r for r in details if r.get("method") == method and r.get("prediction_mode") == mode]
        good = [r for r in method_rows if r.get("valid_for_metrics") == 1]
        errors_norm = [as_float(r.get("error_normalized")) for r in good]
        errors_norm = [x for x in errors_norm if math.isfinite(x)]
        errors_mm = [as_float(r.get("error_mm")) for r in good]
        errors_mm = [x for x in errors_mm if math.isfinite(x)]
        leads = [as_float(r.get("lead_time_ms")) for r in good]
        leads = [x for x in leads if math.isfinite(x)]
        summary = {
            "total_trials": len(touches), "success_rows": sum(valid_row_status(r) for r in method_rows),
            "valid_predictions": len(good), "coverage": len(good) / len(protocol_trials) if protocol_trials else None,
            "mean_error_normalized": float(np.mean(errors_norm)) if errors_norm else None,
            "median_error_normalized": float(np.median(errors_norm)) if errors_norm else None,
            "p90_error_normalized": percentile(errors_norm, 90),
            "lead_time_ms": {"mean": float(np.mean(leads)) if leads else None,
                             "median": float(np.median(leads)) if leads else None,
                             "p90": percentile(leads, 90)},
            "hit_rate_all_protocol_trials": {}, "failure_reasons": dict(),
        }
        if physical:
            summary.update({"mean_error_mm": float(np.mean(errors_mm)) if errors_mm else None,
                            "median_error_mm": float(np.median(errors_mm)) if errors_mm else None,
                            "p90_error_mm": percentile(errors_mm, 90)})
        for radius in radii:
            key = f"hit_at_{int(radius)}mm"
            summary[key] = sum(x <= radius for x in errors_mm) / len(errors_mm) if errors_mm else None
            summary["hit_rate_all_protocol_trials"][key] = sum(
                r.get("valid_for_metrics") == 1 and as_float(r.get("error_mm")) <= radius for r in method_rows
            ) / len(protocol_trials) if protocol_trials and physical else None
        for r in method_rows:
            if r.get("status") != "ok" or r.get("valid_for_metrics") != 1:
                reason = r.get("failure_reason") or "invalid_for_metrics"
                summary["failure_reasons"][reason] = summary["failure_reasons"].get(reason, 0) + 1
        metrics["by_prediction_mode"].setdefault(mode, {})[method] = summary
    make_figures(details, groups, args.output / "figures", physical)
    shutil.copy2(args.config, args.output / "config.yaml")
    manifest = build_manifest(args, cfg, [args.config, args.predictions, args.touches, args.alignment])
    manifest.update({"device_id": device_id, "device": device_cfg})
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "metrics.json", metrics)
    print(f"saved metrics and figures: {args.output}")


def build_manifest(args: argparse.Namespace, cfg: dict, inputs: list[Path]) -> dict:
    hashes = {}
    for path in inputs:
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[str(path)] = {"sha256": digest, "bytes": path.stat().st_size}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                                    capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "opencv": cv2.__version__,
                "pyyaml": getattr(yaml, "__version__", "unknown")}
    try:
        import mediapipe
        versions["mediapipe"] = mediapipe.__version__
    except Exception:
        versions["mediapipe"] = "not_imported"
    session_ids = sorted({r.get("session_id") for r in read_csv(args.touches) if r.get("session_id")})
    return {"schema_version": 1, "experiment_id": args.output.name, "session_ids": session_ids,
            "processing_id": "offline_p0", "inputs": hashes, "config_path": "config.yaml",
            "git_commit": commit, "git_dirty": dirty, "command": " ".join(map(str, sys.argv)),
            "versions": versions, "config": cfg}


def make_figures(details: list[dict], groups: list[tuple[str, str]], output: Path, physical: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    for method, mode in groups:
        rows = [r for r in details if r.get("method") == method and r.get("prediction_mode") == mode
                and r.get("valid_for_metrics") == 1]
        label = f"{method}_{mode}"
        leads = [as_float(r.get("lead_time_ms")) for r in rows]
        errors = [as_float(r.get("error_mm" if physical else "error_normalized")) for r in rows]
        pairs = [(x, y) for x, y in zip(leads, errors) if math.isfinite(x) and math.isfinite(y)]
        if pairs:
            x, y = zip(*pairs)
            plt.figure(figsize=(5, 4)); plt.scatter(x, y, s=18); plt.xlabel("lead time (ms)")
            plt.ylabel("error (mm)" if physical else "error (normalized)"); plt.title(label)
            plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(output / f"error_vs_lead_{label}.png", dpi=150); plt.close()
        if physical:
            plt.figure(figsize=(5, 4))
            for radius in (10, 20, 30):
                xs, ys = [], []
                for lead in (50, 100, 150, 200):
                    eligible = [as_float(r.get("error_mm")) <= radius for r in rows if as_float(r.get("lead_time_ms")) >= lead]
                    if eligible: xs.append(lead); ys.append(float(np.mean(eligible)))
                if xs: plt.plot(xs, ys, "o-", label=f"{radius} mm")
            plt.xlabel("at least this much lead time (ms)"); plt.ylabel("hit rate")
            plt.title(f"hit@r vs lead time: {label}"); plt.ylim(0, 1.05); plt.grid(alpha=.25); plt.legend()
            plt.tight_layout(); plt.savefig(output / f"hit_vs_lead_{label}.png", dpi=150); plt.close()
        if rows:
            plt.figure(figsize=(5, 5)); plt.scatter([as_float(r.get("gt_u")) for r in rows], [as_float(r.get("gt_v")) for r in rows], label="actual", s=20)
            plt.scatter([as_float(r.get("pred_u")) for r in rows], [as_float(r.get("pred_v")) for r in rows], label="predicted", s=20)
            plt.xlim(0, 1); plt.ylim(1, 0); plt.xlabel("u"); plt.ylabel("v"); plt.title(label); plt.legend(); plt.grid(alpha=.25)
            plt.tight_layout(); plt.savefig(output / f"prediction_vs_actual_{label}.png", dpi=150); plt.close()


def norm_to_image(inverse_H: np.ndarray, uv: tuple[float, float]) -> tuple[int, int]:
    point = project(inverse_H, np.asarray([uv], dtype=np.float32))[0]
    return int(round(point[0])), int(round(point[1]))


def cmd_render(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    render_cfg = cfg.get("render", {})
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    H = np.asarray(calibration["homography_image_to_screen"], dtype=np.float32)
    inverse_H = np.linalg.inv(H)
    image_points = np.asarray(calibration["image_points_px"], dtype=np.int32)
    trajectory = {as_int(r.get("frame_index")): r for r in trajectory_rows(args.trajectory)}
    predictions = [r for r in read_csv(args.predictions)
                   if r.get("method") == args.method and r.get("prediction_mode") == args.prediction_mode]
    touches = {r.get("trial_id"): r for r in read_csv(args.touches)}
    alignment = {r.get("trial_id"): r for r in read_csv(args.alignment)}
    predictions.sort(key=lambda r: as_int(r.get("prediction_frame")))
    ensure_not_exists(args.output)
    cap, _ = open_video(args.video)
    fps, width, height = video_meta(cap)
    validate_calibration_size(calibration, width, height)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {args.output}")
    index = 0
    active = None
    tail = int(args.trajectory_tail if args.trajectory_tail is not None
               else render_cfg.get("trajectory_tail", 18))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while predictions and as_int(predictions[0].get("prediction_frame")) <= index:
            active = predictions.pop(0)
        cv2.polylines(frame, [image_points], True, (255, 200, 0), 2)
        center = norm_to_image(inverse_H, (0.5, 0.0)), norm_to_image(inverse_H, (0.5, 1.0))
        cv2.line(frame, center[0], center[1], (255, 0, 255), 2)
        recent = []
        for n in range(max(0, index - tail), index + 1):
            r = trajectory.get(n)
            if r and r.get("valid") == "1":
                recent.append((int(as_float(r.get("thumb_tip_x_px"))), int(as_float(r.get("thumb_tip_y_px")))))
        if len(recent) > 1:
            cv2.polylines(frame, [np.asarray(recent, dtype=np.int32)], False, (0, 220, 255), 3)
        r = trajectory.get(index)
        if r and r.get("valid") == "1":
            cv2.circle(frame, (int(as_float(r.get("thumb_tip_x_px"))), int(as_float(r.get("thumb_tip_y_px")))), 8, (0, 255, 0), -1)
        text = [f"frame={index}  {args.method}/{args.prediction_mode}"]
        if active and active.get("status") == "ok":
            tid = active.get("trial_id"); gt = touches.get(tid, {}); al = alignment.get(tid, {})
            pred_xy = norm_to_image(inverse_H, (as_float(active.get("pred_u")), as_float(active.get("pred_v"))))
            cv2.drawMarker(frame, pred_xy, (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
            if math.isfinite(as_float(gt.get("touch_u"))):
                gt_xy = norm_to_image(inverse_H, (as_float(gt.get("touch_u")), as_float(gt.get("touch_v"))))
                cv2.circle(frame, gt_xy, 12, (255, 0, 0), 3)
            lead = as_float(al.get("touch_time_video_ms")) - as_float(active.get("prediction_time_ms"))
            error = math.hypot(as_float(active.get("pred_u")) - as_float(gt.get("touch_u")),
                               as_float(active.get("pred_v")) - as_float(gt.get("touch_v")))
            text += [f"{tid}  prediction=red  actual=blue", f"lead={lead:.1f} ms  error={error:.4f} normalized"]
        for j, line in enumerate(text):
            cv2.putText(frame, line, (20, 35 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2)
        writer.write(frame)
        index += 1
    cap.release(); writer.release()
    print(f"saved demo video: {args.output}")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("calibrate", help="manually click four screen corners")
    p.add_argument("--video", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--session-id", default="unknown"); p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--points-file", type=Path); p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("extract", help="extract thumb tip and screen coordinates")
    add_common(p); p.add_argument("--video", type=Path, required=True); p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.set_defaults(func=cmd_extract)
    p = sub.add_parser("align", help="match screen flashes to touch rows")
    add_common(p); p.add_argument("--video", type=Path, required=True); p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--touches", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=cmd_align)
    p = sub.add_parser("predict", help="detect crossing and run P0 baselines")
    add_common(p); p.add_argument("--trajectory", type=Path, required=True); p.add_argument("--alignment", type=Path, required=True)
    p.add_argument("--baseline-touches", type=Path); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=cmd_predict)
    p = sub.add_parser("evaluate", help="compute metrics and figures")
    p.add_argument("--device", help="config.devices key; inferred from touches when available")
    add_common(p); p.add_argument("--predictions", type=Path, required=True); p.add_argument("--touches", type=Path, required=True)
    p.add_argument("--alignment", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("render", help="render an auditable overlay video")
    add_common(p); p.add_argument("--video", type=Path, required=True); p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True); p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--touches", type=Path, required=True); p.add_argument("--alignment", type=Path, required=True)
    p.add_argument("--method", default="b2_linear"); p.add_argument("--prediction-mode", default="centerline_crossing")
    p.add_argument("--trajectory-tail", type=int)
    p.add_argument("--output", type=Path, required=True); p.set_defaults(func=cmd_render)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
