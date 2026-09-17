#!/usr/bin/env python3
"""Review one session touch by touch without changing source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml

import visual_pretouch as pipeline


ROOT = Path(__file__).resolve().parents[1]
WINDOW = "visual_pretouch review"
REASONS = {
    ord("1"): "missing_tip",
    ord("2"): "wrong_tip",
    ord("3"): "wrong_hand",
    ord("4"): "occluded",
    ord("5"): "alignment_error",
    ord("6"): "protocol_violation",
    ord("7"): "other",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def load_touches(session_dir: Path) -> list[dict]:
    path = one_file(session_dir, "*_touches.csv")
    rows = pipeline.read_csv(path)
    if not rows:
        raise ValueError(f"touch CSV is empty: {path}")
    return rows


def load_tracking(path: Path, frame_count: int) -> list[dict]:
    source = pipeline.read_csv(path)
    by_frame = {}
    for row in source:
        index = pipeline.as_int(row.get("frame_index"))
        if index < 0 or index in by_frame:
            raise ValueError(f"invalid or duplicate frame_index in {path}: {index}")
        detected = row.get("valid") in {"1", "true", "True"}
        x = row.get("thumb_tip_x_px", "")
        y = row.get("thumb_tip_y_px", "")
        if "full_detected" in row:
            detected = row["full_detected"] == "1"
            x, y = row.get("full_tip_x_native_px", ""), row.get("full_tip_y_native_px", "")
        elif "hand_detected" in row:
            detected = row["hand_detected"] == "1"
        by_frame[index] = {"frame_index": index, "video_time_ms": row.get("video_time_ms", ""),
                           "thumb_tip_x_px": x, "thumb_tip_y_px": y,
                           "valid": int(detected),
                           "invalid_reason": "" if detected else "no_hand"}
    if sorted(by_frame) != list(range(frame_count)):
        raise ValueError(f"tracking must contain exactly frame indices 0..{frame_count - 1}: {path}")
    return [by_frame[index] for index in range(frame_count)]


def load_events(path: Path, touch_count: int) -> list[dict]:
    if path.suffix.lower() == ".csv":
        rows = pipeline.read_csv(path)
        events = [{"frame_index": pipeline.as_int(row.get("flash_frame")),
                   "video_time_ms": pipeline.as_float(row.get("flash_time_video_ms")),
                   "status": row.get("status", "matched")} for row in rows]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "flash_checks" in data:
            checks = [check for check in data["flash_checks"] if check.get("events")]
            matching = [check for check in checks if check.get("count") == touch_count]
            data = (matching or checks)[0]["events"]
        elif isinstance(data, dict):
            data = data.get("events", [])
        events = [{"frame_index": pipeline.as_int(event.get("frame_index")),
                   "video_time_ms": pipeline.as_float(event.get("video_time_ms")),
                   "status": "matched"} for event in data]
    return events


def normalise_events(events: list[dict], touches: list[dict]) -> list[dict]:
    rows = []
    counts_match = len(events) == len(touches)
    for index, touch in enumerate(touches):
        event = events[index] if counts_match else {}
        frame = event.get("frame_index", -1)
        time_ms = event.get("video_time_ms", float("nan"))
        matched = frame >= 0 and np.isfinite(time_ms)
        rows.append({"schema_version": 1, "trial_id": touch.get("trial_id", f"trial_{index:04d}"),
                     "touch_index": touch.get("touch_index", 0),
                     "flash_frame": frame if matched else "",
                     "flash_time_video_ms": f"{time_ms:.3f}" if matched else "",
                     "touch_time_video_ms": f"{time_ms:.3f}" if matched else "",
                     "alignment_method": "existing_event_cache", "uncertainty_ms": "",
                     "status": "matched" if matched else "missing_event"})
    return rows


def atomic_write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temp = path.with_name(path.name + ".tmp")
    pipeline.write_csv(temp, rows, fields)
    os.replace(temp, path)


def atomic_write_json(path: Path, value: object) -> None:
    temp = path.with_name(path.name + ".tmp")
    pipeline.write_json(temp, value)
    os.replace(temp, path)


def review_summary(rows: list[dict]) -> dict:
    decisions = {decision: sum(row["decision"] == decision for row in rows)
                 for decision in ("pending", "keep", "drop")}
    reasons = {}
    for row in rows:
        if row["decision"] == "drop":
            reasons[row["reason_code"]] = reasons.get(row["reason_code"], 0) + 1
    return {"total": len(rows), **decisions, "drop_reasons": reasons}


def save_review(output: Path, rows: list[dict]) -> None:
    fields = ["schema_version", "session_id", "trial_id", "touch_index", "decision",
              "reason_code", "note", "updated_at_utc"]
    atomic_write_csv(output / "review.csv", rows, fields)
    atomic_write_json(output / "review_summary.json", review_summary(rows))


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def draw_frame(frame: np.ndarray, frame_index: int, tracking: list[dict], event: dict,
               start: int, end: int, trial_number: int, total: int, row: dict,
               slow: bool, expanded: bool, max_width: int, trace_tail: int, fps: float) -> np.ndarray:
    frame = frame.copy()
    valid_indices = [i for i in range(max(start, frame_index - trace_tail), frame_index + 1)
                     if tracking[i]["valid"]]
    for previous, current in zip(valid_indices, valid_indices[1:]):
        if current != previous + 1:
            continue
        p = tracking[previous]
        q = tracking[current]
        cv2.line(frame, (round(float(p["thumb_tip_x_px"])), round(float(p["thumb_tip_y_px"]))),
                 (round(float(q["thumb_tip_x_px"])), round(float(q["thumb_tip_y_px"]))),
                 (0, 220, 255), 4, cv2.LINE_AA)
    current = tracking[frame_index]
    if current["valid"]:
        point = (round(float(current["thumb_tip_x_px"])), round(float(current["thumb_tip_y_px"])))
        cv2.circle(frame, point, 18, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.circle(frame, point, 14, (0, 255, 0), 4, cv2.LINE_AA)
    source_status = "DETECTED" if current["valid"] else "NO DETECTION"
    decision = row["decision"].upper()
    flash_frame = pipeline.as_int(event.get("flash_frame"))
    flash_ms = pipeline.as_float(event.get("flash_time_video_ms"))
    from_flash = (float(flash_ms) - float(current["video_time_ms"])
                  if np.isfinite(flash_ms) else float("nan"))
    from_flash_text = f"{from_flash:.0f} ms" if np.isfinite(from_flash) else "unknown"
    visible = tracking[start:end + 1]
    coverage = sum(item["valid"] for item in visible) / max(1, len(visible))
    longest_gap = gap = 0
    for item in visible:
        gap = gap + 1 if not item["valid"] else 0
        longest_gap = max(longest_gap, gap)
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1] - 1, 1250), 216), (20, 20, 20), -1)
    lines = [f"touch {trial_number}/{total}  {decision}  frame {frame_index + 1}/{len(tracking)}",
             f"thumb: {source_status}  time: {float(current['video_time_ms']) / 1000:.2f}s",
             f"flash - frame: {from_flash_text}  window: {start / fps:.2f}-{end / fps:.2f}s",
             f"window coverage: {coverage:.0%}  longest gap: {longest_gap} frames",
             "SPACE play/pause  LEFT/RIGHT frame  K keep  D drop  P/N trial  E expand  M note  Q quit"]
    for line, text in enumerate(lines):
        color = (0, 255, 0) if line == 1 and current["valid"] else (240, 240, 240)
        cv2.putText(frame, text, (20, 38 + line * 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.78, color, 2, cv2.LINE_AA)
    bar_y = frame.shape[0] - 28
    cv2.line(frame, (20, bar_y), (frame.shape[1] - 20, bar_y), (150, 150, 150), 2)
    for index in range(start, end + 1):
        x = 20 + round((frame.shape[1] - 40) * (index - start) / max(1, end - start))
        color = (0, 220, 0) if tracking[index]["valid"] else (0, 80, 220)
        cv2.line(frame, (x, bar_y - 7), (x, bar_y + 7), color, 2)
    if start <= flash_frame <= end:
        x = 20 + round((frame.shape[1] - 40) * (flash_frame - start) / max(1, end - start))
        cv2.line(frame, (x, bar_y - 16), (x, bar_y + 16), (0, 255, 255), 3)
    mode = "SLOW" if slow else "NORMAL"
    cv2.putText(frame, f"{mode}  {'EXPANDED' if expanded else 'DEFAULT'}  range {start}-{end}",
                (20, frame.shape[0] - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if frame.shape[1] > max_width:
        frame = cv2.resize(frame, (max_width, round(frame.shape[0] * max_width / frame.shape[1])))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True,
                        help="cached frame tracking CSV; legacy full_* columns are supported")
    parser.add_argument("--events", type=Path, required=True,
                        help="alignment.csv or diagnostics.json with flash events")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    session_dir = args.session_dir.resolve()
    video_matches = sorted(list(session_dir.glob("*.mp4")) + list(session_dir.glob("*.MP4")))
    if len(video_matches) != 1:
        raise RuntimeError(f"expected one video in {session_dir}, found {len(video_matches)}")
    video = video_matches[0]
    touches_path = one_file(session_dir, "*_touches.csv")
    touches = pipeline.read_csv(touches_path)
    cap, orientation = pipeline.open_video(video)
    fps, width, height = pipeline.video_meta(cap)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cfg = pipeline.load_yaml(args.config)
    rcfg = cfg.get("review", {})
    before_ms = float(rcfg.get("before_flash_ms", 1000))
    after_ms = float(rcfg.get("after_flash_ms", 200))
    max_width = int(rcfg.get("display_max_width_px", 1280))
    trace_tail = int(rcfg.get("trace_tail", 18))
    slow_multiplier = float(rcfg.get("slow_playback_multiplier", 4.0))
    output = args.output.resolve()
    if output.exists():
        required = [output / name for name in ("manifest.json", "review.csv", "alignment.csv", "tracking.csv")]
        if not all(path.is_file() for path in required):
            raise FileExistsError(f"output exists but is not a review directory: {output}")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        source_checks = [("source_video", video), ("source_touches", touches_path),
                         ("tracking_source", args.tracking.resolve()), ("events_source", args.events.resolve())]
        for manifest_key, source_path in source_checks:
            if manifest.get(manifest_key) != str(source_path) or sha256(source_path) != manifest.get(manifest_key + "_sha256"):
                raise ValueError(f"source changed since review preparation: {source_path}")
        cfg = pipeline.load_yaml(output / "config.yaml")
        tracking = load_tracking(output / "tracking.csv", frame_count)
        alignment = pipeline.read_csv(output / "alignment.csv")
        review_rows = pipeline.read_csv(output / "review.csv")
        if len(review_rows) != len(touches):
            raise ValueError("existing review.csv does not match touch count")
    else:
        tracking = load_tracking(args.tracking.resolve(), frame_count)
        events = load_events(args.events.resolve(), len(touches))
        alignment = normalise_events(events, touches)
        output.mkdir(parents=True)
        shutil.copy2(args.config, output / "config.yaml")
        pipeline.write_csv(output / "tracking.csv", tracking, list(tracking[0]))
        pipeline.write_csv(output / "alignment.csv", alignment, list(alignment[0]))
        review_rows = [{"schema_version": 1, "session_id": touch.get("session_id", session_dir.name),
                        "trial_id": touch.get("trial_id", f"trial_{i:04d}"),
                        "touch_index": touch.get("touch_index", 0), "decision": "pending",
                        "reason_code": "", "note": "", "updated_at_utc": ""}
                       for i, touch in enumerate(touches)]
        manifest = {"schema_version": 1, "review_id": output.name, "session_id": session_dir.name,
                    "source_video": str(video), "source_video_sha256": sha256(video),
                    "source_touches": str(touches_path), "source_touches_sha256": sha256(touches_path),
                    "tracking_source": str(args.tracking.resolve()),
                    "tracking_source_sha256": sha256(args.tracking.resolve()),
                    "events_source": str(args.events.resolve()),
                    "events_source_sha256": sha256(args.events.resolve()),
                    "video_size_px": [width, height], "fps": fps, "frame_count": frame_count,
                    "orientation_degrees": orientation, "tracking_columns": list(tracking[0]),
                    "event_count": len(events), "review_window_ms": [before_ms, after_ms],
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                    "code_snapshot": "review_session_snapshot.py", "config": "config.yaml"}
        shutil.copy2(Path(__file__), output / "review_session_snapshot.py")
        pipeline.write_json(output / "manifest.json", manifest)
        save_review(output, review_rows)
    rcfg = cfg.get("review", {})
    before_ms = float(rcfg.get("before_flash_ms", 1000))
    after_ms = float(rcfg.get("after_flash_ms", 200))
    max_width = int(rcfg.get("display_max_width_px", 1280))
    trace_tail = int(rcfg.get("trace_tail", 18))
    slow_multiplier = float(rcfg.get("slow_playback_multiplier", 4.0))
    current_trial = next((i for i, row in enumerate(review_rows) if row["decision"] == "pending"), 0)
    frame_index = 0
    playing = False
    slow = False
    expanded = False
    reason_mode = False
    try:
        while True:
            event = alignment[current_trial]
            flash_frame = pipeline.as_int(event.get("flash_frame"))
            flash_time = pipeline.as_float(event.get("flash_time_video_ms"))
            if flash_frame >= 0:
                if expanded and current_trial > 0:
                    previous = pipeline.as_int(alignment[current_trial - 1].get("flash_frame"))
                    start = max(0, previous)
                else:
                    start = max(0, round((flash_time - before_ms) * fps / 1000))
                end = min(frame_count - 1, round((flash_time + after_ms) * fps / 1000))
            else:
                start, end = 0, min(frame_count - 1, round(2 * fps))
            frame_index = min(max(frame_index, start), end)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"cannot read frame {frame_index}")
            shown = draw_frame(frame, frame_index, tracking, event, start, end, current_trial + 1,
                               len(review_rows), review_rows[current_trial], slow, expanded,
                               max_width, trace_tail, fps)
            if reason_mode:
                cv2.rectangle(shown, (20, 185), (min(shown.shape[1] - 20, 950), 275), (30, 30, 30), -1)
                cv2.putText(shown, "DROP REASON: 1 missing 2 wrong-tip 3 wrong-hand 4 occluded 5 alignment 6 protocol 7 other",
                            (35, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, shown)
            delay = max(1, round(1000 / fps * (slow_multiplier if slow else 1))) if playing else 0
            key = cv2.waitKeyEx(delay)
            if key in (ord("q"), ord("Q"), 27):
                if reason_mode:
                    reason_mode = False
                    continue
                break
            if reason_mode:
                if key in REASONS:
                    review_rows[current_trial].update({"decision": "drop", "reason_code": REASONS[key],
                                                        "updated_at_utc": utc_now()})
                    save_review(output, review_rows)
                    reason_mode = False
                    current_trial = min(current_trial + 1, len(review_rows) - 1)
                    frame_index = 0
                continue
            if key == 32:
                playing = not playing
            elif key in (ord("k"), ord("K")):
                review_rows[current_trial].update({"decision": "keep", "reason_code": "", "updated_at_utc": utc_now()})
                save_review(output, review_rows)
                current_trial = min(current_trial + 1, len(review_rows) - 1)
                frame_index = 0
            elif key in (ord("d"), ord("D")):
                playing = False
                reason_mode = True
            elif key in (ord("p"), ord("P")):
                current_trial = max(0, current_trial - 1)
                frame_index = 0
                playing = False
            elif key in (ord("n"), ord("N")):
                current_trial = min(len(review_rows) - 1, current_trial + 1)
                frame_index = 0
                playing = False
            elif key in (ord("r"), ord("R")):
                frame_index = start
                playing = True
            elif key in (ord("s"), ord("S")):
                slow = not slow
            elif key in (ord("e"), ord("E")):
                expanded = not expanded
                frame_index = 0
            elif key in (ord("m"), ord("M")):
                cv2.destroyWindow(WINDOW)
                note = input(f"note for {review_rows[current_trial]['trial_id']} (empty to cancel): ").strip()
                if note:
                    review_rows[current_trial]["note"] = note
                    review_rows[current_trial]["updated_at_utc"] = utc_now()
                    save_review(output, review_rows)
            elif key in (81, 2424832):
                frame_index = max(start, frame_index - 1)
                playing = False
            elif key in (83, 2555904):
                frame_index = min(end, frame_index + 1)
                playing = False
            elif playing and frame_index >= end:
                playing = False
            elif not playing and key == -1:
                pass
            if playing and frame_index < end:
                frame_index += 1
            elif playing:
                playing = False
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_review(output, review_rows)
        print(json.dumps(review_summary(review_rows), ensure_ascii=False))


if __name__ == "__main__":
    main()
