#!/usr/bin/env python3
"""Prepare and review any session in a browser; choices go to session/review.csv."""

import argparse
import json
import math
import os
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

import visual_pretouch as pipeline


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["trial_id", "touch_index", "keep"]


def one_file(directory, pattern):
    files = sorted(directory.glob(pattern))
    if not files and pattern == "*_touches.csv" and (directory / 'touches.csv').is_file():
        files = [directory / 'touches.csv']
    if len(files) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}, found {len(files)}")
    return files[0]


def source_path(session, supplied, filename):
    if supplied:
        return supplied.resolve()
    # Unversioned diagnostic CSVs cannot establish which hand model produced them.
    if filename == 'frame_diagnostics.csv':
        return None
    local = session / filename
    if filename == 'diagnostics.json' and not local.is_file() and (session / 'alignment.csv').is_file():
        return session / 'alignment.csv'
    candidates = [local] if local.is_file() else sorted(
        (ROOT / "outputs").glob(f"review_{session.name}_*/{filename}"))
    if not candidates:
        return None
    if len(candidates) != 1:
        option = "tracking" if filename.endswith(".csv") else "events"
        raise ValueError(f"Cannot choose {filename}; specify --{option} PATH")
    return candidates[0]


def save_review(path, rows):
    temp = path.with_suffix(".csv.tmp")
    pipeline.write_csv(temp, rows, FIELDS)
    os.replace(temp, path)


def load_review(session, touches):
    rows = [{"trial_id": t["trial_id"], "touch_index": t["touch_index"], "keep": ""}
            for t in touches]
    path = session / "review.csv"
    legacy = sorted((ROOT / "outputs").glob(f"manual_review_{session.name}_*/review.csv"))
    source = path if path.exists() else legacy[0] if len(legacy) == 1 else None
    if source is None:
        return rows
    saved = pipeline.read_csv(source)
    by_key = {(r["trial_id"], r["touch_index"]): r for r in saved}
    keys = {(r["trial_id"], r["touch_index"]) for r in rows}
    if len(by_key) != len(saved) or set(by_key) != keys:
        raise ValueError(f"Review does not match session touches: {source}")
    for row in rows:
        previous = by_key[(row["trial_id"], row["touch_index"])]
        if previous.get("session_id", session.name) != session.name:
            raise ValueError(f"Review belongs to another session: {source}")
        value = previous.get("keep")
        if value is None:
            value = {"keep": "1", "drop": "0", "pending": ""}[previous["decision"]]
        if value not in ("", "0", "1"):
            raise ValueError(f"Invalid keep value in {source}: {value}")
        row["keep"] = value
    if source != path and any(r["keep"] for r in rows):
        save_review(path, rows)
        print(f"Previous choices migrated to {path}")
    return rows


def prepare(session, tracking_path, events_path, config):
    touches = pipeline.read_csv(one_file(session, "*_touches.csv"))
    if not touches or len({(t["trial_id"], t["touch_index"]) for t in touches}) != len(touches):
        raise ValueError("Touch records must be nonempty and have unique identifiers")
    videos = [p for p in session.iterdir() if p.suffix.lower() == ".mp4"]
    if len(videos) != 1:
        raise ValueError(f"Expected one MP4 in {session}")
    video = videos[0]
    cap, _ = pipeline.open_video(video)
    fps, width, height = pipeline.video_meta(cap)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cache = None
    if tracking_path is None or events_path is None:
        from review_prepare import prepare_cache
        cache = prepare_cache(session, video, one_file(session, "*_touches.csv"), config,
                              need_events=events_path is None)
    frames = pipeline.read_csv(tracking_path) if tracking_path is not None else [
        {"frame_index": i, "video_time_ms": t, "valid": "1" if point else "0",
         "thumb_tip_x_px": point[0] if point else "", "thumb_tip_y_px": point[1] if point else ""}
        for i, (t, point) in enumerate(cache['frames'])]
    if sorted(pipeline.as_int(r.get("frame_index")) for r in frames) != list(range(count)):
        raise ValueError("Tracking must contain every video frame exactly once")
    frames.sort(key=lambda r: int(r["frame_index"]))
    tracking = []
    for row in frames:
        legacy = "full_detected" in row
        valid = row.get("full_detected" if legacy else "valid", row.get("hand_detected"))
        x = pipeline.as_float(row.get("full_tip_x_native_px" if legacy else "thumb_tip_x_px"))
        y = pipeline.as_float(row.get("full_tip_y_native_px" if legacy else "thumb_tip_y_px"))
        time = pipeline.as_float(row.get("video_time_ms"))
        if not math.isfinite(time):
            raise ValueError("Tracking must include finite video_time_ms timestamps")
        point = [x, y] if valid in ("1", "true", "True") and math.isfinite(x + y) else None
        tracking.append([time / 1000, point])
    if any(b[0] <= a[0] for a, b in zip(tracking, tracking[1:])):
        raise ValueError("Tracking timestamps must increase")
    if events_path is None:
        times = [e['video_time_ms'] / 1000 for e in cache['events']]
    elif events_path.suffix.lower() == ".csv":
        alignment = pipeline.read_csv(events_path)
        keyed = {(r["trial_id"], r["touch_index"]): r for r in alignment}
        events = [keyed.get((t["trial_id"], t["touch_index"]), {}) for t in touches]
        times = [pipeline.as_float(r.get("flash_time_video_ms")) / 1000 for r in events]
    else:
        data = json.loads(events_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "flash_checks" in data:
            checks = [c for c in data["flash_checks"] if len(c.get("events", [])) == len(touches)]
            if not checks:
                raise ValueError("Flash count does not match touches; provide a checked alignment.csv")
            data = checks[0]["events"]
        elif isinstance(data, dict):
            data = data.get("events", [])
        times = [pipeline.as_float(e.get("video_time_ms")) / 1000 for e in data]
    if len(times) != len(touches) or any(not math.isfinite(t) or t < 0 or t >= count / fps for t in times):
        raise ValueError("Each touch needs a valid flash timestamp")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Flash timestamps must follow touch order")
    rows = load_review(session, touches)
    state = {"session": session.name, "fps": fps, "width": width, "height": height,
             "duration": count / fps, "flashes": times, "tracking": tracking,
             "review": rows, "settings": config["review"], "result_path": str(session / "review.csv")}
    state['tracking_info'] = (
        {'source': str(tracking_path), 'model': 'external'} if tracking_path is not None else
        {'source': str(session / '.review_cache.json'),
         'mediapipe': cache['identity']['mediapipe'],
         'model': {0: 'lite', 1: 'full'}[cache['identity']['tracking']['model_complexity']],
         'input_height_px': cache['identity']['input_height_px']})
    if tracking_path is None:
        state['raw_tracking'] = [[time / 1000, point] for time, point in cache['raw_frames']]
        state['tracking_status'] = cache['tracking_status']
        state['tracking_info']['continuity'] = cache['identity']['tracking'].get('continuity', {}).get('enabled', False)
    return video, state


def make_handler(video, state):
    lock = threading.Lock()
    result = Path(state["result_path"])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_data(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == "/":
                self.send_data(Path(__file__).with_suffix(".html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/session":
                with lock:
                    self.send_data(json.dumps(state, allow_nan=False).encode(), "application/json")
            elif path == "/video":
                self.send_video()
            else:
                self.send_error(404)

        def send_video(self):
            size = video.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                left, right = match.groups()
                start = int(left) if left else max(0, size - int(right))
                end = min(int(right), size - 1) if left and right else size - 1
                if start > end or start >= size:
                    self.send_error(416)
                    return
            self.send_response(206 if requested else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with video.open("rb") as file:
                    file.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        block = file.read(min(remaining, 1024 * 1024))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if self.path != "/api/review":
                self.send_error(404)
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 1024:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                index, keep = request["index"], request["keep"]
                if type(index) is not int or not 0 <= index < len(state["review"]) or keep not in ("", "0", "1"):
                    raise ValueError("Invalid touch or keep value")
                with lock:
                    updated = [dict(row) for row in state["review"]]
                    updated[index]["keep"] = keep
                    save_review(result, updated)
                    state["review"] = updated
                    self.send_data(json.dumps(updated).encode(), "application/json")
            except (KeyError, TypeError, ValueError) as error:
                self.send_data(str(error).encode(), "text/plain; charset=utf-8", 400)
            except OSError as error:
                self.send_data(str(error).encode(), "text/plain; charset=utf-8", 500)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, help="Optional explicit tracking CSV")
    parser.add_argument("--events", type=Path, help="Optional explicit diagnostics.json or alignment.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--port", type=int, default=0, help="0 selects an available local port")
    parser.add_argument("--no-open", action="store_true", help="Print the URL without opening a browser")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare/check inputs without starting the browser")
    args = parser.parse_args()
    session = args.session_dir.resolve()
    tracking = source_path(session, args.tracking, "frame_diagnostics.csv")
    events = source_path(session, args.events, "diagnostics.json")
    video, state = prepare(session, tracking, events, pipeline.load_yaml(args.config))
    if args.prepare_only:
        print(f'Ready: {len(state["review"])} touches, {len(state["tracking"])} frames')
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(video, state))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Review: {url}\nResults: {state['result_path']}\nCtrl+C to stop.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
