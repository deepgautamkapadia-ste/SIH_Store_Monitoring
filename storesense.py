#!/usr/bin/env python3
"""
StoreSense Edge — on-device retail intelligence  (SIH 2026, PS 26179)
One file. Laptop now; target Qualcomm Dragonwing RB3 Gen 2 / Snapdragon (YOLO exported to QNN, Hexagon NPU).

Modules (reference repo each one replaces)
  • Footfall: entry/exit counting, live occupancy, hourly trend     (DeepStream retail analytics + ByteTrack)
  • Floor grid: heatmap + zone dwell time via homography            (DeepStream zone analytics)
  • Shelf grid: per-cell OK / LOW / EMPTY / MISPLACED, occlusion-aware (Retail-Shelf-Monitoring, StoreEye)
  • Depletion forecast: time-to-empty per shelf cell                (demand-forecasting repos)
  • Queue: length, time in queue, +5/+10/+15 min forecast,
    counters-to-open recommendation                                  (QueueLess + multistep forecasting)
  • Rule-based priority decision engine with cooldowns + ack/response time
  • Local SQLite, daily/weekly reports, offline-first cloud sync outbox (EventPulse)
  • Live dashboard: FastAPI + WebSocket + annotated MJPEG feeds
  • POS hook: POST /api/integrations/pos  (conversion KPI uses it when present)
  • Privacy: only track IDs + numbers stored. No frames, no faces. Heads blurred on live feeds.

Install
  pip install ultralytics lap fastapi "uvicorn[standard]" opencv-python numpy
  (plain "uvicorn" has no WebSocket support; the dashboard then falls back to polling)

Run
  python storesense.py                                  # laptop webcam = entry + queue cam
  python storesense.py \
      --cam entry  entry,queue  http://192.168.1.23:8080/video \
      --cam shelfA shelf        http://192.168.1.24:8080/video \
      --cam shelfB shelf        http://192.168.1.25:8080/video
  open http://localhost:8000
  Sources: webcam index, phone stream URL, RTSP, or a video file (loops — good for demos).

Setup
  python storesense.py --pick SOURCE
      click points on a frame -> prints normalized coords to paste into CONFIG["geometry"]
      (entry_line: 2 pts, queue_zone: 4+ pts, floor_quad: 4 pts TL,TR,BR,BL). c = clear, q = done.
  Shelf cams: clear the aisle, fully stock the shelf, hit "Calibrate" on the dashboard.
  The entry feed draws an "IN" arrow — if it points the wrong way, flip "in_side" in CONFIG.
"""
import argparse, json, os, sys, sqlite3, threading, time, urllib.request
from collections import Counter, defaultdict, deque
from datetime import datetime

import cv2
import numpy as np

# ─────────────────────────────── CONFIG ───────────────────────────────
CONFIG = {
    "store_id": "store-001",
    "db_path": "storesense.db",
    "person_model": "yolo11n.pt",   # auto-downloads on first run
    "sku_model": None,              # optional YOLO weights for your SKUs (e.g. SKU-110K fine-tune)
    "imgsz": 640,
    "conf": 0.35,
    "privacy_blur": True,
    "cloud_url": None,              # e.g. "https://hq.example/api/ingest"; buffered while offline
    "alert_cooldown_s": 180,
    "crowd_threshold": 25,
    "geometry": {                   # per camera name; "default" used when a name isn't listed
        "default": {
            "entry_line": [[0.5, 0.0], [0.5, 1.0]],
            "in_side": -1,          # with this vertical line, left -> right counts as IN
                                    # (the live feed draws an "IN" arrow; flip the sign if it points the wrong way)
            "queue_zone": [[0.05, 0.35], [0.45, 0.35], [0.45, 0.95], [0.05, 0.95]],
            "floor_quad": None,
            "floor_grid": [10, 16],
            "zones": {              # floor-grid rects [r0, c0, r1, c1), end exclusive
                "Promo display": [0, 0, 5, 8],
                "Billing area": [5, 0, 10, 8],
                "Main aisle": [0, 8, 10, 16],
            },
        },
    },
    "shelf": {
        "grid": [4, 6],             # rows, cols of the shelf face
        "period_s": 3.0,
        "low": 0.55, "empty": 0.25, # fill ratio vs calibrated full shelf
        "misplace_corr": 0.35,      # colour-histogram similarity below this = product looks different
        "occlusion_overlap": 0.15,
        "eta_window_min": 20,
        "planogram": {},            # {"shelfA": {"0,0": "maggi", ...}} — used only with sku_model
        "weights": {},              # priority multiplier per shelf cam, e.g. {"shelfA": 1.5}
    },
    "queue": {
        "open_counters": 1, "max_counters": 4,
        "default_service_per_min": 1.5,   # prior per counter, learned online
        "target_wait_min": 3.0, "max_wait_min": 6.0,
        "min_time_in_zone_s": 4.0, "rate_window_min": 5.0,
    },
}


def geom(name):
    return CONFIG["geometry"].get(name, CONFIG["geometry"]["default"])


def day_start(ts=None):
    d = datetime.fromtimestamp(ts or time.time())
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def line_side(p, a, b):
    v = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return 1 if v > 1e-6 else (-1 if v < -1e-6 else 0)


def blur_heads(img, boxes):
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0, int(x1)), min(w, int(x2))
        y1 = max(0, int(y1))
        y2 = min(h, int(y1 + 0.25 * (y2 - y1)) + 1)
        if x2 - x1 > 2 and y2 - y1 > 2:
            img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (31, 31), 0)


# ─────────────────────────────── CAPTURE ───────────────────────────────
class Capture:
    """Threaded reader that always holds the latest frame. Reconnects streams, loops files."""

    def __init__(self, src):
        self.src = int(src) if str(src).isdigit() else src
        self.is_file = isinstance(self.src, str) and not self.src.startswith(("http", "rtsp")) and os.path.exists(self.src)
        self.frame, self.lock = None, threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.src)
        if isinstance(self.src, int):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        cap = self._open()
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        while True:
            ok, f = cap.read()
            if not ok:
                if self.is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, f = cap.read()
                if not ok:
                    with self.lock:
                        self.frame = None
                    cap.release()
                    time.sleep(1.0)
                    cap = self._open()
                    continue
            with self.lock:
                self.frame = f
            if self.is_file:
                time.sleep(1.0 / max(fps, 1))

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


# ─────────────────────────────── STORAGE ───────────────────────────────
class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, cam TEXT, type TEXT, data TEXT);
                CREATE TABLE IF NOT EXISTS metrics(ts REAL, key TEXT, value REAL);
                CREATE TABLE IF NOT EXISTS outbox(id INTEGER PRIMARY KEY, ts REAL, payload TEXT, sent INTEGER DEFAULT 0);
                CREATE INDEX IF NOT EXISTS ix_ev ON events(type, ts);
                CREATE INDEX IF NOT EXISTS ix_m ON metrics(key, ts);
            """)

    def q(self, sql, *args):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def event(self, cam, typ, data):
        with self.lock:
            self.db.execute("INSERT INTO events(ts,cam,type,data) VALUES(?,?,?,?)",
                            (time.time(), cam, typ, json.dumps(data)))
            self.db.commit()

    def metrics(self, ts, d):
        with self.lock:
            self.db.executemany("INSERT INTO metrics VALUES(?,?,?)", [(ts, k, float(v)) for k, v in d.items()])
            self.db.commit()

    def outbox_add(self, payload):
        with self.lock:
            self.db.execute("INSERT INTO outbox(ts,payload) VALUES(?,?)", (time.time(), json.dumps(payload)))
            self.db.commit()

    def outbox_pending(self, n=50):
        return self.q("SELECT id,payload FROM outbox WHERE sent=0 ORDER BY id LIMIT ?", n)

    def outbox_mark(self, ids):
        with self.lock:
            self.db.executemany("UPDATE outbox SET sent=1 WHERE id=?", [(i,) for i in ids])
            self.db.commit()

    def report(self, period):
        days = 7 if period == "week" else 1
        since = day_start() - (days - 1) * 86400
        fmt = "%H:00" if days == 1 else "%a %d %b"
        footfall, tot_in = defaultdict(int), 0
        for ts, d in self.q("SELECT ts,data FROM events WHERE type='entry' AND ts>=? ORDER BY ts", since):
            if json.loads(d)["dir"] == "in":
                footfall[datetime.fromtimestamp(ts).strftime(fmt)] += 1
                tot_in += 1
        alerts = Counter()
        for (d,) in self.q("SELECT data FROM events WHERE type='alert' AND ts>=?", since):
            alerts[json.loads(d)["action"]] += 1
        resp = [json.loads(d)["response_s"] for (d,) in self.q("SELECT data FROM events WHERE type='ack' AND ts>=?", since)]
        served = [json.loads(d)["duration_s"] for (d,) in self.q("SELECT data FROM events WHERE type='served' AND ts>=?", since)]
        pos = self.q("SELECT COUNT(*) FROM events WHERE type='pos' AND ts>=?", since)[0][0]
        zones = defaultdict(list)
        for (d,) in self.q("SELECT data FROM events WHERE type='zone_visit' AND ts>=?", since):
            d = json.loads(d)
            zones[d["zone"]].append(d["duration_s"])
        metr = {k: {"avg": round(a, 2), "max": round(m, 2)}
                for k, a, m in self.q("SELECT key,AVG(value),MAX(value) FROM metrics WHERE ts>=? GROUP BY key", since)}
        buyers = pos if pos else len(served)
        return {
            "period": period, "from": datetime.fromtimestamp(since).isoformat(timespec="minutes"),
            "footfall_total": tot_in, "footfall_by_bucket": dict(footfall),
            "peak": max(footfall, key=footfall.get) if footfall else None,
            "customers_billed": buyers, "conversion": round(buyers / tot_in, 3) if tot_in else None,
            "avg_time_in_queue_min": round(np.mean(served) / 60, 2) if served else None,
            "zone_dwell": {z: {"visits": len(v), "avg_dwell_s": round(float(np.mean(v)), 1)} for z, v in zones.items()},
            "alerts_by_action": dict(alerts),
            "avg_alert_response_s": round(float(np.mean(resp)), 1) if resp else None,
            "metrics": metr,
        }


# ─────────────────────────────── DECISION ENGINE ───────────────────────────────
class Engine:
    SEV = {1: "info", 2: "warning", 3: "critical"}

    def __init__(self, store):
        self.store, self.lock = store, threading.RLock()
        self.footfall = {"in": 0, "out": 0}
        self.hourly, self.conversion, self.avg_response_s = {}, None, None
        self.alerts, self.last, self.keys, self.next_id = deque(maxlen=300), {}, {}, 1
        self.queues, self.shelves, self.heat, self.aisle, self.cams = {}, {}, {}, {}, {}
        self.zone_stats = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        self.served = deque(maxlen=200)
        self.open_counters = CONFIG["queue"]["open_counters"]
        self.online = None
        self.refresh_daily()

    def fire(self, key, kind, sev, msg, action, weight=1.0):
        now = time.time()
        with self.lock:
            if now - self.last.get(key, 0) < CONFIG["alert_cooldown_s"]:
                return
            self.last[key] = now
            for a in self.alerts:      # same problem still open: refresh that row, don't stack a new one
                if not a["acked"] and self.keys.get(a["id"]) == key:
                    a.update(ts=now, message=msg, severity=self.SEV[sev],
                             priority=round(sev * weight * 10, 1), count=a.get("count", 1) + 1)
                    return
            a = {"id": self.next_id, "ts": now, "kind": kind, "severity": self.SEV[sev], "count": 1,
                 "priority": round(sev * weight * 10, 1), "message": msg, "action": action, "acked": False}
            self.keys[self.next_id] = key
            self.next_id += 1
            if len(self.keys) > 1000:                       # keep only ids still in the ring buffer
                live = {x["id"] for x in self.alerts}
                self.keys = {k: v for k, v in self.keys.items() if k in live}
            self.alerts.appendleft(a)
        self.store.event("engine", "alert", a)

    def resolve(self, *keys):
        """The condition is gone (shelf refilled, queue drained): drop the alert instead of
        leaving it for a human to dismiss. Keeps the board showing live state, not history."""
        with self.lock:
            for a in list(self.alerts):
                if not a["acked"] and self.keys.get(a["id"]) in keys:
                    self.alerts.remove(a)
                    self.last.pop(self.keys.pop(a["id"], None), None)

    def ack(self, aid):
        """Staff marked it done: clear the cooldown so the alert returns if the problem persists."""
        with self.lock:
            for a in self.alerts:
                if a["id"] == aid and not a["acked"]:
                    a["acked"] = True
                    self.last.pop(self.keys.pop(aid, None), None)
                    self.store.event("engine", "ack", {"id": aid, "kind": a["kind"], "response_s": time.time() - a["ts"]})
                    return True
        return False

    # events from workers
    def on_entry(self, cam, direction):
        with self.lock:
            self.footfall[direction] += 1
            inside = max(0, self.footfall["in"] - self.footfall["out"])
        self.store.event(cam, "entry", {"dir": direction})
        if inside > CONFIG["crowd_threshold"]:
            self.fire("crowd", "crowd", 2, f"{inside} shoppers inside", "Deploy floor staff")
        else:
            self.resolve("crowd")

    def on_zone_visit(self, cam, zone, dur):
        with self.lock:
            s = self.zone_stats[cam][zone]
            s[0] += 1
            s[1] += dur
        self.store.event(cam, "zone_visit", {"zone": zone, "duration_s": round(dur, 1)})

    def on_served(self, cam, dur):
        with self.lock:
            self.served.append(dur)
        self.store.event(cam, "served", {"duration_s": round(dur, 1)})

    def on_aisle(self, cam, n, dt):
        with self.lock:
            a = self.aisle.setdefault(cam, {"people": 0, "person_minutes": 0.0})
            a["people"] = n
            a["person_minutes"] += n * dt / 60

    def on_queue(self, cam, q):
        with self.lock:
            self.queues[cam] = q
            k = self.open_counters
        opn, bld, cls = f"queue:{cam}:open", f"queue:{cam}:build", f"queue:{cam}:close"
        if q["recommend_counters"] > k:
            self.fire(opn, "queue", 3,
                      f"{q['length']} in queue, wait ~{q['wait_min']} min, +10 min forecast {q['forecast']['10']['wait']} min",
                      f"Open {q['recommend_counters'] - k} more counter(s)")
            self.resolve(bld, cls)
        elif q["wait_min"] > CONFIG["queue"]["target_wait_min"]:
            self.fire(bld, "queue", 2, f"Queue building, wait ~{q['wait_min']} min",
                      "Keep a standby cashier ready")
            self.resolve(opn, cls)
        elif k > 1 and q["recommend_counters"] < k and q["length"] == 0:
            self.fire(cls, "queue", 1, "Billing idle", "Close a counter, reassign staff to shelves")
            self.resolve(opn, bld)
        else:
            self.resolve(opn, bld, cls)                  # queue healthy again

    def on_shelf(self, cam, cells):
        with self.lock:
            self.shelves[cam] = cells
        if not cells:
            return
        w = CONFIG["shelf"]["weights"].get(cam, 1.0)
        for c in cells:
            if c["occluded"]:
                continue
            key, loc = f"shelf:{cam}:{c['r']},{c['c']}", f"{cam} row {c['r'] + 1} col {c['c'] + 1}"
            if c["status"] == "EMPTY":
                self.fire(key, "stock", 3, f"{loc} is empty", "Critical refill", w)
            elif c["status"] == "LOW":
                self.fire(key, "stock", 2, f"{loc} low ({c['fill'] * 100:.0f}%)", "Refill", w)
            elif c["eta_min"] is not None and c["eta_min"] < 15:
                self.fire(key, "stock", 2, f"{loc} expected empty in ~{c['eta_min']:.0f} min", "Refill soon", w)
            else:
                self.resolve(key)                        # cell restocked
            if c["status"] == "MISPLACED":
                exp = f" (expected {c['expected']}, found {c['found']})" if c.get("expected") else ""
                self.fire(key + ":pg", "planogram", 1, f"{loc} planogram mismatch{exp}", "Correct shelf", w)
            else:
                self.resolve(key + ":pg")

    def cam_status(self, name, roles, online, fps):
        with self.lock:
            self.cams[name] = {"roles": sorted(roles), "online": online, "fps": round(fps, 1)}

    def refresh_daily(self):
        since = day_start()
        f, hourly = {"in": 0, "out": 0}, defaultdict(int)
        for ts, d in self.store.q("SELECT ts,data FROM events WHERE type='entry' AND ts>=?", since):
            dr = json.loads(d)["dir"]
            f[dr] += 1
            if dr == "in":
                hourly[datetime.fromtimestamp(ts).strftime("%H")] += 1
        served = self.store.q("SELECT COUNT(*) FROM events WHERE type='served' AND ts>=?", since)[0][0]
        pos = self.store.q("SELECT COUNT(*) FROM events WHERE type='pos' AND ts>=?", since)[0][0]
        resp = [json.loads(d)["response_s"] for (d,) in self.store.q("SELECT data FROM events WHERE type='ack' AND ts>=?", since)]
        with self.lock:
            self.footfall = f
            self.hourly = dict(sorted(hourly.items()))
            buyers = pos or served
            self.conversion = round(buyers / f["in"], 3) if f["in"] else None
            self.avg_response_s = float(np.mean(resp)) if resp else None

    def snapshot(self):
        with self.lock:
            alerts = sorted((a for a in self.alerts if not a["acked"]), key=lambda a: (-a["priority"], -a["ts"]))
            heat = {}
            for cam, h in self.heat.items():
                m = float(h.max())
                heat[cam] = (h / m if m > 0 else h).round(3).tolist()
            return {
                "ts": time.time(), "store_id": CONFIG["store_id"], "cloud_online": self.online,
                "footfall": {**self.footfall, "inside": max(0, self.footfall["in"] - self.footfall["out"])},
                "hourly": self.hourly, "conversion": self.conversion, "avg_response_s": self.avg_response_s,
                "open_counters": self.open_counters, "queues": dict(self.queues), "shelves": dict(self.shelves),
                "heat": heat, "aisle": {k: {**v, "person_minutes": round(v["person_minutes"], 1)} for k, v in self.aisle.items()},
                "zones": {cam: {z: {"visits": s[0], "avg_dwell_s": round(s[1] / s[0], 1) if s[0] else 0}
                                for z, s in zs.items()} for cam, zs in self.zone_stats.items()},
                "avg_time_in_queue_min": round(float(np.mean(self.served)) / 60, 2) if self.served else None,
                "cams": dict(self.cams), "alerts": alerts[:50],
            }


# ─────────────────────────────── CAMERA WORKERS ───────────────────────────────
class CamWorker(threading.Thread):
    period = 0.0

    def __init__(self, name, roles, source, engine):
        super().__init__(daemon=True)
        self.name, self.roles, self.engine = name, roles, engine
        self.cap = Capture(source)
        self.jpg, self.fps = None, 0.0
        self.t_prev = self.t_start = time.time()

    def run(self):
        while True:
            f = self.cap.read()
            if f is None:
                self.engine.cam_status(self.name, self.roles, False, 0)
                time.sleep(0.3)
                continue
            t = time.time()
            try:
                vis = self.step(f)
            except Exception as e:
                print(f"[{self.name}] {type(e).__name__}: {e}")
                time.sleep(1.0)
                continue
            self.fps = 0.9 * self.fps + 0.1 / max(time.time() - t, 1e-3)
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                self.jpg = buf.tobytes()
            self.engine.cam_status(self.name, self.roles, True, self.fps)
            sl = self.period - (time.time() - t)
            if sl > 0:
                time.sleep(sl)


class PeopleWorker(CamWorker):
    """Roles: entry (line counting + floor heatmap + zones) and/or queue."""

    def __init__(self, name, roles, source, engine):
        super().__init__(name, roles, source, engine)
        from ultralytics import YOLO
        self.model = YOLO(CONFIG["person_model"])
        self.g = geom(name)
        self.prev_side, self.last_cross = {}, {}
        R, C = self.g["floor_grid"]
        self.heat = np.zeros((R, C), np.float32)
        engine.heat[name] = self.heat
        self.H = None
        self.zone_time, self.zone_last = defaultdict(dict), defaultdict(dict)
        self.cand, self.members, self.missing = {}, {}, {}
        self.arrivals, self.departures = deque(), deque()
        self.mu_c = CONFIG["queue"]["default_service_per_min"]

    def floor_cell(self, fx, fy, w, h):
        R, C = self.g["floor_grid"]
        quad = self.g.get("floor_quad")
        if quad:
            if self.H is None:
                src = np.float32([[x * w, y * h] for x, y in quad])
                self.H = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
            u, v = cv2.perspectiveTransform(np.float32([[[fx, fy]]]), self.H)[0, 0]
        else:
            u, v = fx / w, fy / h
        if not (0 <= u < 1 and 0 <= v < 1):
            return None
        return int(v * R), int(u * C)

    def step(self, frame):
        h, w = frame.shape[:2]
        now = time.time()
        dt = min(now - self.t_prev, 0.5)
        self.t_prev = now
        r = self.model.track(frame, persist=True, classes=[0], conf=CONFIG["conf"], imgsz=CONFIG["imgsz"],
                             tracker="bytetrack.yaml", verbose=False)[0]
        dets = []
        if r.boxes is not None and r.boxes.id is not None:
            for bb, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
                dets.append((*map(float, bb), tid))
        vis = frame.copy()
        if CONFIG["privacy_blur"]:
            blur_heads(vis, [d[:4] for d in dets])

        g = self.g
        a = (g["entry_line"][0][0] * w, g["entry_line"][0][1] * h)
        b = (g["entry_line"][1][0] * w, g["entry_line"][1][1] * h)
        qpoly = np.int32([[x * w, y * h] for x, y in g["queue_zone"]])
        in_queue = set()

        for x1, y1, x2, y2, tid in dets:
            foot = ((x1 + x2) / 2, y2)
            if "entry" in self.roles:
                s = line_side(foot, a, b)
                if s != 0:
                    ps = self.prev_side.get(tid)
                    if ps is not None and s != ps and now - self.last_cross.get(tid, 0) > 1.0:
                        self.engine.on_entry(self.name, "in" if s == g["in_side"] else "out")
                        self.last_cross[tid] = now
                    self.prev_side[tid] = s
                cell = self.floor_cell(foot[0], foot[1], w, h)
                if cell:
                    self.heat[cell] += dt
                    for zname, (r0, c0, r1, c1) in g["zones"].items():
                        if r0 <= cell[0] < r1 and c0 <= cell[1] < c1:
                            self.zone_time[tid][zname] = self.zone_time[tid].get(zname, 0.0) + dt
                            self.zone_last[tid][zname] = now
            if "queue" in self.roles and cv2.pointPolygonTest(qpoly, (float(foot[0]), float(foot[1])), False) >= 0:
                in_queue.add(tid)
            col = (0, 200, 255) if tid in in_queue else (80, 220, 120)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(vis, f"#{tid}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # close finished zone visits
        for tid in list(self.zone_last):
            for zname, last in list(self.zone_last[tid].items()):
                if now - last > 2.0:
                    dur = self.zone_time[tid].pop(zname, 0.0)
                    del self.zone_last[tid][zname]
                    if dur >= 3.0:
                        self.engine.on_zone_visit(self.name, zname, dur)
            if not self.zone_last[tid]:
                del self.zone_last[tid]
                self.zone_time.pop(tid, None)
        for tid in [t for t, ts in self.last_cross.items() if now - ts > 60]:
            del self.last_cross[tid]

        if "entry" in self.roles:
            cv2.line(vis, tuple(map(int, a)), tuple(map(int, b)), (255, 120, 0), 2)
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)          # arrow points at the IN side
            n = np.array([-(b[1] - a[1]), b[0] - a[0]], np.float32) * g["in_side"]
            n /= max(np.linalg.norm(n), 1e-6)
            tip = (int(mid[0] + 45 * n[0]), int(mid[1] + 45 * n[1]))
            cv2.arrowedLine(vis, tuple(map(int, mid)), tip, (255, 120, 0), 2, tipLength=0.3)
            cv2.putText(vis, "IN", tip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
            f = self.engine.footfall
            cv2.putText(vis, f"IN {f['in']}  OUT {f['out']}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if "queue" in self.roles:
            self.update_queue(now, in_queue)
            cv2.polylines(vis, [qpoly], True, (0, 200, 255), 2)
            q = self.engine.queues.get(self.name, {})
            cv2.putText(vis, f"QUEUE {q.get('length', 0)}  WAIT {q.get('wait_min', 0)}m", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return vis

    def update_queue(self, now, in_queue):
        qc = CONFIG["queue"]
        for tid in in_queue:
            self.missing.pop(tid, None)
            if tid in self.members:
                continue
            t0 = self.cand.setdefault(tid, now)
            if now - t0 >= qc["min_time_in_zone_s"]:      # ignore people just walking past
                self.members[tid] = t0
                self.arrivals.append(t0)
                del self.cand[tid]
        for tid in [t for t in self.cand if t not in in_queue]:
            del self.cand[tid]
        for tid in list(self.members):
            if tid not in in_queue:
                m = self.missing.setdefault(tid, now)
                if now - m > 2.0:                          # grace for detection flicker
                    t0 = self.members.pop(tid)
                    self.missing.pop(tid)
                    self.departures.append(m)
                    self.engine.on_served(self.name, m - t0)

        win = qc["rate_window_min"] * 60
        for dq in (self.arrivals, self.departures):
            while dq and dq[0] < now - win:
                dq.popleft()
        elapsed = min(win, now - self.t_start)
        mins = max(elapsed / 60, 0.5)
        lam, dep = len(self.arrivals) / mins, len(self.departures) / mins
        k, L = self.engine.open_counters, len(self.members)
        if L >= k and elapsed > 60 and dep > 0:           # counters busy -> departures reflect service rate
            self.mu_c = 0.9 * self.mu_c + 0.1 * (dep / k)
        cap = max(self.mu_c * k, 1e-3)

        forecast = {}
        for t in (5, 10, 15):                              # fluid queue approximation
            Lt = max(0.0, L + (lam - cap) * t)
            forecast[str(t)] = {"len": round(Lt, 1), "wait": round(Lt / cap, 1)}

        rec = qc["max_counters"]
        for n in range(1, qc["max_counters"] + 1):
            capn = self.mu_c * n
            L10 = max(0.0, L + (lam - capn) * 10)
            if L10 / capn <= qc["target_wait_min"] and lam <= 0.9 * capn and L / capn <= qc["max_wait_min"]:
                rec = n
                break
        self.engine.on_queue(self.name, {
            "length": L, "arrival_per_min": round(lam, 2), "service_per_counter_min": round(self.mu_c, 2),
            "wait_min": round(L / cap, 1), "forecast": forecast, "recommend_counters": rec,
        })


class ShelfWorker(CamWorker):
    """Grid over the opposite shelf. Each cell compared with a calibrated 'fully stocked' reference."""

    def __init__(self, name, roles, source, engine):
        super().__init__(name, roles, source, engine)
        from ultralytics import YOLO
        sc = CONFIG["shelf"]
        self.period = sc["period_s"]
        self.model = YOLO(CONFIG["person_model"])
        self.sku = YOLO(CONFIG["sku_model"]) if CONFIG["sku_model"] else None
        self.R, self.C = sc["grid"]
        self.clahe = cv2.createCLAHE(2.0, (8, 8))
        self.ref_path = f"shelf_ref_{name}.png"
        self.ref = cv2.imread(self.ref_path) if os.path.exists(self.ref_path) else None
        self.ref_feats = self.features(self.ref) if self.ref is not None else None
        self.recent = defaultdict(lambda: deque(maxlen=3))
        self.history = defaultdict(lambda: deque(maxlen=600))
        self.cells_map, self.calib_request, self.calib_msg = {}, False, None

    def rects(self, w, h):
        for r in range(self.R):
            for c in range(self.C):
                yield r, c, int(c * w / self.C), int(r * h / self.R), int((c + 1) * w / self.C), int((r + 1) * h / self.R)

    def features(self, img):
        h, w = img.shape[:2]
        gray = self.clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        edges = cv2.Canny(gray, 60, 160)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        feats = {}
        for r, c, x0, y0, x1, y1 in self.rects(w, h):
            hist = cv2.calcHist([hsv[y0:y1, x0:x1]], [0, 1], None, [18, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            feats[(r, c)] = (float((edges[y0:y1, x0:x1] > 0).mean()), hist)
        return feats

    def eta(self, r, c, now):
        sc = CONFIG["shelf"]
        pts = [(t, f) for t, f in self.history[(r, c)] if t >= now - sc["eta_window_min"] * 60]
        if len(pts) < 4:
            return None
        t = np.array([p[0] for p in pts]) - pts[0][0]
        f = np.array([p[1] for p in pts])
        if t[-1] < 60:
            return None
        slope = np.polyfit(t / 60, f, 1)[0]
        if slope >= -0.005:
            return None
        e = (f[-1] - sc["empty"]) / -slope
        return round(float(e), 1) if 0 < e < 600 else None

    def step(self, frame):
        sc = CONFIG["shelf"]
        h, w = frame.shape[:2]
        now = time.time()
        pr = self.model(frame, classes=[0], conf=CONFIG["conf"], imgsz=CONFIG["imgsz"], verbose=False)[0]
        persons = pr.boxes.xyxy.cpu().numpy().tolist() if pr.boxes is not None else []
        self.engine.on_aisle(self.name, len(persons), max(self.period, 0.1))

        if self.calib_request:
            self.calib_request = False
            if persons:
                self.calib_msg = "Aisle not clear — ask people to step out and retry"
            else:
                cv2.imwrite(self.ref_path, frame)
                self.ref, self.ref_feats = frame.copy(), self.features(frame)
                self.recent.clear()
                self.history.clear()
                self.cells_map = {}
                self.calib_msg = "Calibrated"

        vis = frame.copy()
        if CONFIG["privacy_blur"]:
            blur_heads(vis, persons)
        if self.ref is None or self.ref.shape != frame.shape:
            cv2.putText(vis, "NEEDS CALIBRATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.engine.on_shelf(self.name, None)
            return vis

        feats = self.features(frame)
        found_map = {}
        if self.sku is not None:
            sr = self.sku(frame, conf=CONFIG["conf"], imgsz=CONFIG["imgsz"], verbose=False)[0]
            votes = defaultdict(Counter)
            for bb, cls in zip(sr.boxes.xyxy.cpu().numpy(), sr.boxes.cls.int().cpu().tolist()):
                cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                votes[(min(int(cy * self.R / h), self.R - 1), min(int(cx * self.C / w), self.C - 1))][self.sku.names[cls]] += 1
            found_map = {k: v.most_common(1)[0][0] for k, v in votes.items()}
        plan = sc["planogram"].get(self.name, {})
        colors = {"OK": (80, 200, 80), "LOW": (0, 180, 240), "EMPTY": (60, 60, 230), "MISPLACED": (220, 90, 160)}

        cells = []
        for r, c, x0, y0, x1, y1 in self.rects(w, h):
            area = max((x1 - x0) * (y1 - y0), 1)
            occluded = any(max(0, min(x1, px2) - max(x0, px1)) * max(0, min(y1, py2) - max(y0, py1)) / area
                           >= sc["occlusion_overlap"] for px1, py1, px2, py2 in persons)
            prev = self.cells_map.get((r, c))
            if occluded:
                cell = dict(prev) if prev else {"r": r, "c": c, "fill": None, "corr": None, "status": "OK",
                                                "eta_min": None, "expected": None, "found": None}
                cell["occluded"] = True
            else:
                e, hist = feats[(r, c)]
                re_, rhist = self.ref_feats[(r, c)]
                fill = min(1.0, e / max(re_, 1e-3))
                corr = float(cv2.compareHist(rhist, hist, cv2.HISTCMP_CORREL))
                self.recent[(r, c)].append(fill)
                fill_s = float(np.median(self.recent[(r, c)]))
                self.history[(r, c)].append((now, fill_s))
                status = "EMPTY" if fill_s < sc["empty"] else "LOW" if fill_s < sc["low"] else "OK"
                expected, found = plan.get(f"{r},{c}"), found_map.get((r, c))
                if status != "EMPTY":
                    if self.sku is not None and expected and found and found != expected:
                        status = "MISPLACED"
                    elif self.sku is None and status == "OK" and corr < sc["misplace_corr"]:
                        status = "MISPLACED"
                cell = {"r": r, "c": c, "fill": round(fill_s, 2), "corr": round(corr, 2), "status": status,
                        "eta_min": self.eta(r, c, now), "expected": expected, "found": found, "occluded": False}
            cells.append(cell)
            self.cells_map[(r, c)] = cell
            col = colors[cell["status"]]
            ov = vis.copy()
            cv2.rectangle(ov, (x0, y0), (x1, y1), col, -1)
            cv2.addWeighted(ov, 0.12 if occluded else 0.28, vis, 1 - (0.12 if occluded else 0.28), 0, vis)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 1)
            if cell["fill"] is not None:
                cv2.putText(vis, f"{cell['fill'] * 100:.0f}%", (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        self.engine.on_shelf(self.name, cells)
        return vis


# ─────────────────────────────── BACKGROUND JOBS ───────────────────────────────
def ticker(engine, store):
    while True:
        time.sleep(60)
        try:
            s = engine.snapshot()
            allcells = [c for cells in s["shelves"].values() if cells for c in cells]
            m = {
                "inside": s["footfall"]["inside"], "entries_today": s["footfall"]["in"],
                "queue_len": sum(q["length"] for q in s["queues"].values()),
                "queue_wait_min": max([q["wait_min"] for q in s["queues"].values()] or [0]),
                "empty_cells": sum(c["status"] == "EMPTY" for c in allcells),
                "low_cells": sum(c["status"] == "LOW" for c in allcells),
                "planogram_issues": sum(c["status"] == "MISPLACED" for c in allcells),
                "active_alerts": len(s["alerts"]),
            }
            store.metrics(time.time(), m)
            engine.refresh_daily()
            if CONFIG["cloud_url"]:
                store.outbox_add({"store_id": CONFIG["store_id"], "ts": time.time(), "metrics": m,
                                  "alerts": s["alerts"][:10]})
        except Exception as e:
            print(f"[ticker] {e}")


def cloud_sync(engine, store):
    """Offline-first: summaries queue in SQLite, flushed whenever HQ is reachable."""
    while True:
        time.sleep(30)
        if not CONFIG["cloud_url"]:
            continue
        rows = store.outbox_pending()
        if not rows:
            continue
        try:
            body = json.dumps([json.loads(p) for _, p in rows]).encode()
            req = urllib.request.Request(CONFIG["cloud_url"], body, {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
            store.outbox_mark([i for i, _ in rows])
            engine.online = True
        except Exception:
            engine.online = False


# ─────────────────────────────── API + DASHBOARD ───────────────────────────────
DASHBOARD = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StoreSense Edge</title><style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--text:#e6e8ee;--mute:#8b93a7;--ok:#2fbf71;--low:#f0b429;--empty:#ef4444;--mis:#a78bfa;--acc:#4f8cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif}
header{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;border-bottom:1px solid var(--line);gap:10px;flex-wrap:wrap}
header b{font-size:17px}.pill{padding:3px 10px;border-radius:99px;background:#1f2430;color:var(--mute);font-size:12px}
main{padding:16px 20px;display:grid;gap:14px;grid-template-columns:repeat(12,1fr)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}
.kpis{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi .v{font-size:24px;font-weight:650}.kpi .l{color:var(--mute);font-size:12px}
.s4{grid-column:span 4}.s6{grid-column:span 6}.s8{grid-column:span 8}.s12{grid-column:1/-1}
@media(max-width:1000px){.s4,.s6,.s8{grid-column:1/-1}}
h3{margin:0 0 10px;font-size:12px;color:var(--mute);text-transform:uppercase;letter-spacing:.06em;display:flex;justify-content:space-between;align-items:center;gap:8px}
.alert{display:flex;gap:10px;align-items:center;padding:8px 10px;border-radius:8px;border-left:4px solid;margin-bottom:6px;background:#1c2029}
.critical{border-color:var(--empty)}.warning{border-color:var(--low)}.info{border-color:var(--acc)}
.alert .m{flex:1;min-width:0}.alert .a{font-weight:650}.alert small{color:var(--mute)}
button,input{background:#232838;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 10px;font:inherit}
button{cursor:pointer}button:hover{border-color:var(--acc)}input{width:60px}
.grid{display:grid;gap:3px;margin-bottom:10px}.cell{border-radius:4px;padding:7px 2px;text-align:center;font-size:11px;font-weight:650;color:#0b0d12}
.OK{background:var(--ok)}.LOW{background:var(--low)}.EMPTY{background:var(--empty);color:#fff}.MISPLACED{background:var(--mis)}.occ{opacity:.4}
table{width:100%;border-collapse:collapse}td,th{padding:5px 4px;border-bottom:1px solid var(--line);text-align:left;font-size:13px}th{color:var(--mute);font-weight:500}
.feeds{display:flex;gap:10px;flex-wrap:wrap}.feeds figure{margin:0}.feeds img{width:340px;max-width:100%;border-radius:8px;border:1px solid var(--line);display:block}
.feeds figcaption{color:var(--mute);font-size:12px;margin-top:4px}
canvas{width:100%;border-radius:6px;image-rendering:pixelated;background:#0b0d12}
.bars{display:flex;gap:4px;align-items:flex-end;height:110px;padding-top:14px}.bar{flex:1;background:var(--acc);border-radius:3px 3px 0 0;position:relative;min-height:2px}
.bar i{position:absolute;bottom:-17px;left:0;right:0;text-align:center;font-size:10px;color:var(--mute);font-style:normal}
.bar b{position:absolute;top:-15px;left:0;right:0;text-align:center;font-size:10px;font-weight:500}
.fc{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;text-align:center}.fc div{background:#1c2029;border-radius:8px;padding:8px}.fc small{color:var(--mute)}
.muted{color:var(--mute)}pre{white-space:pre-wrap;margin:0}
</style></head><body>
<header><b>StoreSense Edge</b><span><span class="pill" id="store"></span> <span class="pill" id="conn">connecting…</span> <span class="pill" id="cloud"></span></span></header>
<main>
<section class="kpis" id="kpis"></section>
<section class="card s6"><h3>Active alerts <span class="muted" id="acount"></span></h3><div id="alerts"></div></section>
<section class="card s6"><h3>Queue intelligence <span>counters open <input id="ctr" type="number" min="1" max="10"> <button onclick="setCounters()">Set</button></span></h3><div id="queues"></div></section>
<section class="card s6"><h3>Shelf grid</h3><div id="shelves"></div></section>
<section class="card s6"><h3>Floor heatmap &amp; zone dwell</h3><div id="heat"></div></section>
<section class="card s6"><h3>Footfall today (per hour)</h3><div class="bars" id="hourly"></div><div id="aisle" style="margin-top:26px"></div></section>
<section class="card s6"><h3>Reports <span><button onclick="report('day')">Daily</button> <button onclick="report('week')">Weekly</button></span></h3><div id="report" class="muted">Pick a period.</div></section>
<section class="card s12"><h3>Live cameras (heads blurred, nothing recorded)</h3><div class="feeds" id="feeds"></div></section>
</main>
<script>
const $=id=>document.getElementById(id);let feedsBuilt=false,ctrSet=false,poll=null;
function startPoll(){if(poll)return;$('conn').textContent='● live (polling)';
 poll=setInterval(async()=>{try{render(await(await fetch('/api/state')).json())}catch(e){}},1000)}
function connect(){let opened=false;const ws=new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+'/ws');
ws.onopen=()=>{opened=true;if(poll){clearInterval(poll);poll=null}$('conn').textContent='● live'};
ws.onmessage=e=>render(JSON.parse(e.data));
ws.onclose=()=>{if(!opened)return startPoll();      // no WebSocket support on the server -> poll instead
 $('conn').textContent='reconnecting…';setTimeout(connect,2000)}}connect();
const kpi=(l,v)=>`<div class="card kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`;
const ago=t=>{const s=Math.round(Date.now()/1000-t);return s<60?s+'s ago':Math.round(s/60)+'m ago'};
function heatColor(v){const r=Math.round(255*Math.min(1,v*2)),g=Math.round(255*Math.min(1,Math.max(0,2-2*v))*Math.min(1,v*3)),b=Math.round(160*(1-v));return `rgb(${r},${g},${b})`}
function render(s){
 $('store').textContent=s.store_id;$('cloud').textContent=s.cloud_online==null?'edge only':s.cloud_online?'cloud synced':'offline · buffering';
 if(!ctrSet){$('ctr').value=s.open_counters;ctrSet=true}
 const q=Object.values(s.queues),qlen=q.reduce((a,x)=>a+x.length,0),wait=q.length?Math.max(...q.map(x=>x.wait_min)):0,
 rec=q.length?Math.max(...q.map(x=>x.recommend_counters)):s.open_counters;let low=0,emp=0,mis=0;
 Object.values(s.shelves).forEach(c=>(c||[]).forEach(x=>{if(x.status==='LOW')low++;if(x.status==='EMPTY')emp++;if(x.status==='MISPLACED')mis++}));
 $('kpis').innerHTML=kpi('Inside now',s.footfall.inside)+kpi('Entries today',s.footfall.in)+kpi('Exits today',s.footfall.out)+
 kpi('In queue',qlen)+kpi('Est. wait (min)',wait.toFixed(1))+kpi('Counters open / needed',`${s.open_counters} / ${rec}`)+
 kpi('Low / empty cells',`${low} / ${emp}`)+kpi('Planogram issues',mis)+
 kpi('Billing conversion',s.conversion==null?'—':Math.round(s.conversion*100)+'%')+kpi('Avg staff response',s.avg_response_s==null?'—':Math.round(s.avg_response_s)+'s');
 $('acount').textContent=s.alerts.length;
 $('alerts').innerHTML=s.alerts.length?s.alerts.map(a=>`<div class="alert ${a.severity}"><div class="m"><div class="a">${a.action}</div><div>${a.message}</div><small>${a.kind} · ${ago(a.ts)} · priority ${a.priority}${a.count>1?' · still open, seen '+a.count+'×':''}</small></div><button onclick="ack(${a.id})">Done</button></div>`).join(''):'<div class="muted">All clear.</div>';
 $('queues').innerHTML=Object.entries(s.queues).map(([cam,x])=>`<div class="muted" style="margin-bottom:6px">${cam}: ${x.arrival_per_min}/min arriving · ${x.service_per_counter_min}/min served per counter${s.avg_time_in_queue_min!=null?' · avg time in queue '+s.avg_time_in_queue_min+' min':''}</div>
 <div class="fc"><div><b>${x.length}</b><br><small>now · ${x.wait_min}m</small></div>${['5','10','15'].map(t=>`<div><b>${x.forecast[t].len}</b><br><small>+${t}m · ${x.forecast[t].wait}m wait</small></div>`).join('')}</div>
 <p>Recommended counters: <b>${x.recommend_counters}</b></p>`).join('')||'<div class="muted">No queue camera.</div>';
 $('shelves').innerHTML=Object.entries(s.shelves).map(([cam,cells])=>{
  const head=`<h3>${cam}<button onclick="calib('${cam}')">Calibrate</button></h3>`;
  if(!cells)return head+'<p class="muted">Needs calibration: clear the aisle, stock the shelf fully, press Calibrate.</p>';
  const C=Math.max(...cells.map(c=>c.c))+1;
  return head+`<div class="grid" style="grid-template-columns:repeat(${C},1fr)">${cells.map(c=>`<div class="cell ${c.status} ${c.occluded?'occ':''}" title="${c.status}${c.eta_min!=null?' · empty in ~'+c.eta_min+'m':''}${c.expected?' · expected '+c.expected:''}">${c.fill==null?'?':Math.round(c.fill*100)+'%'}${c.eta_min!=null?'<br>⏱'+Math.round(c.eta_min)+'m':''}</div>`).join('')}</div>`}).join('')||'<div class="muted">No shelf cameras.</div>';
 $('heat').innerHTML=Object.keys(s.heat).map(cam=>{const z=Object.entries((s.zones||{})[cam]||{});
  return `<div class="muted">${cam}</div><canvas id="h_${cam}"></canvas>`+(z.length?
  `<table><tr><th>Zone</th><th>Visits</th><th>Avg dwell</th></tr>${z.map(([n,v])=>`<tr><td>${n}</td><td>${v.visits}</td><td>${v.avg_dwell_s}s</td></tr>`).join('')}</table>`
  :'<div class="muted" style="margin-top:6px">No completed zone visits yet.</div>')}).join('')||'<div class="muted">No entry camera.</div>';
 Object.entries(s.heat).forEach(([cam,g])=>{const cv=$('h_'+cam);if(!cv)return;const R=g.length,C=g[0].length,px=24;cv.width=C*px;cv.height=R*px;
  const x=cv.getContext('2d');g.forEach((row,r)=>row.forEach((v,c)=>{x.fillStyle=v>0?heatColor(v):'#12151c';x.fillRect(c*px,r*px,px-1,px-1)}))});
 const hrs=Object.entries(s.hourly),mx=Math.max(1,...hrs.map(h=>h[1]));
 $('hourly').innerHTML=hrs.length?hrs.map(([h,v])=>`<div class="bar" style="height:${100*v/mx}%"><b>${v}</b><i>${h}</i></div>`).join(''):'<div class="muted">No entries yet today.</div>';
 $('aisle').innerHTML=Object.entries(s.aisle).map(([cam,a])=>`<div class="muted">${cam} aisle: ${a.people} shopper(s) now · ${a.person_minutes} shopper-minutes today</div>`).join('');
 if(!feedsBuilt&&Object.keys(s.cams).length){feedsBuilt=true;$('feeds').innerHTML=Object.entries(s.cams).map(([n,c])=>`<figure><img src="/video/${n}"><figcaption>${n} · ${c.roles.join(', ')}</figcaption></figure>`).join('')}
}
async function ack(id){await fetch('/api/alerts/'+id+'/ack',{method:'POST'})}
async function calib(cam){const r=await fetch('/api/calibrate/'+cam,{method:'POST'});const j=await r.json();alert(j.message)}
async function setCounters(){await fetch('/api/counters?open='+$('ctr').value,{method:'POST'})}
async function report(p){const j=await(await fetch('/api/report?period='+p)).json();
 const rows=[['Footfall',j.footfall_total],['Peak',j.peak||'—'],['Customers billed',j.customers_billed],['Conversion',j.conversion==null?'—':Math.round(j.conversion*100)+'%'],
 ['Avg time in queue',j.avg_time_in_queue_min==null?'—':j.avg_time_in_queue_min+' min'],['Avg staff response',j.avg_alert_response_s==null?'—':j.avg_alert_response_s+' s'],
 ['Avg queue wait',j.metrics.queue_wait_min?j.metrics.queue_wait_min.avg+' min (max '+j.metrics.queue_wait_min.max+')':'—'],
 ['Avg empty cells',j.metrics.empty_cells?j.metrics.empty_cells.avg:'—']];
 $('report').innerHTML=`<table>${rows.map(r=>`<tr><td>${r[0]}</td><td><b>${r[1]}</b></td></tr>`).join('')}</table>
 <p class="muted">Alerts: ${Object.entries(j.alerts_by_action).map(([k,v])=>k+' ×'+v).join(', ')||'none'}</p>
 <p class="muted">Footfall: ${Object.entries(j.footfall_by_bucket).map(([k,v])=>k+': '+v).join(' · ')||'—'}</p>
 <a href="/api/report?period=${p}" target="_blank" style="color:var(--acc)">Download JSON</a>`}
</script></body></html>"""


def make_app(engine, workers, store):
    import asyncio
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
    from fastapi.responses import HTMLResponse, StreamingResponse

    app = FastAPI(title="StoreSense Edge")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return DASHBOARD

    @app.get("/api/state")
    def state():
        return engine.snapshot()

    @app.get("/api/report")
    def report(period: str = "day"):
        return store.report(period)

    @app.post("/api/alerts/{aid}/ack")
    def ack(aid: int):
        return {"ok": engine.ack(aid)}

    @app.post("/api/counters")
    def counters(open: int):
        engine.open_counters = max(1, open)
        return {"open_counters": engine.open_counters}

    @app.post("/api/calibrate/{cam}")
    def calibrate(cam: str):
        w = workers.get(cam)
        if not isinstance(w, ShelfWorker):
            raise HTTPException(404, "not a shelf camera")
        w.calib_msg, w.calib_request = None, True
        for _ in range(int((w.period + 10) / 0.2)):
            if w.calib_msg:
                break
            time.sleep(0.2)
        return {"message": w.calib_msg or "Camera not responding"}

    @app.post("/api/integrations/pos")
    async def pos(req: Request):
        """POS / ERP hook: {"bill_id": "...", "items": [{"sku": "...", "qty": 2}]}"""
        data = await req.json()
        store.event("pos", "pos", data)
        return {"ok": True}

    @app.get("/video/{cam}")
    def video(cam: str):
        w = workers.get(cam)
        if not w:
            raise HTTPException(404)

        def gen():
            while True:
                if w.jpg:
                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + w.jpg + b"\r\n"
                time.sleep(0.1)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                await sock.send_json(engine.snapshot())
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


# ─────────────────────────────── SETUP TOOL + MAIN ───────────────────────────────
def pick(source):
    cap = Capture(source)
    f, t = None, time.time()
    while f is None and time.time() - t < 10:
        f = cap.read()
        time.sleep(0.1)
    if f is None:
        sys.exit("No frame from source")
    h, w = f.shape[:2]
    pts = []

    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append([round(x / w, 3), round(y / h, 3)])
            print(pts[-1])

    cv2.namedWindow("pick")
    cv2.setMouseCallback("pick", cb)
    while True:
        vis = f.copy()
        for i, (x, y) in enumerate(pts):
            p = (int(x * w), int(y * h))
            cv2.circle(vis, p, 5, (0, 255, 255), -1)
            cv2.putText(vis, str(i), (p[0] + 6, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if len(pts) > 1:
            cv2.polylines(vis, [np.int32([[x * w, y * h] for x, y in pts])], False, (0, 255, 255), 1)
        cv2.imshow("pick", vis)
        k = cv2.waitKey(30) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("c"):
            pts.clear()
    cv2.destroyAllWindows()
    print(json.dumps(pts))


def deep_merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = v


def main():
    ap = argparse.ArgumentParser(description="StoreSense Edge — on-device retail intelligence")
    ap.add_argument("--cam", nargs=3, action="append", metavar=("NAME", "ROLES", "SOURCE"),
                    help="roles: entry, queue (can combine: entry,queue) or shelf")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", help="optional JSON merged over CONFIG")
    ap.add_argument("--sku-model", help="optional YOLO SKU weights for planogram checks")
    ap.add_argument("--cloud-url")
    ap.add_argument("--pick", metavar="SOURCE", help="click points on a frame to get geometry coords")
    args = ap.parse_args()

    if args.config:
        with open(args.config) as fh:
            deep_merge(CONFIG, json.load(fh))
    if args.sku_model:
        CONFIG["sku_model"] = args.sku_model
    if args.cloud_url:
        CONFIG["cloud_url"] = args.cloud_url
    if args.pick:
        return pick(args.pick)

    try:                       # nothing leaves the device: kill the model library's usage telemetry
        from ultralytics import settings as ul_settings
        ul_settings.update({"sync": False})
    except Exception:
        pass

    store = Store(CONFIG["db_path"])
    engine = Engine(store)
    workers = {}
    for name, roles, src in args.cam or [["cam0", "entry,queue", "0"]]:
        roles = {r.strip() for r in roles.split(",") if r.strip()}
        if not roles <= {"entry", "queue", "shelf"}:
            sys.exit(f"Unknown role in {roles}")
        if "shelf" in roles and len(roles) > 1:
            sys.exit(f"{name}: a shelf camera can't also be entry/queue")
        cls = ShelfWorker if "shelf" in roles else PeopleWorker
        workers[name] = cls(name, roles, src, engine)
        workers[name].start()
        print(f"[{name}] {','.join(sorted(roles))} <- {src}")

    threading.Thread(target=ticker, args=(engine, store), daemon=True).start()
    threading.Thread(target=cloud_sync, args=(engine, store), daemon=True).start()

    import uvicorn
    try:
        import websockets  # noqa: F401
    except ImportError:
        try:
            import wsproto  # noqa: F401
        except ImportError:
            print("note: no WebSocket library — dashboard will poll. pip install 'uvicorn[standard]' for push updates.")
    print(f"Dashboard: http://localhost:{args.port}")
    uvicorn.run(make_app(engine, workers, store), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
