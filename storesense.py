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
import argparse, json, math, os, sys, sqlite3, threading, time, urllib.request
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
    "layout_path": "layout.json",   # the store model: shelves and cameras in metres
    "store_cell_m": 0.25,           # floor-heatmap resolution in the store frame
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


# ─────────────────────────────── STORE LAYOUT ───────────────────────────────
# The store is modelled top-down in metres. Shelves are rotatable rectangles with up
# to four monitored faces (N/E/S/W in the shelf's own frame); cameras are points with
# a heading, a field of view and a range. Nothing assumes cameras face each other:
# which shelf face a camera actually watches is derived from the geometry, so any
# arrangement works — one camera covering three shelves, or six around an island.

FACES = ("N", "E", "S", "W")
DEFAULT_LAYOUT = {"store": {"w": 12.0, "h": 8.0}, "shelves": [], "cameras": []}


def rot_pt(px, py, cx, cy, deg):
    a = math.radians(deg)
    s, c = math.sin(a), math.cos(a)
    dx, dy = px - cx, py - cy
    return cx + dx * c - dy * s, cy + dx * s + dy * c


def rect_corners(r):
    """Corners of a centre-based rotated rect, clockwise from the local top-left."""
    x, y, w, h, rot = r["x"], r["y"], r["w"], r["h"], r.get("rot", 0)
    pts = [(x - w / 2, y - h / 2), (x + w / 2, y - h / 2), (x + w / 2, y + h / 2), (x - w / 2, y + h / 2)]
    return [rot_pt(px, py, x, y, rot) for px, py in pts]


def face_segment(shelf, face):
    c = rect_corners(shelf)
    return {"N": (c[0], c[1]), "E": (c[1], c[2]), "S": (c[2], c[3]), "W": (c[3], c[0])}[face]


def face_normal(shelf, face):
    n = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}[face]
    return rot_pt(n[0], n[1], 0, 0, shelf.get("rot", 0))


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def seg_cross(a, b, c, d):
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def sight_blocked(p, q, shelves, skip_id):
    """Does any other shelf stand between p and q?"""
    for s in shelves:
        if s.get("id") == skip_id:
            continue
        c = rect_corners(s)
        for i in range(4):
            if seg_cross(p, q, c[i], c[(i + 1) % 4]):
                return True
    return False


def face_visibility(cam, shelf, face, shelves, samples=7):
    """How much of a shelf face this camera really sees: 0..1 over FOV, range and occlusion."""
    (x0, y0), (x1, y1) = face_segment(shelf, face)
    cp = (cam["x"], cam["y"])
    nx, ny = face_normal(shelf, face)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    if (cp[0] - mx) * nx + (cp[1] - my) * ny <= 0:
        return 0.0                                   # camera is behind this face
    half = math.radians(cam.get("fov", 70)) / 2
    rng = cam.get("range", 8.0)
    head = math.radians(cam.get("heading", 0))
    seen = 0
    for i in range(samples):
        t = (i + 0.5) / samples
        p = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        d = math.hypot(p[0] - cp[0], p[1] - cp[1])
        if d > rng or d < 1e-6:
            continue
        ang = (math.atan2(p[1] - cp[1], p[0] - cp[0]) - head + math.pi) % (2 * math.pi) - math.pi
        if abs(ang) > half:
            continue
        if sight_blocked(cp, p, shelves, shelf.get("id")):
            continue
        seen += 1
    return seen / samples


def coverage(layout):
    """Best camera per shelf face, so the UI can show blind spots and workers can self-assign."""
    shelves = layout.get("shelves", [])
    cams = [c for c in layout.get("cameras", []) if "shelf" in c.get("roles", [])]
    out = {}
    for s in shelves:
        for f in s.get("faces", {}):
            best_v, best_c = 0.0, None
            for cm in cams:
                v = face_visibility(cm, s, f, shelves)
                if v > best_v:
                    best_v, best_c = v, cm.get("id")
            out[f"{s['id']}:{f}"] = {"shelf": s["id"], "shelf_name": s.get("name", s["id"]), "face": f,
                                     "visible": round(best_v, 2), "camera": best_c}
    return out


class Layout:
    """The store model, loaded from / saved to a JSON file the editor writes."""

    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        self.data = dict(DEFAULT_LAYOUT)
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    self.data = self.normalise(json.load(fh))
            except Exception as e:
                print(f"[layout] ignoring {path}: {e}")

    @staticmethod
    def normalise(d):
        out = {"store": {"w": float(d.get("store", {}).get("w", 12)), "h": float(d.get("store", {}).get("h", 8))},
               "shelves": [], "cameras": []}
        for i, s in enumerate(d.get("shelves", [])):
            sid = str(s.get("id") or f"S{i + 1}")
            faces = s.get("faces") or {"N": {"grid": list(CONFIG["shelf"]["grid"])}}
            out["shelves"].append({
                "id": sid, "name": s.get("name", sid), "x": float(s["x"]), "y": float(s["y"]),
                "w": float(s["w"]), "h": float(s["h"]), "rot": float(s.get("rot", 0)),
                "height": float(s.get("height", 1.8)),
                "faces": {f: {"grid": [int(v) for v in (faces[f] or {}).get("grid", CONFIG["shelf"]["grid"])]}
                          for f in faces if f in FACES},
            })
        for i, c in enumerate(d.get("cameras", [])):
            cid = str(c.get("id") or f"cam{i + 1}")
            cam = {"id": cid, "name": c.get("name", cid), "x": float(c["x"]), "y": float(c["y"]),
                   "heading": float(c.get("heading", 0)), "fov": float(c.get("fov", 70)),
                   "range": float(c.get("range", 8)), "height": float(c.get("height", 2.2)),
                   "roles": [r for r in c.get("roles", []) if r in ("entry", "queue", "shelf")] or ["shelf"],
                   "source": c.get("source", "")}
            if c.get("watch"):
                cam["watch"] = {"shelf": c["watch"]["shelf"], "face": c["watch"]["face"]}
            if c.get("floor_rect"):
                fr = c["floor_rect"]
                cam["floor_rect"] = {k: float(fr.get(k, 0)) for k in ("x", "y", "w", "h", "rot")}
            out["cameras"].append(cam)
        return out

    def save(self, data):
        with self.lock:
            self.data = self.normalise(data)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self.data, fh, indent=2)
            os.replace(tmp, self.path)
        return self.data

    def cam(self, cam_id):
        return next((c for c in self.data["cameras"] if c["id"] == cam_id), None)

    def shelf(self, shelf_id):
        return next((s for s in self.data["shelves"] if s["id"] == shelf_id), None)

    def watched(self, cam_id):
        """(shelf, face) this camera monitors: explicit if set, otherwise the face it sees best."""
        c = self.cam(cam_id)
        if not c:
            return None, None
        if c.get("watch"):
            s = self.shelf(c["watch"]["shelf"])
            if s and c["watch"]["face"] in s["faces"]:
                return s, c["watch"]["face"]
        best = (0.0, None, None)
        for s in self.data["shelves"]:
            for f in s["faces"]:
                v = face_visibility(c, s, f, self.data["shelves"])
                if v > best[0]:
                    best = (v, s, f)
        return (best[1], best[2]) if best[0] > 0 else (None, None)

    def grid_dims(self):
        m = max(CONFIG["store_cell_m"], 0.05)
        return max(1, int(round(self.data["store"]["h"] / m))), max(1, int(round(self.data["store"]["w"] / m)))


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

    def __init__(self, store, layout=None):
        self.store, self.lock = store, threading.RLock()
        self.layout = layout
        # one floor heatmap for the whole store, in metres — every camera feeds the same grid
        self.store_heat = np.zeros(layout.grid_dims(), np.float32) if layout else None
        self.footfall = {"in": 0, "out": 0}
        self.hourly, self.conversion, self.avg_response_s = {}, None, None
        self.alerts, self.last, self.keys, self.next_id = deque(maxlen=300), {}, {}, 1
        self.queues, self.shelves, self.heat, self.aisle, self.cams = {}, {}, {}, {}, {}
        self.labels = {}              # camera -> human name from the layout ("Aisle 1 A, S face")
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
        where = self.labels.get(cam, cam)
        for c in cells:
            if c["occluded"]:
                continue
            key, loc = f"shelf:{cam}:{c['r']},{c['c']}", f"{where} row {c['r'] + 1} col {c['c'] + 1}"
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
                "store_heat": self.store_heat_norm(), "layout": self.layout.data if self.layout else None,
            }

    def store_heat_norm(self):
        h = self.store_heat
        if h is None:
            return None
        m = float(h.max())
        return (h / m if m > 0 else h).round(3).tolist()

    def add_store_heat(self, mx, my, dt):
        """Drop dwell seconds onto the store-wide floor grid, in metres."""
        h = self.store_heat
        if h is None:
            return
        cell = max(CONFIG["store_cell_m"], 0.05)
        r, c = int(my / cell), int(mx / cell)
        if 0 <= r < h.shape[0] and 0 <= c < h.shape[1]:
            h[r, c] += dt


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
        self.storeH = None            # image -> store metres, built on the first frame

    def store_point(self, fx, fy, w, h):
        """Map a foot position to store metres, if this camera is placed in the layout.

        The four floor points clicked with --pick correspond to the floor_rect the user
        dragged onto the plan, which is what ties every camera into one shared frame.
        """
        lay = self.engine.layout
        quad = self.g.get("floor_quad")
        if lay is None or not quad:
            return None
        cam = lay.cam(self.name)
        if not cam or not cam.get("floor_rect"):
            return None
        if self.storeH is None:
            src = np.float32([[x * w, y * h] for x, y in quad][:4])
            self.storeH = cv2.getPerspectiveTransform(src, np.float32(rect_corners(cam["floor_rect"])))
        mx, my = cv2.perspectiveTransform(np.float32([[[fx, fy]]]), self.storeH)[0, 0]
        return float(mx), float(my)

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
            mp = self.store_point(foot[0], foot[1], w, h)
            if mp:                              # every placed camera feeds the one store heatmap
                self.engine.add_store_heat(mp[0], mp[1], dt)
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
        # the layout decides which shelf face this camera watches and how it is divided
        self.shelf_id = self.face = None
        self.R, self.C = sc["grid"]
        if engine.layout:
            sh, fc = engine.layout.watched(name)
            if sh:
                self.shelf_id, self.face = sh["id"], fc
                self.R, self.C = sh["faces"][fc]["grid"]
                engine.labels[name] = f"{sh['name']} ({fc} face)"
                print(f"[{name}] watching {sh['name']} {fc} face, {self.R}x{self.C} cells")
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


# ─────────────────────────────── 3D STORE VIEW ───────────────────────────────
STATUS_RGB = {"OK": "#2fbf71", "LOW": "#f0b429", "EMPTY": "#ef4444", "MISPLACED": "#a78bfa", None: "#8b93a7"}


def face_status(engine, shelf_id, face):
    """Worst live status on a shelf face, so the 3D box is coloured by what staff must fix."""
    lay = engine.layout
    for cam, cells in engine.shelves.items():
        if not cells:
            continue
        sh, fc = lay.watched(cam)
        if not sh or sh["id"] != shelf_id or fc != face:
            continue
        for want in ("EMPTY", "LOW", "MISPLACED"):
            if any(c["status"] == want for c in cells):
                return want
        return "OK"
    return None


def clip_to_store(p, q, W, H):
    """Trim a sight ray at the store walls so view cones stay inside the plan (Liang-Barsky)."""
    (x0, y0), (x1, y1) = p, q
    dx, dy = x1 - x0, y1 - y0
    t = 1.0
    for num, den in ((x0, -dx), (W - x0, dx), (y0, -dy), (H - y0, dy)):
        if abs(den) < 1e-9:
            if num < 0:
                return p
            continue
        if den > 0:
            t = min(t, max(0.0, num / den))
    return (x0 + dx * t, y0 + dy * t)


def render_3d(engine, el=34, az=-62, dpi=110):
    """Matplotlib 3D axes: floor heatmap, shelves as boxes, cameras with dotted view cones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    lay = engine.layout
    W, H = lay.data["store"]["w"], lay.data["store"]["h"]
    fig = plt.figure(figsize=(9.5, 6.4), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # a filled floor polygon sorts in front of everything in mplot3d, so outline it instead
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], [0] * 5, color="#8b97ab", lw=1.4)

    heat = engine.store_heat
    quads, colors = [], []
    if heat is not None and float(heat.max()) > 0:
        cell, mx = CONFIG["store_cell_m"], float(heat.max())
        cmap = plt.get_cmap("inferno")
        for r, c in zip(*np.nonzero(heat > mx * 0.02)):
            x0, y0 = c * cell, r * cell
            z = 0.004
            quads.append([(x0, y0, z), (x0 + cell, y0, z), (x0 + cell, y0 + cell, z), (x0, y0 + cell, z)])
            colors.append(cmap(float(heat[r, c] / mx)))
    if quads:
        ax.add_collection3d(Poly3DCollection(quads, facecolors=colors, edgecolors="none", zsort="min"))

    for s in lay.data["shelves"]:
        c = rect_corners(s)
        z = s.get("height", 1.8)
        top = [(p[0], p[1], z) for p in c]
        bot = [(p[0], p[1], 0.0) for p in c]
        sides, scolors = [top], ["#c9d2e0"]
        for i, f in enumerate(("N", "E", "S", "W")):
            a, b = c[i], c[(i + 1) % 4]
            sides.append([(a[0], a[1], 0), (b[0], b[1], 0), (b[0], b[1], z), (a[0], a[1], z)])
            scolors.append(STATUS_RGB[face_status(engine, s["id"], f)] if f in s["faces"] else "#dfe4ec")
        ax.add_collection3d(Poly3DCollection(sides + [bot], facecolors=scolors + ["#c9d2e0"],
                                             edgecolors="#31405a", linewidths=0.7))
        cx, cy = s["x"], s["y"]
        ax.text(cx, cy, z + 0.18, s["name"], ha="center", fontsize=7.5, color="#1a2a4a")

    for cm in lay.data["cameras"]:
        cx, cy, cz = cm["x"], cm["y"], cm.get("height", 2.2)
        ax.scatter([cx], [cy], [cz], s=46, c="#1f6feb", marker="o", depthshade=False)
        ax.plot([cx, cx], [cy, cy], [0, cz], color="#1f6feb", lw=0.8, alpha=0.5)
        half, rng, head = math.radians(cm["fov"]) / 2, cm["range"], math.radians(cm["heading"])
        arc = [clip_to_store((cx, cy), (cx + rng * math.cos(head + a), cy + rng * math.sin(head + a)), W, H)
               for a in np.linspace(-half, half, 28)]
        for p in (arc[0], arc[-1]):                       # the two edge rays, dotted
            ax.plot([cx, p[0]], [cy, p[1]], [cz, 0], color="#1f6feb", ls=":", lw=1.1)
        ax.plot([p[0] for p in arc], [p[1] for p in arc], [0] * len(arc), color="#1f6feb", ls=":", lw=1.1)
        ax.text(cx, cy, cz + 0.25, cm["name"], ha="center", fontsize=7.5, color="#1f6feb")

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(0, max(2.6, max([s.get("height", 1.8) for s in lay.data["shelves"]] + [2.2]) + 0.5))
    ax.set_box_aspect((W, H, 2.4))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=el, azim=az)
    ax.set_title(f"{CONFIG['store_id']} — floor heatmap, shelf status and camera coverage", fontsize=11)
    fig.tight_layout()
    buf = __import__("io").BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


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
.lay{display:flex;gap:12px;flex-wrap:wrap}
#plan{background:#0b0d12;border:1px solid var(--line);border-radius:8px;flex:1 1 420px;
 width:100%;max-width:900px;height:auto;touch-action:none;cursor:crosshair}
.panel{flex:0 0 250px;font-size:12.5px}
.panel label{display:flex;justify-content:space-between;align-items:center;gap:8px;margin:5px 0;color:var(--mute)}
.panel input[type=number],.panel input[type=text]{width:92px}
.panel input[type=checkbox]{width:auto}
.panel h4{margin:2px 0 8px;font-size:13px;color:var(--text)}
.panel .row{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
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
<section class="card s12"><h3>Store layout — drag shelves and cameras, top-down (metres)
 <span><button onclick="addShelf()">+ Shelf</button> <button onclick="addCam()">+ Camera</button>
 <button onclick="saveLay()">Save</button> <button onclick="loadLay()">Reload</button></span></h3>
<div class="lay"><canvas id="plan" width="900" height="600"></canvas>
<div class="panel" id="panel"></div></div>
<div class="muted" id="blind" style="margin-top:8px"></div></section>
<section class="card s12"><h3>3D view
 <span><button onclick="spin(-30)">&#8630; rotate</button> <button onclick="spin(30)">rotate &#8631;</button>
 <button onclick="tilt(10)">tilt up</button> <button onclick="tilt(-10)">tilt down</button>
 <button onclick="shot3d()">Refresh</button></span></h3>
<img id="v3d" style="max-width:100%;border-radius:8px;border:1px solid var(--line)">
<div class="muted" style="margin-top:6px">Shelf faces are coloured by live stock status (green OK, amber low,
 red empty, violet misplaced); grey faces are not monitored. Dotted cones are camera coverage, clipped at the
 walls. Rotate to bring a face into view.</div></section>
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
 if(s.store_heat){HEAT=s.store_heat;if(!DRAG)draw()}
 if(!feedsBuilt&&Object.keys(s.cams).length){feedsBuilt=true;$('feeds').innerHTML=Object.entries(s.cams).map(([n,c])=>`<figure><img src="/video/${n}"><figcaption>${n} · ${c.roles.join(', ')}</figcaption></figure>`).join('')}
}
/* ---------- store layout editor: top-down, metres, drag and drop ---------- */
let LAY={store:{w:12,h:8},shelves:[],cameras:[]},SEL=null,DRAG=null,HEAT=null,SC=60;
const CV=$('plan'),CX=CV.getContext('2d'),SNAP=0.25,FACES=['N','E','S','W'];
const m2p=v=>v*SC, p2m=v=>v/SC;
function rot(px,py,cx,cy,d){const a=d*Math.PI/180,s=Math.sin(a),c=Math.cos(a),dx=px-cx,dy=py-cy;
 return [cx+dx*c-dy*s, cy+dx*s+dy*c]}
function corners(r){const {x,y,w,h}=r,d=r.rot||0;
 return [[x-w/2,y-h/2],[x+w/2,y-h/2],[x+w/2,y+h/2],[x-w/2,y+h/2]].map(p=>rot(p[0],p[1],x,y,d))}
function faceSeg(s,f){const c=corners(s);return {N:[c[0],c[1]],E:[c[1],c[2]],S:[c[2],c[3]],W:[c[3],c[0]]}[f]}
function faceNorm(s,f){const n={N:[0,-1],E:[1,0],S:[0,1],W:[-1,0]}[f];return rot(n[0],n[1],0,0,s.rot||0)}
const ccw=(a,b,c)=>(c[1]-a[1])*(b[0]-a[0])>(b[1]-a[1])*(c[0]-a[0]);
const cross=(a,b,c,d)=>ccw(a,c,d)!==ccw(b,c,d)&&ccw(a,b,c)!==ccw(a,b,d);
function blockedBy(p,q,skip){for(const s of LAY.shelves){if(s.id===skip)continue;const c=corners(s);
 for(let i=0;i<4;i++)if(cross(p,q,c[i],c[(i+1)%4]))return true}return false}
function vis(cam,s,f){const [a,b]=faceSeg(s,f),n=faceNorm(s,f),mx=(a[0]+b[0])/2,my=(a[1]+b[1])/2;
 if((cam.x-mx)*n[0]+(cam.y-my)*n[1]<=0)return 0;
 const half=cam.fov*Math.PI/360,head=cam.heading*Math.PI/180;let seen=0,N=7;
 for(let i=0;i<N;i++){const t=(i+0.5)/N,p=[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t];
  const d=Math.hypot(p[0]-cam.x,p[1]-cam.y);if(d>cam.range||d<1e-6)continue;
  let ang=Math.atan2(p[1]-cam.y,p[0]-cam.x)-head;ang=(ang+Math.PI)%(2*Math.PI)-Math.PI;
  if(Math.abs(ang)>half)continue;if(blockedBy([cam.x,cam.y],p,s.id))continue;seen++}
 return seen/N}
function bestVis(s,f){let v=0;for(const c of LAY.cameras)if((c.roles||[]).includes('shelf'))v=Math.max(v,vis(c,s,f));return v}
/* fixed internal resolution (CSS scales it): measuring the element instead would let the
   canvas grow to fill its flex line each frame and push the property panel onto a new row */
function fit(){const IW=1200;if(CV.width!==IW)CV.width=IW;
 SC=IW/LAY.store.w;CV.height=Math.max(200,Math.round(m2p(LAY.store.h)))}
function draw(){fit();const W=CV.width,H=CV.height;CX.clearRect(0,0,W,H);
 CX.fillStyle='#0b0d12';CX.fillRect(0,0,W,H);
 if(HEAT&&HEAT.length){const rows=HEAT.length,cols=HEAT[0].length,cw=W/cols,ch=H/rows;
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const v=HEAT[r][c];if(v>0.02){
   CX.fillStyle=`rgba(${Math.round(255*Math.min(1,v*2))},${Math.round(200*Math.max(0,1-v))},40,${0.15+0.55*v})`;
   CX.fillRect(c*cw,r*ch,cw+0.5,ch+0.5)}}}
 CX.lineWidth=1;
 for(let x=0;x<=LAY.store.w+1e-6;x+=0.5){CX.strokeStyle=(Math.abs(x%1)<1e-6)?'#232838':'#171a21';
  CX.beginPath();CX.moveTo(m2p(x),0);CX.lineTo(m2p(x),H);CX.stroke()}
 for(let y=0;y<=LAY.store.h+1e-6;y+=0.5){CX.strokeStyle=(Math.abs(y%1)<1e-6)?'#232838':'#171a21';
  CX.beginPath();CX.moveTo(0,m2p(y));CX.lineTo(W,m2p(y));CX.stroke()}
 CX.strokeStyle='#45506a';CX.lineWidth=2;CX.strokeRect(1,1,W-2,H-2);
 for(const cm of LAY.cameras){if(cm.floor_rect){const c=corners(cm.floor_rect).map(p=>[m2p(p[0]),m2p(p[1])]);
  CX.setLineDash([5,4]);CX.strokeStyle='#4f8cff';CX.fillStyle='rgba(79,140,255,.07)';
  CX.beginPath();c.forEach((p,i)=>i?CX.lineTo(p[0],p[1]):CX.moveTo(p[0],p[1]));CX.closePath();CX.fill();CX.stroke();
  CX.setLineDash([]);CX.fillStyle='#4f8cff';CX.font='10px system-ui';CX.fillText('floor patch: '+cm.name,c[0][0]+4,c[0][1]+12)}}
 for(const s of LAY.shelves){const c=corners(s).map(p=>[m2p(p[0]),m2p(p[1])]);
  CX.fillStyle=(SEL&&SEL.t==='s'&&SEL.id===s.id)?'#2b3b2f':'#1e2733';
  CX.beginPath();c.forEach((p,i)=>i?CX.lineTo(p[0],p[1]):CX.moveTo(p[0],p[1]));CX.closePath();CX.fill();
  CX.strokeStyle='#3a4658';CX.lineWidth=1;CX.stroke();
  FACES.forEach((f,i)=>{if(!s.faces[f])return;const a=c[i],b=c[(i+1)%4],v=bestVis(s,f);
   CX.lineWidth=4;CX.setLineDash(v>0?[]:[6,4]);
   CX.strokeStyle=v>=0.6?'#2fbf71':(v>0?'#f0b429':'#ef4444');
   CX.beginPath();CX.moveTo(a[0],a[1]);CX.lineTo(b[0],b[1]);CX.stroke();CX.setLineDash([])});
  CX.fillStyle='#e6e8ee';CX.font='11px system-ui';CX.textAlign='center';
  CX.fillText(s.name,m2p(s.x),m2p(s.y)+4);CX.textAlign='left'}
 for(const cm of LAY.cameras){const x=m2p(cm.x),y=m2p(cm.y),half=cm.fov*Math.PI/360,hd=cm.heading*Math.PI/180;
  CX.setLineDash([4,4]);CX.strokeStyle='#4f8cff';CX.fillStyle='rgba(79,140,255,.10)';CX.lineWidth=1.2;
  CX.beginPath();CX.moveTo(x,y);CX.arc(x,y,m2p(cm.range),hd-half,hd+half);CX.closePath();CX.fill();CX.stroke();
  CX.setLineDash([]);CX.fillStyle=(SEL&&SEL.t==='c'&&SEL.id===cm.id)?'#fff':'#4f8cff';
  CX.beginPath();CX.arc(x,y,7,0,6.3);CX.fill();
  CX.fillStyle='#9ab6ff';CX.font='11px system-ui';CX.fillText(cm.name,x+10,y-8)}
 const o=selObj();if(o){const p=handles(o);CX.fillStyle='#ffd166';
  for(const k in p){CX.beginPath();CX.arc(m2p(p[k][0]),m2p(p[k][1]),5,0,6.3);CX.fill()}}}
function selObj(){if(!SEL)return null;
 if(SEL.t==='s')return LAY.shelves.find(s=>s.id===SEL.id);
 if(SEL.t==='c')return LAY.cameras.find(c=>c.id===SEL.id);
 const cm=LAY.cameras.find(c=>c.id===SEL.id);return cm&&cm.floor_rect}
function handles(o){if(SEL.t==='c')return {head:[o.x+Math.cos(o.heading*Math.PI/180)*o.range*0.55,
  o.y+Math.sin(o.heading*Math.PI/180)*o.range*0.55]};
 const c=corners(o);return {rot:rot(o.x,o.y-o.h/2-0.45,o.x,o.y,o.rot||0),size:c[2]}}
function hit(mx,my){const o=selObj();
 if(o){const p=handles(o);for(const k in p)if(Math.hypot(mx-p[k][0],my-p[k][1])<p2m(11))return {t:SEL.t,id:SEL.id,mode:k}}
 for(const cm of LAY.cameras)if(Math.hypot(mx-cm.x,my-cm.y)<p2m(12))return {t:'c',id:cm.id,mode:'move'};
 for(let i=LAY.shelves.length-1;i>=0;i--){const s=LAY.shelves[i];const l=rot(mx,my,s.x,s.y,-(s.rot||0));
  if(Math.abs(l[0]-s.x)<=s.w/2&&Math.abs(l[1]-s.y)<=s.h/2)return {t:'s',id:s.id,mode:'move'}}
 for(const cm of LAY.cameras){if(!cm.floor_rect)continue;const f=cm.floor_rect;
  const l=rot(mx,my,f.x,f.y,-(f.rot||0));
  if(Math.abs(l[0]-f.x)<=f.w/2&&Math.abs(l[1]-f.y)<=f.h/2)return {t:'f',id:cm.id,mode:'move'}}
 return null}
function evM(e){const r=CV.getBoundingClientRect();
 return [p2m((e.clientX-r.left)*CV.width/r.width),p2m((e.clientY-r.top)*CV.height/r.height)]}
CV.addEventListener('pointerdown',e=>{const [mx,my]=evM(e);const h=hit(mx,my);
 if(!h){SEL=null;panel();draw();return}
 if(SEL===null||SEL.t!==h.t||SEL.id!==h.id){SEL={t:h.t,id:h.id};panel()}
 const o=selObj();DRAG={mode:h.mode,ox:mx-(o.x||0),oy:my-(o.y||0)};CV.setPointerCapture(e.pointerId);draw()});
CV.addEventListener('pointermove',e=>{if(!DRAG)return;const [mx,my]=evM(e);const o=selObj();if(!o)return;
 const sn=v=>e.shiftKey?v:Math.round(v/SNAP)*SNAP;
 if(DRAG.mode==='move'){o.x=sn(mx-DRAG.ox);o.y=sn(my-DRAG.oy)}
 else if(DRAG.mode==='head'){o.heading=Math.round(Math.atan2(my-o.y,mx-o.x)*180/Math.PI/5)*5}
 else if(DRAG.mode==='rot'){o.rot=Math.round((Math.atan2(my-o.y,mx-o.x)*180/Math.PI+90)/5)*5}
 else if(DRAG.mode==='size'){const l=rot(mx,my,o.x,o.y,-(o.rot||0));
  o.w=Math.max(0.2,sn(Math.abs(l[0]-o.x)*2));o.h=Math.max(0.2,sn(Math.abs(l[1]-o.y)*2))}
 panel();draw()});
addEventListener('pointerup',()=>{DRAG=null});
addEventListener('keydown',e=>{if(e.key!=='Delete'||!SEL||/INPUT/.test(document.activeElement.tagName))return;
 if(SEL.t==='s')LAY.shelves=LAY.shelves.filter(s=>s.id!==SEL.id);
 else if(SEL.t==='c')LAY.cameras=LAY.cameras.filter(c=>c.id!==SEL.id);
 else {const cm=LAY.cameras.find(c=>c.id===SEL.id);delete cm.floor_rect}
 SEL=null;panel();draw()});
function fld(l,v,on,st){return `<label>${l}<input type="number" step="${st||0.1}" value="${v}" oninput="${on}"></label>`}
function panel(){const o=selObj();let h=`<h4>Store</h4>
 ${fld('width (m)',LAY.store.w,'LAY.store.w=+this.value;draw()',0.5)}
 ${fld('depth (m)',LAY.store.h,'LAY.store.h=+this.value;draw()',0.5)}`;
 if(!o){h+='<p class="muted">Click a shelf or camera to edit it. Drag to move, yellow handles rotate and resize, Delete removes.</p>'}
 else if(SEL.t==='s'){h+=`<h4>Shelf</h4>
  <label>name<input type="text" value="${o.name}" oninput="selObj().name=this.value;draw()"></label>
  ${fld('x (m)',o.x,'selObj().x=+this.value;draw()')}${fld('y (m)',o.y,'selObj().y=+this.value;draw()')}
  ${fld('width (m)',o.w,'selObj().w=+this.value;draw()')}${fld('depth (m)',o.h,'selObj().h=+this.value;draw()')}
  ${fld('height (m)',o.height,'selObj().height=+this.value;draw()')}
  ${fld('rotation (deg)',o.rot||0,'selObj().rot=+this.value;draw()',5)}
  <h4>Monitored faces</h4>`;
  FACES.forEach(f=>{const on=!!o.faces[f],g=on?o.faces[f].grid:[4,6];
   h+=`<div class="row"><label style="flex:1"><span>${f} face</span>
    <input type="checkbox" ${on?'checked':''} onchange="tglFace('${f}',this.checked)"></label>`;
   if(on)h+=`<input type="number" min="1" value="${g[0]}" style="width:52px" title="rows"
     oninput="selObj().faces['${f}'].grid[0]=+this.value">
    <input type="number" min="1" value="${g[1]}" style="width:52px" title="cols"
     oninput="selObj().faces['${f}'].grid[1]=+this.value">`;
   h+=`</div>`;
   if(on)h+=`<div class="muted" style="margin:-2px 0 6px">seen ${Math.round(bestVis(o,f)*100)}% &middot; rows x cols</div>`})}
 else if(SEL.t==='c'){h+=`<h4>Camera</h4>
  <label>name<input type="text" value="${o.name}" oninput="selObj().name=this.value;draw()"></label>
  <label>source<input type="text" value="${o.source||''}" oninput="selObj().source=this.value"
   title="webcam index, http://phone:8080/video, rtsp:// or a file"></label>
  ${fld('x (m)',o.x,'selObj().x=+this.value;draw()')}${fld('y (m)',o.y,'selObj().y=+this.value;draw()')}
  ${fld('heading (deg)',o.heading,'selObj().heading=+this.value;draw()',5)}
  ${fld('field of view (deg)',o.fov,'selObj().fov=+this.value;draw()',5)}
  ${fld('range (m)',o.range,'selObj().range=+this.value;draw()',0.5)}
  ${fld('mount height (m)',o.height||2.2,'selObj().height=+this.value')}
  <div class="row">`+['entry','queue','shelf'].map(r=>`<label style="flex:none">${r}
   <input type="checkbox" ${(o.roles||[]).includes(r)?'checked':''} onchange="tglRole('${r}',this.checked)"></label>`).join('')+
  `</div><button onclick="tglPatch()">${o.floor_rect?'remove':'add'} floor patch</button>
  <div class="muted" style="margin-top:6px">The floor patch is the real-world rectangle matching the 4 floor
  points you clicked with --pick. It puts this camera's shoppers on the shared store heatmap.</div>`}
 else {h+=`<h4>Floor patch</h4>${fld('x (m)',o.x,'selObj().x=+this.value;draw()')}
  ${fld('y (m)',o.y,'selObj().y=+this.value;draw()')}${fld('width (m)',o.w,'selObj().w=+this.value;draw()')}
  ${fld('depth (m)',o.h,'selObj().h=+this.value;draw()')}${fld('rotation (deg)',o.rot||0,'selObj().rot=+this.value;draw()',5)}`}
 $('panel').innerHTML=h;blind()}
function tglFace(f,on){const s=selObj();if(on)s.faces[f]={grid:[4,6]};else delete s.faces[f];panel();draw()}
function tglRole(r,on){const c=selObj();c.roles=c.roles||[];
 c.roles=on?[...new Set([...c.roles,r])]:c.roles.filter(x=>x!==r);panel();draw()}
function tglPatch(){const c=selObj();
 if(c.floor_rect)delete c.floor_rect;else c.floor_rect={x:c.x+2,y:c.y,w:3,h:2,rot:0};panel();draw()}
function blind(){const bad=[];for(const s of LAY.shelves)for(const f in s.faces)
 if(bestVis(s,f)<0.6)bad.push(`${s.name} ${f}${bestVis(s,f)>0?' (partial)':''}`);
 $('blind').innerHTML=bad.length?'⚠ not fully covered: '+bad.join(', ')
  :(LAY.shelves.length?'✓ every monitored shelf face is covered by a camera':'Add shelves and cameras to model the store.')}
function addShelf(){const n=LAY.shelves.length+1;
 LAY.shelves.push({id:'S'+Date.now().toString(36),name:'Shelf '+n,x:LAY.store.w/2,y:LAY.store.h/2,
  w:3,h:0.6,rot:0,height:1.8,faces:{N:{grid:[4,6]},S:{grid:[4,6]}}});
 SEL={t:'s',id:LAY.shelves[LAY.shelves.length-1].id};panel();draw()}
function addCam(){const n=LAY.cameras.length+1;
 LAY.cameras.push({id:'cam'+Date.now().toString(36),name:'Camera '+n,x:1,y:LAY.store.h/2,heading:0,
  fov:70,range:8,height:2.2,roles:['shelf'],source:''});
 SEL={t:'c',id:LAY.cameras[LAY.cameras.length-1].id};panel();draw()}
async function loadLay(){const j=await(await fetch('/api/layout')).json();LAY=j.layout;SEL=null;panel();draw()}
async function saveLay(){const r=await fetch('/api/layout',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify(LAY)});const j=await r.json();
 if(j.ok){LAY=j.layout;panel();draw();shot3d();
  $('blind').innerHTML+=' &middot; saved. Restart to apply new camera sources.'}else alert(j.error||'save failed')}
let AZ=-62,EL=34;
function shot3d(){$('v3d').src=`/api/layout/3d.png?az=${AZ}&el=${EL}&t=${Date.now()}`}
function spin(d){AZ=(AZ+d)%360;shot3d()}
function tilt(d){EL=Math.max(5,Math.min(85,EL+d));shot3d()}
loadLay().then(shot3d);
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
    from fastapi.responses import HTMLResponse, Response, StreamingResponse

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

    @app.get("/api/layout")
    def get_layout():
        lay = engine.layout
        return {"layout": lay.data if lay else DEFAULT_LAYOUT,
                "coverage": coverage(lay.data) if lay else {}}

    @app.post("/api/layout")
    async def post_layout(req: Request):
        lay = engine.layout
        if not lay:
            raise HTTPException(400, "no layout file configured")
        try:
            data = lay.save(await req.json())
        except (KeyError, TypeError, ValueError) as e:
            return {"ok": False, "error": f"bad layout: {e}"}
        with engine.lock:                       # store size may have changed -> regrid the heatmap
            engine.store_heat = np.zeros(lay.grid_dims(), np.float32)
        for w in workers.values():              # cameras may have moved -> rebuild their mappings
            w.storeH = getattr(w, "storeH", None) and None
        return {"ok": True, "layout": data, "coverage": coverage(data)}

    @app.get("/api/layout/3d.png")
    def layout_3d(el: float = 34, az: float = -62):
        if not engine.layout:
            raise HTTPException(404, "no layout")
        return Response(render_3d(engine, el, az), media_type="image/png")

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
    ap.add_argument("--layout", help="store layout JSON written by the dashboard editor")
    ap.add_argument("--pick", metavar="SOURCE", help="click points on a frame to get geometry coords")
    args = ap.parse_args()

    if args.config:
        with open(args.config) as fh:
            deep_merge(CONFIG, json.load(fh))
    if args.sku_model:
        CONFIG["sku_model"] = args.sku_model
    if args.cloud_url:
        CONFIG["cloud_url"] = args.cloud_url
    if args.layout:
        CONFIG["layout_path"] = args.layout
    if args.pick:
        return pick(args.pick)

    try:                       # nothing leaves the device: kill the model library's usage telemetry
        from ultralytics import settings as ul_settings
        ul_settings.update({"sync": False})
    except Exception:
        pass

    store = Store(CONFIG["db_path"])
    layout = Layout(CONFIG["layout_path"])
    engine = Engine(store, layout)
    workers = {}
    # with no --cam, run whatever the layout says: model the store in the editor, then just start
    cams = args.cam or [[c["id"], ",".join(c["roles"]), c["source"]]
                        for c in layout.data["cameras"] if c.get("source")]
    if not cams:
        cams = [["cam0", "entry,queue", "0"]]
        print("no --cam and no camera sources in the layout: using the laptop webcam")
    for name, roles, src in cams:
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
