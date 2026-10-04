#!/usr/bin/env python3
"""
StoreSense Edge — on-device retail intelligence  (SIH 2026, PS 26179)
One file. Laptop now; target Qualcomm Dragonwing RB3 Gen 2 / Snapdragon (YOLO exported to QNN, Hexagon NPU).

Modules (reference repo each one replaces)
  • Footfall: entry/exit counting, live occupancy, hourly trend     (DeepStream retail analytics + ByteTrack)
  • Floor grid: heatmap + zone dwell time via homography            (DeepStream zone analytics)
  • Shelf grid: per-cell OK / LOW / EMPTY / MISPLACED, occlusion-aware (Retail-Shelf-Monitoring, StoreEye)
  • Depth count: units left behind the front row of each column, from one ordinary camera at
    any angle — monocular depth (Depth Anything V2, metric indoor) in shelf coordinates
  • Depletion forecast: time-to-empty per shelf cell                (demand-forecasting repos)
  • Queue: length, time in queue, +5/+10/+15 min forecast,
    counters-to-open recommendation                                  (QueueLess + multistep forecasting)
  • Rule-based priority decision engine with cooldowns + ack/response time
  • Local SQLite, daily/weekly reports, offline-first cloud sync outbox (EventPulse)
  • Live dashboard: FastAPI + WebSocket + annotated MJPEG feeds
  • POS hook: POST /api/integrations/pos  (conversion KPI uses it when present)
  • Privacy: only track IDs + numbers stored. No frames, no faces. Heads blurred on live feeds.

Install
  pip install ultralytics lap fastapi "uvicorn[standard]" opencv-python numpy transformers
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
        "slot_low": 0.55,           # a facing counts as gone below this share of its calibrated edges
        "grid": [4, 6],             # rows, cols of the shelf face — fallback when no slots are marked
        "period_s": 3.0,            # stock is re-read this often
        "detect_s": 0.25,           # people in front of the shelf are checked this often
        "low": 0.55, "empty": 0.25, # fill ratio vs calibrated full shelf
        "misplace_corr": 0.35,      # colour-histogram similarity below this = product looks different
        "occlusion_overlap": 0.15,
        "eta_window_min": 20,
        "planogram": {},            # {"shelfA": {"0,0": "maggi", ...}} — used only with sku_model
        "weights": {},              # priority multiplier per shelf cam, e.g. {"shelfA": 1.5}
        # Monocular depth: counts units BEHIND the front row from one ordinary camera at any angle.
        # Used for product boxes that have "unit depth (cm)" set; others fall back to facings × depth.
        "depth": {
            "enabled": True,        # needs: pip install transformers  (weights download on first use)
            "model": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            "every_s": 6.0,         # one depth pass per shelf camera this often (~0.3-1 s on a laptop CPU)
            "smooth": 3,            # median over this many passes per column
            "hfov_deg": 65,         # camera's horizontal field of view if the store plan doesn't give one
            "calib_passes": 3,      # depth passes averaged into the full-shelf reference after Calibrate
        },
    },
    "crowd": {                      # a crowd = this many people in one camera's view, steady for a few seconds
        "people": 6, "hold_s": 8,
        "clear_ratio": 0.7,         # over once the count drops to this share of the threshold
        "trend_window_s": 150,      # how far back the "when will it thin out" estimate looks
    },
    "queue": {
        "open_counters": 1, "max_counters": 4,
        "default_service_per_min": 1.5,   # prior per counter, learned online
        "target_wait_min": 3.0, "max_wait_min": 6.0,
        "min_time_in_zone_s": 4.0, "rate_window_min": 5.0,
    },
    "shopper": {                    # everyone who walks in gets an anonymous ID, checked again at the till
        "require_id": False,        # True: a bill can't be made until the shopper's ID is verified
        "match_min": 0.55,          # a camera match needs at least this clothing-colour similarity...
        "match_margin": 0.04,       # ...and has to beat the runner-up by this much
        "keep_h": 12,               # hours a shopper who has left stays in the list
    },
    "store_name": "StoreSense Mart", # printed on bills
    "bills_dir": "bills",           # PNG + PDF of every bill
    "analytics_dir": "analytics",   # per-minute CSV for forecasting models
    "pos": {"rescan_s": 2.0},       # an item must leave the checkout camera's view this long to count again
    "layout_path": "layout.json",   # the store model: shelves and cameras in metres
    "store_cell_m": 0.25,           # floor-heatmap resolution in the store frame
    "heat": {"spread_m": 0.35,      # a person warms the floor around their feet, not one square
             "live_tau_s": 45},     # the "now" heatmap forgets a visit after about this long
    # POS / inventory / ERP systems told about what happens, as JSON POSTs. A plain URL gets every
    # event; {"url": ..., "events": ["alert", "bill", "stock", "restock"]} picks some. Queued in
    # SQLite and retried, so nothing is lost while the store is offline.
    "webhooks": [],
    "max_width": 1280,              # camera frames are shrunk to this width (phones send 1080p+)
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
# a small shop is the common case, so that is the default canvas — not a warehouse
DEFAULT_LAYOUT = {"store": {"w": 6.0, "h": 4.0}, "shelves": [], "cameras": [], "fixtures": []}
# things on the plan that aren't shelves: doors (entry / exit), checkout counters, anything else
# (pillar, freezer, promo stand). Counters and fixtures block camera views; doors don't.
FIXTURE_KINDS = {"door": 2.1, "counter": 1.0, "fixture": 1.5}      # kind -> default height (m)


def blockers(layout):
    """Everything a camera can't see through: shelves, counters, other fixtures (not doors)."""
    return list(layout.get("shelves", [])) + [f for f in layout.get("fixtures", []) if f.get("kind") != "door"]


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
    walls = blockers(layout)
    cams = [c for c in layout.get("cameras", []) if "shelf" in c.get("roles", [])]
    out = {}
    for s in shelves:
        for f in s.get("faces", {}):
            best_v, best_c = 0.0, None
            for cm in cams:
                v = face_visibility(cm, s, f, walls)
                if v > best_v:
                    best_v, best_c = v, cm.get("id")
            out[f"{s['id']}:{f}"] = {"shelf": s["id"], "shelf_name": s.get("name", s["id"]), "face": f,
                                     "visible": round(best_v, 2), "camera": best_c}
    return out


def slot_locations(slots):
    """Row and column of each product box, read off where the boxes sit on the shelf picture:
    boxes whose centres are within half a box-height of a row's centre share that row."""
    if not slots:
        return {}
    med_h = float(np.median([s["h"] for s in slots])) or 0.1
    rows = []
    for s in sorted(slots, key=lambda s: s["y"] + s["h"] / 2):
        c = s["y"] + s["h"] / 2
        if rows and abs(c - np.mean([r["y"] + r["h"] / 2 for r in rows[-1]])) <= med_h * 0.5:
            rows[-1].append(s)
        else:
            rows.append([s])
    out = {}
    for ri, row in enumerate(rows):
        for ci, s in enumerate(sorted(row, key=lambda s: s["x"])):
            out[s["id"]] = (ri + 1, ci + 1)
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
        out = {"store": {"w": float(d.get("store", {}).get("w", DEFAULT_LAYOUT["store"]["w"])),
                         "h": float(d.get("store", {}).get("h", DEFAULT_LAYOUT["store"]["h"]))},
               "shelves": [], "cameras": [], "fixtures": []}
        for i, f in enumerate(d.get("fixtures", [])):
            kind = f.get("kind") if f.get("kind") in FIXTURE_KINDS else "fixture"
            fx = {"id": str(f.get("id") or f"F{i + 1}"), "kind": kind, "name": f.get("name") or kind.title(),
                  "x": float(f["x"]), "y": float(f["y"]), "w": float(f.get("w", 1.0)), "h": float(f.get("h", 0.5)),
                  "rot": float(f.get("rot", 0)), "height": float(f.get("height", FIXTURE_KINDS[kind]))}
            if kind == "door":
                fx["dir"] = f.get("dir") if f.get("dir") in ("in", "out", "both") else "both"
            out["fixtures"].append(fx)
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
                   "heading": float(c.get("heading", 0)), "fov": float(c.get("fov", 60)),
                   "range": float(c.get("range", 5)), "height": float(c.get("height", 2.2)),
                   "roles": [r for r in c.get("roles", []) if r in ("entry", "queue", "shelf", "checkout")] or ["shelf"],
                   "source": c.get("source", "")}
            # geometry drawn on the camera's own picture, in 0..1 image coordinates
            for k, n in (("entry_line", 2), ("queue_zone", 3), ("floor_quad", 4)):
                pts = c.get(k)
                if pts and len(pts) >= n:
                    cam[k] = [[float(p[0]), float(p[1])] for p in pts]
            if int(c.get("in_side", 0)) in (1, -1):
                cam["in_side"] = int(c["in_side"])
            # product slots drawn on this camera's picture: one per product block on the shelf,
            # with how many units stand side by side (facings) and how many deep they stack
            slots = []
            for k, sl in enumerate(c.get("slots", [])):
                slots.append({
                    "id": str(sl.get("id") or f"p{k + 1}"), "name": sl.get("name", f"Product {k + 1}"),
                    "sku": sl.get("sku", ""),
                    "x": float(sl["x"]), "y": float(sl["y"]), "w": float(sl["w"]), "h": float(sl["h"]),
                    "facings": max(1, int(sl.get("facings", 1))), "deep": max(1, int(sl.get("deep", 1))),
                    # front-to-back size of one unit; enables the depth-model count for this box
                    "unit_cm": max(0.0, float(sl.get("unit_cm") or 0)),
                })
            if slots:
                cam["slots"] = slots
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
                v = face_visibility(c, s, f, blockers(self.data))
                if v > best[0]:
                    best = (v, s, f)
        return (best[1], best[2]) if best[0] > 0 else (None, None)

    def grid_dims(self):
        m = max(CONFIG["store_cell_m"], 0.05)
        return max(1, int(round(self.data["store"]["h"] / m))), max(1, int(round(self.data["store"]["w"] / m)))


# ─────────────────────────────── CAPTURE ───────────────────────────────
def normalise_source(src):
    """Fix the usual phone-camera URL slips: '192.168.1.5:8080' → 'http://192.168.1.5:8080/video'.
    IP Webcam and DroidCam both stream MJPEG at /video; the bare address is their web page."""
    s = str(src).strip()
    if s.isdigit() or not s:
        return s
    from urllib.parse import urlparse
    if not s.startswith(("http://", "https://", "rtsp://")) and not os.path.exists(s) and \
            (s[0].isdigit() and ("." in s or ":" in s)):
        s = "http://" + s
    u = urlparse(s)
    if u.scheme in ("http", "https") and u.path in ("", "/"):
        s = s.rstrip("/") + "/video"
    return s


class Capture:
    """Threaded reader that always holds the latest frame. Reconnects streams, loops files.
    Keeps a plain-words status ('live', or why it isn't) for the dashboard."""

    def __init__(self, src):
        src = normalise_source(src)
        self.src = int(src) if str(src).isdigit() else src
        self.is_file = isinstance(self.src, str) and not self.src.startswith(("http", "rtsp")) and os.path.exists(self.src)
        self.frame, self.lock = None, threading.Lock()
        self.alive, self.status, self.fails = True, "connecting…", 0
        self.seq = 0                          # bumps on every new frame, so workers skip repeats
        self.jpeg, self.jpeg_seq, self.dec_seq = None, 0, 0
        http = isinstance(self.src, str) and self.src.startswith(("http://", "https://"))
        threading.Thread(target=self._run_http if http else self._run, daemon=True).start()

    def stop(self):
        self.alive = False

    def why(self):
        s = self.src
        if isinstance(s, int):
            return f"webcam {s} not available — another app using it, or wrong index"
        if isinstance(s, str) and s.startswith("http"):
            return (f"no picture from {s} — is the phone's camera server started, is it on the same Wi-Fi "
                    f"(campus Wi-Fi often blocks phone-to-laptop; use a hotspot), and does the URL open in a browser?")
        if isinstance(s, str) and s.startswith("rtsp"):
            return f"no picture from {s} — check the address, username/password and that the camera is on"
        return f"can't open {s} — file not found or unreadable"

    def _open(self):
        if isinstance(self.src, str) and self.src.startswith(("http", "rtsp")):
            try:                                # don't hang ~30 s on an address that isn't there
                cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG,
                                       [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            except (TypeError, cv2.error):      # older OpenCV without open parameters
                cap = cv2.VideoCapture(self.src)
        else:
            cap = cv2.VideoCapture(self.src)
        if isinstance(self.src, int):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _fit(self, f):
        """Phones send 1080p+; nothing here needs more than max_width, and big frames cost CPU and lag."""
        mw = CONFIG.get("max_width") or 0
        if f is not None and mw and f.shape[1] > mw:
            f = cv2.resize(f, (mw, int(f.shape[0] * mw / f.shape[1])), interpolation=cv2.INTER_AREA)
        return f

    def _run_http(self):
        """Phone cameras (IP Webcam, DroidCam) stream MJPEG over HTTP. Read the bytes ourselves and keep
        only the newest complete JPEG: if the laptop falls behind, old frames are dropped instead of
        queueing up — which is what made OpenCV's reader lag further and further behind the phone.
        Frames are decoded only when a worker asks for one."""
        import urllib.request
        while self.alive:
            try:
                r = urllib.request.urlopen(self.src, timeout=5)
                ctype = r.headers.get("Content-Type", "")
                if "multipart" not in ctype.lower():
                    r.close()
                    return self._run()           # not MJPEG (e.g. an HLS page): let OpenCV try
                bnd = ctype.split("boundary=")[-1].strip().strip('"').lstrip("-").encode() if "boundary=" in ctype else b""
                buf = b""
                while self.alive:
                    chunk = r.read1(262144) if hasattr(r, "read1") else r.read(65536)
                    if not chunk:
                        raise IOError("stream ended")
                    buf += chunk
                    got = None
                    if bnd:                      # complete parts sit between two boundaries
                        cuts = [i for i in self._find_all(buf, bnd)]
                        if len(cuts) >= 2:
                            part = buf[cuts[-2]:cuts[-1]]
                            got = part[part.find(b"\xff\xd8"):] if b"\xff\xd8" in part else None
                            buf = buf[cuts[-1]:]
                    else:                        # no boundary given: split on JPEG start/end markers
                        e = buf.rfind(b"\xff\xd9")
                        st = buf.rfind(b"\xff\xd8", 0, e) if e > 0 else -1
                        if st >= 0:
                            got, buf = buf[st:e + 2], buf[e + 2:]
                    if got:
                        with self.lock:
                            self.jpeg, self.jpeg_seq = got, self.jpeg_seq + 1
                        self.fails, self.status = 0, "live"
                    if len(buf) > 16_000_000:
                        buf = b""
                r.close()
            except Exception:
                with self.lock:
                    self.jpeg, self.frame = None, None
                self.fails += 1
                self.status = self.why() + (f" (retrying, attempt {self.fails})" if self.fails > 1 else "")
                time.sleep(min(1.0 + self.fails * 0.5, 5.0))

    @staticmethod
    def _find_all(buf, pat):
        i = buf.find(pat)
        while i != -1:
            yield i
            i = buf.find(pat, i + len(pat))

    def _run(self):
        cap = self._open()
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        while self.alive:
            ok, f = cap.read()
            if not ok:
                if self.is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, f = cap.read()
                if not ok:
                    with self.lock:
                        self.frame = None
                    self.fails += 1
                    self.status = self.why() + (f" (retrying, attempt {self.fails})" if self.fails > 1 else "")
                    cap.release()
                    time.sleep(min(1.0 + self.fails * 0.5, 5.0))
                    if not self.alive:
                        break
                    cap = self._open()
                    continue
            self.fails, self.status = 0, "live"
            f = self._fit(f)
            with self.lock:
                self.frame, self.seq = f, self.seq + 1
            if self.is_file:
                time.sleep(1.0 / max(fps, 1))
        cap.release()

    def read(self):
        with self.lock:
            if self.jpeg is not None and self.jpeg_seq != self.dec_seq:
                jpg, self.dec_seq = self.jpeg, self.jpeg_seq
            else:
                jpg = None
        if jpg is not None:                      # decode outside the lock; only frames someone uses
            f = self._fit(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR))
            if f is not None:
                with self.lock:
                    self.frame, self.seq = f, self.seq + 1
        with self.lock:
            return None if self.frame is None else self.frame.copy()


# ─────────────────────────────── BARCODES & BILLS ───────────────────────────────
_EAN_L = ["0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011"]
_EAN_G = ["0100111", "0110011", "0011011", "0100001", "0011101", "0111001", "0000101", "0010001", "0001001", "0010111"]
_EAN_R = ["1110010", "1100110", "1101100", "1000010", "1011100", "1001110", "1010000", "1000100", "1001000", "1110100"]
_EAN_PARITY = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG", "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]


def ean_check(d12):
    s = sum(int(c) * (3 if i % 2 else 1) for i, c in enumerate(d12))
    return str((10 - s % 10) % 10)


def internal_ean(sku):
    """A stable in-store EAN-13 for products that arrive without one. GS1 reserves the 20-29
    prefixes for exactly this (restricted, in-store use), so it can't collide with a real code."""
    h = int.from_bytes(__import__("hashlib").sha1(sku.encode()).digest()[:6], "big") % 10**10
    d12 = f"20{h:010d}"
    return d12 + ean_check(d12)


def ean13_bits(code):
    code = "".join(ch for ch in str(code) if ch.isdigit())
    if len(code) == 12:
        code += ean_check(code)
    if len(code) != 13 or ean_check(code[:12]) != code[12]:
        return None
    par = _EAN_PARITY[int(code[0])]
    left = "".join((_EAN_L if par[i] == "L" else _EAN_G)[int(code[1 + i])] for i in range(6))
    right = "".join(_EAN_R[int(c)] for c in code[7:13])
    return "101" + left + "01010" + right + "101", code


def ean13_image(code, label="", module=3, height=90):
    """Printable label: stick it on the product so the checkout camera or a USB scanner can read it."""
    from PIL import Image, ImageDraw
    r = ean13_bits(code)
    if not r:
        raise ValueError(f"{code} is not a valid EAN-13")
    bits, code = r
    quiet = 11 * module
    W = len(bits) * module + 2 * quiet
    top = 22 if label else 8
    img = Image.new("RGB", (W, height + top + 26), "white")
    d = ImageDraw.Draw(img)
    for i, b in enumerate(bits):
        if b == "1":
            x = quiet + i * module
            guard = i < 3 or 45 <= i < 50 or i >= 92
            d.rectangle([x, top, x + module - 1, top + height + (8 if guard else 0)], fill="black")
    f = _font(15)
    d.text((W / 2, top + height + 12), code, fill="black", font=f, anchor="mt")
    if label:
        d.text((W / 2, 4), label[:34], fill="black", font=_font(13), anchor="mt")
    return img


def _font(size, bold=False):
    from PIL import ImageFont
    names = (["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"] if bold else
             ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"])
    dirs = ["", "/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/TTF/", "/Library/Fonts/",
            "C:/Windows/Fonts/"]
    for n in names:
        for d in dirs:
            try:
                return ImageFont.truetype(d + n, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def render_bill(bill):
    """Receipt as an image (thermal-printer proportions) and the same page as a PDF."""
    from PIL import Image, ImageDraw
    W, pad = 620, 28
    rows = len(bill["lines"])
    H = 460 + rows * 44
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f, fb, fs, ft = _font(17), _font(17, True), _font(14), _font(26, True)
    y = 24
    d.text((W / 2, y), CONFIG.get("store_name", "StoreSense Store"), font=ft, fill="black", anchor="mt")
    y += 40
    d.text((W / 2, y), f"{CONFIG['store_id']}  ·  Customer bill", font=fs, fill="#444", anchor="mt")
    y += 30
    when = datetime.fromtimestamp(bill["ts"]).strftime("%d %b %Y  %H:%M")
    d.text((pad, y), f"Bill {bill['id']}", font=fb, fill="black")
    d.text((W - pad, y), when, font=f, fill="black", anchor="ra")
    y += 34
    sh = bill.get("shopper")
    if sh:
        d.text((pad, y - 4), f"Shopper {sh['id']}", font=fs, fill="#444")
        d.text((W - pad, y - 4), "ID verified" if sh.get("check", "ok") == "ok" else f"ID check: {sh['check']}",
               font=fs, fill="#444", anchor="ra")
        y += 24
    d.line([pad, y, W - pad, y], fill="black", width=2)
    y += 10
    cols = [pad, 330, 400, W - pad]
    d.text((cols[0], y), "Item", font=fb, fill="black")
    d.text((cols[1], y), "Qty", font=fb, fill="black")
    d.text((cols[2], y), "Rate", font=fb, fill="black")
    d.text((cols[3], y), "Amount", font=fb, fill="black", anchor="ra")
    y += 30
    d.line([pad, y, W - pad, y], fill="#999", width=1)
    y += 8
    for ln in bill["lines"]:
        d.text((cols[0], y), ln["name"][:30], font=f, fill="black")
        d.text((cols[1], y), str(ln["qty"]), font=f, fill="black")
        d.text((cols[2], y), f"{ln['price']:.2f}", font=f, fill="black")
        d.text((cols[3], y), f"{ln['amount']:.2f}", font=f, fill="black", anchor="ra")
        if ln["mrp"] > ln["price"]:
            d.text((cols[0], y + 21), f"MRP ₹{ln['mrp']:.2f}", font=fs, fill="#666")
        y += 44
    d.line([pad, y, W - pad, y], fill="black", width=2)
    y += 14
    for k, v, bold in (("Items", str(bill["n_items"]), False),
                       ("Total at MRP", f"₹{bill['mrp_total']:.2f}", False),
                       ("You saved", f"₹{bill['savings']:.2f}", False),
                       ("Amount payable", f"₹{bill['total']:.2f}", True)):
        d.text((pad, y), k, font=fb if bold else f, fill="black")
        d.text((W - pad, y), v, font=fb if bold else f, fill="black", anchor="ra")
        y += 32 if bold else 28
    y += 14
    d.rectangle([pad, y, W - pad, y + 58], outline="black", width=2)
    d.text((W / 2, y + 12), "Please move to the payment counter", font=fb, fill="black", anchor="mt")
    d.text((W / 2, y + 36), "to complete your purchase", font=f, fill="black", anchor="mt")
    y += 76
    d.text((W / 2, y), "MRP inclusive of all taxes  ·  Thank you!", font=fs, fill="#444", anchor="mt")
    img = img.crop((0, 0, W, y + 34))
    io_ = __import__("io")
    png, pdf = io_.BytesIO(), io_.BytesIO()
    img.save(png, "PNG")
    img.save(pdf, "PDF", resolution=150)
    return png.getvalue(), pdf.getvalue()


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
                CREATE TABLE IF NOT EXISTS hooks(id INTEGER PRIMARY KEY, ts REAL, url TEXT, payload TEXT,
                                                 sent INTEGER DEFAULT 0, tries INTEGER DEFAULT 0);
                CREATE INDEX IF NOT EXISTS ix_ev ON events(type, ts);
                CREATE INDEX IF NOT EXISTS ix_m ON metrics(key, ts);
                CREATE TABLE IF NOT EXISTS products(sku TEXT PRIMARY KEY, barcode TEXT, name TEXT,
                    brand TEXT, mrp REAL, price REAL, updated REAL);
                CREATE TABLE IF NOT EXISTS bills(id TEXT PRIMARY KEY, ts REAL, items TEXT, mrp_total REAL,
                    total REAL, n_items INTEGER, demo INTEGER DEFAULT 0);
                CREATE INDEX IF NOT EXISTS ix_bill ON bills(ts);
            """)
            cols = [r[1] for r in self.db.execute("PRAGMA table_info(bills)")]
            for c in ("shopper_id", "shopper_how"):     # which shopper the bill was made for, and how it was checked
                if c not in cols:
                    self.db.execute(f"ALTER TABLE bills ADD COLUMN {c} TEXT")
            for tbl in ("events", "metrics"):           # rows made by --demo-history are tagged, never mixed silently
                cols = [r[1] for r in self.db.execute(f"PRAGMA table_info({tbl})")]
                if "demo" not in cols:
                    self.db.execute(f"ALTER TABLE {tbl} ADD COLUMN demo INTEGER DEFAULT 0")
            self.db.commit()
        from inventory.database import InventoryRepository
        self.inventory_repo = InventoryRepository(self)

    # ── product catalog ──────────────────────────────────────────────
    PRODUCT_FIELDS = ("sku", "barcode", "name", "brand", "mrp", "price")

    def product_upsert(self, p):
        sku = str(p.get("sku", "")).strip()
        if not sku:
            raise ValueError("sku is required")
        cur = self.product_get(sku) or {}
        row = {**{k: cur.get(k) for k in self.PRODUCT_FIELDS}, **{k: p[k] for k in self.PRODUCT_FIELDS if k in p}}
        row["sku"] = sku
        row["barcode"] = str(row.get("barcode") or "").strip() or internal_ean(sku)
        row["name"] = (row.get("name") or sku).strip()
        row["brand"] = (row.get("brand") or "").strip()
        row["mrp"] = round(float(row.get("mrp") or 0), 2)
        row["price"] = round(float(row.get("price") if row.get("price") not in (None, "") else row["mrp"]), 2)
        if row["price"] > row["mrp"] > 0:
            raise ValueError("selling price can't be above MRP")
        clash = self.q("SELECT sku FROM products WHERE barcode=? AND sku<>?", row["barcode"], sku)
        if clash:
            raise ValueError(f"barcode {row['barcode']} already belongs to {clash[0][0]}")
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO products VALUES(?,?,?,?,?,?,?)",
                            (sku, row["barcode"], row["name"], row["brand"], row["mrp"], row["price"], time.time()))
            self.db.commit()
        return row

    def product_get(self, sku):
        r = self.q("SELECT sku,barcode,name,brand,mrp,price FROM products WHERE sku=?", sku)
        return dict(zip(self.PRODUCT_FIELDS, r[0])) if r else None

    def product_by_code(self, code):
        """A scan can be the printed barcode or, typed by hand, the SKU."""
        code = str(code).strip()
        r = self.q("SELECT sku,barcode,name,brand,mrp,price FROM products WHERE barcode=? OR sku=?", code, code)
        return dict(zip(self.PRODUCT_FIELDS, r[0])) if r else None

    def products(self):
        return [dict(zip(self.PRODUCT_FIELDS, r))
                for r in self.q("SELECT sku,barcode,name,brand,mrp,price FROM products ORDER BY name")]

    def product_delete(self, sku):
        with self.lock:
            if self.db.execute("SELECT 1 FROM inventory_stock WHERE sku=?", (sku,)).fetchone():
                raise ValueError("remove inventory before deleting this catalog product")
            self.db.execute("DELETE FROM products WHERE sku=?", (sku,))
            self.db.commit()

    # ── bills ────────────────────────────────────────────────────────
    def next_bill_no(self, ts=None):
        day = datetime.fromtimestamp(ts or time.time()).strftime("%Y%m%d")
        n = self.q("SELECT COUNT(*) FROM bills WHERE id LIKE ?", f"SS-{day}-%")[0][0]
        return f"SS-{day}-{n + 1:04d}"

    def bill_save(self, b, demo=0):
        with self.lock:
            sh = b.get("shopper") or {}
            self.db.execute("INSERT INTO bills(id,ts,items,mrp_total,total,n_items,demo,shopper_id,shopper_how) "
                            "VALUES(?,?,?,?,?,?,?,?,?)",
                            (b["id"], b["ts"], json.dumps(b["lines"]), b["mrp_total"], b["total"], b["n_items"], demo,
                             sh.get("id"), sh.get("how")))
            self.db.commit()

    def bill_get(self, bid):
        r = self.q("SELECT id,ts,items,mrp_total,total,n_items,shopper_id,shopper_how FROM bills WHERE id=?", bid)
        if not r:
            return None
        i, ts, items, mrp, tot, n, sid, how = r[0]
        return {"id": i, "ts": ts, "lines": json.loads(items), "mrp_total": mrp, "total": tot, "n_items": n,
                "savings": round(mrp - tot, 2), "shopper": {"id": sid, "how": how} if sid else None}

    def q(self, sql, *args):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def event(self, cam, typ, data):
        with self.lock:
            self.db.execute("INSERT INTO events(ts,cam,type,data) VALUES(?,?,?,?)",
                            (time.time(), cam, typ, json.dumps(data)))
            self.db.commit()

    def events_at(self, rows, demo=0):
        """rows: (ts, cam, type, data) — for backfilled history, tagged so it is never mistaken for real."""
        with self.lock:
            self.db.executemany("INSERT INTO events(ts,cam,type,data,demo) VALUES(?,?,?,?,?)",
                                [(t, c, ty, json.dumps(d), demo) for t, c, ty, d in rows])
            self.db.commit()

    def metrics_at(self, rows, demo=0):
        with self.lock:
            self.db.executemany("INSERT INTO metrics(ts,key,value,demo) VALUES(?,?,?,?)",
                                [(t, k, float(v), demo) for t, k, v in rows])
            self.db.commit()

    def has_demo(self):
        return bool(self.q("SELECT 1 FROM events WHERE demo=1 LIMIT 1") or
                    self.q("SELECT 1 FROM bills WHERE demo=1 LIMIT 1"))

    def clear_demo(self):
        with self.lock:
            for t in ("events", "metrics", "bills"):
                self.db.execute(f"DELETE FROM {t} WHERE demo=1")
            self.db.commit()

    def metrics(self, ts, d):
        with self.lock:
            self.db.executemany("INSERT INTO metrics(ts,key,value) VALUES(?,?,?)", [(ts, k, float(v)) for k, v in d.items()])
            self.db.commit()

    def outbox_add(self, payload):
        with self.lock:
            self.db.execute("INSERT INTO outbox(ts,payload) VALUES(?,?)", (time.time(), json.dumps(payload)))
            self.db.commit()

    def hook_add(self, url, payload):
        with self.lock:
            self.db.execute("INSERT INTO hooks(ts,url,payload) VALUES(?,?,?)", (time.time(), url, json.dumps(payload)))
            self.db.commit()

    def hook_pending(self, n=50):
        return self.q("SELECT id,url,payload,tries FROM hooks WHERE sent=0 ORDER BY id LIMIT ?", n)

    def hook_done(self, hid, ok):
        with self.lock:
            self.db.execute("UPDATE hooks SET sent=1 WHERE id=?" if ok else "UPDATE hooks SET tries=tries+1 WHERE id=?", (hid,))
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
        self.store_heat = np.zeros(layout.grid_dims(), np.float32) if layout else None   # dwell today, seconds
        self.store_live = np.zeros_like(self.store_heat) if layout else None            # who is where lately (fades)
        self.heat_t, self.heat_day, self.heat_src, self._kern = time.time(), day_start(), {}, None
        self.footfall = {"in": 0, "out": 0}
        self.hourly, self.conversion, self.avg_response_s = {}, None, None
        self.alerts, self.last, self.keys, self.next_id = deque(maxlen=300), {}, {}, 1
        self.queues, self.shelves, self.heat, self.aisle, self.cams = {}, {}, {}, {}, {}
        self.labels = {}              # camera -> human name from the layout ("Aisle 1 A, S face")
        self.cam_face = {}            # camera -> "shelfId:FACE", so the 3D view can colour it
        self.depth_info = {}          # shelf camera -> depth model status
        self.attention = {}           # (camera, product box) -> today's stops and dwell seconds
        self.last_status = {}         # (camera, product box) -> last shelf status, for change events
        self.hook_state = {"sent": 0, "failed": 0, "last_error": None, "last_ok": None}
        self.stock, self.sold = {}, {}   # sku -> units at last restock / sold since then (POS)
        self.carts, self.active_cart, self.last_scan = {}, None, None
        self.zone_stats = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        self.served = deque(maxlen=200)
        self.open_counters = CONFIG["queue"]["open_counters"]
        self.online = None
        self.crowds, self.crowd_pending, self.crowd_hist = {}, {}, defaultdict(lambda: deque(maxlen=900))
        self.queue_peak = {"day": day_start(), "len": 0, "wait": 0.0}
        self.shoppers, self.track_shopper = {}, {}      # id -> record; (camera, track id) -> id
        self.till, self.sh_stats = None, {"issued": 0, "left_unbilled": 0, "unverified": 0}
        self.refresh_daily()
        from inventory.service import InventoryService
        self.inventory = InventoryService(store, self)

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
        self.emit("alert", a)

    # ── stock counting ───────────────────────────────────────────────
    # One ordinary camera sees the front row only: it cannot look behind a product. So a unit
    # count from vision alone is an ESTIMATE (facings seen × how deep the user says they stack).
    # Where the POS is connected we also track it properly — restock sets the level, each sale
    # decrements it — and a gap between the two is worth flagging rather than hiding.
    def restock(self, slots):
        with self.lock:
            for s in slots:
                self.stock[s["sku"]] = max(1, s["facings"]) * max(1, s["deep"])
                self.sold.pop(s["sku"], None)
        if slots:
            self.store.event("engine", "restock", {s["sku"]: self.stock[s["sku"]] for s in slots})
            self.emit("restock", {"source": "calibration", "levels": {s["sku"]: self.stock[s["sku"]] for s in slots}})

    def delivered(self, items, mode="add"):
        """Stock arriving from an inventory/ERP system: add to (or set) the till-tracked level."""
        out = {}
        with self.lock:
            for it in items:
                sku, qty = str(it.get("sku", "")), int(it.get("qty", 0))
                if not sku:
                    continue
                left = self.stock.get(sku, 0) - self.sold.get(sku, 0)
                self.stock[sku] = qty if mode == "set" else max(0, left) + qty
                self.sold.pop(sku, None)
                out[sku] = self.stock[sku]
        self.store.event("erp", "restock", out)
        self.emit("restock", {"source": "integration", "levels": out})
        return out

    def on_pos(self, data):
        with self.lock:
            for it in data.get("items", []):
                sku = str(it.get("sku", ""))
                if sku:
                    self.sold[sku] = self.sold.get(sku, 0) + int(it.get("qty", 1))

    # ── carts and checkout ───────────────────────────────────────────
    def cart_new(self):
        cid = f"C{int(time.time() * 1000) % 10**9:09d}"
        with self.lock:
            self.carts[cid] = {"id": cid, "created": time.time(), "items": {}}
            self.active_cart = cid
        return cid

    def cart_view(self, cid):
        with self.lock:
            c = self.carts.get(cid)
            if not c:
                return None
            items = dict(c["items"])
        lines, mrp_total, total, n = [], 0.0, 0.0, 0
        for sku, qty in items.items():
            p = self.store.product_get(sku)
            if not p:
                continue
            amt = round(p["price"] * qty, 2)
            lines.append({"sku": sku, "name": p["name"], "brand": p["brand"], "barcode": p["barcode"],
                          "qty": qty, "mrp": p["mrp"], "price": p["price"], "amount": amt})
            mrp_total += p["mrp"] * qty
            total += amt
            n += qty
        return {"id": cid, "lines": lines, "n_items": n, "mrp_total": round(mrp_total, 2),
                "total": round(total, 2), "savings": round(mrp_total - total, 2),
                "shopper": (self.carts.get(cid) or {}).get("shopper")}

    def cart_add(self, cid, code, qty=1, source="manual"):
        p = self.store.product_by_code(code)
        if not p:
            self.last_scan = {"ts": time.time(), "code": code, "ok": False, "source": source,
                              "msg": f"Unknown code {code}"}
            return None, f"No product with barcode or SKU {code}"
        with self.lock:
            if cid not in self.carts:
                self.carts[cid] = {"id": cid, "created": time.time(), "items": {}}
            it = self.carts[cid]["items"]
            it[p["sku"]] = max(0, it.get(p["sku"], 0) + int(qty))
            if it[p["sku"]] == 0:
                del it[p["sku"]]
            self.active_cart = cid
        self.last_scan = {"ts": time.time(), "code": code, "ok": True, "name": p["name"],
                          "price": p["price"], "source": source, "msg": f"{p['name']} added"}
        return p, None

    def cart_set(self, cid, sku, qty):
        with self.lock:
            c = self.carts.get(cid)
            if not c:
                return False
            if int(qty) <= 0:
                c["items"].pop(sku, None)
            else:
                c["items"][sku] = int(qty)
        return True

    def scan(self, code, source):
        """A barcode read by a checkout camera lands in the open cart (a new one if none)."""
        cid = self.active_cart if self.active_cart in self.carts else self.cart_new()
        return self.cart_add(cid, code, 1, source)

    def checkout(self, cid, shopper=None):
        v = self.cart_view(cid)
        if not v or not v["lines"]:
            return None, "Cart is empty"
        who = self.verify_shopper(cid, shopper)
        if CONFIG["shopper"]["require_id"] and not (who and who["check"] == "ok"):
            why = f" ({who['check']})" if who else ""
            return None, f"Shopper ID not verified{why} — pick the shopper at the till first"
        ts = time.time()
        bill = {**v, "id": self.store.next_bill_no(ts), "ts": ts, "shopper": who}
        self.store.bill_save(bill)
        png, pdf = render_bill(bill)
        os.makedirs(CONFIG["bills_dir"], exist_ok=True)
        for ext, data in (("png", png), ("pdf", pdf)):
            with open(os.path.join(CONFIG["bills_dir"], f"{bill['id']}.{ext}"), "wb") as fh:
                fh.write(data)
        items = [{"sku": ln["sku"], "qty": ln["qty"], "amount": ln["amount"]} for ln in bill["lines"]]
        self.store.event("pos", "pos", {"bill": bill["id"], "items": items, "total": bill["total"],
                                        "n_items": bill["n_items"], "shopper": (who or {}).get("id")})
        if who and who["check"] == "ok":
            with self.lock:
                sh = self.shoppers.get(who["id"])
                if sh:
                    sh["bills"].append(bill["id"])
                    sh["status"] = "billed"
        elif self.shoppers or self.sh_stats["issued"]:     # entry IDs are in use, yet nobody checked out against one
            with self.lock:
                self.sh_stats["unverified"] += 1
            self.store.event("pos", "unverified_checkout", {"bill": bill["id"], "why": (who or {}).get("check", "no shopper identified")})
            self.fire("checkout:unverified", "checkout", 1,
                      f"Bill {bill['id']} was made without a verified shopper ID" +
                      (f" ({who['id']}: {who['check']})" if who else ""), "Check the shopper ID")
        self.on_pos({"items": items})
        self.refresh_daily()
        self.emit("bill", bill)
        with self.lock:
            self.carts.pop(cid, None)
            if self.active_cart == cid:
                self.active_cart = None
        return bill, None

    def pos_units(self, sku, full):
        """Units left according to the till, or None when this product isn't tracked there."""
        if not sku:
            return None
        with self.lock:
            if sku not in self.stock:
                return None
            return max(0, self.stock[sku] - self.sold.get(sku, 0))

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
    def on_entry(self, cam, direction, sig=None, tid=None):
        """Someone crossed the entry line. Coming in they are given a shopper ID (returned); going out
        their ID is found from the track or, failing that, from their clothing signature."""
        with self.lock:
            self.footfall[direction] += 1
            if direction == "in":
                hh = datetime.now().strftime("%H")
                self.hourly[hh] = self.hourly.get(hh, 0) + 1
                sid = self.shopper_enter(cam, tid, sig)
            else:
                sid = self.shopper_leave(cam, tid, sig)
            inside = max(0, self.footfall["in"] - self.footfall["out"])
        self.store.event(cam, "entry", {"dir": direction, **({"shopper": sid} if sid else {})})
        if inside > CONFIG["crowd_threshold"]:
            self.fire("crowd", "crowd", 2, f"{inside} shoppers inside", "Deploy floor staff")
        else:
            self.resolve("crowd")
        return sid

    # ── shoppers: an anonymous ID for everyone who walks in, checked again at the till ──
    # Only a number and a clothing-colour histogram are kept (in memory, dropped when the shopper
    # leaves). No face, no picture. A camera match at the till is a hint staff can overrule by hand.
    def shopper_enter(self, cam, tid, sig):
        now = time.time()
        sid = self.track_shopper.get((cam, tid)) if tid is not None else None
        sh = self.shoppers.get(sid) if sid else None
        if sh and sh["status"] != "left":                 # the line was crossed twice with no exit between
            return sid
        if sh and now - (sh["t_out"] or 0) < 90:          # stepped out and straight back in: same shopper
            sh.update(status="billed" if sh["bills"] else "inside", t_out=None, sig=sig)
            self.store.event(cam, "shopper_in", {"id": sid, "again": True})
            return sid
        day = datetime.now().strftime("%y%m%d")
        n = self.store.q("SELECT COUNT(*) FROM events WHERE type='shopper_in' AND ts>=? AND data NOT LIKE '%again%'",
                         day_start())[0][0] + 1
        sid = f"SH-{day}-{n:03d}"
        self.shoppers[sid] = {"id": sid, "t_in": now, "t_out": None, "status": "inside", "sig": sig,
                              "cam": cam, "bills": []}
        if tid is not None:
            self.track_shopper[(cam, tid)] = sid
        self.sh_stats["issued"] += 1
        self.store.event(cam, "shopper_in", {"id": sid})
        return sid

    def shopper_leave(self, cam, tid, sig):
        sid = self.track_shopper.get((cam, tid)) if tid is not None else None
        if not (sid and self.shoppers.get(sid, {}).get("status") in ("inside", "billed")):
            sid = self.identify(sig)[0] if sig is not None else None
        sh = self.shoppers.get(sid) if sid else None
        if sh is None:
            return None
        sh.update(status="left", t_out=time.time(), sig=None)
        if not sh["bills"]:
            self.sh_stats["left_unbilled"] += 1
        self.store.event(cam, "shopper_out", {"id": sid, "billed": bool(sh["bills"])})
        horizon = time.time() - CONFIG["shopper"]["keep_h"] * 3600
        for k in [k for k, v in self.shoppers.items() if v["status"] == "left" and (v["t_out"] or 0) < horizon]:
            del self.shoppers[k]
        self.track_shopper = {k: v for k, v in self.track_shopper.items() if v in self.shoppers}
        return sid

    def identify(self, sig):
        """(shopper id or None, best similarity, runner-up) among the people inside. A match must be
        good enough and clearly better than the next one — otherwise it is None, never a guess."""
        best, bs, second = None, 0.0, 0.0
        with self.lock:
            for sid, sh in self.shoppers.items():
                if sh["status"] == "left" or sh.get("sig") is None:
                    continue
                sc = sig_similarity(sig, sh["sig"])
                if sc > bs:
                    best, bs, second = sid, sc, bs
                elif sc > second:
                    second = sc
        c = CONFIG["shopper"]
        if best is None or bs < c["match_min"] or bs - second < c["match_margin"]:
            return None, bs, second
        return best, bs, second

    def till_seen(self, cam, sig):
        """The checkout camera saw someone at the till. The match is only trusted once the same shopper
        has come up twice in a row, so a passer-by doesn't get a stranger's cart."""
        if sig is None:
            return
        sid, score, _ = self.identify(sig)
        now = time.time()
        t = self.till
        if sid is None:
            self.till = {"id": None, "score": round(score, 2), "ts": now, "cam": cam, "hits": 0}
        elif t and t["id"] == sid and now - t["ts"] < 8:
            self.till = {**t, "score": round(score, 2), "ts": now, "hits": t["hits"] + 1}
        else:
            self.till = {"id": sid, "score": round(score, 2), "ts": now, "cam": cam, "hits": 1}

    def till_current(self):
        t = self.till
        if t and t["id"] and t["hits"] >= 2 and time.time() - t["ts"] < 8:
            return dict(t)
        return None

    def cart_shopper(self, cid, sid):
        """Staff picks (or types) the shopper a cart belongs to. '' clears it. Returns an error or None."""
        with self.lock:
            c = self.carts.get(cid)
            if not c:
                return "no such cart"
            sid = (sid or "").strip().upper()
            if not sid:
                c.pop("shopper", None)
                return None
            if sid not in self.shoppers:
                return f"No shopper {sid} entered today"
            if self.shoppers[sid]["status"] == "left":
                return f"{sid} has already left the store"
            c["shopper"] = sid
        return None

    def verify_shopper(self, cid, given=None):
        """Who is this checkout for, and does the entry record agree?
        {"id", "how": manual|camera, "check": "ok" or the reason it isn't} — or None if nobody was identified."""
        with self.lock:
            sid = (given or (self.carts.get(cid) or {}).get("shopper") or "").strip().upper()
            if sid:
                sh = self.shoppers.get(sid)
                chk = "ok" if sh and sh["status"] != "left" else ("not entered today" if not sh else "already left")
                return {"id": sid, "how": "manual", "check": chk}
            t = self.till_current()
            if t:
                return {"id": t["id"], "how": "camera", "check": "ok", "score": t["score"]}
        return None

    def shopper_view(self):
        now = time.time()
        inside = sorted((s for s in self.shoppers.values() if s["status"] != "left"), key=lambda s: -s["t_in"])
        return {"inside": len(inside), "issued": self.sh_stats["issued"], "left_unbilled": self.sh_stats["left_unbilled"],
                "unverified": self.sh_stats["unverified"], "till": self.till_current(),
                "list": [{"id": s["id"], "mins": round((now - s["t_in"]) / 60, 1), "billed": bool(s["bills"])}
                         for s in inside[:60]]}

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
            if self.queue_peak["day"] != day_start():
                self.queue_peak = {"day": day_start(), "len": 0, "wait": 0.0}
            self.queue_peak["len"] = max(self.queue_peak["len"], q["length"])
            self.queue_peak["wait"] = max(self.queue_peak["wait"], q["wait_min"])
        opn, bld, cls = f"queue:{cam}:open", f"queue:{cam}:build", f"queue:{cam}:close"
        gone = (f", clears in ~{q['clear_min']} min" if q.get("clear_min") else
                ", not clearing at this rate" if q["length"] and q.get("clear_min") is None else "")
        if q["recommend_counters"] > k:
            more = q.get("clear_min_if_opened")
            self.fire(opn, "queue", 3,
                      f"{q['length']} in queue, wait ~{q['wait_min']} min{gone}; +10 min forecast {q['forecast']['10']['wait']} min",
                      f"Open {q['recommend_counters'] - k} more counter(s)" +
                      (f" — clears in ~{more} min" if more else ""))
            self.resolve(bld, cls)
        elif q["wait_min"] > CONFIG["queue"]["target_wait_min"]:
            self.fire(bld, "queue", 2, f"Queue building, wait ~{q['wait_min']} min{gone}",
                      "Keep a standby cashier ready")
            self.resolve(opn, cls)
        elif k > 1 and q["recommend_counters"] < k and q["length"] == 0:
            self.fire(cls, "queue", 1, "Billing idle", "Close a counter, reassign staff to shelves")
            self.resolve(opn, bld)
        else:
            self.resolve(opn, bld, cls)                  # queue healthy again

    # ── crowds: anywhere a camera sees too many people, call staff and say when it should thin out ──
    def cam_name(self, cam):
        c = self.layout.cam(cam) if self.layout else None
        return (c or {}).get("name") or cam

    def crowd_eta(self, cam, n, now=None):
        """Minutes until the count falls back under the threshold, from the trend over the last couple of
        minutes: 0 if it already has, None if it isn't thinning (or there isn't enough to go on yet)."""
        c, now = CONFIG["crowd"], now or time.time()
        target = c["people"] * c["clear_ratio"]
        if n <= target:
            return 0.0
        pts = [(t, v) for t, v in self.crowd_hist[cam] if t >= now - c["trend_window_s"]]
        if len(pts) < 8 or pts[-1][0] - pts[0][0] < 30:
            return None
        t = np.array([p[0] for p in pts]) - pts[0][0]
        slope = float(np.polyfit(t / 60, np.array([p[1] for p in pts], float), 1)[0])    # people per minute
        if slope > -0.15:
            return None
        return round(min(60.0, (n - target) / -slope), 1)

    def on_crowd(self, cam, n):
        c, now = CONFIG["crowd"], time.time()
        key, ended = f"crowd:{cam}", None
        with self.lock:
            h = self.crowd_hist[cam]
            h.append((now, n))
            near = sorted(v for t, v in h if t >= now - 3)
            now_n = near[len(near) // 2]                    # median of the last few seconds: one bad frame isn't a crowd
            st = self.crowds.get(cam)
            if st is None:
                if now_n >= c["people"]:
                    t0 = self.crowd_pending.setdefault(cam, now)
                    if now - t0 >= c["hold_s"]:
                        st = self.crowds[cam] = {"since": now, "peak": now_n}
                        self.crowd_pending.pop(cam, None)
                else:
                    self.crowd_pending.pop(cam, None)
            else:
                st["peak"] = max(st["peak"], now_n)
                if now_n <= c["people"] * c["clear_ratio"]:
                    ended = self.crowds.pop(cam)
                    st = None
        if ended:                                           # it has thinned out: log it, drop the alert
            self.store.event(cam, "crowd", {"where": self.cam_name(cam), "people": ended["peak"],
                                            "duration_s": round(now - ended["since"], 1)})
            self.resolve(key)
        if st is None:
            return
        where = self.cam_name(cam)
        eta = self.crowd_eta(cam, now_n, now)
        staff = int(min(3, max(1, round(now_n / c["people"]))))
        when = ("thinning — about " + f"{eta} min to go" if eta else "not thinning yet") if eta != 0 else "easing"
        self.fire(key, "crowd", 3 if now_n >= 1.5 * c["people"] else 2,
                  f"{now_n} people at {where} — {when}",
                  f"Send {staff} staff to {where}")

    def crowd_view(self):
        now = time.time()
        out = []
        for cam, st in self.crowds.items():
            h = [v for t, v in self.crowd_hist[cam] if t >= now - 3]
            n = sorted(h)[len(h) // 2] if h else 0
            eta = self.crowd_eta(cam, n, now)
            alert = any(not a["acked"] and self.keys.get(a["id"]) == f"crowd:{cam}" for a in self.alerts)
            out.append({"cam": cam, "where": self.cam_name(cam), "people": n, "peak": st["peak"],
                        "for_s": round(now - st["since"]), "eta_min": eta, "staff_alert": alert})
        return sorted(out, key=lambda x: -x["people"])

    def on_shelf(self, cam, cells):
        with self.lock:
            self.shelves[cam] = cells
        if not cells:
            return
        self.inventory.observe_cells(cells)
        inventoried = {p["sku"] for p in self.inventory.list_products()}
        w = CONFIG["shelf"]["weights"].get(cam, 1.0)
        where = self.labels.get(cam, cam)
        for c in cells:
            if c["occluded"]:
                continue
            if "slot" in c:                          # a marked product, not an anonymous grid cell
                key = f"shelf:{cam}:{c['slot']}"
                loc = f"{c['name']} — {where}, {c.get('loc', '')}"
            else:
                key, loc = f"shelf:{cam}:{c['r']},{c['c']}", f"{where} row {c['r'] + 1} col {c['c'] + 1}"
            if c.get("sku") in inventoried:
                self.resolve(key)
            elif c["status"] == "EMPTY":
                self.fire(key, "stock", 3, f"{loc} is empty", "Critical refill", w)
            elif c["status"] == "LOW":
                left = f" — about {c['est_units']} of {c['full_units']} left" if "slot" in c else \
                       f" ({c['fill'] * 100:.0f}%)"
                self.fire(key, "stock", 2, f"{loc} low{left}", "Refill", w)
            elif c["eta_min"] is not None and c["eta_min"] < 15:
                self.fire(key, "stock", 2, f"{loc} expected empty in ~{c['eta_min']:.0f} min", "Refill soon", w)
            else:
                self.resolve(key)                        # cell restocked
            if c.get("misplaced"):                # a product box holding something else
                self.fire(key + ":pg", "planogram", 2, f"{loc}: a different product is in this box",
                          "Put the right product back", w)
            elif c["status"] == "MISPLACED":
                exp = f" (expected {c['expected']}, found {c['found']})" if c.get("expected") else ""
                self.fire(key + ":pg", "planogram", 1, f"{loc} planogram mismatch{exp}", "Correct shelf", w)
            else:
                self.resolve(key + ":pg")
        for c in cells:                           # tell integrations when a product's shelf status changes
            if "slot" in c and not c["occluded"]:
                k = (cam, c["slot"])
                if self.last_status.get(k) not in (None, c["status"]):
                    self.emit("stock", {"cam": cam, "slot": c["slot"], "sku": c.get("sku"), "name": c["name"],
                                        "where": f"{where}, {c.get('loc', '')}", "status": c["status"],
                                        "was": self.last_status[k], "units": c["est_units"], "full": c["full_units"],
                                        "method": c.get("method")})
                self.last_status[k] = c["status"]
        # alerts for boxes or grid cells this camera no longer reports (boxes redrawn, grid replaced
        # by product boxes) would otherwise sit on the Home page forever
        live = {f"shelf:{cam}:{c['slot']}" if "slot" in c else f"shelf:{cam}:{c['r']},{c['c']}" for c in cells}
        with self.lock:
            stale = [k for i, k in self.keys.items() if k.startswith(f"shelf:{cam}:")
                     and k.split(":pg")[0] not in live and any(a["id"] == i and not a["acked"] for a in self.alerts)]
        if stale:
            self.resolve(*set(stale))

    # ── integrations: tell POS / inventory / ERP systems what happened ──
    def emit(self, kind, data):
        for h in CONFIG.get("webhooks") or []:
            url, events = (h, None) if isinstance(h, str) else (h.get("url"), h.get("events"))
            if url and (not events or kind in events):
                self.store.hook_add(url, {"type": kind, "store_id": CONFIG["store_id"], "ts": time.time(), "data": data})

    # ── shopper attention per product ────────────────────────────────
    def on_product_dwell(self, cam, slot, dur):
        day = day_start()
        with self.lock:
            a = self.attention.get((cam, slot["id"]))
            if not a or a["day"] != day:
                a = self.attention[(cam, slot["id"])] = {"day": day, "visits": 0, "total_s": 0.0}
            a["visits"] += 1
            a["total_s"] += dur
        self.store.event(cam, "product_dwell", {"slot": slot["id"], "sku": slot.get("sku", ""),
                                                "name": slot.get("name", ""), "s": round(dur, 1)})

    def attention_today(self, cam, sid):
        with self.lock:
            a = self.attention.get((cam, sid))
        if not a or a["day"] != day_start():
            return {"visits": 0, "avg_s": None}
        return {"visits": a["visits"], "avg_s": round(a["total_s"] / a["visits"], 1)}

    def cam_status(self, name, roles, online, fps, people=0, msg=""):
        with self.lock:
            self.cams[name] = {"roles": sorted(roles), "online": online, "fps": round(fps, 1), "people": people,
                               "msg": msg}

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
        nb, rev = self.store.q("SELECT COUNT(*), COALESCE(SUM(total),0) FROM bills WHERE ts>=? AND demo=0", since)[0]
        issued = self.store.q("SELECT COUNT(*) FROM events WHERE type='shopper_in' AND ts>=? AND data NOT LIKE '%again%'", since)[0][0]
        unbilled = sum(1 for (d,) in self.store.q("SELECT data FROM events WHERE type='shopper_out' AND ts>=?", since)
                       if not json.loads(d).get("billed"))
        unverified = self.store.q("SELECT COUNT(*) FROM events WHERE type='unverified_checkout' AND ts>=?", since)[0][0]
        with self.lock:
            self.sh_stats = {"issued": issued, "left_unbilled": unbilled, "unverified": unverified}
            self.bills_today, self.sales_today = nb, round(rev, 2)
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
                "inventory": self.inventory.status(),
                "labels": dict(self.labels), "cam_face": dict(self.cam_face),
                "depth": dict(self.depth_info), "depth_model_err": DEPTH["err"],
                "pos_tracked": len(self.stock), "shoppers": self.shopper_view(),
                "crowds": self.crowd_view(), "crowd_threshold": CONFIG["crowd"]["people"],
                "queue_peak": dict(self.queue_peak),
                "pos": {"active_cart": self.active_cart, "last_scan": self.last_scan},
                "sales_today": getattr(self, "sales_today", 0), "bills_today": getattr(self, "bills_today", 0),
                "hour_now": datetime.now().hour,
                "store_heat": self.store_heat_norm(), "store_live": self.store_live_norm(),
                "heat_src": dict(self.heat_src), "layout": self.layout.data if self.layout else None,
            }

    def store_heat_norm(self):
        """Today's dwell time on the floor, scaled so the busiest spot is 1."""
        h = self.store_heat
        if h is None:
            return None
        m = float(h.max())
        return (h / m if m > 0 else h).round(3).tolist()

    def store_live_norm(self):
        """Where people have been in the last minute or so (older visits fade out). Scaled so one person
        standing still for ~10 s is already 1 — a single shopper browsing shows up straight away, instead of
        being drowned by the day's busiest spot."""
        h = self.store_live
        if h is None:
            return None
        with self.lock:
            self.fade_live(time.time())
            ref = max(float(h.max()), 0.25 * CONFIG["heat"]["live_tau_s"] * self.kernel()[0].max())
            return (h / ref).round(3).tolist()

    def fade_live(self, now):
        self.store_live *= math.exp(-max(0.0, now - self.heat_t) / CONFIG["heat"]["live_tau_s"])
        self.heat_t = now

    def kernel(self):
        """Gaussian that spreads one second of someone's presence over the cells around their feet."""
        cell = max(CONFIG["store_cell_m"], 0.05)
        key = (cell, CONFIG["heat"]["spread_m"])
        if self._kern is None or self._kern[1] != key:
            sig = max(CONFIG["heat"]["spread_m"], 0.05) / cell
            r = max(1, int(math.ceil(3 * sig)))
            ax = np.arange(-r, r + 1, dtype=np.float32)
            k = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2 * sig * sig))
            self._kern = (k / k.sum(), key, r)
        return self._kern

    def add_store_heat(self, mx, my, dt):
        """Drop dwell seconds onto the store-wide floor grid, in metres: into today's total and into the
        fading "now" layer."""
        h = self.store_heat
        if h is None:
            return
        cell = max(CONFIG["store_cell_m"], 0.05)
        r0, c0 = int(my / cell), int(mx / cell)
        if not (0 <= r0 < h.shape[0] and 0 <= c0 < h.shape[1]):
            return
        k, _, rad = self.kernel()
        with self.lock:
            now = time.time()
            if day_start(now) != self.heat_day:           # a new trading day starts with a clean floor
                self.heat_day = day_start(now)
                h[:] = 0
            self.fade_live(now)
            ya, yb = max(0, r0 - rad), min(h.shape[0], r0 + rad + 1)
            xa, xb = max(0, c0 - rad), min(h.shape[1], c0 + rad + 1)
            sub = k[ya - (r0 - rad):yb - (r0 - rad), xa - (c0 - rad):xb - (c0 - rad)] * dt
            h[ya:yb, xa:xb] += sub
            self.store_live[ya:yb, xa:xb] += sub


def approx_store_point(cam, u, v):
    """Estimate where someone's feet are on the plan from the camera's placement alone. u, v: the foot's
    position in the picture, 0..1. Sideways angle comes straight from the field of view; distance runs from
    close to the camera (bottom of the picture) out to its range (top). Returns None outside the cone."""
    if not (0 <= u <= 1 and 0 <= v <= 1):
        return None
    hd, half = math.radians(cam["heading"]), math.radians(cam["fov"]) / 2
    near = 0.2 * cam["range"]
    d = near + (cam["range"] - near) * (1 - v)
    lateral = d * math.tan(half) * (2 * u - 1)
    return (cam["x"] + d * math.cos(hd) - lateral * math.sin(hd),
            cam["y"] + d * math.sin(hd) + lateral * math.cos(hd))


# ─────────────────────────────── CAMERA WORKERS ───────────────────────────────
class CamWorker(threading.Thread):
    period = 0.0

    def __init__(self, name, roles, source, engine):
        super().__init__(daemon=True)
        self.name, self.roles, self.engine = name, roles, engine
        self.source, self.alive = source, True
        self.cap = Capture(source)
        self.jpg, self.fps = None, 0.0
        self.t_prev = self.t_start = self.t_shown = time.time()
        self.cctv_jpg, self.cctv_want, self.people = None, 0.0, 0
        # for the live view: last person boxes (to blur), the overlay of the last processed frame
        self.blur_boxes, self.blur_t, self.cctv_boxes = [], 0.0, []
        self.ov, self._live_key, self._live = None, None, None
        self.storeH, self._sp_sig = None, None

    def store_point(self, fx, fy, w, h, quad=None):
        """Where on the store plan a person's feet are (metres), or None. Exact when this camera has its four
        floor points and the floor patch placed on the plan; otherwise estimated from where the camera sits
        (position, heading, field of view, range) — good enough to show where people are, not to the
        centimetre. Which one is in use is reported to the dashboard."""
        lay = self.engine.layout
        cam = lay.cam(self.name) if lay else None
        if not cam:
            return None
        quad = quad or cam.get("floor_quad")
        if quad and cam.get("floor_rect"):
            sig = (json.dumps(quad), json.dumps(cam["floor_rect"]), w, h)
            if self.storeH is None or getattr(self, "_sp_sig", None) != sig:
                src = np.float32([[x * w, y * h] for x, y in quad][:4])
                self.storeH = cv2.getPerspectiveTransform(src, np.float32(rect_corners(cam["floor_rect"])))
                self._sp_sig = sig
            mx, my = cv2.perspectiveTransform(np.float32([[[fx, fy]]]), self.storeH)[0, 0]
            self.engine.heat_src[self.name] = "exact"
            return float(mx), float(my)
        self.engine.heat_src[self.name] = "approx"
        return approx_store_point(cam, fx / w, fy / h)

    def cctv(self, frame, boxes, labels=None):
        """Plain security view: the picture plus people boxes, no analytics overlays. Heads are
        still blurred — privacy is a property of the system, not of which tab you open. Only
        encoded while someone is watching the CCTV tab, to spare the edge CPU."""
        self.people = len(boxes)
        self.engine.on_crowd(self.name, self.people)
        self.cctv_boxes = list(boxes)
        if time.time() - self.cctv_want > 6:
            return
        v = frame.copy()
        if CONFIG["privacy_blur"]:
            blur_heads(v, [b[:4] for b in boxes])
        for b in boxes:
            x1, y1, x2, y2 = map(int, b[:4])
            cv2.rectangle(v, (x1, y1), (x2, y2), (60, 60, 230), 2)
            if len(b) > 4:
                cv2.putText(v, (labels or {}).get(b[4], f"#{b[4]}"), (x1, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 230), 1)
        cv2.putText(v, datetime.now().strftime("%d-%m-%Y %H:%M:%S"), (10, v.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(v, f"{self.name}  people: {len(boxes)}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            self.cctv_jpg = buf.tobytes()

    def stop(self):
        self.alive = False
        self.cap.stop()

    def blur(self, vis, boxes, t=None):
        """Blur heads and remember where people were (and when) for the live view."""
        self.blur_boxes, self.blur_t = [list(b[:4]) for b in boxes], t or time.time()
        if CONFIG["privacy_blur"]:
            blur_heads(vis, self.blur_boxes)

    def live_jpg(self, view="analytics"):
        """The newest camera frame with the last analysis drawn on it, so the picture moves at the
        camera's rate even when detection runs slower. Privacy first: people are blurred where they
        were last seen, with a margin that grows with the age of that detection; if detection has
        stalled, return None and the caller shows the last fully processed frame instead."""
        age = time.time() - self.blur_t
        if age > 2.0:
            return None
        f = self.cap.read()
        if f is None:
            return None
        key = (self.cap.seq, view, self.blur_t)
        if key == self._live_key:
            return self._live
        v = f.copy()
        ov = self.ov
        if view == "analytics" and ov is not None and ov[0].shape == v.shape:
            v[ov[1]] = ov[0][ov[1]]
        if CONFIG["privacy_blur"] and self.blur_boxes:
            h, w = v.shape[:2]
            grow = 0.15 + 0.6 * min(age, 1.0)          # people move: the older the boxes, the wider the blur
            for x1, y1, x2, y2 in self.blur_boxes:
                bw, bh = x2 - x1, y2 - y1
                a, b = max(0, int(x1 - grow * bw)), max(0, int(y1 - grow * bh))
                c = min(w, int(x2 + grow * bw))
                d = min(h, int(y1 + (0.25 if age < 0.5 else 1.0) * bh + grow * bh))
                if c - a > 2 and d - b > 2:
                    v[b:d, a:c] = cv2.GaussianBlur(v[b:d, a:c], (31, 31), 0)
        if view == "cctv":
            for bx in self.cctv_boxes:
                x1, y1, x2, y2 = map(int, bx[:4])
                cv2.rectangle(v, (x1, y1), (x2, y2), (60, 60, 230), 2)
            cv2.putText(v, datetime.now().strftime("%d-%m-%Y %H:%M:%S"), (10, v.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(v, f"{self.name}  people: {self.people}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            return None
        self._live_key, self._live = key, buf.tobytes()
        return self._live

    def run(self):
        last = -1
        while self.alive:
            f = self.cap.read()
            if f is None:
                self.engine.cam_status(self.name, self.roles, False, 0, msg=self.cap.status)
                time.sleep(0.3)
                continue
            if self.cap.seq == last:             # nothing new from the camera yet: don't redo work
                time.sleep(0.005)
                continue
            last = self.cap.seq
            t = time.time()
            try:
                vis = self.step(f)
            except Exception as e:
                print(f"[{self.name}] {type(e).__name__}: {e}")
                time.sleep(1.0)
                continue
            # frames actually shown per second (was: how fast one frame is processed — misleading)
            self.fps = 0.9 * self.fps + 0.1 / max(t - self.t_shown, 1e-3)
            self.t_shown = t
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                self.jpg = buf.tobytes()
            try:                                  # remember what the analysis drew, to lay over newer frames
                if vis.shape == f.shape:
                    base = f.copy()
                    if CONFIG["privacy_blur"]:
                        blur_heads(base, self.blur_boxes)
                    self.ov = (vis, cv2.absdiff(vis, base).max(axis=2) > 0)
            except Exception:
                self.ov = None
            if self.alive:
                self.engine.cam_status(self.name, self.roles, True, self.fps, self.people, msg="live")
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
        self.shopper_of = {}                  # track id -> the shopper ID given when they came in
        R, C = self.g["floor_grid"]
        self.heat = np.zeros((R, C), np.float32)
        engine.heat[name] = self.heat
        self.H = None
        self.zone_time, self.zone_last = defaultdict(dict), defaultdict(dict)
        self.cand, self.members, self.missing = {}, {}, {}
        self.arrivals, self.departures = deque(), deque()
        self.mu_c = CONFIG["queue"]["default_service_per_min"]
        self._quad = self.g.get("floor_quad")
        self._rect = None

    def geo(self):
        """Geometry for this camera: CONFIG defaults, overridden by whatever was drawn on its
        picture in the dashboard. Editing in the UI takes effect on the next frame."""
        g = dict(geom(self.name))
        lay = self.engine.layout
        cam = lay.cam(self.name) if lay else None
        if cam:
            for k in ("entry_line", "in_side", "queue_zone", "floor_quad"):
                if cam.get(k) is not None:
                    g[k] = cam[k]
        rect = cam.get("floor_rect") if cam else None
        if g.get("floor_quad") != self._quad or rect != self._rect:
            self._quad, self._rect = g.get("floor_quad"), rect
            self.H = self.storeH = None          # remap on the next point
        return g

    def store_point(self, fx, fy, w, h, quad=None):
        return CamWorker.store_point(self, fx, fy, w, h, quad=quad or self.g.get("floor_quad"))

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
        self.g = self.geo()
        r = self.model.track(frame, persist=True, classes=[0], conf=CONFIG["conf"], imgsz=CONFIG["imgsz"],
                             tracker="bytetrack.yaml", verbose=False)[0]
        dets = []
        if r.boxes is not None and r.boxes.id is not None:
            for bb, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
                dets.append((*map(float, bb), tid))
        self.cctv(frame, dets, {t: "SH" + sid.rsplit("-", 1)[1] for t, sid in self.shopper_of.items()})
        vis = frame.copy()
        self.blur(vis, dets)

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
                        sid = self.engine.on_entry(self.name, "in" if s == g["in_side"] else "out",
                                                   sig=body_signature(frame, (x1, y1, x2, y2)), tid=tid)
                        if sid:
                            self.shopper_of[tid] = sid
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
            sid = self.shopper_of.get(tid)
            cv2.putText(vis, "SH" + sid.rsplit("-", 1)[1] if sid else f"#{tid}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

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
        if len(self.shopper_of) > 300:        # tracks come and go; keep only the recent ones
            self.shopper_of = dict(list(self.shopper_of.items())[-150:])

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
        def clears(n):
            """Minutes for today's line to disappear with n counters, at the current arrival rate: the line
            shrinks by (service − arrivals) a minute. None = arrivals outpace service, it will keep growing."""
            if L == 0:
                return 0.0
            net = self.mu_c * n - lam
            return round(min(120.0, L / net), 1) if net > 0.05 else None
        self.engine.on_queue(self.name, {
            "length": L, "arrival_per_min": round(lam, 2), "service_per_counter_min": round(self.mu_c, 2),
            "wait_min": round(L / cap, 1), "forecast": forecast, "recommend_counters": rec,
            "clear_min": clears(k), "clear_min_if_opened": clears(rec) if rec > k else None,
            "what_if": [{"counters": n, "wait_min": round(L / max(self.mu_c * n, 1e-3), 1), "clear_min": clears(n),
                         "keeps_up": lam <= 0.9 * self.mu_c * n} for n in range(1, qc["max_counters"] + 1)],
        })


# ─────────────────────────── MONOCULAR DEPTH ───────────────────────────
# One ordinary camera sees only the front unit of each column. When that unit is taken, the next
# one is still there — just further back. A monocular depth model (Depth Anything V2, metric indoor)
# estimates distance per pixel, so "how far back is the front unit now, compared with the full
# shelf" divided by the size of one unit = units gone from that column. Works at any camera angle.
DEPTH = {"fn": None, "err": None, "tried": False, "lock": threading.Lock(), "name": None}


def depth_model():
    """The depth function (BGR frame -> metres per pixel, same size), or None with DEPTH['err'] set.
    Loaded once, shared by every shelf camera. Tests put a stand-in in DEPTH['fn']."""
    if DEPTH["fn"] is not None or DEPTH["tried"]:
        return DEPTH["fn"]
    DEPTH["tried"] = True
    dc = CONFIG["shelf"]["depth"]
    if not dc.get("enabled"):
        DEPTH["err"] = "turned off in CONFIG"
        return None
    try:
        import torch
        from PIL import Image
        from transformers import pipeline
        dev = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("depth-estimation", model=dc["model"], device=dev)

        def fn(bgr):
            h, w = bgr.shape[:2]
            out = pipe(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            d = out["predicted_depth"]
            d = (d.detach().float().cpu().numpy() if hasattr(d, "detach") else np.asarray(d, np.float32)).squeeze()
            return cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        DEPTH["fn"], DEPTH["name"] = fn, dc["model"].split("/")[-1]
        print(f"[depth] {DEPTH['name']} loaded on {'GPU' if dev == 0 else 'CPU'}")
    except ImportError:
        DEPTH["err"] = "not installed — pip install transformers"
    except Exception as e:
        DEPTH["err"] = f"could not load: {type(e).__name__}: {str(e)[:120]}"
    if DEPTH["err"]:
        print(f"[depth] {DEPTH['err']} (depth counts off; using facings × depth)")
    return DEPTH["fn"]


def run_depth(bgr):
    fn = depth_model()
    if fn is None:
        return None
    with DEPTH["lock"]:                    # one model, several shelf cameras
        return fn(bgr)


def align_depth(d, d0, mask):
    """Fit d0 ≈ a·d + b on pixels that should not change (shelf frame, walls, floor) and apply it
    to d. A monocular model's scale drifts a few percent frame to frame — at 1.5 m that is a whole
    unit — so live depth is always re-anchored to the calibration pass. Returns (aligned, noise_m)."""
    x, y = d[mask], d0[mask]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 500:
        return d, None
    step = max(1, x.size // 40000)
    x, y = x[::step].astype(np.float64), y[::step].astype(np.float64)
    keep = np.ones(x.size, bool)
    a, b = 1.0, 0.0
    for _ in range(3):                     # trimmed least squares: drop what did change
        A = np.stack([x[keep], np.ones(int(keep.sum()))], 1)
        (a, b), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        r = np.abs(a * x + b - y)
        keep = r <= np.percentile(r, 70)
    if not 0.6 < a < 1.6:                  # implausible — the view itself changed; don't trust it
        return d, None
    return (a * d + b).astype(np.float32), float(np.median(np.abs(a * x[keep] + b - y[keep])))


def depth_region(box, w, h):
    """Centre of a facing: where the front unit's face is, away from the gaps between columns."""
    _, x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * 0.2, (y1 - y0) * 0.15
    a, b = min(max(int(x0 + mx), 0), w - 1), min(max(int(y0 + my), 0), h - 1)
    return a, b, min(max(int(x1 - mx), a + 1), w), min(max(int(y1 - my), b + 1), h)


def nearest_surface(dist, unit, min_pts=6):
    """Distance of the nearest real surface among the points in a column's tube. Depth models smear
    the rim of a gap ('flying pixels' between near and far), so the nearest few points can't be
    trusted; a unit's face is a dense cluster. Bins of half a unit: take the first bin holding a fair
    share of the points, then the median of the points around it."""
    if dist.size < min_pts:
        return None
    step = unit / 2
    bins = np.floor(dist / step).astype(int)
    lo = bins.min()
    counts = np.bincount(bins - lo)
    if counts.max() < min_pts:              # only smeared rim pixels: the column can't be seen
        return None
    need = max(min_pts, 0.3 * counts.max())
    first = int(np.argmax(counts >= need))
    centre = (first + lo + 0.5) * step
    near = dist[np.abs(dist - centre) <= step]
    return float(np.median(near))


def backproject(d, hfov_deg):
    """Depth map -> 3-D point per pixel (metres, camera frame: x right, y down, z forward)."""
    h, w = d.shape
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    u = (np.arange(w, dtype=np.float32) - w / 2 + 0.5) / f
    v = (np.arange(h, dtype=np.float32) - h / 2 + 0.5) / f
    return np.stack([d * u[None, :], d * v[:, None], d], -1)


def shelf_frame(P0, slots, w, h, plane_slots=None):
    """From the calibrated (full) shelf: the plane the product fronts stand on, and for every column
    the patch of that plane its front unit covers. A unit taken from the front leaves the next one
    further back along the plane's normal — inside the same 'tube', wherever that lands in the picture.
    The plane is fitted through the fronts of every box on the shelf (`plane_slots`, default `slots`), so a
    box with a single column can be counted too; tubes are built only for `slots`."""
    per = {}
    for s in (plane_slots or slots):
        for box in ShelfWorker.facing_boxes(s, w, h):
            a, b, c2, d2 = depth_region(box, w, h)
            q = P0[b:d2, a:c2].reshape(-1, 3)
            q = q[np.isfinite(q).all(1)]
            if len(q) >= 12:
                per[(s["id"], box[0])] = (q, s)
    counted = {s["id"] for s in slots}
    if not any(k[0] in counted for k in per):
        return None
    X = np.concatenate([q for q, _ in per.values()])
    c = np.median(X, 0)
    for _ in range(2):                       # plane through the fronts, robust to the odd stray point
        _, _, vt = np.linalg.svd(X - c, full_matrices=False)
        n = vt[2]
        dist = np.abs((X - c) @ n)
        X = X[dist <= np.percentile(dist, 80)]
        c = X.mean(0)
    n = n if n @ c > 0 else -n               # pointing away from the camera, into the shelf
    e1 = np.array([1.0, 0, 0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    cols, raw = {}, {}
    for k, (q, s) in per.items():
        if k[0] not in counted:
            continue
        r = q - c
        dn = r @ n
        front0 = float(np.median(dn))
        on = np.abs(dn - front0) < max(0.02, s["unit_cm"] / 200)
        if on.sum() < 8:
            continue
        pa, pb = r[on] @ e1, r[on] @ e2
        a0, a1 = np.percentile(pa, [5, 95])
        b0, b1 = np.percentile(pb, [5, 95])
        ma, mb = (a1 - a0) * 0.1, (b1 - b0) * 0.1
        raw[k] = (a0, a1, b0, b1, front0, s)
        # last: points a unit face must show (live passes use every 2nd pixel each way)
        cols[k] = (a0 + ma, a1 - ma, b0 + mb, b1 - mb, front0, max(6, int(on.sum() / 4 * 0.15)))
    return {"c": c, "n": n, "e1": e1, "e2": e2, "cols": cols, "ev": slice_evidence(P0, c, n, e1, e2, raw)}


def slice_evidence(P0, c, n, e1, e2, raw, min_pts=8):
    """Where on the FULL shelf the camera could see a unit's top or side, behind the front row. Straight on
    there is nothing to see but the front pack; from above or from the side, the tops/sides of the packs
    behind it show up as surface at known depths. Later, if those pixels have moved away, those packs are gone —
    even when the front one is still there. Returns {column: {unit position: (pixel indices, depth then)}}.
    Positions with too few visible pixels are simply absent: that is 'cannot tell', never a guess."""
    Ps = P0[::2, ::2].reshape(-1, 3) - c                  # live passes sample every 2nd pixel each way
    ok = np.isfinite(Ps).all(1)
    A, B, D = Ps @ e1, Ps @ e2, Ps @ n
    keys = sorted(raw, key=lambda k: raw[k][0])
    out = {}
    for k in keys:
        a0, a1, b0, b1, f0, s = raw[k]
        u, hh, ww = s["unit_cm"] / 100.0, b1 - b0, a1 - a0
        if s["deep"] < 2 or u < 0.04:                    # packs thinner than ~4 cm are within the model's noise
            continue
        left = [raw[o] for o in keys if raw[o][1] < a0 and raw[o][2] < b1 and raw[o][3] > b0]
        right = [raw[o] for o in keys if raw[o][0] > a1 and raw[o][2] < b1 and raw[o][3] > b0]
        la = max(a0 - 0.5 * ww, (max(r[1] for r in left) + a0) / 2) if left else a0 - 0.5 * ww
        ra = min(a1 + 0.5 * ww, (min(r[0] for r in right) + a1) / 2) if right else a1 + 0.5 * ww
        top = (A > a0) & (A < a1) & (B > b0 - 0.2 * hh) & (B < b0 + 0.1 * hh)
        sides = (B > b0) & (B < b1) & (((A > la) & (A < a0)) | ((A > a1) & (A < ra)))
        m = (top | sides) & ok
        ev = {}
        for pos in range(1, s["deep"]):
            sel = np.flatnonzero(m & (D > f0 + (pos + 0.25) * u) & (D < f0 + (pos + 0.75) * u))
            if sel.size >= min_pts:
                ev[pos] = (sel.astype(np.int32), D[sel].astype(np.float32))
        if ev:
            out[k] = ev
    return out


def depth_colour(d, lo=None, hi=None):
    """Depth map as a picture: near = bright red, far = dark (the dashboard's palette)."""
    v = d[np.isfinite(d)]
    lo = float(np.percentile(v, 2)) if lo is None else lo
    hi = float(np.percentile(v, 98)) if hi is None else hi
    n = np.clip((hi - d) / max(hi - lo, 1e-6), 0, 1)
    n = np.nan_to_num(n)
    img = np.zeros(d.shape + (3,), np.uint8)
    img[..., 2] = (30 + 202 * n).astype(np.uint8)          # red rises as things get nearer
    img[..., 1] = (22 + 32 * n ** 2).astype(np.uint8)
    img[..., 0] = (25 + 54 * n ** 2).astype(np.uint8)
    return img


def hs_hist(region):
    """Hue-saturation histogram of a picture region: a product's colour fingerprint."""
    if region.size == 0:
        return np.zeros((18, 8), np.float32)
    hist = cv2.calcHist([region], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def body_signature(frame, box):
    """Clothing-colour fingerprint of a person: for each of three bands (shoulders to waist, waist to knee,
    shin down) a histogram of hues, plus how much of the band is dark, grey or light. Brightness is ignored
    for coloured cloth, so the door camera and the till camera can agree despite different lighting. The head
    is left out and no picture is kept: it tells the till "this is who the door camera saw", not who anyone is."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    bw, bh = x2 - x1, y2 - y1
    if bw < 12 or bh < 36:
        return None
    xa, xb = int(max(0, x1 + 0.2 * bw)), int(min(w, x2 - 0.2 * bw))
    bands = []
    for a, b in ((0.18, 0.45), (0.45, 0.72), (0.72, 1.0)):
        ya, yb = int(max(0, y1 + a * bh)), int(min(h, y1 + b * bh))
        if yb - ya < 4 or xb - xa < 4:
            return None
        hsv = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2HSV).reshape(-1, 3)
        hue, sat, val = hsv[:, 0].astype(int), hsv[:, 1], hsv[:, 2]
        colour = (sat >= 60) & (val >= 60)
        hist = np.zeros(15, np.float32)
        hues = np.bincount(np.minimum(hue[colour] * 12 // 180, 11), minlength=12).astype(np.float32)
        hist[:12] = 0.5 * hues + 0.25 * np.roll(hues, 1) + 0.25 * np.roll(hues, -1)     # red wraps round
        hist[12:] = np.bincount(np.digitize(val[~colour], (85, 170)), minlength=3)      # dark / grey / light
        bands.append(hist / max(float(hist.sum()), 1e-6))
    return np.stack(bands).astype(np.float32)


def sig_similarity(a, b):
    """0..1 — how alike two body signatures are (Bhattacharyya coefficient per band, averaged)."""
    return float(np.clip(np.sqrt(a * b).sum(1).mean(), 0, 1))


def count_badge(img, x_right, y_top, text, colour):
    """A filled tag like '4/5' tucked into a box's top-right corner."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x0, y0 = max(0, x_right - tw - 14), max(0, y_top + 3)
    cv2.rectangle(img, (x0, y0), (x_right - 3, y0 + th + 12), colour, -1)
    cv2.putText(img, text, (x0 + 6, y0 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


class ShelfWorker(CamWorker):
    """Grid over the opposite shelf. Each cell compared with a calibrated 'fully stocked' reference."""

    def __init__(self, name, roles, source, engine):
        super().__init__(name, roles, source, engine)
        from ultralytics import YOLO
        sc = CONFIG["shelf"]
        # people are checked often (smooth, blurred live view); stock is re-read every period_s
        self.period, self.t_read = min(sc["period_s"], sc.get("detect_s", 0.25)), 0.0
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
                engine.cam_face[name] = f"{sh['id']}:{fc}"
                n_sl = len((engine.layout.cam(name) or {}).get("slots") or [])
                print(f"[{name}] watching {sh['name']} {fc} face, " +
                      (f"{n_sl} product boxes" if n_sl else f"{self.R}x{self.C} grid (no product boxes yet)"))
        self.t_heat = time.time()
        self.clahe = cv2.createCLAHE(2.0, (8, 8))
        self.ref_path = f"shelf_ref_{name}.png"
        self.ref = cv2.imread(self.ref_path) if os.path.exists(self.ref_path) else None
        self.ref_feats = self.features(self.ref) if self.ref is not None else None
        self.recent = defaultdict(lambda: deque(maxlen=3))
        self.history = defaultdict(lambda: deque(maxlen=600))
        self.cells_map, self.calib_request, self.calib_msg = {}, False, None
        self.slot_ref, self.slot_sig = {}, None      # per-facing reference edges, rebuilt when slots change
        self.slot_hist, self.recent_corr = {}, defaultdict(lambda: deque(maxlen=3))
        self.engage = {}                                # slot -> [first seen, last seen] of someone in front of it
        # depth: the calibration pass (from the stored reference picture), per-column history, last result
        self.depth_ref, self.depth_t, self.depth_jpg = None, 0.0, None
        self.depth_geo, self.depth_geo_sig, self.depth_stack = None, None, []
        self.depth_want = 0.0                     # last time someone looked at the depth view
        self.depth_back = defaultdict(lambda: deque(maxlen=CONFIG["shelf"]["depth"]["smooth"]))
        self.depth_slices = defaultdict(lambda: deque(maxlen=CONFIG["shelf"]["depth"]["smooth"]))   # per pass: unit position -> absent/present/None
        self.ledger = {}                                  # column -> {"j": front position last seen, "lo": units for sure}
        self.depth_state = {"on": False, "msg": "no product box has a unit depth set", "noise_cm": None}

    # ── product slots ────────────────────────────────────────────────
    def slots(self):
        lay = self.engine.layout
        cam = lay.cam(self.name) if lay else None
        return (cam or {}).get("slots") or []

    @staticmethod
    def facing_boxes(slot, w, h):
        """Pixel box of each facing: a slot is split across its width, one column per unit."""
        x0, y0 = slot["x"] * w, slot["y"] * h
        bw, bh = slot["w"] * w, slot["h"] * h
        n = max(1, slot["facings"])
        for i in range(n):
            a = int(x0 + bw * i / n)
            b = int(x0 + bw * (i + 1) / n)
            yield i, a, int(y0), max(a + 1, b), int(y0 + bh)

    def edge_map(self, img):
        gray = self.clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        return cv2.Canny(gray, 60, 160) > 0

    def build_slot_ref(self, slots):
        """Measure each facing on the calibrated 'full' picture. Re-derived from the stored
        reference image, so editing slots never means re-shooting the shelf."""
        self.slot_ref = {}
        if self.ref is None:
            return
        e = self.edge_map(self.ref)
        h, w = self.ref.shape[:2]
        hsv = cv2.cvtColor(self.ref, cv2.COLOR_BGR2HSV)
        self.slot_hist = {}
        for s in slots:
            self.slot_ref[s["id"]] = [float(e[y0:y1, x0:x1].mean()) if e[y0:y1, x0:x1].size else 0.0
                                      for _, x0, y0, x1, y1 in self.facing_boxes(s, w, h)]
            # colour fingerprint of each facing on the full shelf: a different product in the box won't match it
            self.slot_hist[s["id"]] = [hs_hist(hsv[y0:y1, x0:x1]) for _, x0, y0, x1, y1 in self.facing_boxes(s, w, h)]
        self.slot_sig = json.dumps(slots, sort_keys=True)

    def track_attention(self, frame, persons, now):
        """Dwell at each product: someone standing in front of a product box (their box covers part of
        it) starts a stop; it ends when nobody has been there for 1.5 s. Stops under 1.5 s are people
        walking past and aren't counted. Anonymous — only durations are kept."""
        h, w = frame.shape[:2]
        for s in self.slots():
            x0, y0, x1, y1 = s["x"] * w, s["y"] * h, (s["x"] + s["w"]) * w, (s["y"] + s["h"]) * h
            area = max((x1 - x0) * (y1 - y0), 1)
            here = any(max(0, min(x1, a2) - max(x0, a1)) * max(0, min(y1, b2) - max(y0, b1)) / area >= 0.15
                       for a1, b1, a2, b2 in persons)
            st = self.engage.get(s["id"])
            if here:
                if st is None:
                    self.engage[s["id"]] = [now, now]
                else:
                    st[1] = now
            elif st is not None and now - st[1] > 1.5:
                dur = st[1] - st[0]
                del self.engage[s["id"]]
                if dur >= 1.5:
                    self.engine.on_product_dwell(self.name, s, dur)

    def depth_pass(self, frame, slots, persons):
        """Measure how far back the front unit of every depth-enabled column now sits, versus the
        calibration picture. Adds one reading (in units) per column to its short history."""
        want = [s for s in slots if s.get("unit_cm", 0) > 0]
        watching = time.time() - self.depth_want < 15
        if not want and not watching:            # nothing to count and nobody looking: save the CPU
            self.depth_state = {"on": False, "msg": "set 'one unit, front to back (cm)' on a product box to count units behind the front row", "noise_cm": None}
            return
        if self.ref is None or self.ref.shape != frame.shape:
            return
        if self.depth_ref is None:          # derived from the stored calibration picture
            self.depth_ref = run_depth(self.ref)
            self.depth_stack = [] if self.depth_ref is None else [self.depth_ref]
        d = run_depth(frame) if self.depth_ref is not None else None
        if d is None:
            self.depth_state = {"on": False, "msg": DEPTH["err"] or "unavailable", "noise_cm": None}
            return
        h, w = frame.shape[:2]
        mask = np.isfinite(self.depth_ref) & (self.depth_ref > 0)
        for s in slots:                     # products may change; everything else should not
            mask[int(s["y"] * h):int((s["y"] + s["h"]) * h), int(s["x"] * w):int((s["x"] + s["w"]) * w)] = False
        for px1, py1, px2, py2 in persons:
            mask[max(0, int(py1)):int(py2), max(0, int(px1)):int(px2)] = False
        da, noise = align_depth(d, self.depth_ref, mask)
        if noise is None:
            self.depth_state = {"on": False, "msg": "view changed since calibration — re-calibrate", "noise_cm": None}
            return
        n_cal = CONFIG["shelf"]["depth"]["calib_passes"]
        if not want:                             # depth picture only, nothing to count
            self.depth_state = {"on": False, "msg": "set 'one unit, front to back (cm)' on a product box to count units behind the front row", "noise_cm": round(noise * 100, 1)}
            self.depth_picture(da, mask, slots, [], w, h)
            return
        if len(self.depth_stack) < n_cal:   # the full-shelf depth is the median of a few passes
            self.depth_stack.append(da)
            self.depth_ref = np.median(np.stack(self.depth_stack), 0).astype(np.float32)
            self.depth_geo = None
            self.depth_state = {"on": False, "noise_cm": None,
                                "msg": f"measuring the full shelf ({len(self.depth_stack)}/{n_cal}) — leave it untouched"}
            return
        hfov = self.hfov()
        if self.depth_geo is None or self.depth_geo_sig != (self.slot_sig, hfov):
            self.depth_geo = shelf_frame(backproject(self.depth_ref, hfov), want, w, h, plane_slots=slots)
            self.depth_geo_sig = (self.slot_sig, hfov)
        geo = self.depth_geo
        if geo is None:
            self.depth_state = {"on": False, "msg": "could not find the shelf front in the depth map", "noise_cm": None}
            return
        # every pixel as a point in shelf coordinates: across (A), up/down (B), into the shelf (D)
        P = backproject(da, hfov)[::2, ::2]
        keep = np.isfinite(P).all(-1)
        for px1, py1, px2, py2 in persons:
            keep[max(0, int(py1) // 2):int(py2) // 2 + 1, max(0, int(px1) // 2):int(px2) // 2 + 1] = False
        keepf, Pf = keep.reshape(-1), P.reshape(-1, 3)
        R = P[keep] - geo["c"]
        A, B, D = R @ geo["e1"], R @ geo["e2"], R @ geo["n"]
        for s in want:
            x0, y0 = s["x"] * w, s["y"] * h
            x1, y1 = x0 + s["w"] * w, y0 + s["h"] * h
            blocked = any(max(0, min(x1, px2) - max(x0, px1)) * max(0, min(y1, py2) - max(y0, py1))
                          > 0 for px1, py1, px2, py2 in persons)
            if blocked:                     # someone in front: keep the last readings
                continue
            u = s["unit_cm"] / 100.0
            for i in range(max(1, s["facings"])):
                col = geo["cols"].get((s["id"], i))
                if not col:
                    continue
                a0, a1, b0, b1, f0, npts = col
                m = (A > a0) & (A < a1) & (B > b0) & (B < b1) & (D > f0 - u / 2) & (D < f0 + s["deep"] * u + 0.6)
                front = nearest_surface(D[m] - f0, u, npts)     # metres behind the full front
                # None: nothing solid visible in the tube — neighbours hide it from this angle
                self.depth_back[(s["id"], i)].append(np.nan if front is None else front / u)
                self.depth_slices[(s["id"], i)].append(self.judge_slices(geo, (s["id"], i), s, keepf, Pf))
        self.depth_state = {"on": True, "msg": f"{DEPTH['name'] or 'depth model'} · re-anchored every pass",
                            "noise_cm": round(noise * 100, 1)}
        self.depth_picture(da, mask, slots, want, w, h)

    def depth_picture(self, da, mask, slots, want, w, h):
        """Depth for the dashboard: nearer = brighter red. Counted products get units left per column
        (? = hidden from this angle) and a total badge; the others just their outline."""
        src = want or slots
        fronts = np.concatenate([self.depth_ref[int(s["y"] * h):int((s["y"] + s["h"]) * h),
                                                 int(s["x"] * w):int((s["x"] + s["w"]) * w)].ravel() for s in src]) \
            if src else self.depth_ref[mask]
        fronts = fronts[np.isfinite(fronts)]
        lo = float(np.percentile(fronts, 5)) - 0.05 if fronts.size else 0.5
        reach = (max(s["deep"] * s["unit_cm"] / 100 for s in want) if want else 0.45) + 0.15
        img = depth_colour(da, lo, lo + reach)          # colour range = the shelf's own depth
        counted = {s["id"] for s in want}
        for s in slots:
            left_total, known = 0, True
            for box in self.facing_boxes(s, w, h):
                i, a, b, c2, d2 = box
                if s["id"] not in counted:
                    continue
                back = self.depth_units_back(s, i)
                hidden = self.depth_hidden(s, i)
                if back is None or hidden:
                    known = False
                else:
                    left_total += max(0, s["deep"] - back)
                label = "?" if hidden else ("" if back is None else str(max(0, s["deep"] - back)))
                if label:
                    cv2.putText(img, label, (a + 4, d2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        msg = "nearer = brighter  ·  numbers = units left per column  ·  ? = hidden" if want else \
            "depth picture  ·  set a unit size on a product box to count units behind the front row"
        cv2.putText(img, msg, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.depth_jpg = buf.tobytes()

    def hfov(self):
        cam = self.engine.layout.cam(self.name) if self.engine.layout else None
        return float((cam or {}).get("fov") or CONFIG["shelf"]["depth"]["hfov_deg"])

    def depth_hidden(self, s, i):
        raw = self.depth_back.get((s["id"], i))
        return bool(raw) and not np.isfinite(raw[-1]) and self.depth_state["on"]

    def depth_units_back(self, s, i):
        """Units missing from the front of column i (median of recent passes), or None."""
        raw = self.depth_back.get((s["id"], i))
        if not raw or not np.isfinite(raw[-1]):     # never seen, or hidden on the latest pass
            return None
        hist = [v for v in raw if np.isfinite(v)]
        return int(np.clip(round(float(np.median(hist))), 0, s["deep"]))

    @staticmethod
    def judge_slices(geo, key, s, keepf, Pf):
        """For each unit position behind the front, from the pixels that showed its top or side on the full
        shelf: 'absent' if they now lie clearly further back (the pack is gone), 'present' if unchanged,
        None if there is nothing to judge by (straight-on, thin pack, someone in the way)."""
        u = s["unit_cm"] / 100.0
        out = {}
        for pos, (idx, d0) in (geo.get("ev", {}).get(key) or {}).items():
            v = keepf[idx]
            if int(v.sum()) < 8:
                continue
            dn = (Pf[idx[v]] - geo["c"]) @ geo["n"] - d0[v]
            if np.mean(dn > max(0.5 * u, 0.02)) >= 0.6:
                out[pos] = "absent"
            elif np.mean(np.abs(dn) < max(0.3 * u, 0.012)) >= 0.6:
                out[pos] = "present"
        return out

    def depth_column(self, s, i):
        """(units in column i, whether some of them are assumed) from the front pack's position plus what
        the camera can see of the packs behind it. Packs behind the front are taken to follow it unless the
        surfaces where they should be have moved away. 'Assumed' = at least one position nobody could check."""
        j = self.depth_units_back(s, i)
        if j is None:
            return None
        if j >= s["deep"]:
            return 0, False
        hist = list(self.depth_slices.get((s["id"], i), ()))
        n, assumed = 1, False
        for pos in range(j + 1, s["deep"]):
            votes = [h.get(pos) for h in hist]
            a, pr = votes.count("absent"), votes.count("present")
            if a > pr:
                break
            if pr == 0:
                assumed = True
            n += 1
        return n, assumed

    def ledger_update(self, s, i, j, upper, assumed):
        """What is certain about a column. After a pack is put back at the front, the front surface moves
        forward — but with nothing seen behind it, the column could hold anything from that one pack up to a
        full stack. So: taking packs away lowers the lower bound by as many; the front coming forward raises
        it by one (something was added, no more is proven); a column nobody doubts has lower = upper."""
        key = (s["id"], i)
        led = self.ledger.get(key)
        if not assumed or led is None or upper == 0:
            lo = upper
        elif j > led["j"]:
            lo = led["lo"] - (j - led["j"])
        elif j < led["j"]:
            lo = led["lo"] + 1
        else:
            lo = led["lo"]
        lo = int(max(0, min(lo, upper)))
        self.ledger[key] = {"j": j, "lo": lo}
        return lo

    def refilled(self, slot_id):
        """Staff say a product box was topped up: forget the doubt about it."""
        for k in [k for k in self.ledger if k[0] == slot_id]:
            del self.ledger[k]

    def read_slots(self, frame, slots, persons):
        sc = CONFIG["shelf"]
        h, w = frame.shape[:2]
        now = time.time()
        sig = json.dumps(slots, sort_keys=True)
        if sig != self.slot_sig:
            self.build_slot_ref(slots)
            self.depth_back.clear()
            self.depth_slices.clear()
            self.ledger.clear()
            self.recent_corr.clear()
        if now - self.depth_t >= sc["depth"]["every_s"]:
            self.depth_t = now
            self.depth_pass(frame, slots, persons)
        e = self.edge_map(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        locs = slot_locations(slots)
        out = []
        for s in slots:
            x0, y0 = int(s["x"] * w), int(s["y"] * h)
            x1, y1 = int((s["x"] + s["w"]) * w), int((s["y"] + s["h"]) * h)
            area = max((x1 - x0) * (y1 - y0), 1)
            occluded = any(max(0, min(x1, px2) - max(x0, px1)) * max(0, min(y1, py2) - max(y0, py1)) / area
                           >= sc["occlusion_overlap"] for px1, py1, px2, py2 in persons)
            prev = self.cells_map.get(s["id"])
            if occluded and prev:
                out.append({**prev, "occluded": True})
                continue
            ref = self.slot_ref.get(s["id"]) or []
            present, fills, cols, corrs, lows, doubts = 0, [], [], [], [], []
            nf = max(1, s["facings"])
            backs = [self.depth_units_back(s, i) for i in range(nf)] \
                if s.get("unit_cm", 0) > 0 and self.depth_state["on"] else [None] * nf
            for i, a, b, c2, d2 in self.facing_boxes(s, w, h):
                cur = float(e[b:d2, a:c2].mean()) if e[b:d2, a:c2].size else 0.0
                r = ref[i] if i < len(ref) else 0.0
                f = min(1.0, cur / r) if r > 1e-4 else (1.0 if cur > 0.02 else 0.0)
                fills.append(f)
                hl = self.slot_hist.get(s["id"]) or []
                refh = hl[i] if i < len(hl) else None
                if f >= sc["slot_low"] and refh is not None:     # only facings that hold something
                    corrs.append(float(cv2.compareHist(refh, hs_hist(hsv[b:d2, a:c2]), cv2.HISTCMP_CORREL)))
                if backs[i] is not None:     # depth: how many are gone from the front of this column
                    left, assumed = self.depth_column(s, i)
                    lo = self.ledger_update(s, i, backs[i], left, assumed)
                else:                        # not measured by depth: front-row rule
                    left = s["deep"] if f >= sc["slot_low"] else 0
                    if self.depth_hidden(s, i):
                        # nothing solid in the column's tube: its front unit is certainly gone,
                        # but neighbours hide how far back the rest goes from this angle
                        left = min(left, s["deep"] - 1)
                    assumed, lo = s["deep"] > 1 and left > 0, left
                cols.append(left)
                lows.append(lo)
                doubts.append(bool(assumed))
                if left > 0:
                    present += 1
            n = max(1, s["facings"])
            est = int(sum(cols))
            nd = sum(v is not None for v in backs)
            method = "depth" if nd == nf else ("mixed" if nd else "front")
            full = n * s["deep"]
            fill = float(np.mean(fills)) if fills else 0.0
            self.recent[s["id"]].append(fill)
            fill_s = float(np.median(self.recent[s["id"]]))
            self.history[s["id"]].append((now, fill_s))
            est_min = int(sum(lows))
            # stock status goes by what is certain: a pack put back at the front doesn't make a shelf look full
            status = "EMPTY" if est == 0 else ("LOW" if est_min / full <= 0.5 else "OK")
            # planogram: the facings that hold something should look like the product calibrated there
            if corrs:
                self.recent_corr[s["id"]].append(float(np.median(corrs)))
            match = float(np.median(self.recent_corr[s["id"]])) if self.recent_corr[s["id"]] else None
            misplaced = bool(est and match is not None and len(self.recent_corr[s["id"]]) >= 2
                             and match < sc["misplace_corr"])
            p = self.engine.store.product_get(s.get("sku", "")) if s.get("sku") else None
            r_, c_ = locs.get(s["id"], (0, 0))
            cell = {"slot": s["id"], "name": p["name"] if p else s["name"], "sku": s.get("sku", ""),
                    "brand": p["brand"] if p else "", "price": p["price"] if p else None,
                    "row": r_, "col": c_, "loc": f"row {r_} · col {c_}",
                    "facings": n, "deep": s["deep"], "present": present,
                    # "depth": each column counted by how far back its front unit sits (depth model)
                    # "front": facings still visible × stated depth — cannot see behind row 1
                    "est_units": est, "method": method,
                    # est_units counts packs behind the front as following it; est_min is what is certain,
                    # and exact says the two agree because every position could be checked
                    "est_min": est_min, "exact": not any(doubts), "assumed": [i for i, d in enumerate(doubts) if d],
                    "columns": cols if method != "front" else None,
                    "hidden": [i for i in range(nf) if self.depth_hidden(s, i)],
                    "full_units": full, "fill": round(fill_s, 2),
                    "status": status, "occluded": False,
                    "misplaced": misplaced, "match": None if match is None else round(match, 2),
                    "attention": self.engine.attention_today(self.name, s["id"]),
                    "eta_min": self.eta(s["id"], now),
                    "pos_units": self.engine.pos_units(s.get("sku", ""), n * s["deep"])}
            out.append(cell)
            self.cells_map[s["id"]] = cell
        return out

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

    def eta(self, key, now):
        sc = CONFIG["shelf"]
        pts = [(t, f) for t, f in self.history[key] if t >= now - sc["eta_window_min"] * 60]
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
        dt, self.t_heat = min(now - self.t_heat, 0.5), now
        for x1, y1, x2, y2 in persons:               # someone browsing is on the floor map as well
            mp = self.store_point((x1 + x2) / 2, y2, w, h)
            if mp:
                self.engine.add_store_heat(mp[0], mp[1], dt)
        self.track_attention(frame, persons, now)

        if self.calib_request:
            self.calib_request = False
            if persons:
                self.calib_msg = "Aisle not clear — ask people to step out and retry"
            else:
                cv2.imwrite(self.ref_path, frame)
                self.ref, self.ref_feats = frame.copy(), self.features(frame)
                self.recent.clear()
                self.history.clear()
                self.recent_corr.clear()
                self.cells_map = {}
                self.slot_sig = None                 # remeasure every facing against the new picture
                self.depth_ref, self.depth_t, self.depth_geo = None, 0.0, None   # re-derived from the new picture
                self.t_read, self.ov = 0.0, None     # read the shelf right away
                self.depth_back.clear()
                self.depth_slices.clear()
                self.ledger.clear()
                sl = self.slots()
                if sl:
                    self.build_slot_ref(sl)
                    self.engine.restock([s for s in sl if s.get("sku")])
                self.calib_msg = "Calibrated"

        self.cctv(frame, persons)
        vis = frame.copy()
        self.blur(vis, persons)
        if now - self.t_read < sc["period_s"] and self.ov is not None and self.ov[0].shape == vis.shape \
                and self.ref is not None and self.ref.shape == frame.shape:
            vis[self.ov[1]] = self.ov[0][self.ov[1]]     # between stock reads: last overlay, fresh picture
            return vis
        self.t_read = now
        if self.ref is None or self.ref.shape != frame.shape:
            cv2.putText(vis, "NEEDS CALIBRATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.engine.on_shelf(self.name, None)
            return vis

        marked = self.slots()
        if marked:                                   # product slots beat the uniform grid
            cells = self.read_slots(frame, marked, persons)
            self.engine.depth_info[self.name] = dict(self.depth_state)
            self.engine.on_shelf(self.name, cells)
            by = {c["slot"]: c for c in cells}
            for s in marked:
                c = by.get(s["id"])
                if not c:
                    continue
                x0, y0 = int(s["x"] * w), int(s["y"] * h)
                x1, y1 = int((s["x"] + s["w"]) * w), int((s["y"] + s["h"]) * h)
                col = {"OK": (80, 200, 80), "LOW": (0, 180, 240), "EMPTY": (60, 60, 230)}[c["status"]]
                ov = vis.copy()
                cv2.rectangle(ov, (x0, y0), (x1, y1), col, -1)
                cv2.addWeighted(ov, 0.10 if c["occluded"] else 0.22, vis,
                                1 - (0.10 if c["occluded"] else 0.22), 0, vis)
                cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
                for _, a, b2, c3, d3 in self.facing_boxes(s, w, h):
                    cv2.line(vis, (a, b2), (a, d3), col, 1)
                tag = "~" if c["method"] == "front" or not c.get("exact", True) else ""
                cv2.putText(vis, c["name"][:28], (x0 + 2, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (240, 240, 240), 1, cv2.LINE_AA)
                lo = c.get("est_min", c["est_units"])
                count_badge(vis, x1, y0, f"{tag}{lo if lo == c['est_units'] else str(lo) + '-' + str(c['est_units'])}/{c['full_units']}", col)
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
                        "eta_min": self.eta((r, c), now), "expected": expected, "found": found, "occluded": False}
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


class CheckoutWorker(CamWorker):
    """Self-checkout camera: reads product barcodes held up to it and adds them to the open cart.
    A USB barcode scanner works too — it types into the checkout screen like a keyboard."""

    def __init__(self, name, roles, source, engine):
        super().__init__(name, roles, source, engine)
        from ultralytics import YOLO
        self.model = YOLO(CONFIG["person_model"])
        self.det = cv2.barcode.BarcodeDetector()
        self.seen = {}          # code -> last time it was in view
        self.persons, self.n, self.persons_t = [], 0, 0.0

    def decode(self, frame):
        """OpenCV's detector only locks on over a narrow band of barcode sizes, and a code held up
        to a till camera can be anywhere from tiny to filling the frame. Trying a few image
        scales (with the light blur every real lens adds anyway) reads it across that range."""
        for s in (1.0, 0.6, 0.4, 1.5):
            im = frame if s == 1.0 else cv2.resize(frame, None, fx=s, fy=s,
                                                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            im = cv2.GaussianBlur(im, (3, 3), 0)
            ok, infos, pts, _ = self.det.detectAndDecodeMulti(im)   # (ok, codes, corners, straight)
            found = [(c, None if pts is None or i >= len(pts) else pts[i] / s)
                     for i, c in enumerate(infos or ()) if c]
            if found:
                return found
        return []

    def step(self, frame):
        self.n += 1
        if self.n % 3 == 1:                      # people move slowly; barcodes need every frame
            pr = self.model(frame, classes=[0], conf=CONFIG["conf"], imgsz=416, verbose=False)[0]
            self.persons = pr.boxes.xyxy.cpu().numpy().tolist() if pr.boxes is not None else []
            self.persons_t = time.time()
            if self.persons:     # whoever stands closest (largest box) is the one at the till
                near = max(self.persons, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                self.engine.till_seen(self.name, body_signature(frame, near))
        self.cctv(frame, self.persons)
        vis = frame.copy()
        self.blur(vis, self.persons, self.persons_t)
        now = time.time()
        for code, pts in self.decode(frame):
            if pts is not None:
                cv2.polylines(vis, [np.int32(pts)], True, (80, 200, 80), 3)
            held = now - self.seen.get(code, 0) < CONFIG["pos"]["rescan_s"]
            self.seen[code] = now                # an item kept in view is counted once
            if not held:
                self.engine.scan(code, self.name)
        ls = self.engine.last_scan
        if ls and ls.get("source") == self.name and now - ls["ts"] < 3:   # only what this camera read
            col = (80, 200, 80) if ls["ok"] else (60, 60, 230)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (20, 20, 20), -1)
            cv2.putText(vis, ls["msg"], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
        return vis


# ─────────────────────────────── BACKGROUND JOBS ───────────────────────────────
TS_FIELDS = ["ts", "date", "hour", "minute", "weekday", "inside", "entries", "exits", "queue_len",
             "queue_wait_min", "counters_open", "empty_slots", "low_slots", "active_alerts",
             "bills", "revenue", "items_sold"]


def minute_row(engine, store, now=None):
    """One row per minute: what a recurrent model needs to learn the store's rhythm."""
    now = now or time.time()
    s = engine.snapshot()
    since = now - 60
    ent = ext = 0
    for (d,) in store.q("SELECT data FROM events WHERE type='entry' AND ts>=? AND ts<?", since, now):
        if json.loads(d)["dir"] == "in":
            ent += 1
        else:
            ext += 1
    bills = rev = items = 0
    for (d,) in store.q("SELECT data FROM events WHERE type='pos' AND ts>=? AND ts<?", since, now):
        d = json.loads(d)
        bills += 1
        rev += float(d.get("total", 0))
        items += int(d.get("n_items", sum(i.get("qty", 1) for i in d.get("items", []))))
    cells = [c for cs in s["shelves"].values() if cs for c in cs]
    t = datetime.fromtimestamp(now)
    return {"ts": t.strftime("%Y-%m-%d %H:%M"), "date": t.strftime("%Y-%m-%d"), "hour": t.hour,
            "minute": t.minute, "weekday": t.weekday(), "inside": s["footfall"]["inside"],
            "entries": ent, "exits": ext,
            "queue_len": sum(q["length"] for q in s["queues"].values()),
            "queue_wait_min": max([q["wait_min"] for q in s["queues"].values()] or [0]),
            "counters_open": s["open_counters"],
            "empty_slots": sum(c["status"] == "EMPTY" for c in cells),
            "low_slots": sum(c["status"] == "LOW" for c in cells),
            "active_alerts": len(s["alerts"]), "bills": bills, "revenue": round(rev, 2), "items_sold": items}


def append_csv(row):
    os.makedirs(CONFIG["analytics_dir"], exist_ok=True)
    path = os.path.join(CONFIG["analytics_dir"], "timeseries.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = __import__("csv").DictWriter(fh, fieldnames=TS_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def ticker(engine, store):
    while True:
        time.sleep(60)
        try:
            row = minute_row(engine, store)
            store.metrics(time.time(), {k: v for k, v in row.items() if isinstance(v, (int, float))})
            append_csv(row)
            engine.refresh_daily()
            if CONFIG["cloud_url"]:
                store.outbox_add({"store_id": CONFIG["store_id"], "ts": time.time(), "metrics": row,
                                  "alerts": engine.snapshot()["alerts"][:10]})
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


def webhook_sender(engine, store):
    """Deliver queued integration events; failures stay queued and are retried (offline-first)."""
    while True:
        time.sleep(3)
        if not CONFIG.get("webhooks"):
            continue
        for hid, url, payload, tries in store.hook_pending():
            try:
                req = urllib.request.Request(url, payload.encode(), {"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=8)
                store.hook_done(hid, True)
                engine.hook_state.update(sent=engine.hook_state["sent"] + 1, last_ok=time.time())
            except Exception as e:
                store.hook_done(hid, False)
                engine.hook_state.update(failed=engine.hook_state["failed"] + 1,
                                         last_error=f"{url}: {type(e).__name__}: {str(e)[:80]}")
                time.sleep(min(30, 2 + tries))      # back off while that system is unreachable
                break


# ─────────────────────────────── ANALYTICS ───────────────────────────────
def _local(ts):
    return datetime.fromtimestamp(ts)


def analytics(store, days=14):
    """Everything the Analytics tab draws, computed from the local database."""
    days = max(1, min(int(days), 90))
    since = day_start() - (days - 1) * 86400
    dates = [(_local(since + i * 86400)).strftime("%Y-%m-%d") for i in range(days)]
    daily = {d: {"date": d, "entries": 0, "bills": 0, "revenue": 0.0, "items": 0} for d in dates}
    hour_tot = [0] * 24
    wk_hour = [[0] * 24 for _ in range(7)]
    wk_days = [set() for _ in range(7)]
    today = datetime.now().strftime("%Y-%m-%d")
    today_hourly = [0] * 24
    for ts, d in store.q("SELECT ts,data FROM events WHERE type='entry' AND ts>=?", since):
        if json.loads(d)["dir"] != "in":
            continue
        t = _local(ts)
        k = t.strftime("%Y-%m-%d")
        if k in daily:
            daily[k]["entries"] += 1
        hour_tot[t.hour] += 1
        wk_hour[t.weekday()][t.hour] += 1
        wk_days[t.weekday()].add(k)
        if k == today:
            today_hourly[t.hour] += 1
    active_days = max(1, sum(1 for d in daily.values() if d["entries"]))
    wk_avg = [[round(v / max(1, len(wk_days[w])), 1) for v in wk_hour[w]] for w in range(7)]

    units, revenue_by, names = Counter(), Counter(), {}
    rev_hour = [0.0] * 24
    basket = Counter()
    bill_vals = []
    for bid, ts, items, mrp, tot, n in store.q(
            "SELECT id,ts,items,mrp_total,total,n_items FROM bills WHERE ts>=?", since):
        k = _local(ts).strftime("%Y-%m-%d")
        if k in daily:
            daily[k]["bills"] += 1
            daily[k]["revenue"] += tot
            daily[k]["items"] += n
        rev_hour[_local(ts).hour] += tot
        bill_vals.append(tot)
        b = n if n <= 5 else ("6-8" if n <= 8 else "9+")
        basket[str(b)] += 1
        for ln in json.loads(items):
            units[ln["sku"]] += ln["qty"]
            revenue_by[ln["sku"]] += ln["amount"]
            names[ln["sku"]] = ln.get("name", ln["sku"])
    for d in daily.values():
        d["revenue"] = round(d["revenue"], 2)
        d["conversion"] = round(d["bills"] / d["entries"], 3) if d["entries"] else None

    waits = [v for (v,) in store.q("SELECT value FROM metrics WHERE key='queue_wait_min' AND ts>=?", since)]
    edges = [0, 1, 2, 3, 4, 6, 8]
    wait_hist = [{"bin": (f"{a}-{b}m" if b else f"{a}m+"),
                  "n": sum(1 for w in waits if w >= a and (b is None or w < b))}
                 for a, b in zip(edges, edges[1:] + [None])]
    occ = defaultdict(list)
    for ts, v in store.q("SELECT ts,value FROM metrics WHERE key='inside' AND ts>=?", day_start() - 6 * 86400):
        t = _local(ts)
        occ[(t.weekday(), t.hour)].append(v)
    occupancy = [[round(float(np.mean(occ[(w, h)])), 1) if occ[(w, h)] else 0 for h in range(24)] for w in range(7)]

    stock = Counter()
    for (d,) in store.q("SELECT data FROM events WHERE type='alert' AND ts>=?", since):
        d = json.loads(d)
        if d.get("kind") == "stock" and d.get("severity") == "critical":
            stock[d["message"].split(" — ")[0].replace(" is empty", "")] += 1
    look = defaultdict(list)                   # shopper stops in front of each product
    for (d,) in store.q("SELECT data FROM events WHERE type='product_dwell' AND ts>=?", since):
        d = json.loads(d)
        look[d.get("name") or d.get("sku") or "?"].append(float(d["s"]))
    zones = defaultdict(list)
    for (d,) in store.q("SELECT data FROM events WHERE type='zone_visit' AND ts>=?", since):
        d = json.loads(d)
        zones[d["zone"]].append(d["duration_s"])

    tot_e = sum(d["entries"] for d in daily.values())
    tot_b = sum(d["bills"] for d in daily.values())
    tot_r = sum(d["revenue"] for d in daily.values())
    peak = max(range(24), key=lambda h: hour_tot[h]) if tot_e else None
    top = sorted(units, key=lambda k: -revenue_by[k])[:10]
    return {
        "days": days, "from": dates[0], "to": dates[-1], "has_demo": store.has_demo(),
        "kpis": {"footfall": tot_e, "bills": tot_b, "revenue": round(tot_r, 2),
                 "avg_basket": round(tot_r / tot_b, 2) if tot_b else None,
                 "items": sum(d["items"] for d in daily.values()),
                 "conversion": round(tot_b / tot_e, 3) if tot_e else None,
                 "avg_wait_min": round(float(np.mean(waits)), 2) if waits else None,
                 "peak_hour": f"{peak:02d}:00" if peak is not None else None,
                 "stockouts": sum(stock.values()), "footfall_per_day": round(tot_e / active_days, 1)},
        "daily": list(daily.values()),
        "hourly_avg": [round(v / active_days, 2) for v in hour_tot],
        "today_hourly": today_hourly,
        "weekday_hour": wk_avg,
        "occupancy_weekday_hour": occupancy,
        "revenue_by_hour": [round(v / active_days, 2) for v in rev_hour],
        "wait_hist": wait_hist,
        "basket_hist": [{"bin": k, "n": basket.get(k, 0)} for k in ("1", "2", "3", "4", "5", "6-8", "9+")],
        "top_products": [{"sku": k, "name": names[k], "units": units[k], "revenue": round(revenue_by[k], 2)}
                         for k in top],
        "stockouts": [{"name": k, "count": v} for k, v in stock.most_common(10)],
        "attention": sorted([{"name": k, "stops": len(v), "avg_s": round(float(np.mean(v)), 1),
                              "total_min": round(sum(v) / 60, 1)} for k, v in look.items()],
                            key=lambda x: -x["stops"])[:12],
        "zones": [{"zone": z, "visits": len(v), "avg_dwell_s": round(float(np.mean(v)), 1)}
                  for z, v in sorted(zones.items())],
    }


def queue_analytics(store, days=14):
    """What the Queue tab draws besides the live numbers: today minute by minute (real trading only), how
    busy each weekday/hour usually is, crowd episodes, and the day's headline figures."""
    t0 = day_start()
    since = t0 - (max(1, min(int(days), 90)) - 1) * 86400
    rows = defaultdict(dict)
    for ts, k, v in store.q("SELECT ts,key,value FROM metrics WHERE ts>=? AND demo=0 AND key IN "
                            "('queue_len','queue_wait_min','counters_open') ORDER BY ts", t0):
        rows[round(ts / 60) * 60][k] = v
    today = [{"t": t, "len": d.get("queue_len"), "wait": d.get("queue_wait_min"), "counters": d.get("counters_open")}
             for t, d in sorted(rows.items())]
    by = defaultdict(list)
    for ts, v in store.q("SELECT ts,value FROM metrics WHERE key='queue_len' AND ts>=?", since):
        t = _local(ts)
        by[(t.weekday(), t.hour)].append(v)
    busy = [[round(float(np.mean(by[(w, h)])), 2) if by[(w, h)] else 0 for h in range(24)] for w in range(7)]
    served = [json.loads(d)["duration_s"] for (d,) in store.q("SELECT data FROM events WHERE type='served' AND ts>=? AND demo=0", t0)]
    waits = [r["wait"] for r in today if r["wait"] is not None]
    lens = [r["len"] for r in today if r["len"] is not None]
    crowds = [{"ts": ts, **json.loads(d)} for ts, d in
              store.q("SELECT ts,data FROM events WHERE type='crowd' AND ts>=? ORDER BY ts DESC LIMIT 20", t0)]
    peak_h = None
    hr = defaultdict(list)
    for r in today:
        if r["len"] is not None:
            hr[_local(r["t"]).hour].append(r["len"])
    if hr:
        peak_h = max(hr, key=lambda h: np.mean(hr[h]))
    return {"today": today, "weekday_hour": busy, "has_demo": store.has_demo(), "crowds": crowds,
            "stats": {"served": len(served), "avg_in_queue_min": round(float(np.mean(served)) / 60, 2) if served else None,
                      "peak_len": max(lens) if lens else 0, "peak_wait_min": max(waits) if waits else 0,
                      "avg_wait_min": round(float(np.mean(waits)), 2) if waits else None,
                      "busiest_hour": f"{peak_h:02d}:00" if peak_h is not None else None}}


def hourly_timeseries(store, days=30):
    """One row per hour, footfall + sales + queue: the training table for a forecasting model."""
    since = day_start() - (max(1, int(days)) - 1) * 86400
    rows = {}

    def row(ts):
        t = _local(ts).replace(minute=0, second=0, microsecond=0)
        k = t.strftime("%Y-%m-%d %H:00")
        if k not in rows:
            rows[k] = {"hour_start": k, "date": t.strftime("%Y-%m-%d"), "hour": t.hour, "weekday": t.weekday(),
                       "entries": 0, "exits": 0, "bills": 0, "revenue": 0.0, "items_sold": 0,
                       "avg_inside": None, "avg_queue_wait_min": None, "stock_alerts": 0, "demo": 0}
        return rows[k]
    for ts, d, demo in store.q("SELECT ts,data,demo FROM events WHERE type='entry' AND ts>=?", since):
        r = row(ts)
        r["entries" if json.loads(d)["dir"] == "in" else "exits"] += 1
        r["demo"] = max(r["demo"], demo or 0)
    for ts, tot, n, demo in store.q("SELECT ts,total,n_items,demo FROM bills WHERE ts>=?", since):
        r = row(ts)
        r["bills"] += 1
        r["revenue"] = round(r["revenue"] + tot, 2)
        r["items_sold"] += n
        r["demo"] = max(r["demo"], demo or 0)
    for key, col in (("inside", "avg_inside"), ("queue_wait_min", "avg_queue_wait_min")):
        acc = defaultdict(list)
        for ts, v in store.q("SELECT ts,value FROM metrics WHERE key=? AND ts>=?", key, since):
            acc[row(ts)["hour_start"]].append(v)
        for k, vs in acc.items():
            rows[k][col] = round(float(np.mean(vs)), 2)
    for ts, d in store.q("SELECT ts,data FROM events WHERE type='alert' AND ts>=?", since):
        if json.loads(d).get("kind") == "stock":
            row(ts)["stock_alerts"] += 1
    return [rows[k] for k in sorted(rows)]


def to_csv(rows):
    if not rows:
        return ""
    buf = __import__("io").StringIO()
    w = __import__("csv").DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


DEMO_CATALOG = [   # sku, name, brand, mrp, price, how often it sells, shelf capacity
    ("maggi-70", "Maggi Masala Noodles 70g", "Nestle", 15, 14, 9, 36),
    ("parleg-250", "Parle-G Biscuits 250g", "Parle", 25, 25, 8, 30),
    ("amul-butter-100", "Amul Butter 100g", "Amul", 58, 56, 5, 20),
    ("tata-salt-1kg", "Tata Salt 1kg", "Tata", 28, 27, 4, 24),
    ("aashirvaad-5kg", "Aashirvaad Atta 5kg", "ITC", 285, 259, 3, 10),
    ("fortune-oil-1l", "Fortune Sunflower Oil 1L", "Adani Wilmar", 165, 149, 3, 12),
    ("dove-sh-180", "Dove Shampoo 180ml", "HUL", 199, 169, 2, 12),
    ("colgate-200", "Colgate Toothpaste 200g", "Colgate", 125, 112, 3, 18),
    ("surf-1kg", "Surf Excel Detergent 1kg", "HUL", 155, 145, 2, 10),
    ("lays-52", "Lay's Classic Salted 52g", "PepsiCo", 20, 20, 7, 30),
    ("coke-750", "Coca-Cola 750ml", "Coca-Cola", 40, 38, 5, 24),
    ("britannia-bread", "Britannia Bread 400g", "Britannia", 45, 45, 5, 12),
    ("dettol-soap", "Dettol Soap 125g", "Reckitt", 52, 48, 3, 20),
    ("red-label-500", "Red Label Tea 500g", "HUL", 290, 265, 2, 10),
    ("kurkure-90", "Kurkure Masala Munch 90g", "PepsiCo", 20, 20, 6, 30),
]


def seed_demo(store, days=14, seed=7):
    """Generate plausible past trading so the Analytics tab has something to show in a demo.
    Every row is tagged demo=1: the UI says so, and --clear-demo removes exactly these rows."""
    rng = np.random.default_rng(seed)
    store.clear_demo()
    for sku, name, brand, mrp, price, _, _ in DEMO_CATALOG:
        if not store.product_get(sku):
            store.product_upsert({"sku": sku, "name": name, "brand": brand, "mrp": mrp, "price": price})
    pop = np.array([c[5] for c in DEMO_CATALOG], float)
    pop /= pop.sum()
    # shelf holds about a day's sales plus a margin that varies by product, so fast movers run
    # out on busy days and slow ones rarely do — the pattern a real stock-out chart shows
    cap = {c[0]: int(390 * p * rng.uniform(1.0, 1.5)) + 4 for c, p in zip(DEMO_CATALOG, pop)}
    g = lambda h, m, s: np.exp(-((h - m) ** 2) / (2 * s * s))
    ev, mt, bills = [], [], []
    zones = list(CONFIG["geometry"]["default"]["zones"])
    for dback in range(days, 0, -1):
        day0 = day_start() - dback * 86400
        wd = _local(day0).weekday()
        f = (1.35 if wd >= 5 else 1.0) * rng.normal(1.0, 0.08)
        sold = Counter()
        visits = []
        for h in range(8, 22):
            lam = f * (5 + 9 * g(h, 11, 1.4) + 7 * g(h, 13.5, 0.9) + 16 * g(h, 19.2, 1.5))
            for _ in range(rng.poisson(lam)):
                t_in = day0 + h * 3600 + rng.uniform(0, 3600)
                dwell = float(np.clip(rng.lognormal(np.log(11 * 60), 0.45), 120, 45 * 60))
                visits.append((t_in, t_in + dwell))
        visits.sort()
        billno, first_bill = 0, len(bills)
        for t_in, t_out in visits:
            ev.append((t_in, "entry", "entry", {"dir": "in"}))
            ev.append((t_out, "entry", "entry", {"dir": "out"}))
            for z in rng.choice(zones, size=rng.integers(1, 3), replace=False):
                ev.append((t_in + 60, "entry", "zone_visit",
                           {"zone": str(z), "duration_s": round(float(rng.uniform(20, 240)), 1)}))
            for i in rng.choice(len(DEMO_CATALOG), size=rng.integers(1, 4), p=pop, replace=False):   # stops at shelves
                ev.append((t_in + 90, "shelfA", "product_dwell", {"sku": DEMO_CATALOG[i][0], "name": DEMO_CATALOG[i][1],
                                                                  "s": round(float(rng.gamma(2.0, 4.5) + 1.5), 1)}))
            if rng.random() < (0.66 if wd >= 5 else 0.61):
                n = int(min(12, 1 + rng.poisson(2.2)))
                pick = rng.choice(len(DEMO_CATALOG), size=n, p=pop)
                lines = {}
                for i in pick:
                    sku, name, brand, mrp, price, _, _ = DEMO_CATALOG[i]
                    q = 2 if rng.random() < 0.15 else 1
                    ln = lines.setdefault(sku, {"sku": sku, "name": name, "brand": brand, "qty": 0,
                                                "mrp": mrp, "price": price, "amount": 0.0})
                    ln["qty"] += q
                    ln["amount"] = round(ln["qty"] * price, 2)
                    before = sold[sku]
                    sold[sku] += q
                    if before < cap[sku] <= sold[sku]:
                        ev.append((t_out, "shelf", "alert",
                                   {"kind": "stock", "severity": "critical", "action": "Critical refill",
                                    "message": f"{name} — shelf, is empty"}))
                billno += 1
                ls = list(lines.values())
                bills.append({"id": f"SS-{_local(day0).strftime('%Y%m%d')}-{billno:04d}", "ts": t_out - 45,
                              "lines": ls, "n_items": sum(l["qty"] for l in ls),
                              "mrp_total": round(sum(l["mrp"] * l["qty"] for l in ls), 2),
                              "total": round(sum(l["amount"] for l in ls), 2)})
                ev.append((t_out - 45, "pos", "pos", {"bill": bills[-1]["id"], "total": bills[-1]["total"],
                                                      "n_items": bills[-1]["n_items"],
                                                      "items": [{"sku": l["sku"], "qty": l["qty"]} for l in ls]}))
        today_bills = [b["ts"] for b in bills[first_bill:]]
        for step in range(8 * 12, 22 * 12):                  # every 5 minutes while open
            t = day0 + step * 300
            inside = sum(1 for a, b in visits if a <= t < b)
            arriving = sum(1 for bt in today_bills if t <= bt < t + 300)
            ql = max(0, int(round(arriving * rng.uniform(0.9, 1.7))))     # one counter, ~1.2 served/min
            mt += [(t, "inside", inside), (t, "queue_len", ql), (t, "queue_wait_min", round(ql / 1.2, 2))]
    store.events_at([(t, c, ty, d) for t, c, ty, d in ev], demo=1)
    store.metrics_at(mt, demo=1)
    for b in bills:
        store.bill_save(b, demo=1)
    return {"days": days, "visits": sum(1 for e in ev if e[2] == "entry" and e[3]["dir"] == "in"),
            "bills": len(bills)}


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

    for fx in lay.data.get("fixtures", []):          # doors as a floor marker, counters/fixtures as boxes
        c = rect_corners(fx)
        if fx["kind"] == "door":
            ax.add_collection3d(Poly3DCollection([[(p[0], p[1], 0.01) for p in c]], facecolors="#9be3a8",
                                                 edgecolors="#2f8f46", linewidths=0.8))
            ax.text(fx["x"], fx["y"], 0.25, fx["name"], ha="center", fontsize=7.5, color="#2f8f46")
            continue
        z = fx.get("height", 1.0)
        faces = [[(p[0], p[1], z) for p in c]] + [[(c[i][0], c[i][1], 0), (c[(i + 1) % 4][0], c[(i + 1) % 4][1], 0),
                                                   (c[(i + 1) % 4][0], c[(i + 1) % 4][1], z), (c[i][0], c[i][1], z)]
                                                  for i in range(4)]
        col = "#b7c8e6" if fx["kind"] == "counter" else "#d8d2c4"
        ax.add_collection3d(Poly3DCollection(faces, facecolors=col, edgecolors="#31405a", linewidths=0.6))
        ax.text(fx["x"], fx["y"], z + 0.18, fx["name"], ha="center", fontsize=7.5, color="#1a2a4a")

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
:root{--bg:#0b0c0e;--card:#141619;--card2:#1b1e23;--line:#282c33;--line2:#1f2329;
 --text:#e9ebef;--mute:#9298a3;--acc:#e4002b;--acc2:#ff3b52;--accsoft:#2a1114;--accline:#5c1a22;
 --ok:#3fb950;--low:#d9a441;--bad:#ff4d4f;--mis:#a371f7}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);
 display:flex;align-items:center;gap:14px;padding:0 20px;height:54px}
header b{font-size:16px;letter-spacing:-.01em}
.tabs{display:flex;gap:2px;margin-left:8px}
.tab{padding:6px 14px;border-radius:7px;cursor:pointer;color:var(--mute);font-weight:550;user-select:none}
.tab:hover{background:var(--bg)}.tab.on{background:var(--accsoft);color:var(--acc2)}
.right{margin-left:auto;display:flex;align-items:center;gap:8px}
.pill{font-size:12px;color:var(--mute);background:var(--bg);border:1px solid var(--line);
 border-radius:99px;padding:3px 10px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--ok);display:inline-block;margin-right:5px}
main{padding:18px 20px 40px;display:grid;gap:14px;grid-template-columns:repeat(12,1fr);
 max-width:1560px;align-items:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0}
.s4{grid-column:span 4}.s6{grid-column:span 6}.s7{grid-column:span 7}.s5{grid-column:span 5}.s12{grid-column:1/-1}
@media(max-width:1100px){.s4,.s5,.s6,.s7{grid-column:1/-1}}
h3{margin:0 0 12px;font-size:13px;font-weight:650;letter-spacing:.01em;display:flex;
 justify-content:space-between;align-items:center;gap:10px}
h3 .sub{font-weight:450;color:var(--mute);font-size:12px}
.kpis{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(122px,1fr));gap:10px}
.kpi{background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:11px 13px}
.kpi .v{font-size:21px;font-weight:660;letter-spacing:-.02em}
.kpi .l{color:var(--mute);font-size:11.5px;margin-top:1px}
button{background:var(--card2);color:var(--text);border:1px solid var(--line);border-radius:8px;
 padding:6px 11px;font:inherit;font-size:13px;cursor:pointer}
button:hover{background:#22262c;border-color:#3a3f47}
button.pri{background:var(--acc);border-color:var(--acc);color:#fff}
button.pri:hover{background:var(--acc2)}
button.on{background:var(--accsoft);border-color:var(--accline);color:var(--acc2)}
button.danger:hover{border-color:var(--bad);color:var(--bad)}
input[type=number],input[type=text]{background:#0f1114;border:1px solid var(--line);border-radius:7px;
 padding:5px 8px;font:inherit;font-size:13px;width:96px;color:var(--text)}
input:focus{outline:2px solid var(--accsoft);border-color:var(--acc);background:#121418}
.alert{display:flex;gap:11px;align-items:center;padding:10px 12px;border:1px solid var(--line);
 border-left:3px solid;border-radius:9px;margin-bottom:7px;background:var(--card2)}
.critical{border-left-color:var(--bad)}.warning{border-left-color:var(--low)}.info{border-left-color:var(--acc)}
.alert .m{flex:1;min-width:0}.alert .a{font-weight:600}
.alert small,.muted{color:var(--mute);font-size:12px}
table{width:100%;border-collapse:collapse}
td,th{padding:6px 4px;border-bottom:1px solid var(--line2);text-align:left;font-size:13px}
th{color:var(--mute);font-weight:500;font-size:12px}
.grid{display:grid;gap:4px}
.cell{border-radius:6px;padding:9px 2px;text-align:center;font-size:11px;font-weight:600;color:#fff}
.OK{background:var(--ok)}.LOW{background:var(--low)}.EMPTY{background:var(--bad)}.MISPLACED{background:var(--mis)}
.occ{opacity:.35}
.feeds{display:flex;gap:12px;flex-wrap:wrap}.feeds figure{margin:0}
.feeds img{width:330px;max-width:100%;border-radius:9px;border:1px solid var(--line);display:block}
figcaption{color:var(--mute);font-size:12px;margin-top:5px}
.bars{display:flex;gap:5px;align-items:flex-end;height:96px;padding-top:15px}
.bar{flex:0 0 30px;background:#4a2a30;border-radius:4px 4px 0 0;position:relative;min-height:2px}
.bar:hover{background:var(--acc)}
.bar i{position:absolute;bottom:-17px;left:0;right:0;text-align:center;font-size:10px;color:var(--mute);font-style:normal}
.bar b{position:absolute;top:-15px;left:0;right:0;text-align:center;font-size:10px;font-weight:500}
.fc{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;text-align:center;margin:4px 0 10px}
.fc div{background:var(--card2);border:1px solid var(--line);border-radius:9px;padding:9px 4px}
.fc b{font-size:17px}.fc small{color:var(--mute);display:block;margin-top:2px}
/* ---- setup ---- */
.bar2{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
.sep{width:1px;height:22px;background:var(--line)}
.wrap{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
canvas.plan{background:#0f1114;border:1px solid var(--line);border-radius:10px;
 flex:1 1 440px;width:100%;max-width:820px;height:auto;touch-action:none}
.side{flex:0 0 268px;min-width:248px}
.side h4{margin:0 0 8px;font-size:12px;color:var(--mute);text-transform:uppercase;letter-spacing:.05em}
.side .box{border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px;background:var(--card2)}
.f{display:flex;justify-content:space-between;align-items:center;gap:10px;margin:7px 0}
.f span{color:var(--mute);font-size:12.5px}
.f input[type=number]{width:84px}.f input[type=text]{width:150px}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{border:1px solid var(--line);border-radius:7px;padding:4px 9px;font-size:12.5px;cursor:pointer;user-select:none}
.chip.on{background:var(--accsoft);border-color:var(--accline);color:var(--acc2);font-weight:600}
.shop{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:var(--card2);border:1px solid var(--line);
 border-radius:10px;padding:10px 12px;margin-bottom:12px}
.shop .who{flex:1 1 180px;min-width:0}.shop .lbl{display:block;color:var(--mute);font-size:11.5px}
.shop .sid{font-size:17px;letter-spacing:.02em}.shop .sid small{font-size:11.5px;color:var(--mute);font-weight:450;margin-left:6px}
.shop select{background:#0f1114;border:1px solid var(--line);border-radius:7px;color:var(--text);padding:6px 8px;font:inherit;font-size:13px;max-width:100%}
.shop.ok{border-color:#2c5a36}.shop.warn{border-color:#6b5420}
.hint{color:var(--mute);font-size:12px;line-height:1.45}
.empty{text-align:center;color:var(--mute);padding:26px 10px;font-size:13px}
.warnline{margin-top:10px;font-size:12.5px}
.warnline.bad{color:var(--bad)}.warnline.good{color:var(--ok)}
.feedwrap{position:relative;display:inline-block;line-height:0}
.feedwrap canvas{position:absolute;left:0;top:0;cursor:crosshair}
.feedwrap img{border-radius:9px;border:1px solid var(--line);display:block;width:420px;max-width:100%}
.s8{grid-column:span 8}
@media(max-width:1100px){.s8{grid-column:1/-1}}
.tabs .tab{white-space:nowrap}
a.lnk,.sub a{color:var(--acc2);text-decoration:none}a.lnk:hover,.sub a:hover{text-decoration:underline}
select{background:#0f1114;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 9px;font:inherit;font-size:13px}
.kpis.inner{grid-column:auto;margin-bottom:8px}
.kpi .d{font-size:11.5px;color:var(--mute);margin-top:2px}
/* charts */
.chart{position:relative;min-height:60px}
.chart svg{display:block;width:100%;height:auto;overflow:visible}
.chart .ax{fill:var(--mute);font-size:11px}
.chart .lab{fill:var(--text);font-size:11.5px}
.chart .val{fill:var(--mute);font-size:11px}
.legend{display:flex;gap:14px;font-size:12px;color:var(--mute);margin:0 0 6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.tt{position:fixed;z-index:50;display:none;pointer-events:none;background:#0b0c0e;border:1px solid var(--line);
 border-radius:8px;padding:7px 10px;font-size:12px;color:var(--text);box-shadow:0 6px 20px rgba(0,0,0,.4);max-width:260px}
.tt b{font-weight:600}.tt .r{display:flex;justify-content:space-between;gap:14px}
.tt i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px}
.tbl{max-height:280px;overflow:auto;margin-top:4px}
.ttog{font-size:11.5px;color:var(--mute);cursor:pointer;user-select:none;margin-left:12px;padding:1px 7px;border:1px solid var(--line);border-radius:6px}.ttog:hover{color:var(--text)}
.demobanner{background:#1f1a0c;border:1px solid #5a4a18;color:#e9d9a8;border-radius:10px;padding:10px 14px;font-size:13px}
.exports{font-size:12.5px;color:var(--mute);margin-top:4px}
/* status chips */
.st{display:inline-block;font-size:11px;font-weight:650;padding:2px 8px;border-radius:99px;letter-spacing:.02em}
.st.OK{background:rgba(63,185,80,.15);color:#56d364}.st.LOW{background:rgba(217,164,65,.15);color:#e3b341}
.st.EMPTY{background:rgba(255,77,79,.15);color:#ff7b7d}.st.MISPLACED{background:rgba(163,113,247,.15);color:#bc8cff}
.st.occ{background:#22262c;color:var(--mute)}
.bar-in{height:6px;border-radius:3px;background:#2a2e35;overflow:hidden;min-width:70px}
.bar-in>div{height:100%;background:var(--ok);border-radius:3px}
/* shelves */
.shwrap{position:relative;line-height:0;flex:1 1 520px;max-width:760px}
.shwrap img{width:100%;border-radius:10px;border:1px solid var(--line);display:block;min-height:120px;background:#0f1114}
.shwrap canvas{position:absolute;left:0;top:0;touch-action:none}
.pick{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 8px}
/* checkout */
.scanrow{display:flex;gap:8px}.scanrow input{flex:1;width:auto;font-size:15px;padding:10px 12px}
.sugg{border:1px solid var(--line);border-radius:9px;margin-top:6px;overflow:hidden}
.sugg div{padding:8px 12px;cursor:pointer;display:flex;justify-content:space-between;border-bottom:1px solid var(--line2)}
.sugg div:hover{background:var(--card2)}
.cl{display:grid;grid-template-columns:1fr auto 90px 90px 28px;gap:10px;align-items:center;padding:9px 2px;
 border-bottom:1px solid var(--line2);font-size:13.5px}
.cl .n small{display:block;color:var(--mute);font-size:11.5px}
.qty{display:flex;align-items:center;gap:4px}.qty button{padding:2px 9px}
.cl .amt{text-align:right;font-weight:600}.cl .rate{text-align:right;color:var(--mute)}
.tot{display:flex;justify-content:space-between;padding:5px 2px;font-size:13.5px;color:var(--mute)}
.tot.big{font-size:19px;color:var(--text);font-weight:700;padding-top:9px}
button.big{padding:10px 18px;font-size:14.5px;font-weight:600}
.flash{animation:fl .9s ease}@keyframes fl{0%{background:rgba(63,185,80,.18)}100%{background:transparent}}
.billwrap{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start}
.billwrap img{width:340px;max-width:100%;border-radius:8px;border:1px solid var(--line);background:#fff}
.paynote{flex:1;min-width:240px;font-size:15px}
.paynote .h{font-size:22px;font-weight:700;margin-bottom:6px;color:var(--text)}
.toast{margin-top:10px;padding:10px 12px;border-radius:9px;font-size:13.5px;border:1px solid var(--line)}
.toast.ok{border-color:#2d5f37;background:rgba(63,185,80,.08)}.toast.bad{border-color:var(--accline);background:var(--accsoft)}
/* cctv */
.cctvgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}
.cctvgrid figure{margin:0;position:relative;cursor:zoom-in}
.cctvgrid figure.big{grid-column:1/-1;cursor:zoom-out}
.cctvgrid img{width:100%;border-radius:10px;border:1px solid var(--line);display:block;background:#0f1114;min-height:180px}
.cctvgrid .badge{position:absolute;top:10px;right:10px;background:rgba(11,12,14,.78);border:1px solid var(--line);
 border-radius:99px;font-size:12px;padding:3px 10px}
/* ---- v2 type scale + layout: readable at a glance on a shop-floor screen ---- */
body{font-size:15px}
header{height:62px;padding:0 28px}header b{font-size:18px}
.tab{font-size:15px;padding:8px 16px}.pill{font-size:13px;padding:4px 12px}
main{max-width:none;padding:22px 28px 48px;gap:18px}
.card{padding:20px 22px;border-radius:14px}
h3{font-size:16px;margin-bottom:14px}h3 .sub{font-size:13.5px}
.kpis{grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:14px}
.kpi{padding:16px 18px;border-radius:12px}.kpi .v{font-size:30px;line-height:1.15}
.kpi .l{font-size:13.5px;margin-top:4px;color:#b3b8c2}.kpi .d{font-size:12.5px}
.kpi.warn .v{color:#e3b341}.kpi.bad .v{color:#ff7b7d}
button{font-size:14px;padding:7px 13px}select,input[type=number],input[type=text]{font-size:14px}
td,th{font-size:14.5px;padding:9px 6px}th{font-size:13px}
.muted,.alert small{font-size:13px}.hint{font-size:13.5px}.chip{font-size:13.5px;padding:5px 11px}
.st{font-size:12px;padding:3px 9px}figcaption{font-size:13px}.legend{font-size:13px}
.chart .ax{font-size:12.5px}.chart .lab,.chart .val{font-size:13px}.tt{font-size:13px}
.f span{font-size:13.5px}.exports{font-size:13.5px}.ttog{font-size:12.5px}
.s3{grid-column:span 3}@media(max-width:1300px){.s3{grid-column:span 6}}@media(max-width:1100px){.s3{grid-column:1/-1}}
/* home: every row the same height, long lists scroll inside their card */
#home{align-items:stretch}
.hcard{display:flex;flex-direction:column;height:460px}.hcard2{display:flex;flex-direction:column;height:520px}
.hcard .body,.hcard2 .body{flex:1;min-height:0;overflow:auto}
.body.mini{display:flex;flex-direction:column;gap:10px}
.body.mini canvas{width:100%;flex:1;min-height:0;object-fit:contain;border-radius:10px;cursor:grab}
.seg{display:inline-flex}.seg button{border-radius:0;padding:3px 11px;font-size:12px}
.seg button:first-child{border-radius:7px 0 0 7px}.seg button:last-child{border-radius:0 7px 7px 0;border-left:0}
.hlegend{display:flex;align-items:center;gap:8px;font-size:11.5px;color:var(--mute)}
.hlegend i{flex:0 0 120px;height:7px;border-radius:4px;background:linear-gradient(90deg,#266ec8,#1ebeaa,#ebc83c,#fa7828,#ff3246)}
.hlegend em{margin-left:auto;font-style:normal;text-align:right}
@media(max-width:1100px){.hcard,.hcard2{height:auto}.hcard .body,.hcard2 .body{max-height:420px}}
.alert{padding:12px 14px;margin-bottom:8px;gap:14px}
.alert .top{display:flex;justify-content:space-between;gap:10px;align-items:baseline}
.alert .a{font-size:15px}.alert .t{color:var(--mute);font-size:12.5px;white-space:nowrap}
.alert .msg{margin-top:2px;color:#d5d8de}
.sevsum{display:flex;gap:8px}.sevsum span{font-size:12.5px;padding:2px 9px;border-radius:99px;font-weight:600}
.sevsum .c{background:rgba(255,77,79,.14);color:#ff7b7d}.sevsum .w{background:rgba(217,164,65,.14);color:#e3b341}
.sevsum .i{background:var(--accsoft);color:var(--acc2)}
.qbig{display:flex;align-items:baseline;gap:10px;margin:2px 0 14px}.qbig b{font-size:44px;line-height:1}
.qbig span{color:var(--mute)}
.fc{grid-template-columns:repeat(3,1fr);gap:10px}.fc b{font-size:22px}.fc small{font-size:12.5px}
.rec{margin-top:14px;padding:12px 14px;border-radius:10px;background:var(--card2);border:1px solid var(--line)}
.rec b{font-size:20px}
.lowrow{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;padding:10px 2px;border-bottom:1px solid var(--line2)}
.lowrow .n{font-weight:600}.lowrow .w{color:var(--mute);font-size:13px}.lowrow .u{text-align:right;font-size:17px;font-weight:650}
.lowrow .u small{color:var(--mute);font-weight:400;font-size:13px}
.rep{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.rep div{background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.rep b{display:block;font-size:20px}.rep span{color:var(--mute);font-size:13px}
/* live analytics */
.livedot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--acc2);margin-right:8px;
 box-shadow:0 0 0 0 rgba(255,59,82,.6);animation:pulse 1.6s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,59,82,.55)}70%{box-shadow:0 0 0 9px rgba(255,59,82,0)}100%{box-shadow:0 0 0 0 rgba(255,59,82,0)}}
.livegrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
.livegrid>div{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:14px 16px 8px}
.lt{display:flex;justify-content:space-between;align-items:baseline;font-weight:600;font-size:14.5px;margin-bottom:6px}
.lt b{font-size:24px;font-weight:700}.lt small{color:var(--mute);font-weight:400;font-size:12.5px;margin-left:6px}
/* ---- analytics ---- */
#analytics{gap:16px;align-items:stretch}
.ahead{grid-column:1/-1;display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;padding:4px 2px 2px}
.ahead h2{margin:0;font-size:28px;letter-spacing:-.025em;line-height:1.1}.ahead p{margin:5px 0 0;color:var(--mute);font-size:13.5px}
.ahr{display:flex;flex-direction:column;align-items:flex-end;gap:8px}
.sech{grid-column:1/-1;display:flex;align-items:center;gap:12px;margin:20px 2px 0}
.sech h4{margin:0;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--text);font-weight:650}
.sech i{flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent)}.sech .muted{font-size:12px}
.akp{grid-column:1/-1;display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px}
.hk{background:linear-gradient(180deg,#1c1f25,var(--card));border:1px solid var(--line);border-radius:14px;padding:16px 16px 10px;min-width:0;position:relative;overflow:hidden}
.hk .l{color:var(--mute);font-size:12.5px;font-weight:500}
.hk .v{font-size:30px;font-weight:700;letter-spacing:-.03em;line-height:1.15;margin:2px 0 6px;white-space:nowrap}
.hk .dl{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--mute);min-height:22px;white-space:nowrap}
.hk svg{display:block;width:calc(100% + 4px);height:38px;margin:8px -2px 0}
.pill{padding:1px 8px;border-radius:99px;font-weight:650;font-size:11.5px;border:1px solid}
.pill.up{color:#6fdc8c;border-color:#2c5a36;background:#12261a}.pill.dn{color:#ff8d8f;border-color:#6b2a2c;background:#2a1315}
.pill.flat{color:var(--mute);border-color:var(--line);background:var(--card2)}
.ins{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.in{border:1px solid var(--line);border-radius:14px;padding:14px 16px;background:var(--card);display:flex;gap:12px;align-items:flex-start}
.in .ic{flex:0 0 34px;height:34px;border-radius:10px;background:var(--accsoft);color:var(--acc2);display:grid;place-items:center;font-size:16px;border:1px solid var(--accline)}
.in b{display:block;font-size:11.5px;color:var(--mute);font-weight:550;letter-spacing:.04em;text-transform:uppercase;margin-bottom:3px}
.in span{font-size:14px;line-height:1.4}.in span em{font-style:normal;font-weight:650}
.livestrip{background:none!important;border:0!important;padding:0!important}
#analytics .livegrid{grid-template-columns:repeat(4,minmax(0,1fr))}
#analytics .livegrid>div{border-radius:14px;background:var(--card);padding:14px 14px 6px}
#analytics .lt{font-size:13px;color:var(--mute);font-weight:550}#analytics .lt b{color:var(--text);font-size:26px;letter-spacing:-.02em}
#analytics .card{border-radius:14px;padding:18px 18px 14px}
#analytics .card h3{font-size:14.5px;margin-bottom:14px}
.ashop{display:grid;grid-template-columns:1fr 1fr;gap:10px}.ashop>div{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.ashop b{display:block;font-size:26px;letter-spacing:-.02em;line-height:1.1}.ashop span{color:var(--mute);font-size:12.5px}
@media(max-width:1250px){.akp{grid-template-columns:repeat(3,minmax(0,1fr))}#analytics .livegrid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:700px){.akp{grid-template-columns:repeat(2,minmax(0,1fr))}.ahr{align-items:flex-start}}
.warnbox{background:#2b2210;border:1px solid #6b5420;color:#f0d9a0;border-radius:9px;padding:9px 11px;font-size:12.5px;line-height:1.45;margin:8px 0}
.tabdot{display:none;width:8px;height:8px;border-radius:50%;background:var(--bad);margin-left:5px;vertical-align:1px}
.tabdot.on{display:inline-block;animation:qpulse 1.6s infinite}.tabdot.warn{background:var(--low)}
@keyframes qpulse{50%{opacity:.35}}
.verdict{display:flex;gap:16px;align-items:center;padding:14px 16px;border-radius:12px;border:1px solid var(--line);background:var(--card2);margin-bottom:14px}
.verdict .big{font-size:34px;font-weight:700;letter-spacing:-.02em;line-height:1;min-width:112px}
.verdict .why{color:var(--mute);font-size:13px}.verdict .why b{color:var(--text);font-weight:600}
.verdict.ok{border-color:#2c5a36}.verdict.ok .big{color:var(--ok)}
.verdict.warn{border-color:#6b5420}.verdict.warn .big{color:var(--low)}
.verdict.bad{border-color:#6b2a2c}.verdict.bad .big{color:var(--bad)}
table.wi td,table.wi th{padding:9px 8px}table.wi tr.cur td{background:#1d2026}
table.wi tr.rec td{background:var(--accsoft)}table.wi .tag{font-size:11px;border-radius:99px;padding:2px 8px;margin-left:6px;border:1px solid var(--line);color:var(--mute)}
table.wi tr.rec .tag{border-color:var(--accline);color:var(--acc2)}
.crowd{display:grid;grid-template-columns:auto 1fr auto;gap:6px 14px;align-items:center;padding:12px 14px;border:1px solid var(--line);
 border-left:3px solid var(--low);border-radius:10px;background:var(--card2);margin-bottom:8px}
.crowd.crit{border-left-color:var(--bad)}.crowd .n{font-size:30px;font-weight:700;line-height:1;grid-row:span 2}
.crowd .w{font-weight:600}.crowd .t{color:var(--mute);font-size:12.5px}.crowd .eta{grid-column:2/4;font-size:13px}
.qimg{width:100%;border-radius:10px;border:1px solid var(--line);display:block}
.qstat{display:grid;grid-template-columns:1fr auto;gap:2px 12px}.qstat div{padding:9px 0;border-bottom:1px solid var(--line2)}
.qstat div:nth-child(even){text-align:right;font-weight:650}.qstat div:nth-child(odd){color:var(--mute)}
</style></head><body>
<header><b>StoreSense Edge</b>
 <nav class="tabs" id="tabs">
  <div class="tab on" data-t="home">Home</div><div class="tab" data-t="shelves">Shelves</div>
  <div class="tab" data-t="checkout">Checkout</div><div class="tab" data-t="queue">Queue <span class="tabdot" id="qdot"></span></div><div class="tab" data-t="cctv">CCTV</div>
  <div class="tab" data-t="analytics">Analytics</div><div class="tab" data-t="setup">Setup</div></nav>
 <div class="right"><span class="pill" id="store"></span><span class="pill" id="conn">connecting…</span>
  <span class="pill" id="cloud"></span></div></header>

<main id="home">
<section class="kpis" id="kpis"></section>
<section class="card s5 hcard"><h3>Needs attention <span class="sevsum" id="acount"></span></h3><div class="body" id="alerts"></div></section>
<section class="card s4 hcard"><h3>Running low <span class="sub"><a href="#" onclick="tab('shelves');return false">all shelves →</a></span></h3>
 <div class="body" id="lowstock"></div></section>
<section class="card s3 hcard"><h3>Queue <span class="sub">counters
 <input id="ctr" type="number" min="1" max="10" style="width:58px"> <button onclick="setCounters()">Set</button></span></h3>
 <div class="body" id="queues"></div></section>
<section class="card s6 hcard2"><h3>Footfall today <span class="sub">entries per hour</span></h3><div class="body"><div class="chart" id="c_today"></div></div></section>
<section class="card s6 hcard2"><h3>Floor heatmap <span class="sub"><span class="seg"><button id="hm_now" class="on" onclick="heatMode('now')">Now</button><button id="hm_day" onclick="heatMode('day')">Today</button></span></span></h3>
 <div class="body mini"><canvas id="h3d" title="drag to turn · scroll to zoom · double-click to reset"></canvas>
  <div class="hlegend"><span>quiet</span><i></i><span>busy</span><em id="hnote"></em></div><div id="zones"></div></div></section>
<section class="card s12"><h3>Reports <span class="sub"><button onclick="report('day')">Today</button>
 <button onclick="report('week')">7 days</button></span></h3><div id="report" class="muted">Pick a period to see a summary.</div></section>
</main>

<main id="shelves" style="display:none">
<section class="card s12"><h3>Mark products on the shelf
  <span class="sub">draw one box per product · set how many sit side by side and how deep they stack</span></h3>
 <div class="bar2"><select id="shcam" onchange="shLoad(this.value)"></select><div class="sep"></div>
  <button id="b_draw" onclick="shDrawMode()">Draw product box</button>
  <button onclick="shCalibrate()">Calibrate (shelf full, aisle clear)</button><div class="sep"></div>
  <button id="b_shsave" onclick="shSave()">Save products</button><div class="sep"></div>
  <button id="b_vstill" onclick="shView('still')">Still</button><button id="b_vlive" onclick="shView('live')">Live</button><button id="b_depth" onclick="shView('depth')">Depth</button>
  <span class="muted" id="shmsg"></span></div>
 <div class="hint" id="shdepth" style="margin:-4px 0 8px"></div>
 <div class="wrap"><div class="shwrap" id="shwrap"><img id="shimg" alt="" onerror="shImgErr()"><canvas id="shov"></canvas></div>
  <div class="side" id="shside"></div></div></section>
<section class="card s12"><h3>Stock on this shelf
  <span class="sub">depth = units counted per column by the depth model · front row = facings seen × units deep · till = restocked − sold</span></h3><div id="shtable"></div></section>
<section class="card s12"><h3>Product catalog <span class="sub"><button onclick="prodEdit()">+ Product</button>
  <button onclick="$('csvin').click()">Import CSV</button><input type="file" id="csvin" accept=".csv,text/csv" style="display:none" onchange="prodImport(this)">
  <a class="lnk" href="/api/export/products.csv">Export CSV</a><span class="muted" id="impmsg"></span></span></h3>
 <div id="prodform"></div><div id="catalog"></div></section>
</main>

<main id="checkout" style="display:none">
<section class="card s7"><h3>Cart <span class="sub" id="cartid"></span></h3>
 <div id="shopbar"></div>
 <div class="scanrow"><input id="scan" placeholder="Scan a barcode — or type a SKU or product name" autocomplete="off">
  <button class="pri" onclick="scanSubmit()">Add</button></div>
 <div id="suggest"></div><div id="scanmsg" class="muted"></div>
 <div id="cartlines"></div><div id="carttotals"></div>
 <div class="bar2" style="margin-top:14px"><button onclick="cartNew()">Clear cart</button>
  <button class="pri big" id="b_bill" onclick="cartBill()">Finish &amp; print bill</button></div></section>
<section class="card s5"><h3>Checkout camera <span class="sub">hold the barcode 15–25 cm away</span></h3>
 <div id="tillcam"></div></section>
<section class="card s12" id="billbox" style="display:none"><h3>Bill <span class="sub" id="billlinks"></span></h3>
 <div class="billwrap"><img id="billimg" alt="bill"><div class="paynote" id="paynote"></div></div></section>
</main>

<main id="queue" style="display:none">
<section class="kpis" id="qkpis"></section>
<section class="card s7"><h3>Will the line clear? <span class="sub">at today's arrival and service rates</span></h3>
 <div id="qverdict"></div>
 <table class="wi" id="qwhatif"></table>
 <div class="bar2" style="margin:14px 0 0"><span class="muted">Counters open</span>
  <input id="qctr" type="number" min="1" max="10" style="width:64px"> <button onclick="qSetCounters()">Set</button>
  <button class="pri" id="qopen" style="display:none" onclick="qOpenRec()"></button></div></section>
<section class="card s5"><h3>Queue camera <span class="sub">faces blurred</span></h3><div id="qcam"></div></section>
<section class="card s6"><h3>Crowds <span class="sub" id="qcrowdsub"></span></h3><div id="qcrowds"></div></section>
<section class="card s6"><h3>Staff calls <span class="sub" id="qresp"></span></h3><div id="qcalls"></div></section>
<section class="card s8"><h3>Queue today <span class="sub">people in line and wait, per minute</span></h3><div class="chart" id="q_today"></div></section>
<section class="card s4"><h3>Today so far</h3><div class="qstat" id="qstats"></div></section>
<section class="card s12"><h3>When queues build <span class="sub">average people in line, by weekday and hour</span></h3><div class="chart" id="q_heat"></div></section>
</main>

<main id="cctv" style="display:none">
<section class="card s12"><h3>Cameras <span class="sub">
  <span class="chip on" id="cv_plain" onclick="cctvMode('cctv')">Plain</span>
  <span class="chip" id="cv_ana" onclick="cctvMode('analytics')">With analytics</span>
  · faces are blurred on every view · nothing is recorded</span></h3>
 <div class="cctvgrid" id="cctvgrid"></div></section>
</main>

<main id="analytics" style="display:none">
<header class="ahead"><div><h2>Analytics</h2><p id="asub">How the store has been doing</p></div>
 <div class="ahr"><span class="seg" id="arange"><button data-d="7">7 days</button><button class="on" data-d="14">14 days</button><button data-d="30">30 days</button></span>
  <span class="exports"><a class="lnk" href="/api/export/timeseries_hourly.csv?days=90">hourly series</a> · <a class="lnk" href="/api/export/timeseries_minute.csv">per-minute log</a> ·
  <a class="lnk" href="/api/export/bills.csv?days=90">bills</a> · <a class="lnk" href="/api/export/products.csv">products</a></span></div></header>
<section class="s12 demobanner" id="demobanner" style="display:none"></section>

<section class="akp" id="akpis"></section>
<section class="ins" id="ains"></section>

<div class="sech"><h4>Right now</h4><i></i><span class="muted">moves every second · real trading only, no demo data</span>
 <span class="seg" id="lwin"><button class="on" data-w="300">5 min</button><button data-w="1800">30 min</button><button data-w="3600">60 min</button></span></div>
<section class="s12 livestrip"><div class="livegrid">
  <div><div class="lt"><span><span class="livedot"></span>People inside</span><b id="lv_inside">—</b></div><div class="chart" id="l_inside"></div></div>
  <div><div class="lt"><span>Entries per minute</span><b id="lv_ent">—</b></div><div class="chart" id="l_ent"></div></div>
  <div><div class="lt"><span>Queue length</span><b id="lv_queue">—</b></div><div class="chart" id="l_queue"></div></div>
  <div><div class="lt"><span>Sales today</span><b id="lv_sales">—</b></div><div class="chart" id="l_sales"></div></div>
 </div></section>

<div class="sech"><h4>Traffic and sales</h4><i></i></div>
<section class="card s8"><h3>Footfall and bills per day <span class="sub">complete days</span></h3><div class="chart" id="c_daily"></div></section>
<section class="card s4"><h3>Conversion <span class="sub">bills ÷ entries</span></h3><div class="chart" id="c_conv"></div></section>
<section class="card s12"><h3>When the store is busy <span class="sub">average entries by weekday and hour · darker = quieter</span></h3>
 <div class="chart" id="c_heat"></div></section>
<section class="card s4"><h3>Footfall by hour of day <span class="sub">average per day</span></h3><div class="chart" id="c_hour"></div></section>
<section class="card s4"><h3>Revenue by hour <span class="sub">average per day</span></h3><div class="chart" id="c_revh"></div></section>
<section class="card s4"><h3>Revenue per day</h3><div class="chart" id="c_rev"></div></section>

<div class="sech"><h4>Products and shelves</h4><i></i></div>
<section class="card s6"><h3>Top products <span class="sub">by revenue</span></h3><div class="chart" id="c_top"></div></section>
<section class="card s6"><h3>Shopper attention by product <span class="sub">stops of 1.5 s or more in front of it</span></h3><div class="chart" id="c_look"></div></section>
<section class="card s6"><h3>Stock-outs <span class="sub">times a product ran empty</span></h3><div class="chart" id="c_stock"></div></section>
<section class="card s6"><h3>Store zones <span class="sub">visits · hover for average dwell</span></h3><div class="chart" id="c_zones"></div></section>

<div class="sech"><h4>Checkout and service</h4><i></i></div>
<section class="card s4"><h3>Basket size <span class="sub">items per bill</span></h3><div class="chart" id="c_basket"></div></section>
<section class="card s4"><h3>Queue wait</h3><div class="chart" id="c_wait"></div></section>
<section class="card s4"><h3>Shopper IDs today <span class="sub">issued at the door, checked at the till</span></h3><div id="a_shop"></div></section>
</main>

<main id="setup" style="display:none">
<section class="card s12"><h3>Store plan <span class="sub">drag to move · scroll a number to nudge · Delete to remove</span></h3>
 <div class="bar2"><button class="pri" onclick="addShelf()">+ Shelf</button>
  <button onclick="addCam()">+ Camera</button><button onclick="addFix('door')">+ Door</button>
  <button onclick="addFix('counter')">+ Counter</button><button onclick="addFix('fixture')">+ Fixture</button><div class="sep"></div>
  <button id="b_save" onclick="saveLay()">Save plan</button><button onclick="loadLay()">Reload</button>
  <div class="sep"></div><span class="muted" id="saved"></span></div>
 <div class="wrap"><canvas id="plan" class="plan"></canvas><div class="side" id="side"></div></div>
 <div class="warnline" id="blind"></div></section>
<section class="card s12" id="camsetup" style="display:none"><h3>Camera view
  <span class="sub">click on the picture to place points</span></h3><div id="camsetupbody"></div></section>
<section class="card s12"><h3>Integrations <span class="sub">connect a POS, inventory or ERP system</span></h3>
 <div class="hint" style="margin-bottom:10px">Other systems can call these on this box, or receive events by webhook
  (set <code>"webhooks"</code> in the config: a URL list; events queue while offline and are retried).</div>
 <table><tr><th>Call</th><th>What it does</th></tr>
  <tr><td><code>POST /api/integrations/pos</code></td><td>Record a sale made on another till</td></tr>
  <tr><td><code>POST /api/integrations/products</code></td><td>Import the catalog (JSON list or CSV: sku, name, brand, barcode, mrp, price)</td></tr>
  <tr><td><code>POST /api/integrations/restock</code></td><td>A delivery arrived: add to (or set) each product's stock</td></tr>
  <tr><td><code>GET /api/integrations/stock</code></td><td>Every product: where it sits, camera count, till count, status, wrong-product flag</td></tr>
  <tr><td><code>GET /api/integrations/sales?since=</code></td><td>Bills since a time</td></tr>
  <tr><td>webhook events</td><td><code>alert</code> · <code>bill</code> · <code>stock</code> (a product's shelf status changed) · <code>restock</code></td></tr></table>
 <div class="muted" id="intstat" style="margin-top:10px"></div></section>
<section class="card s12"><h3>3D view <span class="sub">drag to orbit · scroll to zoom · double-click to reset</span></h3>
 <canvas id="v3d" style="width:100%;height:auto;cursor:grab;border:1px solid var(--line);border-radius:10px;background:#0f1114"></canvas>
 <div class="hint" style="margin-top:8px">Shelf faces are coloured by live stock: green in stock, amber low,
  red empty, violet misplaced. Grey faces are not monitored. Dotted cones are what each camera can see.</div></section>
</main>

<script>
const $=i=>document.getElementById(i);
let LAY={store:{w:6,h:4},shelves:[],cameras:[]},SEL=null,DRAG=null,HEAT=null,HEATDAY=null,HEATNOW=null,HEATMODE='now',HOV=null,DIRTY=false,S=null;
const FACES=['N','E','S','W'],SNAP=0.25;

/* ---------- geometry ---------- */
function rot(px,py,cx,cy,d){const a=d*Math.PI/180,s=Math.sin(a),c=Math.cos(a),dx=px-cx,dy=py-cy;
 return [cx+dx*c-dy*s,cy+dx*s+dy*c]}
function corners(r){const {x,y,w,h}=r,d=r.rot||0;
 return [[x-w/2,y-h/2],[x+w/2,y-h/2],[x+w/2,y+h/2],[x-w/2,y+h/2]].map(p=>rot(p[0],p[1],x,y,d))}
function faceSeg(s,f){const c=corners(s);return {N:[c[0],c[1]],E:[c[1],c[2]],S:[c[2],c[3]],W:[c[3],c[0]]}[f]}
function faceNorm(s,f){const n={N:[0,-1],E:[1,0],S:[0,1],W:[-1,0]}[f];return rot(n[0],n[1],0,0,s.rot||0)}
const ccw=(a,b,c)=>(c[1]-a[1])*(b[0]-a[0])>(b[1]-a[1])*(c[0]-a[0]);
const xs=(a,b,c,d)=>ccw(a,c,d)!==ccw(b,c,d)&&ccw(a,b,c)!==ccw(a,b,d);
const FIX=()=>LAY.fixtures||(LAY.fixtures=[]);
// counters and fixtures block a camera's view like shelves do; doors don't
function blocked(p,q,skip){for(const s of [...LAY.shelves,...FIX().filter(f=>f.kind!=='door')]){if(s.id===skip)continue;const c=corners(s);
 for(let i=0;i<4;i++)if(xs(p,q,c[i],c[(i+1)%4]))return true}return false}
function vis(cam,s,f){const [a,b]=faceSeg(s,f),n=faceNorm(s,f),mx=(a[0]+b[0])/2,my=(a[1]+b[1])/2;
 if((cam.x-mx)*n[0]+(cam.y-my)*n[1]<=0)return 0;
 const half=cam.fov*Math.PI/360,head=cam.heading*Math.PI/180;let seen=0;const N=7;
 for(let i=0;i<N;i++){const t=(i+0.5)/N,p=[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t];
  const d=Math.hypot(p[0]-cam.x,p[1]-cam.y);if(d>cam.range||d<1e-6)continue;
  let an=Math.atan2(p[1]-cam.y,p[0]-cam.x)-head;an=(an+Math.PI)%(2*Math.PI)-Math.PI;
  if(Math.abs(an)>half)continue;if(blocked([cam.x,cam.y],p,s.id))continue;seen++}
 return seen/N}
function bestVis(s,f){let v=0;for(const c of LAY.cameras)if((c.roles||[]).includes('shelf'))v=Math.max(v,vis(c,s,f));return v}
const faceCol=v=>v>=.6?'#3fb950':(v>0?'#d9a441':'#ff4d4f');
/* ---------- 2D plan ---------- */
function planDraw(cv,opts){const IW=opts.w||1100;if(cv.width!==IW)cv.width=IW;
 const sc=IW/LAY.store.w,H=Math.max(140,Math.round(sc*LAY.store.h));if(cv.height!==H)cv.height=H;
 const X=cv.getContext('2d'),m=v=>v*sc;X.clearRect(0,0,IW,H);
 X.fillStyle='#0f1114';X.fillRect(0,0,IW,H);
 if(HEAT&&HEAT.length){const R=HEAT.length,C=HEAT[0].length,oc=document.createElement('canvas');oc.width=C;oc.height=R;   /* one pixel per cell, scaled up smoothly: a soft glow, not squares */
  const ox=oc.getContext('2d'),im=ox.createImageData(C,R);
  for(let r=0;r<R;r++)for(let c=0;c<C;c++){const v=HEAT[r][c],k=(r*C+c)*4,m=hcol(v).match(/[\d.]+/g);
   im.data[k]=+m[0];im.data[k+1]=+m[1];im.data[k+2]=+m[2];im.data[k+3]=v>0.02?Math.round(255*Math.min(.85,.25+.7*v)):0}
  ox.putImageData(im,0,0);X.imageSmoothingEnabled=true;X.imageSmoothingQuality='high';X.drawImage(oc,0,0,IW,H)}
 X.lineWidth=1;for(let x=0;x<=LAY.store.w+1e-6;x+=0.5){X.strokeStyle=Math.abs(x%1)<1e-6?'#23272e':'#191c21';
  X.beginPath();X.moveTo(m(x),0);X.lineTo(m(x),H);X.stroke()}
 for(let y=0;y<=LAY.store.h+1e-6;y+=0.5){X.strokeStyle=Math.abs(y%1)<1e-6?'#23272e':'#191c21';
  X.beginPath();X.moveTo(0,m(y));X.lineTo(IW,m(y));X.stroke()}
 X.strokeStyle='#3a3f48';X.lineWidth=2;X.strokeRect(1,1,IW-2,H-2);
 if(opts.edit)for(const c of LAY.cameras){if(!c.floor_rect)continue;const p=corners(c.floor_rect).map(q=>[m(q[0]),m(q[1])]);
  X.setLineDash([6,5]);X.strokeStyle='#7a5a60';X.fillStyle='rgba(228,0,43,.05)';X.lineWidth=1.5;
  X.beginPath();p.forEach((q,i)=>i?X.lineTo(q[0],q[1]):X.moveTo(q[0],q[1]));X.closePath();X.fill();X.stroke();X.setLineDash([])}
 if(opts.cones!==false)for(const c of LAY.cameras){const x=m(c.x),y=m(c.y),half=c.fov*Math.PI/360,hd=c.heading*Math.PI/180;
  const act=SEL&&SEL.id===c.id||HOV&&HOV.id===c.id;   // only the active cone is filled, else they stack into mud
  X.fillStyle=act?'rgba(228,0,43,.12)':'rgba(228,0,43,.03)';
  X.strokeStyle=act?'rgba(255,59,82,.75)':'rgba(228,0,43,.25)';X.lineWidth=act?1.6:1;X.setLineDash([5,4]);
  X.beginPath();X.moveTo(x,y);X.arc(x,y,m(c.range),hd-half,hd+half);X.closePath();X.fill();X.stroke();X.setLineDash([])}
 for(const s of LAY.shelves){const c=corners(s).map(p=>[m(p[0]),m(p[1])]);
  const on=opts.edit&&SEL&&SEL.t==='s'&&SEL.id===s.id,hov=HOV&&HOV.t==='s'&&HOV.id===s.id;
  X.fillStyle=on?'#2a2126':(hov?'#23262c':'#1d2025');X.strokeStyle=on?'var(--acc)':'#454b55';X.lineWidth=on?2:1;
  X.beginPath();c.forEach((p,i)=>i?X.lineTo(p[0],p[1]):X.moveTo(p[0],p[1]));X.closePath();X.fill();X.stroke();
  FACES.forEach((f,i)=>{if(!s.faces[f])return;const a=c[i],b=c[(i+1)%4],v=bestVis(s,f);
   X.lineWidth=5;X.setLineDash(v>0?[]:[7,5]);X.strokeStyle=faceCol(v);
   X.beginPath();X.moveTo(a[0],a[1]);X.lineTo(b[0],b[1]);X.stroke();X.setLineDash([])});
  X.fillStyle='#c3c9d2';X.font='600 12px system-ui';X.textAlign='center';
  X.fillText(s.name,m(s.x),m(s.y)+4);X.textAlign='left'}
 for(const f of FIX()){const c=corners(f).map(p=>[m(p[0]),m(p[1])]);
  const on=opts.edit&&SEL&&SEL.t==='x'&&SEL.id===f.id,hov=HOV&&HOV.t==='x'&&HOV.id===f.id;
  const st={door:['rgba(63,185,80,.16)','#3fb950',[7,4]],counter:['#1b2533','#5b8fd1',[]],fixture:['#221f1a','#8a8170',[]]}[f.kind];
  X.fillStyle=hov&&!on?'#262a31':st[0];X.strokeStyle=on?'#ff3b52':st[1];X.lineWidth=on?2.5:1.6;X.setLineDash(st[2]);
  X.beginPath();c.forEach((p,i)=>i?X.lineTo(p[0],p[1]):X.moveTo(p[0],p[1]));X.closePath();X.fill();X.stroke();X.setLineDash([]);
  if(f.kind==='fixture'){X.save();X.beginPath();c.forEach((p,i)=>i?X.lineTo(p[0],p[1]):X.moveTo(p[0],p[1]));X.closePath();X.clip();
   X.strokeStyle='rgba(138,129,112,.35)';X.lineWidth=1;const bx=Math.min(...c.map(p=>p[0])),by=Math.min(...c.map(p=>p[1])),
   ex=Math.max(...c.map(p=>p[0])),ey=Math.max(...c.map(p=>p[1]));
   for(let t=bx-(ey-by);t<ex;t+=9){X.beginPath();X.moveTo(t,ey);X.lineTo(t+(ey-by),by);X.stroke()}X.restore()}
  const lab=f.kind==='door'?`${f.dir==='in'?'↓ ':f.dir==='out'?'↑ ':'⇅ '}${f.name}`:f.name;
  X.font='600 12px system-ui';X.textAlign='center';X.fillStyle=f.kind==='door'?'#7ee2a0':f.kind==='counter'?'#a9c6ee':'#c9c0ad';
  const tw=X.measureText(lab).width/2+6,lx=Math.max(tw,Math.min(IW-tw,m(f.x)));   // a door on a wall keeps its label on the plan
  const ly=Math.max(14,Math.min(H-6,m(f.y)+4));X.fillText(lab,lx,ly);X.textAlign='left'}
 for(const c of LAY.cameras){const x=m(c.x),y=m(c.y);
  const on=opts.edit&&SEL&&SEL.t==='c'&&SEL.id===c.id;
  X.beginPath();X.arc(x,y,on?9:7,0,6.3);X.fillStyle=on?'#ff3b52':'#e4002b';X.fill();
  X.strokeStyle='#0f1114';X.lineWidth=2;X.stroke();
  const hd=c.heading*Math.PI/180;X.strokeStyle=on?'#ff3b52':'#e4002b';X.lineWidth=2;
  X.beginPath();X.moveTo(x,y);X.lineTo(x+Math.cos(hd)*14,y+Math.sin(hd)*14);X.stroke();
  X.fillStyle='#c3c9d2';X.font='500 11.5px system-ui';X.fillText(c.name,x+12,y-9)}
 if(opts.edit){const o=selObj();if(o){const hs=handles(o);
  for(const k in hs){const p=hs[k];X.beginPath();X.arc(m(p[0]),m(p[1]),6,0,6.3);
   X.fillStyle='#0f1114';X.fill();X.strokeStyle='#ffb020';X.lineWidth=2;X.stroke()}}}
 return sc}
let SC=60;
function draw(){SC=planDraw($('plan'),{edit:true})}
function drawMini(){draw3d('h3d')}
function heatMode(m){HEATMODE=m;HEAT=m==='now'?HEATNOW:HEATDAY;
 $('hm_now').className=m==='now'?'on':'';$('hm_day').className=m==='day'?'on':'';
 if(TABV==='home')drawMini();if(TABV==='setup'){draw();draw3d('v3d')}}
function selObj(){if(!SEL)return null;
 if(SEL.t==='s')return LAY.shelves.find(s=>s.id===SEL.id);
 if(SEL.t==='c')return LAY.cameras.find(c=>c.id===SEL.id);
 if(SEL.t==='x')return FIX().find(f=>f.id===SEL.id);
 const c=LAY.cameras.find(c=>c.id===SEL.id);return c&&c.floor_rect}
function handles(o){if(SEL.t==='c')return {head:[o.x+Math.cos(o.heading*Math.PI/180)*Math.min(o.range*.5,1.1),
  o.y+Math.sin(o.heading*Math.PI/180)*Math.min(o.range*.5,1.1)]};
 return {rot:rot(o.x,o.y-o.h/2-0.38,o.x,o.y,o.rot||0),size:corners(o)[2]}}
function pick(mx,my){const o=selObj();
 if(o){const hs=handles(o);for(const k in hs)if(Math.hypot(mx-hs[k][0],my-hs[k][1])<10/SC)return {t:SEL.t,id:SEL.id,mode:k}}
 for(const c of LAY.cameras)if(Math.hypot(mx-c.x,my-c.y)<12/SC)return {t:'c',id:c.id,mode:'move'};
 for(let i=FIX().length-1;i>=0;i--){const s=FIX()[i],l=rot(mx,my,s.x,s.y,-(s.rot||0));
  if(Math.abs(l[0]-s.x)<=s.w/2&&Math.abs(l[1]-s.y)<=s.h/2)return {t:'x',id:s.id,mode:'move'}}
 for(let i=LAY.shelves.length-1;i>=0;i--){const s=LAY.shelves[i],l=rot(mx,my,s.x,s.y,-(s.rot||0));
  if(Math.abs(l[0]-s.x)<=s.w/2&&Math.abs(l[1]-s.y)<=s.h/2)return {t:'s',id:s.id,mode:'move'}}
 for(const c of LAY.cameras){if(!c.floor_rect)continue;const f=c.floor_rect,l=rot(mx,my,f.x,f.y,-(f.rot||0));
  if(Math.abs(l[0]-f.x)<=f.w/2&&Math.abs(l[1]-f.y)<=f.h/2)return {t:'f',id:c.id,mode:'move'}}
 return null}
function evM(e){const cv=$('plan'),r=cv.getBoundingClientRect();
 return [(e.clientX-r.left)*cv.width/r.width/SC,(e.clientY-r.top)*cv.height/r.height/SC]}
const PL=$('plan');
PL.addEventListener('pointermove',e=>{if(DRAG){const [mx,my]=evM(e),o=selObj();if(!o)return;
  const sn=v=>e.shiftKey?Math.round(v*100)/100:Math.round(v/SNAP)*SNAP;
  if(DRAG.mode==='move'){o.x=sn(mx-DRAG.ox);o.y=sn(my-DRAG.oy)}
  else if(DRAG.mode==='head')o.heading=Math.round(Math.atan2(my-o.y,mx-o.x)*180/Math.PI/5)*5;
  else if(DRAG.mode==='rot')o.rot=Math.round((Math.atan2(my-o.y,mx-o.x)*180/Math.PI+90)/5)*5;
  else if(DRAG.mode==='size'){const l=rot(mx,my,o.x,o.y,-(o.rot||0));
   o.w=Math.max(0.2,sn(Math.abs(l[0]-o.x)*2));o.h=Math.max(0.2,sn(Math.abs(l[1]-o.y)*2))}
  dirty();side();draw();return}
 const [mx,my]=evM(e),h=pick(mx,my);HOV=h;
 PL.style.cursor=h?(h.mode==='move'?'grab':'crosshair'):'default';draw()});
PL.addEventListener('pointerdown',e=>{const [mx,my]=evM(e),h=pick(mx,my);
 if(!h){SEL=null;side();draw();return}
 if(!SEL||SEL.t!==h.t||SEL.id!==h.id){SEL={t:h.t,id:h.id};side()}
 const o=selObj();DRAG={mode:h.mode,ox:mx-o.x,oy:my-o.y};PL.setPointerCapture(e.pointerId);
 PL.style.cursor='grabbing';draw()});
addEventListener('pointerup',()=>{if(DRAG){DRAG=null;PL.style.cursor='grab'}});
addEventListener('keydown',e=>{if(TABV!=='setup')return;
 if(/INPUT/.test(document.activeElement.tagName))return;
 if(e.key==='Delete'||e.key==='Backspace'){if(!SEL)return;e.preventDefault();del()}
 if(e.key==='Escape'){SEL=null;side();draw()}});
/* ---------- side panel ---------- */
function f(l,v,on,st){return `<div class="f"><span>${l}</span><input type="number" step="${st||0.1}" value="${v}" oninput="${on}"></div>`}
function side(){const o=selObj();let h=`<div class="box"><h4>Store size</h4>
 ${f('width (m)',LAY.store.w,'LAY.store.w=Math.max(1,+this.value);dirty();draw()',0.5)}
 ${f('depth (m)',LAY.store.h,'LAY.store.h=Math.max(1,+this.value);dirty();draw()',0.5)}</div>`;
 if(!o){h+=`<div class="box"><div class="empty">Nothing selected.<br><br>
  Add shelves, cameras, doors, counters or other fixtures, then drag them into place.<br>Click any item to edit it.</div></div>`}
 else if(SEL.t==='s'){h+=`<div class="box"><h4>Shelf</h4>
  <div class="f"><span>name</span><input type="text" value="${o.name}" oninput="selObj().name=this.value;dirty();draw()"></div>
  ${f('x (m)',o.x,'selObj().x=+this.value;dirty();draw()')}${f('y (m)',o.y,'selObj().y=+this.value;dirty();draw()')}
  ${f('width (m)',o.w,'selObj().w=+this.value;dirty();draw()')}${f('depth (m)',o.h,'selObj().h=+this.value;dirty();draw()')}
  ${f('height (m)',o.height,'selObj().height=+this.value;dirty()')}
  ${f('rotation °',o.rot||0,'selObj().rot=+this.value;dirty();draw()',5)}
  <button class="danger" onclick="del()" style="margin-top:6px">Delete shelf</button></div>
  <div class="box"><h4>Monitored faces</h4><div class="chips">`+
  FACES.map(x=>`<div class="chip ${o.faces[x]?'on':''}" onclick="tglFace('${x}')">${x}</div>`).join('')+`</div>`;
  for(const x of FACES){if(!o.faces[x])continue;const v=bestVis(o,x);
   h+=`<div class="f" style="margin-top:9px"><span>${x}: ${Math.round(v*100)}% seen</span><span>
    <input type="number" min="1" style="width:46px" value="${o.faces[x].grid[0]}"
     oninput="selObj().faces['${x}'].grid[0]=+this.value;dirty()">×
    <input type="number" min="1" style="width:46px" value="${o.faces[x].grid[1]}"
     oninput="selObj().faces['${x}'].grid[1]=+this.value;dirty()"></span></div>`}
  h+=`<div class="hint">rows × columns of product slots</div></div>`}
 else if(SEL.t==='c'){h+=`<div class="box"><h4>Camera</h4>
  <div class="f"><span>name</span><input type="text" value="${o.name}" oninput="selObj().name=this.value;dirty();draw()"></div>
  <div class="f"><span>source</span><input type="text" value="${o.source||''}" placeholder="0, or phone IP e.g. 192.168.1.5:8080"
   oninput="selObj().source=this.value;dirty()"></div>
  <div class="hint" id="camfeed" style="margin:-2px 0 8px">${feedLine(o)}</div>
  ${f('x (m)',o.x,'selObj().x=+this.value;dirty();draw()')}${f('y (m)',o.y,'selObj().y=+this.value;dirty();draw()')}
  ${f('facing °',o.heading,'selObj().heading=+this.value;dirty();draw()',5)}
  ${f('lens angle °',o.fov,'selObj().fov=+this.value;dirty();draw()',5)}
  ${f('sees up to (m)',o.range,'selObj().range=+this.value;dirty();draw()',0.5)}
  <button class="danger" onclick="del()" style="margin-top:6px">Delete camera</button></div>
  <div class="box"><h4>This camera does</h4><div class="chips">`+
  ['entry','queue','shelf','checkout'].map(r=>`<div class="chip ${(o.roles||[]).includes(r)?'on':''}"
   onclick="tglRole('${r}')">${{entry:'count people',queue:'watch queue',shelf:'watch shelves',checkout:'self-checkout'}[r]}</div>`).join('')+
  `</div><div class="hint" style="margin-top:8px">${camNote(o)}</div></div>`}
 else if(SEL.t==='x'){const K={door:'Door',counter:'Checkout counter',fixture:'Fixture'}[o.kind];
  h+=`<div class="box"><h4>${K}</h4>
  <div class="f"><span>name</span><input type="text" value="${esc(o.name)}" oninput="selObj().name=this.value;dirty();draw()"></div>
  <div class="chips" style="margin:6px 0 8px">`+['door','counter','fixture'].map(k=>`<div class="chip ${o.kind===k?'on':''}"
   onclick="fixKind('${k}')">${{door:'door',counter:'counter',fixture:'fixture'}[k]}</div>`).join('')+`</div>`+
  (o.kind==='door'?`<div class="chips" style="margin-bottom:8px">`+[['in','entry'],['out','exit'],['both','entry & exit']].map(([k,l])=>
   `<div class="chip ${o.dir===k?'on':''}" onclick="doorDir('${k}')">${l}</div>`).join('')+`</div>`:'')+`
  ${f('x (m)',o.x,'selObj().x=+this.value;dirty();draw()')}${f('y (m)',o.y,'selObj().y=+this.value;dirty();draw()')}
  ${f('width (m)',o.w,'selObj().w=+this.value;dirty();draw()')}${f('depth (m)',o.h,'selObj().h=+this.value;dirty();draw()')}
  ${o.kind!=='door'?f('height (m)',o.height,'selObj().height=+this.value;dirty();draw3d()'):''}
  ${f('rotation °',o.rot||0,'selObj().rot=+this.value;dirty();draw()',5)}
  <div class="hint" style="margin:6px 0">${o.kind==='door'?'Shown on the plan and heatmap. Put the entry camera’s count line across it. Doesn’t block camera views.'
   :o.kind==='counter'?'A billing counter. Point a camera with the “watch queue” job at the line in front of it. Blocks camera views like a shelf.'
   :'Anything that isn’t a shelf — pillar, freezer, promo stand. Blocks camera views, so blind spots stay accurate.'}</div>
  <button class="danger" onclick="del()">Delete ${K.toLowerCase()}</button></div>`}
 else h+=`<div class="box"><h4>Floor patch</h4>${f('x (m)',o.x,'selObj().x=+this.value;dirty();draw()')}
  ${f('y (m)',o.y,'selObj().y=+this.value;dirty();draw()')}${f('width (m)',o.w,'selObj().w=+this.value;dirty();draw()')}
  ${f('depth (m)',o.h,'selObj().h=+this.value;dirty();draw()')}${f('rotation °',o.rot||0,'selObj().rot=+this.value;dirty();draw()',5)}
  <div class="hint">The real floor area matching the 4 points marked on this camera's picture.</div></div>`;
 $('side').innerHTML=h;warn();camSetup()}
function camNote(c){const r=c.roles||[];
 if(r.includes('shelf')){let best=null,bv=0;for(const s of LAY.shelves)for(const fc in s.faces){
   const v=vis(c,s,fc);if(v>bv){bv=v;best=s.name+' '+fc}}
  return best?`Watching <b>${best}</b> (${Math.round(bv*100)}% of the face).`:'No shelf face in view yet — move it or widen the lens angle.'}
 if(r.includes('checkout'))return 'Self-checkout: hold a barcode 15–25 cm from this camera and it goes into the open cart.';
 const near=FIX().filter(f=>f.kind==='door').map(f=>[f,Math.hypot(f.x-c.x,f.y-c.y)]).sort((a,b)=>a[1]-b[1])[0];
 const tips=[];if(r.includes('entry'))tips.push('mark the count line on the picture below'+(near?`, across <b>${esc(near[0].name)}</b>`:''));
 if(r.includes('queue'))tips.push('mark the queue area in front of the counter');
 return tips.length?tips.join('; ').replace(/^./,x=>x.toUpperCase())+'.':'Pick what this camera does.'}
function tglFace(x){const s=selObj();if(s.faces[x])delete s.faces[x];else s.faces[x]={grid:[4,6]};dirty();side();draw()}
function tglRole(r){const c=selObj();c.roles=c.roles||[];
 if(c.roles.includes(r))c.roles=c.roles.filter(v=>v!==r);
 else if(r==='shelf'||r==='checkout')c.roles=[r];              // these need the camera to themselves
 else c.roles=[...c.roles.filter(v=>v!=='shelf'&&v!=='checkout'),r];
 dirty();side();draw()}
function feedLine(c){if(!c)return'';const k=S&&S.cams&&S.cams[c.id];
 if(!String(c.source||'').trim())return 'No source yet — a webcam number (0) or the phone\'s address from its camera app.';
 if(DIRTY)return 'Save the plan to connect.';
 if(!k)return 'Not running — save the plan to start it.';
 return k.online?`<span style="color:#3fb950">● live</span> · ${k.fps} fps`:`<span style="color:#ff4d4f">● not connected</span> — ${esc(k.msg||'connecting…')}`}
function del(){if(!SEL)return;
 if(SEL.t==='s')LAY.shelves=LAY.shelves.filter(s=>s.id!==SEL.id);
 else if(SEL.t==='c')LAY.cameras=LAY.cameras.filter(c=>c.id!==SEL.id);
 else if(SEL.t==='x')LAY.fixtures=FIX().filter(f=>f.id!==SEL.id);
 else delete LAY.cameras.find(c=>c.id===SEL.id).floor_rect;
 SEL=null;dirty();side();draw()}
function addShelf(){const id='S'+Date.now().toString(36);
 LAY.shelves.push({id,name:'Shelf '+(LAY.shelves.length+1),x:LAY.store.w/2,y:LAY.store.h/2,
  w:Math.min(2.5,LAY.store.w-1),h:0.5,rot:0,height:1.8,faces:{N:{grid:[4,6]}}});
 SEL={t:'s',id};dirty();side();draw()}
function addFix(kind){const id='F'+Date.now().toString(36),W=LAY.store.w,H=LAY.store.h,n=FIX().filter(f=>f.kind===kind).length+1;
 const d={door:{name:'Entry',x:W/2,y:H-0.12,w:1.2,h:0.2,height:2.1,dir:'in'},
  counter:{name:'Counter '+n,x:W-1.2,y:H-1.2,w:1.4,h:0.6,height:1.0},
  fixture:{name:'Fixture '+n,x:W/2,y:H/2,w:0.6,h:0.6,height:1.5}}[kind];
 FIX().push({id,kind,rot:0,...d});SEL={t:'x',id};dirty();side();draw();draw3d()}
function fixKind(k){const o=selObj();o.kind=k;o.height={door:2.1,counter:1.0,fixture:1.5}[k];
 if(k==='door'&&!o.dir)o.dir='both';dirty();side();draw();draw3d()}
function doorDir(d){const o=selObj(),auto=['Entry','Exit','Entry / exit','Door'].includes(o.name);o.dir=d;
 if(auto)o.name={in:'Entry',out:'Exit',both:'Entry / exit'}[d];dirty();side();draw()}
function addCam(){const id='cam'+Date.now().toString(36);
 LAY.cameras.push({id,name:'Camera '+(LAY.cameras.length+1),x:0.5,y:LAY.store.h/2,heading:0,
  fov:60,range:5,height:2.2,roles:['shelf'],source:''});
 SEL={t:'c',id};dirty();side();draw()}
function warn(){const bad=[];for(const s of LAY.shelves)for(const fc in s.faces){const v=bestVis(s,fc);
  if(v<0.6)bad.push(`${s.name} ${fc}${v>0?` (only ${Math.round(v*100)}%)`:''}`)}
 const el=$('blind');
 if(!LAY.shelves.length){el.className='warnline muted';el.textContent='Add shelves and cameras to model the shop.'}
 else if(bad.length){el.className='warnline bad';el.textContent='Not fully covered: '+bad.join(', ')}
 else {el.className='warnline good';el.textContent='Every monitored shelf face is covered.'}}
function dirty(){DIRTY=true;$('b_save').className='pri';$('saved').textContent='unsaved changes'}
/* ---------- camera picture setup ---------- */
let PICK=null,CAMUI=null;
const NEED={entry_line:2,queue_zone:4,floor_quad:4};
const MODE_LABEL={entry_line:'Entry line',queue_zone:'Queue area',floor_quad:'Floor points'};
function camModes(c){const r=c.roles||[],m=[];
 if(r.includes('entry'))m.push('entry_line');
 if(r.includes('queue'))m.push('queue_zone');
 m.push('floor_quad');return m}
function camSetup(){const box=$('camsetup');
 if(!SEL||SEL.t!=='c'){box.style.display='none';CAMUI=null;PICK=null;return}
 const c=selObj();
 if(!c.source){box.style.display='none';CAMUI=null;
  return}
 box.style.display='';
 if(CAMUI===c.id){modeBtns(c);ovDraw();return}          // same camera: don't reload the feed
 CAMUI=c.id;PICK=null;
 $('camsetupbody').innerHTML=`<div class="bar2" id="pkbar"></div>
  <div class="feedwrap"><img id="feed" class="live" data-cam="${esc(c.id)}"><canvas id="ov"></canvas></div>
  <div class="hint" style="margin-top:8px">Pick a tool, then click on the picture. The entry line is where
  people are counted; the arrow shows which way counts as coming in. Floor points are the four corners of a
  floor rectangle, which ties this camera into the store map.</div>`;
 modeBtns(c);
 $('ov').onclick=ev=>{if(!PICK)return;const ov=$('ov'),r=ov.getBoundingClientRect();
  const cam=selObj();let cur=(cam[PICK]||[]).slice();
  if(cur.length>=NEED[PICK])cur=[];
  cur.push([+((ev.clientX-r.left)/r.width).toFixed(3),+((ev.clientY-r.top)/r.height).toFixed(3)]);
  cam[PICK]=cur;dirty();
  if(cur.length===NEED[PICK]){if(PICK==='entry_line'&&cam.in_side==null)cam.in_side=-1;PICK=null}
  modeBtns(cam);ovDraw()};
 sizeOv()}
function modeBtns(c){const bar=$('pkbar');if(!bar)return;
 bar.innerHTML=camModes(c).map(m=>{const n=(c[m]||[]).length,need=NEED[m];
  const state=n>=need?'✓':(PICK===m?`${need-n} left`:'');
  return `<button class="${PICK===m?'on':''}" onclick="setPick('${m}')">${MODE_LABEL[m]} ${state}</button>`}).join('')+
  ((c.roles||[]).includes('entry')?`<div class="sep"></div><button onclick="flipIn()">Flip in/out</button>`:'')+
  `<div class="sep"></div><button onclick="clearGeo()">Clear this</button>
   <span class="muted" style="margin-left:4px">${PICK?'click on the picture':'pick a tool'}</span>`}
function setPick(m){const c=selObj();PICK=PICK===m?null:m;if(PICK)c[PICK]=[];dirty();modeBtns(c);ovDraw()}
function flipIn(){const c=selObj();c.in_side=(c.in_side===1?-1:1);dirty();ovDraw()}
function clearGeo(){const c=selObj();const m=PICK||camModes(c)[0];delete c[m];dirty();modeBtns(c);ovDraw()}
function sizeOv(){const img=$('feed'),ov=$('ov');if(!img||!ov)return;
 const w=img.clientWidth,h=img.clientHeight;      // MJPEG never fires a useful load event, so poll
 if(w&&h&&(ov.width!==w||ov.height!==h)){ov.width=w;ov.height=h;
  ov.style.width=w+'px';ov.style.height=h+'px';ovDraw()}}
setInterval(sizeOv,400);
function ovDraw(){const ov=$('ov');if(!ov)return;const X=ov.getContext('2d'),W=ov.width,H=ov.height;
 X.clearRect(0,0,W,H);const c=selObj();if(!c||!SEL||SEL.t!=='c')return;
 const P=p=>[p[0]*W,p[1]*H];
 if(c.floor_quad&&c.floor_quad.length){X.strokeStyle='#ff3b52';X.setLineDash([6,4]);X.lineWidth=2;
  X.fillStyle='rgba(228,0,43,.16)';X.beginPath();c.floor_quad.forEach((p,i)=>{const q=P(p);
   i?X.lineTo(q[0],q[1]):X.moveTo(q[0],q[1])});if(c.floor_quad.length>2)X.closePath();
  if(c.floor_quad.length>2)X.fill();X.stroke();X.setLineDash([])}
 if(c.queue_zone&&c.queue_zone.length){X.strokeStyle='#d9a441';X.lineWidth=2;X.fillStyle='rgba(217,164,65,.16)';
  X.beginPath();c.queue_zone.forEach((p,i)=>{const q=P(p);i?X.lineTo(q[0],q[1]):X.moveTo(q[0],q[1])});
  if(c.queue_zone.length>2){X.closePath();X.fill()}X.stroke()}
 if(c.entry_line&&c.entry_line.length===2){const a=P(c.entry_line[0]),b=P(c.entry_line[1]);
  X.strokeStyle='#3fb950';X.lineWidth=3;X.beginPath();X.moveTo(a[0],a[1]);X.lineTo(b[0],b[1]);X.stroke();
  const mx=(a[0]+b[0])/2,my=(a[1]+b[1])/2,sg=c.in_side===1?1:-1;
  let nx=-(b[1]-a[1]),ny=b[0]-a[0];const L=Math.hypot(nx,ny)||1;nx=nx/L*sg*32;ny=ny/L*sg*32;
  X.beginPath();X.moveTo(mx,my);X.lineTo(mx+nx,my+ny);X.stroke();
  X.beginPath();X.moveTo(mx+nx*1.25,my+ny*1.25);X.lineTo(mx+nx-ny*.25,my+ny+nx*.25);
  X.lineTo(mx+nx+ny*.25,my+ny-nx*.25);X.closePath();X.fillStyle='#3fb950';X.fill();
  X.font='700 12px system-ui';X.fillText('IN',mx+nx*1.6-7,my+ny*1.6+4)}
 for(const k in NEED){if(!c[k])continue;
  X.fillStyle=k==='entry_line'?'#3fb950':k==='queue_zone'?'#d9a441':'#ff3b52';
  for(const p of c[k]){const q=P(p);X.beginPath();X.arc(q[0],q[1],5,0,6.3);X.fill();
   X.strokeStyle='#0f1114';X.lineWidth=1.5;X.stroke()}}}
/* ---------- 3D ---------- */
/* One renderer, two views of the store: Setup shows everything (cameras and what they see), Home shows a
   medium-sized model with only the shelf and fixture names. Heat is a surface that rises where people have been. */
let YAW=-0.62,PIT=0.92,ZOOM=1;
const V3={v3d:{yaw:-0.62,pit:0.92,zoom:1},h3d:{yaw:-0.62,pit:1.0,zoom:1.18}};
function raw(p){const ca=Math.cos(YAW),sa=Math.sin(YAW),cp=Math.cos(PIT),sp=Math.sin(PIT);
 const x=p[0]-LAY.store.w/2,y=p[1]-LAY.store.h/2,z=p[2];
 const X=x*ca-y*sa,Y=x*sa+y*ca;
 return [X,-(Y*sp+z*cp)*0.72,Y*cp-z*sp]}
let VX=0,VY=0,VS=60;
function proj(p){const r=raw(p);return [VX+r[0]*VS,VY+r[1]*VS,r[2]]}
function fit3d(W,H,zmin){const zs=[0,...LAY.shelves.map(s=>s.height||1.8),...LAY.cameras.map(c=>c.height||2.2),...FIX().map(f=>f.height||1)];
 const zmax=Math.max(...zs);let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
 for(const X of [0,LAY.store.w])for(const Y of [0,LAY.store.h])for(const Z of [0,zmax]){
  const r=raw([X,Y,Z]);x0=Math.min(x0,r[0]);x1=Math.max(x1,r[0]);y0=Math.min(y0,r[1]);y1=Math.max(y1,r[1])}
 VS=Math.min(W/(x1-x0+0.6),H/(y1-y0+0.6))*ZOOM;
 VX=W/2-(x0+x1)/2*VS;VY=H/2-(y0+y1)/2*VS}
const HSTOPS=[[0,[38,110,200]],[.25,[30,190,170]],[.5,[235,200,60]],[.75,[250,120,40]],[1,[255,50,70]]];
function hcol(v,a){v=Math.max(0,Math.min(1,v));let i=0;while(i<HSTOPS.length-2&&v>HSTOPS[i+1][0])i++;
 const [t0,c0]=HSTOPS[i],[t1,c1]=HSTOPS[i+1],f=(v-t0)/(t1-t0);
 return `rgba(${c0.map((c,k)=>Math.round(c+(c1[k]-c)*f)).join(',')},${a===undefined?(0.2+0.7*v).toFixed(2):a})`}
function draw3d(id,o){id=id||'v3d';o=o||(id==='v3d'?{cams:true}:{cams:false});
 const cv=$(id);if(!cv)return;const st=V3[id];YAW=st.yaw;PIT=st.pit;ZOOM=st.zoom;
 const W=id==='v3d'?1320:980,H=id==='v3d'?620:600;if(cv.width!==W){cv.width=W;cv.height=H}
 const X=cv.getContext('2d');X.clearRect(0,0,W,H);X.fillStyle='#0f1114';X.fillRect(0,0,W,H);
 fit3d(W,H);
 const polys=[];
 const push=(pts,fill,stroke,lw,dash)=>{const pr=pts.map(p=>proj(p));
  polys.push({pr,fill,stroke,lw:lw||1,dash:dash||null,d:pr.reduce((a,b)=>a+b[2],0)/pr.length})};
 for(let x=0;x<=LAY.store.w+1e-6;x+=1)push([[x,0,0],[x,LAY.store.h,0]],null,'#23272e',1);
 for(let y=0;y<=LAY.store.h+1e-6;y+=1)push([[0,y,0],[LAY.store.w,y,0]],null,'#23272e',1);
 push([[0,0,0],[LAY.store.w,0,0],[LAY.store.w,LAY.store.h,0],[0,LAY.store.h,0]],null,'#454b55',2);
 if(HEAT&&HEAT.length){                      // heat as a surface: cells rise with how busy they are, corners are averaged so it is smooth
  const R=HEAT.length,C=HEAT[0].length,cw=LAY.store.w/C,ch=LAY.store.h/R,HM=0.75;
  const hv=(r,c)=>HEAT[Math.max(0,Math.min(R-1,r))][Math.max(0,Math.min(C-1,c))];
  const vh=(r,c)=>(hv(r-1,c-1)+hv(r-1,c)+hv(r,c-1)+hv(r,c))/4;
  for(let r=0;r<R;r++)for(let c=0;c<C;c++){const v=HEAT[r][c];
   const z=[vh(r,c),vh(r,c+1),vh(r+1,c+1),vh(r+1,c)];if(v<=0.02&&Math.max(...z)<=0.02)continue;
   const col=hcol((v+z[0]+z[1]+z[2]+z[3])/5);
   push([[c*cw,r*ch,z[0]*HM+.002],[(c+1)*cw,r*ch,z[1]*HM+.002],[(c+1)*cw,(r+1)*ch,z[2]*HM+.002],[c*cw,(r+1)*ch,z[3]*HM+.002]],col,col,0.8)}}
 if(o.cams)for(const c of LAY.cameras){const half=c.fov*Math.PI/360,hd=c.heading*Math.PI/180,z=c.height||2.2;
  const arc=[];for(let i=0;i<=16;i++){const a=hd-half+2*half*i/16;
   arc.push(clip([c.x,c.y],[c.x+Math.cos(a)*c.range,c.y+Math.sin(a)*c.range]))}
  push([[c.x,c.y,z],[arc[0][0],arc[0][1],0]],null,'rgba(228,0,43,.6)',1,[4,4]);
  push([[c.x,c.y,z],[arc[16][0],arc[16][1],0]],null,'rgba(228,0,43,.6)',1,[4,4]);
  push(arc.map(p=>[p[0],p[1],0]),null,'rgba(228,0,43,.6)',1,[4,4]);
  push([[c.x,c.y,0],[c.x,c.y,z]],null,'rgba(228,0,43,.45)',1);}
 for(const s of LAY.shelves){const c=corners(s),z=s.height||1.8;
  push(c.map(p=>[p[0],p[1],z]),'#2b3037','#565d68',1);
  FACES.forEach((fc,i)=>{const a=c[i],b=c[(i+1)%4];
   const st=s.faces[fc]?(S&&S.faceStatus?S.faceStatus[s.id+':'+fc]:null):null;
   const col=!s.faces[fc]?'#22262c':(st==='EMPTY'?'#ff4d4f':st==='LOW'?'#d9a441':st==='MISPLACED'?'#a371f7':
    st==='OK'?'#3fb950':'#3a4048');
   push([[a[0],a[1],0],[b[0],b[1],0],[b[0],b[1],z],[a[0],a[1],z]],col,'#565d68',1)})}
 for(const f of FIX()){const c=corners(f);
  if(f.kind==='door'){push(c.map(p=>[p[0],p[1],0.004]),'rgba(63,185,80,.28)','#3fb950',1.5,[6,4]);continue}
  const z=f.height||1,top=f.kind==='counter'?'#2a3a52':'#3a352c',side=f.kind==='counter'?'#223047':'#2e2a23';
  push(c.map(p=>[p[0],p[1],z]),top,'#5d6573',1);
  for(let i=0;i<4;i++){const a=c[i],b=c[(i+1)%4];push([[a[0],a[1],0],[b[0],b[1],0],[b[0],b[1],z],[a[0],a[1],z]],side,'#5d6573',1)}}
 polys.sort((a,b)=>b.d-a.d);
 for(const p of polys){X.beginPath();p.pr.forEach((q,i)=>i?X.lineTo(q[0],q[1]):X.moveTo(q[0],q[1]));
  if(p.fill){X.closePath();X.fillStyle=p.fill;X.fill()}
  if(p.stroke){X.setLineDash(p.dash||[]);X.strokeStyle=p.stroke;X.lineWidth=p.lw;X.stroke();X.setLineDash([])}}
 X.font=(id==='v3d'?'600 12px':'600 13px')+' system-ui';X.textAlign='center';
 const tag=(txt,q,col)=>{X.strokeStyle='#0f1114';X.lineWidth=3.5;X.strokeText(txt,q[0],q[1]);X.fillStyle=col;X.fillText(txt,q[0],q[1])};
 for(const s of LAY.shelves)tag(s.name,proj([s.x,s.y,(s.height||1.8)+0.14]),'#c3c9d2');
 for(const f of FIX()){if(f.kind==='door'&&!o.cams)continue;
  tag(f.name,proj([f.x,f.y,(f.kind==='door'?0:f.height||1)+0.14]),f.kind==='door'?'#7ee2a0':f.kind==='counter'?'#a9c6ee':'#c9c0ad')}
 if(o.cams)for(const c of LAY.cameras){const q=proj([c.x,c.y,c.height||2.2]);
  X.beginPath();X.arc(q[0],q[1],4.5,0,6.3);X.fillStyle='#ff3b52';X.fill();
  X.textAlign='left';X.strokeStyle='#0f1114';X.lineWidth=3.5;X.strokeText(c.name,q[0]+9,q[1]+4);
  X.fillStyle='#ff6b7d';X.fillText(c.name,q[0]+9,q[1]+4);X.textAlign='center'}
 X.textAlign='left'}
function clip(p,q){const W=LAY.store.w,H=LAY.store.h;let t=1;const dx=q[0]-p[0],dy=q[1]-p[1];
 for(const [num,den] of [[p[0],-dx],[W-p[0],dx],[p[1],-dy],[H-p[1],dy]]){
  if(Math.abs(den)<1e-9){if(num<0)return p;continue}
  if(den>0)t=Math.min(t,Math.max(0,num/den))}
 return [p[0]+dx*t,p[1]+dy*t]}
['v3d','h3d'].forEach(id=>{const cv=$(id);if(!cv)return;let d=null;const st=V3[id],home={...st};
 cv.addEventListener('pointerdown',e=>{d=[e.clientX,e.clientY];cv.style.cursor='grabbing';cv.setPointerCapture(e.pointerId)});
 cv.addEventListener('pointermove',e=>{if(!d)return;st.yaw+=(e.clientX-d[0])*0.01;
  st.pit=Math.max(0.12,Math.min(1.48,st.pit-(e.clientY-d[1])*0.008));d=[e.clientX,e.clientY];draw3d(id)});
 addEventListener('pointerup',()=>{d=null;cv.style.cursor='grab'});
 cv.addEventListener('wheel',e=>{e.preventDefault();st.zoom=Math.max(0.4,Math.min(3,st.zoom*(e.deltaY>0?0.92:1.09)));draw3d(id)},{passive:false});
 cv.addEventListener('dblclick',()=>{Object.assign(st,home);draw3d(id)})});
/* ---------- tabs ---------- */
const TABS=['home','shelves','checkout','queue','cctv','analytics','setup'];
let TABV='home';
function tab(t){if(!TABS.includes(t))t='home';TABV=t;
 for(const x of TABS)$(x).style.display=x===t?'grid':'none';
 document.querySelectorAll('#tabs .tab').forEach(el=>el.classList.toggle('on',el.dataset.t===t));
 try{history.replaceState(null,'','#'+t)}catch(e){}
 streams();
 if(t==='setup'){draw();draw3d();intStatus()}
 if(t==='home'){drawMini();if(S)homeCharts(S)}
 if(t==='shelves')shEnter();
 if(t==='checkout')coEnter();
 if(t==='queue'){qLoad();if(S)qRender(S)}
 if(t==='analytics'){anLoad();liveSeed()}
 if(t==='cctv')cctvBuild()}
document.querySelectorAll('#tabs .tab').forEach(el=>el.onclick=()=>tab(el.dataset.t));
/* live video only streams on the tab you are looking at — each MJPEG feed costs the edge box */
function streams(){}   // live pictures pace themselves (liveTick) and pause when hidden
/* live pictures: every <img class="live" data-cam data-view> asks for its next frame only after the last
   one has arrived. A slow link drops frames instead of falling behind, nothing holds a connection open
   (browsers allow only 6 per site), and hidden tabs make no requests at all. */
function liveTick(){document.querySelectorAll('img.live').forEach(im=>{
 if(im._busy||!im.isConnected||im.offsetParent===null)return;
 im._busy=true;im.onload=im.onerror=()=>{im._busy=false};
 im.src='/api/frame/'+encodeURIComponent(im.dataset.cam)+'.jpg?view='+(im.dataset.view||'analytics')+'&t='+Date.now()})}
setInterval(liveTick,60);

/* ---------- chart kit (dark surface, validated pair red #e8364f / slate #5b8fd1) ---------- */
const CC={s1:'#e8364f',s2:'#5b8fd1',rest:'#454b55',grid:'#20242a',axis:'#343941'};
const TT=document.createElement('div');TT.className='tt';document.body.appendChild(TT);
function tipAt(e,h){TT.innerHTML=h;TT.style.display='block';const w=TT.offsetWidth,hh=TT.offsetHeight;
 let x=e.clientX+14,y=e.clientY+14;if(x+w>innerWidth-8)x=e.clientX-w-14;if(y+hh>innerHeight-8)y=e.clientY-hh-14;
 TT.style.left=x+'px';TT.style.top=y+'px'}
function untip(){TT.style.display='none'}
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function nice(m){if(!(m>0))return 1;const p=Math.pow(10,Math.floor(Math.log10(m))),f=m/p;
 return (f<=1?1:f<=2?2:f<=2.5?2.5:f<=5?5:10)*p}
const fmtN=v=>v==null?'—':Math.abs(v)>=1e5?(v/1e5).toFixed(1).replace(/\.0$/,'')+'L':
 Math.abs(v)>=1000?(v/1000).toFixed(1).replace(/\.0$/,'')+'k':(+v).toLocaleString('en-IN',{maximumFractionDigits:1});
const fmtR=v=>v==null?'—':'₹'+Math.round(v).toLocaleString('en-IN');
const fmtRs=v=>v==null?'—':'₹'+fmtN(v);
const fmtP=v=>v==null?'—':Math.round(v*100)+'%';
function barPath(x,y,w,h){const r=Math.min(4,w/2,h);
 return `M${x},${y+h}V${y+r}Q${x},${y} ${x+r},${y}H${x+w-r}Q${x+w},${y} ${x+w},${y+r}V${y+h}Z`}
function chartShell(el,o,draw,table){          // every chart can flip to a table (a11y / exact values)
 el._o=o;el._draw=draw;el._table=table;
 const h3=el.closest('.card')&&el.closest('.card').querySelector('h3');
 if(h3&&!h3.querySelector('.ttog')){const t=document.createElement('span');t.className='ttog';t.textContent='table';
  t.onclick=()=>{el._tab=!el._tab;t.textContent=el._tab?'chart':'table';rerender(el)};
  (h3.querySelector('.sub')||h3).appendChild(t)}
 rerender(el)}
function rerender(el){if(el._tab){const t=el._table(el._o);
  el.innerHTML=`<div class="tbl"><table><tr>${t.cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr>`+
  t.rows.map(r=>`<tr>${r.map(c=>`<td>${esc(c)}</td>`).join('')}</tr>`).join('')+`</table></div>`}
 else el._draw(el,el._o)}
function ticks(raw,int,fixed){if(fixed)return {mx:fixed,n:4};if(int)raw=Math.max(raw,4);if(!(raw>0))raw=1;
 const p=Math.pow(10,Math.floor(Math.log10(raw/4)));
 for(const k of [1,2,2.5,5,10,20]){const st=k*p;if(int&&(st<1||!Number.isInteger(st)))continue;
  const n=Math.ceil(raw/st-1e-9);if(n<=5)return {mx:n*st,n:Math.max(1,n)}}return {mx:raw,n:4}}
const lblEvery=(labels,iw)=>{const L=Math.max(...labels.map(l=>String(l).length));
 return Math.max(1,Math.ceil(labels.length/Math.max(1,Math.floor(iw/(L*7.6+18)))))};
function yAxis(m,W,ih,mx,fmt,n){n=n||4;let g='';for(let k=0;k<=n;k++){const v=mx*k/n,y=m.t+ih-ih*k/n;
 g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y}" y2="${y}" stroke="${k?CC.grid:CC.axis}" stroke-width="1"/>`+
  `<text class="ax" x="${m.l-8}" y="${y+4}" text-anchor="end">${(fmt||fmtN)(v)}</text>`}return g}
function barChart(el,o){chartShell(el,o,(el,o)=>{
 const W=Math.max(260,el.clientWidth||600),H=o.h||200,m={l:54,r:8,t:10,b:28},n=o.values.length;
 const iw=W-m.l-m.r,ih=H-m.t-m.b,intish=o.values.every(v=>Number.isInteger(v||0));
 const T=ticks(Math.max(0,...o.values.map(v=>v||0))*1.02,intish),mx=T.mx;
 const step=iw/Math.max(1,n),bw=Math.max(2,Math.min(40,step-2)),Y=v=>m.t+ih-(v/mx)*ih;
 const every=lblEvery(o.labels,iw);
 let g=yAxis(m,W,ih,mx,o.yfmt,T.n);
 o.values.forEach((v,i)=>{const x=m.l+i*step+(step-bw)/2,top=Y(v||0),hh=m.t+ih-top;
  const col=o.hi?(o.hi(i)?CC.s1:CC.rest):(o.color||CC.s1);
  if(hh>0.5)g+=`<path d="${barPath(x,top,bw,hh)}" fill="${col}"/>`;
  if(i%every===0)g+=`<text class="ax" x="${x+bw/2}" y="${H-7}" text-anchor="middle">${esc(o.labels[i])}</text>`;
  g+=`<rect data-i="${i}" x="${m.l+i*step}" y="${m.t}" width="${step}" height="${ih}" fill="transparent"/>`});
 el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${g}</svg>`;
 el.querySelectorAll('rect[data-i]').forEach(r=>{r.onmousemove=e=>{const i=+r.dataset.i;
  tipAt(e,`<b>${esc(o.tlabels?o.tlabels[i]:o.labels[i])}</b><div class="r"><span>${esc(o.name||'')}</span>
   <span>${(o.fmt||fmtN)(o.values[i])}</span></div>`)};r.onmouseleave=untip})},
 o=>({cols:[o.xname||'',o.name||'value'],rows:o.labels.map((l,i)=>[o.tlabels?o.tlabels[i]:l,(o.fmt||fmtN)(o.values[i])])}))}
function lineChart(el,o){chartShell(el,o,(el,o)=>{
 const multi=o.series.length>1,W=Math.max(260,el.clientWidth||600),H=o.h||220;
 const m={l:54,r:multi?84:18,t:12,b:28},n=o.labels.length,iw=W-m.l-m.r,ih=H-m.t-m.b;
 const all=o.series.flatMap(s=>s.values.filter(v=>v!=null));
 const T=ticks(Math.max(0,...all)*1.04,all.every(v=>Number.isInteger(v)),o.max),mx=T.mx;
 const X=i=>m.l+(n<2?iw/2:i*iw/(n-1)),Y=v=>m.t+ih-(v/mx)*ih;
 const every=lblEvery(o.labels,iw);
 let g=yAxis(m,W,ih,mx,o.yfmt,T.n);
 // every n-th label, and the last one — dropping the regular label just before it when they'd touch
 o.labels.forEach((l,i)=>{const last=i===n-1,show=last||(i%every===0&&(n-1-i)>=every*0.75);
  if(show)g+=`<text class="ax" x="${X(i)}" y="${H-7}" text-anchor="${last&&n>1?'end':'middle'}">${esc(l)}</text>`});
 const ends=[];
 for(const s of o.series){let d='',pen=false;s.values.forEach((v,i)=>{if(v==null){pen=false;return}
   d+=(pen?'L':'M')+X(i).toFixed(1)+','+Y(v).toFixed(1);pen=true});
  g+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  if(multi){let li=s.values.length-1;while(li>0&&s.values[li]==null)li--;
   if(s.values[li]!=null){g+=`<circle cx="${X(li)}" cy="${Y(s.values[li])}" r="3.5" fill="${s.color}"/>`;
    ends.push({x:X(li)+9,y:Y(s.values[li])+4,name:s.name})}}}
 ends.sort((a,b)=>a.y-b.y);for(let k=1;k<ends.length;k++)if(ends[k].y-ends[k-1].y<16)ends[k].y=ends[k-1].y+16;   // direct labels never overlap
 for(const e of ends)g+=`<text class="lab" x="${e.x}" y="${e.y}">${esc(e.name)}</text>`;
 g+=`<g class="xh"></g><rect class="hit" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/>`;
 el.innerHTML=(multi?`<div class="legend">${o.series.map(s=>`<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('')}</div>`:'')+
  `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${g}</svg>`;
 const svg=el.querySelector('svg'),xh=el.querySelector('.xh'),hit=el.querySelector('.hit');
 hit.onmousemove=e=>{const r=svg.getBoundingClientRect(),px=(e.clientX-r.left)*W/r.width;
  const i=Math.max(0,Math.min(n-1,Math.round((px-m.l)/(n<2?1:iw/(n-1)))));
  xh.innerHTML=`<line x1="${X(i)}" x2="${X(i)}" y1="${m.t}" y2="${m.t+ih}" stroke="#5d636e" stroke-width="1"/>`+
   o.series.filter(s=>s.values[i]!=null).map(s=>`<circle cx="${X(i)}" cy="${Y(s.values[i])}" r="4.5" fill="${s.color}" stroke="#141619" stroke-width="2"/>`).join('');
  tipAt(e,`<b>${esc(o.tlabels?o.tlabels[i]:o.labels[i])}</b>`+o.series.map(s=>
   `<div class="r"><span><i style="background:${s.color}"></i>${esc(s.name)}</span><span>${(o.fmt||fmtN)(s.values[i])}</span></div>`).join(''))};
 hit.onmouseleave=()=>{xh.innerHTML='';untip()}},
 o=>({cols:['',...o.series.map(s=>s.name)],rows:o.labels.map((l,i)=>[o.tlabels?o.tlabels[i]:l,...o.series.map(s=>(o.fmt||fmtN)(s.values[i]))])}))}
function hbar(el,o){chartShell(el,o,(el,o)=>{
 if(!o.items.length){el.innerHTML='<div class="empty">No data in this range.</div>';return}
 const mx=Math.max(...o.items.map(i=>i.value))||1;
 el.innerHTML=o.items.map((it,k)=>`<div class="hb" data-k="${k}" style="display:grid;grid-template-columns:minmax(120px,38%) 1fr 70px;
  gap:10px;align-items:center;padding:5px 0"><span style="font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(it.label)}</span>
  <span style="height:14px;display:block"><span style="display:block;height:14px;width:${Math.max(1,100*it.value/mx)}%;
   background:${o.color||CC.s1};border-radius:0 4px 4px 0"></span></span>
  <span style="text-align:right;font-size:12.5px;color:var(--mute)">${(o.fmt||fmtN)(it.value)}</span></div>`).join('');
 el.querySelectorAll('.hb').forEach(r=>{r.onmousemove=e=>{const it=o.items[+r.dataset.k];
  tipAt(e,`<b>${esc(it.label)}</b>`+(it.rows||[[o.name||'value',(o.fmt||fmtN)(it.value)]]).map(x=>
   `<div class="r"><span>${esc(x[0])}</span><span>${esc(x[1])}</span></div>`).join(''))};r.onmouseleave=untip})},
 o=>({cols:['',o.name||'value'],rows:o.items.map(i=>[i.label,(o.fmt||fmtN)(i.value)])}))}
const RAMP=[[30,20,23],[76,20,32],[140,24,42],[200,32,54],[240,62,82],[255,138,152]];
function ramp(t){t=Math.max(0,Math.min(1,t))*(RAMP.length-1);const i=Math.min(RAMP.length-2,Math.floor(t)),f=t-i;
 return `rgb(${RAMP[i].map((c,k)=>Math.round(c+(RAMP[i+1][k]-c)*f)).join(',')})`}
function heat(el,o){chartShell(el,o,(el,o)=>{
 const W=Math.max(360,el.clientWidth||800),rows=o.rows.length,cols=o.cols.length,m={l:40,r:8,t:6,b:42};
 const cw=(W-m.l-m.r)/cols,ch=Math.min(26,Math.max(16,cw*0.8)),H=m.t+rows*ch+m.b;
 const flat=o.m.flat().filter(v=>v>0).sort((x,y)=>x-y),mx=Math.max(...o.m.flat())||1;
 const cap=flat.length>10?Math.max(flat[Math.floor(flat.length*0.96)],mx*0.25):mx;   /* the top 4% share the brightest colour: one outlier can't wash the rest out */
 let g='';
 o.m.forEach((row,r)=>{g+=`<text class="ax" x="${m.l-8}" y="${m.t+r*ch+ch/2+4}" text-anchor="end">${o.rows[r]}</text>`;
  row.forEach((v,c)=>{g+=`<rect data-r="${r}" data-c="${c}" x="${m.l+c*cw+1}" y="${m.t+r*ch+1}" width="${Math.max(1,cw-2)}"
   height="${ch-2}" rx="3" fill="${v>0?ramp(Math.pow(Math.min(1,v/cap),0.8)):'#171a1e'}"/>`})});
 o.cols.forEach((c,i)=>{if(i%2===0)g+=`<text class="ax" x="${m.l+i*cw+cw/2}" y="${m.t+rows*ch+14}" text-anchor="middle">${c}</text>`});
 const lx=W-m.r-180,ly=H-14;
 g+=`<defs><linearGradient id="hg${el.id}">${RAMP.map((c,i)=>`<stop offset="${i/(RAMP.length-1)}" stop-color="rgb(${c})"/>`).join('')}</linearGradient></defs>
  <text class="ax" x="${lx-8}" y="${ly+4}" text-anchor="end">0</text><rect x="${lx}" y="${ly-5}" width="150" height="9" rx="3" fill="url(#hg${el.id})"/>
  <text class="ax" x="${lx+158}" y="${ly+4}">${fmtN(cap)}${cap<mx?'+':''}</text>`;
 el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${g}</svg>`;
 el.querySelectorAll('rect[data-r]').forEach(r=>{r.onmousemove=e=>{const a=+r.dataset.r,b=+r.dataset.c;
  tipAt(e,`<b>${o.rows[a]} ${o.cols[b]}:00</b><div class="r"><span>${esc(o.name)}</span><span>${(o.fmt||fmtN)(o.m[a][b])}</span></div>`)};
  r.onmouseleave=untip})},
 o=>({cols:['',...o.cols.map(c=>c+':00')],rows:o.m.map((r,i)=>[o.rows[i],...r.map(v=>fmtN(v))])}))}
let RSZ=null;addEventListener('resize',()=>{clearTimeout(RSZ);RSZ=setTimeout(()=>{
 document.querySelectorAll('main#'+TABV+' .chart').forEach(el=>{if(el._draw)rerender(el)})},150)});

/* ---------- home ---------- */
let ctrSet=false,poll=null;
function startPoll(){if(poll)return;$('conn').innerHTML='<span class="dot"></span>live';
 poll=setInterval(async()=>{try{render(await(await fetch('/api/state')).json())}catch(e){}},1000)}
function connect(){let open=false;const ws=new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+'/ws');
 ws.onopen=()=>{open=true;if(poll){clearInterval(poll);poll=null}$('conn').innerHTML='<span class="dot"></span>live'};
 ws.onmessage=e=>render(JSON.parse(e.data));
 ws.onclose=()=>{if(!open)return startPoll();$('conn').textContent='reconnecting…';setTimeout(connect,2000)}}
connect();
const kpi=(l,v,d,cls)=>`<div class="kpi ${cls||''}"><div class="v">${v}</div><div class="l">${l}</div>${d?`<div class="d">${d}</div>`:''}</div>`;
let ASIG='';
const ago=t=>{const s=Math.round(Date.now()/1000-t);return s<60?s+'s ago':s<3600?Math.round(s/60)+'m ago':Math.round(s/3600)+'h ago'};
function allCells(s){const out=[];for(const cam in s.shelves)for(const c of (s.shelves[cam]||[]))out.push({...c,cam});return out}
function render(s){S=s;
 $('store').textContent=s.store_id;
 $('cloud').textContent=s.cloud_online==null?'edge only':s.cloud_online?'cloud synced':'offline · buffering';
 if(!ctrSet){$('ctr').value=s.open_counters;ctrSet=true}
 HEATDAY=s.store_heat;HEATNOW=s.store_live;HEAT=HEATMODE==='now'?HEATNOW:HEATDAY;
 const approx=Object.entries(s.heat_src||{}).filter(([,v])=>v==='approx').map(([k])=>((s.labels||{})[k]||k));
 if($('hnote'))$('hnote').textContent=approx.length?`${approx.join(', ')}: positions estimated from where the camera is placed — add floor points in Setup for exact`:'';
 if(s.layout&&!DIRTY)LAY=s.layout;
 s.faceStatus={};for(const cam in s.shelves){const cells=s.shelves[cam];if(!cells)continue;
  const w=(s.cam_face||{})[cam];if(!w)continue;let st='OK';
  for(const c of cells){if(c.status==='EMPTY'){st='EMPTY';break}if(c.status==='LOW')st='LOW';
   else if(c.status==='MISPLACED'&&st==='OK')st='MISPLACED'}s.faceStatus[w]=st}
 const cells=allCells(s),low=cells.filter(c=>c.status==='LOW').length,emp=cells.filter(c=>c.status==='EMPTY').length;
 const q=Object.values(s.queues),wait=q.length?Math.max(...q.map(x=>x.wait_min)):0;
 $('kpis').innerHTML=kpi('Inside now',s.footfall.inside,'in the store')+kpi('Entries today',s.footfall.in,`${s.footfall.out} left`)+
  kpi('Sales today',fmtR(s.sales_today||0),`${s.bills_today||0} bill${s.bills_today===1?'':'s'}`)+
  kpi('Conversion',s.conversion==null?'—':fmtP(s.conversion),'bills ÷ entries')+
  kpi('Queue wait',wait.toFixed(1)+' min',`${q.reduce((a,x)=>a+x.length,0)} in line`,wait>=4?'warn':'')+
  kpi('Low / empty',`${low} / ${emp}`,'products',emp?'bad':low?'warn':'')+
  kpi('Open alerts',s.alerts.length,'to act on',s.alerts.some(a=>a.severity==='critical')?'bad':'')+
  kpi('Staff response',s.avg_response_s==null?'—':Math.round(s.avg_response_s)+'s','average today');
 const sv={critical:0,warning:0,info:0};s.alerts.forEach(a=>sv[a.severity]=(sv[a.severity]||0)+1);
 $('acount').innerHTML=(sv.critical?`<span class="c">${sv.critical} critical</span>`:'')+
  (sv.warning?`<span class="w">${sv.warning} warning</span>`:'')+(sv.info?`<span class="i">${sv.info} info</span>`:'');
 const asig=s.alerts.map(a=>a.id+':'+a.ts+':'+a.count).join('|')+Math.floor(Date.now()/60000);
 if(asig!==ASIG){ASIG=asig;
  $('alerts').innerHTML=s.alerts.length?s.alerts.slice(0,60).map(a=>`<div class="alert ${a.severity}"><div class="m">
   <div class="top"><span class="a">${esc(a.action)}</span><span class="t">${ago(a.ts)}${a.count>1?' · '+a.count+'×':''}</span></div>
   <div class="msg">${esc(a.message)}</div></div>
   <button onclick="ack(${a.id})">Done</button></div>`).join(''):'<div class="empty">All clear — nothing needs doing right now.</div>'}
 $('queues').innerHTML=Object.entries(s.queues).map(([cam,x])=>`
  <div class="qbig"><b>${x.length}</b><span>in line · ${x.wait_min} min wait</span></div>
  <div class="muted" style="margin-bottom:8px">Forecast</div>
  <div class="fc">${['5','10','15'].map(t=>`<div><b>${x.forecast[t].len}</b><small>in ${t} min · ${x.forecast[t].wait}m</small></div>`).join('')}</div>
  <div class="rec">Open <b>${x.recommend_counters}</b> counter${x.recommend_counters===1?'':'s'}
   <div class="muted">${x.arrival_per_min}/min arriving · ${x.service_per_counter_min}/min served each</div></div>`).join('')
  ||'<div class="empty">No queue camera yet.<br><span class="muted">In Setup, give a camera the “watch queue” job.</span></div>';
 const lows=cells.filter(c=>c.slot&&(c.status==='LOW'||c.status==='EMPTY'))
  .sort((a,b)=>(a.status==='EMPTY'?0:1)-(b.status==='EMPTY'?0:1)||a.est_units/a.full_units-b.est_units/b.full_units);
 $('lowstock').innerHTML=lows.length?lows.slice(0,40).map(c=>`<div class="lowrow"><span class="st ${c.status}">${c.status}</span>
  <div><div class="n">${esc(c.name)}</div><div class="w">${esc((s.labels||{})[c.cam]||c.cam)} · ${esc(c.loc)}</div></div>
  <div class="u">${cnt(c)}<small> / ${c.full_units}</small></div></div>`).join('')
  :cells.some(c=>c.slot)?'<div class="empty">Every marked product is stocked.</div>':'<div class="empty">No products marked yet.<br><span class="muted">Open Shelves and draw a box round each product.</span></div>';
 const z=Object.entries((s.zones||{})[Object.keys(s.zones||{})[0]]||{});
 $('zones').innerHTML=z.length?`<table><tr><th>Zone</th><th>Visits</th><th>Avg dwell</th></tr>`+
  z.map(([n,v])=>`<tr><td>${esc(n)}</td><td>${v.visits}</td><td>${v.avg_dwell_s}s</td></tr>`).join('')+`</table>`:'';
 if(TABV==='home'){drawMini();homeCharts(s)}
 if(TABV==='setup'&&!DRAG)draw3d();
 if(TABV==='shelves'){shTable();shOv();shDepthLine();if(SH.view==='depth'&&s.depth_model_err)shImg()}
 if(TABV==='cctv'&&Object.keys(S.cams||{}).join('|')!==CCTVSIG)cctvBuild();   // a camera joined or left
 if(TABV==='setup'&&$('camfeed')&&SEL&&SEL.t==='c')$('camfeed').innerHTML=feedLine(selObj());
 if(TABV==='cctv')cctvBadges();
 coSync(s);if(TABV==='checkout')shopDraw();qRender(s);liveSample(s)}
let HOMEKEY='';
function homeCharts(s){const key=JSON.stringify(s.hourly);if(key===HOMEKEY&&$('c_today')._draw)return;HOMEKEY=key;
 const hrs=[];for(let h=7;h<=22;h++)hrs.push(h);const now=s.hour_now==null?new Date().getHours():s.hour_now;
 barChart($('c_today'),{labels:hrs.map(h=>h+''),tlabels:hrs.map(h=>`${h}:00–${h+1}:00`),xname:'hour',
  values:hrs.map(h=>s.hourly[String(h).padStart(2,'0')]||0),name:'entries',hi:i=>hrs[i]===now,h:300})}

/* ---------- shelves: mark products on the calibrated picture ---------- */
let SH={cam:null,slots:[],sel:null,mode:null,drag:null,dirty:false},PRODS=[];
async function prodsLoad(){PRODS=(await(await fetch('/api/products')).json()).products}
function shelfCams(){return (LAY.cameras||[]).filter(c=>(c.roles||[]).includes('shelf'))}
async function shEnter(){const cams=shelfCams(),sel=$('shcam');
 sel.innerHTML=cams.map(c=>`<option value="${esc(c.id)}">${esc(c.name)} (${esc(c.id)})</option>`).join('')
  ||'<option value="">no shelf cameras — add one in Setup</option>';
 await prodsLoad();catalog();
 if(cams.length){const want=SH.cam&&cams.find(c=>c.id===SH.cam)?SH.cam:cams[0].id;sel.value=want;
  if(want!==SH.cam||!SH.slots.length)await shLoad(want);else{shSide();shTable();shOv()}}}
async function shLoad(cam){if(!cam)return;SH={cam,slots:[],sel:null,mode:null,drag:null,dirty:false,view:SH.view||'live'};
 const r=await fetch('/api/slots/'+cam);if(r.ok){const j=await r.json();
  SH.slots=j.slots.map(({row,col,product,...rest})=>rest);SH.calibrated=j.calibrated}
 shImg();$('b_shsave').className='';$('shmsg').textContent='';
 shSide();shTable();setTimeout(shSize,300)}
function shSize(){const im=$('shimg'),ov=$('shov');if(!im||!ov)return;const w=im.clientWidth,h=im.clientHeight;
 if(w&&h&&(ov.width!==w||ov.height!==h)){ov.width=w;ov.height=h;ov.style.width=w+'px';ov.style.height=h+'px'}shOv()}
setInterval(()=>{if(TABV==='shelves')shSize()},500);
function liveCell(id){const cs=(S&&S.shelves[SH.cam])||[];return cs.find(c=>c.slot===id)}
const SCOL={OK:'#3fb950',LOW:'#d9a441',EMPTY:'#ff4d4f'};
function shOv(){const ov=$('shov');if(!ov||!ov.width)return;const X=ov.getContext('2d'),W=ov.width,H=ov.height;
 X.clearRect(0,0,W,H);
 for(const s of SH.slots){const lc=liveCell(s.id),on=SH.sel===s.id,col=lc&&!SH.dirty?SCOL[lc.status]||'#e8364f':'#e8364f';
  const x=s.x*W,y=s.y*H,w=s.w*W,h=s.h*H;
  X.fillStyle=on?'rgba(232,54,79,.16)':'rgba(0,0,0,.18)';X.fillRect(x,y,w,h);
  X.strokeStyle=on?'#fff':col;X.lineWidth=on?2.5:2;X.strokeRect(x,y,w,h);
  X.strokeStyle=col;X.lineWidth=1;X.setLineDash([3,3]);
  for(let i=1;i<s.facings;i++){const fx=x+w*i/s.facings;X.beginPath();X.moveTo(fx,y);X.lineTo(fx,y+h);X.stroke()}
  X.setLineDash([]);
  const p=PRODS.find(p=>p.sku===s.sku),nm=p?p.name:(s._new&&s._new.name)||s.name||'unnamed';
  X.font='600 13px system-ui';const tw=X.measureText(nm).width+12;
  X.fillStyle='rgba(11,12,14,.85)';X.fillRect(x,Math.max(0,y-21),tw,20);
  X.fillStyle='#e9ebef';X.fillText(nm,x+6,Math.max(15,y-6));
  if(lc&&!SH.dirty){       // units left / full, in the box's top-right corner ("~" = front-row estimate, "?" = part hidden)
   const b=`${lc.method==='front'?'~':''}${lc.est_units}/${lc.full_units}${lc.hidden&&lc.hidden.length?'?':''}`;
   X.font='700 15px system-ui';const bw=X.measureText(b).width+14;
   X.fillStyle=SCOL[lc.status]||'#e8364f';X.beginPath();X.roundRect?X.roundRect(x+w-bw-4,y+4,bw,24,6):X.rect(x+w-bw-4,y+4,bw,24);X.fill();
   X.fillStyle='#0b0c0e';X.fillText(b,x+w-bw+3,y+21);
   if(lc.misplaced){X.strokeStyle='#a371f7';X.lineWidth=3;X.setLineDash([7,4]);X.strokeRect(x+2,y+2,w-4,h-4);X.setLineDash([]);
    X.font='700 13px system-ui';const t='WRONG PRODUCT',ww=X.measureText(t).width+12;
    X.fillStyle='#a371f7';X.fillRect(x+4,y+h-26,ww,22);X.fillStyle='#0b0c0e';X.fillText(t,x+10,y+h-10)}}
  if(on){X.fillStyle='#fff';X.fillRect(x+w-6,y+h-6,12,12)}}
 if(SH.drag&&SH.drag.mode==='draw'){const d=SH.drag;X.strokeStyle='#fff';X.setLineDash([6,4]);X.lineWidth=2;
  X.strokeRect(Math.min(d.x0,d.x1)*W,Math.min(d.y0,d.y1)*H,Math.abs(d.x1-d.x0)*W,Math.abs(d.y1-d.y0)*H);X.setLineDash([])}}
function shPt(e){const r=$('shov').getBoundingClientRect();
 return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]}
function shHit(px,py){const W=$('shov').width,H=$('shov').height;
 const s0=SH.slots.find(s=>s.id===SH.sel);
 if(s0&&Math.abs(px-(s0.x+s0.w))*W<10&&Math.abs(py-(s0.y+s0.h))*H<10)return {id:s0.id,mode:'size'};
 for(let i=SH.slots.length-1;i>=0;i--){const s=SH.slots[i];
  if(px>=s.x&&px<=s.x+s.w&&py>=s.y&&py<=s.y+s.h)return {id:s.id,mode:'move'}}return null}
(function(){const ov=$('shov');
 ov.addEventListener('pointerdown',e=>{const [px,py]=shPt(e);ov.setPointerCapture(e.pointerId);
  if(SH.mode==='draw'){SH.drag={mode:'draw',x0:px,y0:py,x1:px,y1:py};return}
  const h=shHit(px,py);if(!h){SH.sel=null;shSide();shOv();return}
  SH.sel=h.id;const s=SH.slots.find(s=>s.id===h.id);SH.drag={mode:h.mode,ox:px-s.x,oy:py-s.y};shSide();shOv()});
 ov.addEventListener('pointermove',e=>{const [px,py]=shPt(e);
  if(!SH.drag){const h=SH.mode==='draw'?null:shHit(px,py);
   ov.style.cursor=SH.mode==='draw'?'crosshair':h?(h.mode==='size'?'nwse-resize':'grab'):'default';return}
  const d=SH.drag;if(d.mode==='draw'){d.x1=px;d.y1=py;shOv();return}
  const s=SH.slots.find(s=>s.id===SH.sel);if(!s)return;
  if(d.mode==='move'){s.x=Math.max(0,Math.min(1-s.w,px-d.ox));s.y=Math.max(0,Math.min(1-s.h,py-d.oy))}
  else{s.w=Math.max(0.02,Math.min(1-s.x,px-s.x));s.h=Math.max(0.02,Math.min(1-s.y,py-s.y))}
  shDirty();shOv()});
 ov.addEventListener('pointerup',()=>{const d=SH.drag;SH.drag=null;if(!d)return;
  if(d.mode==='draw'){const x=Math.min(d.x0,d.x1),y=Math.min(d.y0,d.y1),w=Math.abs(d.x1-d.x0),h=Math.abs(d.y1-d.y0);
   if(w>0.02&&h>0.02){const id='p'+Date.now().toString(36);
    SH.slots.push({id,name:'New product',sku:'',x:+x.toFixed(4),y:+y.toFixed(4),w:+w.toFixed(4),h:+h.toFixed(4),facings:1,deep:1,unit_cm:0});
    SH.sel=id;shDirty()}
   shDrawMode(false)}
  shSide();shOv()})})();
function shDrawMode(on){SH.mode=(on===undefined?SH.mode!=='draw':on)?'draw':null;
 $('b_draw').className=SH.mode==='draw'?'on':'';$('shov').style.cursor=SH.mode==='draw'?'crosshair':'default'}
function shDirty(){SH.dirty=true;$('b_shsave').className='pri';$('shmsg').textContent='unsaved changes'}
function shSide(){const s=SH.slots.find(x=>x.id===SH.sel);let h='';
 if(!SH.cam){$('shside').innerHTML='<div class="box"><div class="empty">Add a shelf camera in Setup first.</div></div>';return}
 if(!SH.calibrated)h+=`<div class="box" style="border-color:#5a4a18"><h4>First, calibrate</h4><div class="hint">
  Fill the shelf, clear the aisle and press <b>Calibrate</b>. That picture is what you mark products on,
  and what each product is compared against.</div></div>`;
 if(!s){h+=`<div class="box"><h4>Products on this shelf</h4>`+(SH.slots.length?SH.slots.map(x=>{
   const p=PRODS.find(p=>p.sku===x.sku);return `<div class="f" style="cursor:pointer" onclick="SH.sel='${x.id}';shSide();shOv()">
   <span>${esc(p?p.name:x.name)}</span><span class="muted">${x.facings} × ${x.deep}</span></div>`}).join('')
  :`<div class="hint">Press <b>Draw product box</b>, then drag a box around one product block. Draw one per product —
   a box can be as narrow as a single item, so taking one out is noticed.</div>`)+`</div>`}
 else{const p=PRODS.find(p=>p.sku===s.sku),nw=s._new||{};
  h+=`<div class="box"><h4>Product in this box</h4>
   <div class="f"><span>product</span><select style="width:170px" onchange="shPick(this.value)">
    <option value="">— choose —</option>${PRODS.map(x=>`<option value="${esc(x.sku)}" ${x.sku===s.sku?'selected':''}>${esc(x.name)}</option>`).join('')}
    <option value="__new" ${s._new?'selected':''}>+ new product…</option></select></div>`;
  if(s._new)h+=['name','sku','barcode','brand'].map(k=>`<div class="f"><span>${k}${k==='barcode'?' <small>(optional)</small>':''}</span>
    <input type="text" id="shn_${k}" value="${esc(nw[k]||'')}" oninput="SHN('${k}',this.value)"></div>`).join('')+
    `<div class="f"><span>MRP ₹</span><input type="number" step="0.5" value="${nw.mrp||''}" oninput="SHN('mrp',this.value)"></div>
     <div class="f"><span>selling price ₹</span><input type="number" step="0.5" value="${nw.price||''}" oninput="SHN('price',this.value)"></div>`;
  else if(p)h+=`<div class="hint">${esc(p.brand)} · MRP ₹${p.mrp} · sells at ₹${p.price}<br>barcode ${esc(p.barcode)}</div>`;
  h+=`</div><div class="box"><h4>How it sits</h4>
   <div class="f"><span>side by side (facings)</span><input type="number" min="1" max="30" value="${s.facings}"
    oninput="SHS('facings',this.value)"></div>
   <div class="f"><span>units deep</span><input type="number" min="1" max="30" value="${s.deep}" oninput="SHS('deep',this.value)"></div>
   <div class="f"><span>one unit, front to back (cm)</span><input type="number" min="0" max="60" step="0.5" value="${s.unit_cm||''}"
    placeholder="e.g. 7" oninput="SHU(this.value)"></div>
   ${s.deep>1&&!s.unit_cm?`<div class="warnbox">⚠ <b>${s.deep} deep, but no unit size.</b> Without it the camera can only see the front pack of each
     column, so a pack taken from the front still shows as full and one put back shows as full again. Enter how deep one pack is.</div>`:''}
   ${s.deep>1&&s.unit_cm&&SH.cam&&S&&S.depth&&S.depth[SH.cam]&&!S.depth[SH.cam].on?`<div class="warnbox">⚠ <b>Depth counting is off for this camera:</b> ${esc(S.depth[SH.cam].msg)}</div>`:''}
   <div class="hint">A full box holds <b id="sh_full">${s.facings*s.deep}</b> units. With the unit size set, the depth model
    measures how far back the front unit of each column sits and counts what's left behind it. Without it, the count is
    an estimate: facings still visible × units deep.</div>
   <button class="danger" style="margin-top:8px" onclick="shDel()">Remove box</button></div>`;
  const lc=liveCell(s.id);
  if(lc&&!SH.dirty)h+=`<div class="box"><h4>Right now</h4><div class="f"><span>status</span><span class="st ${lc.status}">${lc.status}</span></div>
   <div class="f"><span>facings visible</span><b>${lc.present} / ${lc.facings}</b></div>
   <div class="f"><span>camera count</span><b>${cnt(lc)} / ${lc.full_units}</b></div>
   <div class="f"><span>counted by</span><span>${lc.method==='front'?'front row × units deep':'depth model, per column: '+lc.columns.join(' · ')+(lc.hidden&&lc.hidden.length?` (column ${lc.hidden.map(i=>i+1).join(', ')} hidden from this angle — front unit gone, rest unseen)`:'')}</span></div>
   <div class="f"><span>by the till</span><b>${lc.pos_units==null?'—':lc.pos_units}</b></div>
   <div class="f"><span>where</span><span>${esc(lc.loc)}</span></div></div>`}
 $('shside').innerHTML=h}
function shPick(v){const s=SH.slots.find(x=>x.id===SH.sel);
 if(v==='__new'){s._new=s._new||{name:'',sku:'',barcode:'',brand:'',mrp:'',price:''};s.sku=''}
 else{delete s._new;s.sku=v;const p=PRODS.find(p=>p.sku===v);if(p)s.name=p.name}
 shDirty();shSide();shOv()}
function SHN(k,v){const s=SH.slots.find(x=>x.id===SH.sel);s._new[k]=v;
 if(k==='name'){s.name=v;if(!s._new._skuTouched)s._new.sku=v.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'').slice(0,30);
  const i=$('shn_sku');if(i)i.value=s._new.sku}
 if(k==='sku')s._new._skuTouched=true;shDirty();shOv()}
function SHS(k,v){const s=SH.slots.find(x=>x.id===SH.sel);s[k]=Math.max(1,Math.min(30,parseInt(v)||1));shDirty();shOv();
 const f=$('sh_full');if(f)f.textContent=s.facings*s.deep}
function SHU(v){const s=SH.slots.find(x=>x.id===SH.sel);s.unit_cm=Math.max(0,Math.min(60,parseFloat(v)||0));shDirty()}
function shView(v){SH.view=v;shImg()}
function shImg(){if(!SH.cam)return;const im=$('shimg');
 if(!SH.calibrated&&SH.view==='still')SH.view='live';            // nothing to show still before calibrating
 if(SH.view==='depth'&&S&&S.depth_model_err){SH.view='live';
  $('shmsg').textContent='Depth needs the depth model: run  pip install transformers  and restart the app.'}
 for(const [id,v] of [['b_vstill','still'],['b_vlive','live'],['b_depth','depth']])$(id).className=SH.view===v?'on':'';
 if(SH.view==='live'){im.classList.add('live');im.dataset.cam=SH.cam;im.dataset.view='raw';im._busy=false;return}
 im.classList.remove('live');im.onload=null;im.onerror=shImgErr;
 im.src=SH.view==='depth'?'/api/depth/'+SH.cam+'.jpg?t='+Date.now():'/api/shelf/'+SH.cam+'/still.jpg?t='+Date.now()}
setInterval(()=>{if(TABV==='shelves'&&SH.view==='depth')shImg()},3000);
function shImgErr(){if(SH.view!=='depth')return;SH.view='live';shImg();
 const d=S&&S.depth&&S.depth[SH.cam];$('shmsg').textContent='No depth picture yet — '+(d&&!d.on?d.msg:'set a unit size on a product box and wait a few seconds')}
function shDepthLine(){const el=$('shdepth');if(!el)return;const d=S&&S.depth&&S.depth[SH.cam];
 el.innerHTML=!d?'':d.on?`Depth model on — ${esc(d.msg)}${d.noise_cm!=null?` · fit error ${d.noise_cm} cm`:''}`
  :`Depth model off — ${esc(d.msg)}`}
function shDel(){SH.slots=SH.slots.filter(s=>s.id!==SH.sel);SH.sel=null;shDirty();shSide();shOv()}
addEventListener('keydown',e=>{if(TABV!=='shelves'||!SH.sel||/INPUT|SELECT/.test(document.activeElement.tagName))return;
 if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();shDel()}});
async function shSave(){const msg=$('shmsg');
 for(const s of SH.slots){if(!s._new)continue;const n=s._new;
  if(!n.name||!n.sku||!n.mrp){msg.textContent=`Fill name, SKU and MRP for “${n.name||'new product'}”`;return}
  const r=await(await fetch('/api/products',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({sku:n.sku,name:n.name,brand:n.brand,barcode:n.barcode,mrp:+n.mrp,price:n.price===''?+n.mrp:+n.price})})).json();
  if(!r.ok){msg.textContent=r.error;return}s.sku=r.product.sku;s.name=r.product.name;delete s._new}
 const body={slots:SH.slots.map(({_new,...s})=>s)};
 const r=await(await fetch('/api/slots/'+SH.cam,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
 if(!r.ok){msg.textContent=r.error;return}
 SH.dirty=false;$('b_shsave').className='';msg.textContent='saved — the camera uses these boxes from its next frame';
 await prodsLoad();catalog();shSide();shOv()}
async function shCalibrate(){if(!SH.cam)return;$('shmsg').textContent='calibrating…';
 const j=await(await fetch('/api/calibrate/'+SH.cam,{method:'POST'})).json();$('shmsg').textContent=j.message;
 if(/Calibrated/.test(j.message)){SH.calibrated=true;shImg();shSide()}}
/* a camera count: "3" when every position was checked, "1–3" when only part of it could be, "~" when it rests on the
   front row alone. The range is the honest answer after a pack is put back at the front: one pack, or a full stack. */
function cnt(c){const lo=c.est_min==null?c.est_units:c.est_min,hi=c.est_units;
 return (c.method==='front'||c.exact===false?'~':'')+(lo<hi?`${lo}–${hi}`:hi)}
async function refilled(cam,slot){await fetch('/api/shelf/'+encodeURIComponent(cam)+'/refilled',{method:'POST',
 headers:{'Content-Type':'application/json'},body:JSON.stringify({slot})})}
function shTable(){const el=$('shtable');if(!el||!S)return;const cells=(S.shelves[SH.cam]||[]);
 if(!SH.cam){el.innerHTML='';return}
 if(!cells.length){el.innerHTML=`<div class="empty">${SH.calibrated?'Waiting for the camera…':'Not calibrated yet.'}</div>`;return}
 if(!cells[0].slot){el.innerHTML=`<div class="hint" style="margin-bottom:8px">No product boxes yet — showing the fallback grid.
   Draw boxes above for per-product counts.</div>`+gridCells(cells);return}
 el.innerHTML=`<table><tr><th>Where</th><th>Product</th><th>Facings seen</th><th>Camera count</th><th>By the till</th>
  <th>Status</th><th>Shoppers today</th><th>Runs out</th></tr>`+cells.slice().sort((a,b)=>a.row-b.row||a.col-b.col).map(c=>`<tr>
  <td class="muted">${esc(c.loc)}</td><td>${esc(c.name)}${c.brand?`<div class="muted">${esc(c.brand)}</div>`:''}</td>
  <td><div style="display:flex;align-items:center;gap:8px"><div class="bar-in" style="flex:1"><div style="width:${100*c.present/c.facings}%;
   background:${SCOL[c.status]||'#3fb950'}"></div></div><span>${c.present}/${c.facings}</span></div></td>
  <td>${cnt(c)} <span class="muted">/ ${c.full_units} · ${{depth:'depth',mixed:'depth, part hidden',front:'front row'}[c.method]||''}</span>
   ${c.est_min!=null&&c.est_min<c.est_units?`<div class="muted" style="font-size:11.5px">at least ${c.est_min} for sure · <a href="#" class="lnk" onclick="refilled('${esc(SH.cam)}','${esc(c.slot)}');return false">refilled</a></div>`:''}</td><td>${c.pos_units==null?'<span class="muted">—</span>':c.pos_units}</td>
  <td><span class="st ${c.occluded?'occ':c.status}">${c.occluded?'BLOCKED':c.status}</span>${c.misplaced?' <span class="st MISPLACED">WRONG PRODUCT</span>':''}</td>
  <td>${c.attention&&c.attention.visits?`${c.attention.visits} stop${c.attention.visits===1?'':'s'} <span class="muted">· avg ${c.attention.avg_s}s</span>`:'<span class="muted">—</span>'}</td>
  <td class="muted">${c.eta_min!=null?'~'+Math.round(c.eta_min)+' min':''}</td></tr>`).join('')+`</table>`}
function gridCells(cells){const C=Math.max(...cells.map(c=>c.c))+1;
 return `<div class="grid" style="grid-template-columns:repeat(${C},1fr)">`+cells.map(c=>`<div class="cell ${c.status} ${c.occluded?'occ':''}">
  ${c.fill==null?'?':Math.round(c.fill*100)+'%'}</div>`).join('')+`</div>`}
async function prodImport(inp){const f=inp.files[0];if(!f)return;const txt=await f.text();inp.value='';
 const j=await(await fetch('/api/integrations/products',{method:'POST',headers:{'Content-Type':'text/csv'},body:txt})).json();
 $('impmsg').textContent=` imported ${j.imported}`+(j.errors.length?` · ${j.errors.length} problem${j.errors.length>1?'s':''}: ${j.errors.slice(0,2).join('; ')}`:'');
 await prodsLoad();catalog()}
let PEDIT=null;
function catalog(){const el=$('catalog');if(!el)return;
 el.innerHTML=PRODS.length?`<table><tr><th>Product</th><th>Brand</th><th>SKU</th><th>Barcode</th><th style="text-align:right">MRP</th>
  <th style="text-align:right">Price</th><th></th></tr>`+PRODS.map(p=>`<tr><td>${esc(p.name)}</td><td class="muted">${esc(p.brand)}</td>
  <td class="muted">${esc(p.sku)}</td><td class="muted">${esc(p.barcode)}</td><td style="text-align:right">₹${p.mrp}</td>
  <td style="text-align:right">₹${p.price}</td><td style="text-align:right;white-space:nowrap">
  <a class="lnk" href="/api/products/${encodeURIComponent(p.sku)}/label.png" target="_blank">label</a> ·
  <a class="lnk" href="#" onclick="prodEdit('${esc(p.sku)}');return false">edit</a> ·
  <a class="lnk" href="#" onclick="prodDel('${esc(p.sku)}');return false">delete</a></td></tr>`).join('')+`</table>`
  :'<div class="empty">No products yet. Add them here, or while marking boxes on a shelf.</div>'}
function prodEdit(sku){const p=sku?PRODS.find(x=>x.sku===sku):{sku:'',name:'',brand:'',barcode:'',mrp:'',price:''};PEDIT=sku||null;
 $('prodform').innerHTML=`<div class="box" style="border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px;background:var(--card2)">
  <div class="bar2" style="margin:0;gap:10px">`+
  [['name','text',200],['sku','text',130],['barcode','text',140],['brand','text',120],['mrp','number',80],['price','number',80]].map(([k,t,w])=>
   `<label class="muted" style="font-size:12px">${k==='mrp'?'MRP ₹':k==='price'?'price ₹':k}<br><input id="pf_${k}" type="${t}" style="width:${w}px"
    value="${esc(p[k])}" ${k==='sku'&&sku?'disabled':''}></label>`).join('')+
  `<button class="pri" onclick="prodSave()">Save</button><button onclick="$('prodform').innerHTML=''">Cancel</button>
   <span class="muted" id="pf_msg"></span></div><div class="hint" style="margin-top:6px">Leave barcode empty to get an
   in-store code (GS1 20–29 range) — print its label and stick it on the product.</div></div>`}
async function prodSave(){const b={};for(const k of ['name','sku','barcode','brand','mrp','price'])b[k]=$('pf_'+k).value;
 if(b.price==='')delete b.price;const r=await(await fetch('/api/products',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(b)})).json();if(!r.ok){$('pf_msg').textContent=r.error;return}
 $('prodform').innerHTML='';await prodsLoad();catalog();shSide()}
async function prodDel(sku){if(!confirm('Delete '+sku+'?'))return;await fetch('/api/products/'+encodeURIComponent(sku),{method:'DELETE'});
 await prodsLoad();catalog()}

/* ---------- checkout ---------- */
let CART=null,CARTV=null,LASTSCAN=0;
async function coEnter(){await prodsLoad();
 if(S&&S.pos&&S.pos.active_cart&&!CART)CART=S.pos.active_cart;await cartRefresh();
 const till=Object.entries((S&&S.cams)||{}).find(([n,c])=>(c.roles||[]).includes('checkout'));
 $('tillcam').innerHTML=till?`<div class="feeds"><figure><img class="live" data-cam="${esc(till[0])}" style="width:100%"><figcaption>${esc(till[0])}</figcaption></figure></div>
  <div id="scantoast"></div>`:`<div class="hint">No checkout camera. Add a camera with the <b>checkout</b> role, or use a USB barcode
  scanner — it types into the box on the left. Typing a product name works too.</div><div id="scantoast"></div>`;
 setTimeout(()=>$('scan').focus(),50)}
async function cartRefresh(){if(!CART){CARTV=null;cartDraw();return}
 const r=await fetch('/api/cart/'+CART);CARTV=r.ok?await r.json():null;if(!CARTV)CART=null;cartDraw()}
function cartDraw(flash){const v=CARTV;SHSIG='';shopDraw();$('cartid').textContent=v?`cart ${v.id} · ${v.n_items} item${v.n_items===1?'':'s'}`:'';
 $('cartlines').innerHTML=v&&v.lines.length?v.lines.map(l=>`<div class="cl ${flash===l.sku?'flash':''}"><div class="n">${esc(l.name)}
  <small>${esc(l.brand)}${l.mrp>l.price?` · MRP ₹${l.mrp}`:''}</small></div>
  <div class="qty"><button onclick="cartQty('${esc(l.sku)}',${l.qty-1})">−</button><b>${l.qty}</b>
   <button onclick="cartQty('${esc(l.sku)}',${l.qty+1})">+</button></div>
  <div class="rate">₹${l.price.toFixed(2)}</div><div class="amt">₹${l.amount.toFixed(2)}</div>
  <button class="danger" title="remove" onclick="cartQty('${esc(l.sku)}',0)">×</button></div>`).join('')
  :'<div class="empty">Scan a product to start.</div>';
 $('carttotals').innerHTML=v&&v.lines.length?`<div class="tot"><span>Total at MRP</span><span>₹${v.mrp_total.toFixed(2)}</span></div>
  <div class="tot"><span>You save</span><span>₹${v.savings.toFixed(2)}</span></div>
  <div class="tot big"><span>Payable</span><span>₹${v.total.toFixed(2)}</span></div>`:'';
 $('b_bill').disabled=!(v&&v.lines.length)}
async function ensureCart(){if(!CART){const j=await(await fetch('/api/cart',{method:'POST'})).json();CART=j.id}}
async function addCode(code){await ensureCart();
 const r=await(await fetch('/api/cart/'+CART+'/scan',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({code})})).json();
 if(r.ok){CARTV=r.cart;cartDraw(r.product.sku);$('scanmsg').textContent=`${r.product.name} added`;return true}
 return false}
async function scanSubmit(){const inp=$('scan'),v=inp.value.trim();if(!v)return;
 if(await addCode(v)){inp.value='';$('suggest').innerHTML='';inp.focus();return}
 const m=matches(v);if(m.length===1){await addCode(m[0].sku);inp.value='';$('suggest').innerHTML=''}
 else{$('scanmsg').textContent=m.length?'Pick one below':`Nothing matches “${v}”`;suggest(v)}}
function matches(v){v=v.toLowerCase();return PRODS.filter(p=>p.name.toLowerCase().includes(v)||p.brand.toLowerCase().includes(v)||p.sku.includes(v)).slice(0,6)}
function suggest(v){const m=/^\d+$/.test(v)||v.length<2?[]:matches(v);
 $('suggest').innerHTML=m.length?`<div class="sugg">${m.map(p=>`<div onclick="addCode('${esc(p.sku)}');$('scan').value='';
  $('suggest').innerHTML='';$('scan').focus()"><span>${esc(p.name)} <span class="muted">${esc(p.brand)}</span></span><b>₹${p.price}</b></div>`).join('')}</div>`:''}
$('scan').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();scanSubmit()}});
$('scan').addEventListener('input',e=>suggest(e.target.value.trim()));
/* a USB scanner types into whatever has focus, so keep the scan box focused on this tab */
document.addEventListener('click',e=>{if(TABV==='checkout'&&!e.target.closest('input,button,select,a,.sugg'))$('scan').focus()});
async function cartQty(sku,q){if(!CART)return;CARTV=await(await fetch('/api/cart/'+CART+'/set',{method:'POST',
 headers:{'Content-Type':'application/json'},body:JSON.stringify({sku,qty:q})})).json();cartDraw()}
async function cartNew(){if(CARTV&&CARTV.lines.length&&!confirm('Clear this cart?'))return;
 const j=await(await fetch('/api/cart',{method:'POST'})).json();CART=j.id;CARTV=j;cartDraw();$('billbox').style.display='none';$('scan').focus()}
async function cartBill(){if(!CART)return;const j=await(await fetch('/api/cart/'+CART+'/checkout',{method:'POST'})).json();
 if(!j.ok)return alert(j.error);const b=j.bill;CART=null;CARTV=null;cartDraw();
 $('billbox').style.display='';$('billimg').src='/api/bills/'+b.id+'.png';
 $('billlinks').innerHTML=`<a class="lnk" href="/api/bills/${b.id}.pdf" target="_blank">PDF</a> ·
  <a class="lnk" href="/api/bills/${b.id}.png" download="${b.id}.png">PNG</a>`;
 $('paynote').innerHTML=`<div class="h">Please move to the payment counter</div>
  <div>to complete your purchase.</div><p style="font-size:28px;font-weight:700;margin:14px 0 4px">₹${b.total.toFixed(2)}</p>
  <div class="muted">Bill ${b.id} · ${b.n_items} item${b.n_items===1?'':'s'} · you saved ₹${b.savings.toFixed(2)}</div>
  <div class="muted" style="margin-top:6px">${b.shopper?`Shopper ${esc(b.shopper.id)} · ${b.shopper.check==='ok'?'ID verified by '+(b.shopper.how==='camera'?'camera':'staff'):'ID check failed: '+esc(b.shopper.check)}`:'No shopper ID was verified for this bill'}</div>
  <div class="bar2" style="margin-top:16px"><button class="pri" onclick="window.open('/api/bills/${b.id}.pdf')">Print bill</button>
   <button onclick="cartNew()">Next customer</button></div>`;
 $('billbox').scrollIntoView({behavior:'smooth'})}
let SHSIG='';
/* who is at the till: the door camera's ID for them, found again by clothing colour, or picked by staff */
function shopDraw(){const el=$('shopbar');if(!el||!S||!S.shoppers)return;const sh=S.shoppers,t=sh.till,cur=CARTV&&CARTV.shopper;
 const sig=[cur,t&&t.id,t&&Math.round(t.score*100),sh.list.map(x=>x.id+x.billed).join(',')].join('|');
 if(sig===SHSIG)return;SHSIG=sig;
 const name=cur?`${esc(cur)}<small>picked by staff</small>`:t?`${esc(t.id)}<small>camera match ${Math.round(t.score*100)}%</small>`:
  `<span class="muted" style="font-size:14px">${sh.inside?'not identified yet — pick the shopper':'nobody has entered yet'}</span>`;
 el.innerHTML=`<div class="shop ${cur||t?'ok':'warn'}"><div class="who"><span class="lbl">Shopper ID · checked at checkout</span>
  <b class="sid">${name}</b></div><select id="shsel" onchange="shopPick(this.value)">
  <option value="">${t&&!cur?'use the camera match':'choose a shopper…'}</option>${sh.list.map(x=>
  `<option value="${esc(x.id)}" ${cur===x.id?'selected':''}>${esc(x.id)} · ${x.mins<1?'just came in':Math.round(x.mins)+' min inside'}${x.billed?' · already billed':''}</option>`).join('')}</select></div>`}
async function shopPick(v){await ensureCart();
 const r=await(await fetch('/api/cart/'+CART+'/shopper',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({shopper:v})})).json();
 if(!r.ok){alert(r.error);SHSIG='';shopDraw();return}CARTV=r.cart;SHSIG='';cartDraw()}
function coSync(s){const ls=s.pos&&s.pos.last_scan;if(!ls||ls.ts<=LASTSCAN)return;LASTSCAN=ls.ts;
 if(TABV!=='checkout'||ls.source==='screen')return;const t=$('scantoast');   /* screen scans report inline */
 if(t)t.innerHTML=`<div class="toast ${ls.ok?'ok':'bad'}">${esc(ls.msg)}${ls.ok&&ls.price!=null?` · ₹${ls.price}`:''}</div>`;
 if(ls.ok&&s.pos.active_cart){CART=s.pos.active_cart;cartRefresh()}}

/* ---------- queue ---------- */
let QA=null,QT=0,QCTRSET=false;
async function qLoad(){if(Date.now()-QT<20000)return;QT=Date.now();
 try{QA=await(await fetch('/api/queue')).json()}catch(e){}qCharts()}
const mins=m=>m==null?'—':m<1?'under a minute':m<60?'~'+Math.round(m*10)/10+' min':'over an hour';
function qMain(s){const qs=Object.entries(s.queues||{});            // the busiest queue camera drives the page
 return qs.length?qs.sort((a,b)=>b[1].length-a[1].length)[0]:null}
function qRender(s){const m=qMain(s),q=m&&m[1],crowds=s.crowds||[],calls=s.alerts.filter(a=>a.kind==='queue'||a.kind==='crowd');
 $('qdot').className='tabdot'+(calls.some(a=>a.severity==='critical')?' on':calls.length?' on warn':'');
 if(TABV!=='queue')return;
 if(!QCTRSET&&document.activeElement!==$('qctr')){$('qctr').value=s.open_counters;QCTRSET=true}
 const k=s.open_counters;
 $('qkpis').innerHTML=q?kpi('In line now',q.length,'people')+kpi('Wait',q.wait_min+' min','for the last in line',q.wait_min>=4?'warn':'')+
  kpi('Clears in',q.length===0?'—':q.clear_min==null?'not clearing':mins(q.clear_min),`with ${k} counter${k===1?'':'s'}`,q.length&&q.clear_min==null?'bad':'')+
  kpi('Arriving',q.arrival_per_min+'/min','last 5 min')+kpi('Served',q.service_per_counter_min+'/min','per counter, learned')+
  kpi('Peak today',(s.queue_peak||{}).len||0,'people in line'):
  kpi('Queue camera','none','give a camera the “watch queue” job in Setup');
 let v='';
 if(!q)v='<div class="empty">No queue camera yet.<br><span class="muted">In Setup, give a camera the “watch queue” job.</span></div>';
 else{const rec=q.recommend_counters,cls=q.length===0?'ok':q.clear_min==null?'bad':q.clear_min>CFGW?'warn':'ok';
  const head=q.length===0?'Empty':q.clear_min==null?'No':mins(q.clear_min);
  const why=q.length===0?'Nobody is waiting.':q.clear_min==null?
   `<b>${q.arrival_per_min}</b> people arrive a minute but ${k} counter${k===1?' serves':'s serve'} only <b>${(q.service_per_counter_min*k).toFixed(1)}</b>. The line keeps growing${q.clear_min_if_opened?` — with <b>${rec}</b> counters it would clear in <b>${mins(q.clear_min_if_opened)}</b>`:''}.`:
   `<b>${q.length}</b> in line, shrinking by about <b>${(q.service_per_counter_min*k-q.arrival_per_min).toFixed(1)}</b> a minute at ${k} counter${k===1?'':'s'}.`;
  v=`<div class="verdict ${cls}"><div class="big">${head}</div><div class="why">${why}</div></div>`}
 $('qverdict').innerHTML=v;
 $('qwhatif').innerHTML=q?`<tr><th>Counters open</th><th>Wait for the last in line</th><th>Line gone in</th><th></th></tr>`+
  q.what_if.map(w=>`<tr class="${w.counters===k?'cur':''} ${w.counters===q.recommend_counters&&w.counters!==k?'rec':''}"><td><b>${w.counters}</b>${w.counters===k?'<span class="tag">now</span>':''}${w.counters===q.recommend_counters&&w.counters!==k?'<span class="tag">suggested</span>':''}</td>
   <td>${w.wait_min} min</td><td>${w.clear_min==null?'<span style="color:var(--bad)">never — still growing</span>':w.clear_min===0?'—':mins(w.clear_min)}</td>
   <td>${w.keeps_up?'<span style="color:var(--ok)">keeps up</span>':'<span class="muted">falls behind</span>'}</td></tr>`).join(''):'';
 const need=q&&q.recommend_counters>k;$('qopen').style.display=need?'':'none';if(need)$('qopen').textContent=`Open ${q.recommend_counters-k} more (${q.recommend_counters} total)`;
 const cam=qMain(s)&&qMain(s)[0];if(cam&&$('qcam').dataset.cam!==cam){$('qcam').dataset.cam=cam;
  $('qcam').innerHTML=`<img class="live qimg" data-cam="${esc(cam)}" alt="queue camera"><div class="muted" style="margin-top:6px">${esc((s.labels||{})[cam]||cam)} · the yellow outline is the queue zone</div>`}
 $('qcrowdsub').textContent=`alert at ${s.crowd_threshold}+ people in one camera's view`;
 $('qcrowds').innerHTML=crowds.length?crowds.map(c=>`<div class="crowd ${c.people>=1.5*s.crowd_threshold?'crit':''}"><div class="n">${c.people}</div>
  <div><div class="w">${esc(c.where)}</div><div class="t">for ${c.for_s<90?c.for_s+' s':Math.round(c.for_s/60)+' min'} · peak ${c.peak}</div></div>
  <div class="t">${c.staff_alert?'staff alerted':'—'}</div>
  <div class="eta">${c.eta_min===0?'Easing off — back under the limit.':c.eta_min==null?'<b>Not thinning yet</b> — no estimate until the count starts to fall.':
   `Expected to thin out in <b>${mins(c.eta_min)}</b> <span class="muted">(trend of the last couple of minutes)</span>`}</div></div>`).join(''):
  `<div class="empty">No crowds right now.<br><span class="muted">Every camera is watched; staff are called when one sees ${s.crowd_threshold}+ people for a few seconds.</span></div>`;
 $('qcalls').innerHTML=calls.length?calls.map(a=>`<div class="alert ${a.severity}"><div class="m"><div class="top"><span class="a">${esc(a.action)}</span><span class="t">${ago(a.ts)}</span></div>
  <div class="msg">${esc(a.message)}</div></div><button onclick="ack(${a.id})">On it</button></div>`).join(''):'<div class="empty">Nobody needs calling.</div>';
 $('qresp').textContent=s.avg_response_s==null?'':`average response today ${Math.round(s.avg_response_s)} s`}
const CFGW=6;
function qCharts(){const a=QA;if(!a||TABV!=='queue')return;
 const rows=a.today;
 lineChart($('q_today'),{labels:rows.map(r=>new Date(r.t*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})),
  series:[{name:'In line',color:CC.s1,values:rows.map(r=>r.len)},{name:'Wait (min)',color:CC.s2,values:rows.map(r=>r.wait)}],h:260});
 const hrs=[];for(let h=7;h<=22;h++)hrs.push(h);
 heat($('q_heat'),{rows:WD,cols:hrs.map(h=>String(h).padStart(2,'0')),m:a.weekday_hour.map(r=>hrs.map(h=>r[h])),name:'people in line (avg)'});
 const t=a.stats;$('qstats').innerHTML=[['Served',t.served],['Average time in queue',t.avg_in_queue_min==null?'—':t.avg_in_queue_min+' min'],
  ['Longest line',t.peak_len+' people'],['Longest wait',t.peak_wait_min+' min'],['Average wait',t.avg_wait_min==null?'—':t.avg_wait_min+' min'],
  ['Busiest hour',t.busiest_hour||'—'],['Crowds today',a.crowds.length]].map(r=>`<div>${r[0]}</div><div>${r[1]}</div>`).join('')}
async function qSetCounters(){await fetch('/api/counters?open='+$('qctr').value,{method:'POST'});QCTRSET=false}
async function qOpenRec(){const q=qMain(S)[1];$('qctr').value=q.recommend_counters;await qSetCounters()}
setInterval(()=>{if(TABV==='queue'){QT=0;qLoad()}},30000);

/* ---------- cctv ---------- */
let CCTVV='cctv';
function cctvMode(m){CCTVV=m;$('cv_plain').className='chip'+(m==='cctv'?' on':'');$('cv_ana').className='chip'+(m==='analytics'?' on':'');cctvBuild()}
let CCTVSIG='';
function cctvBuild(){const cams=Object.entries((S&&S.cams)||{});CCTVSIG=cams.map(([n])=>n).join('|');
 $('cctvgrid').innerHTML=cams.length?cams.map(([n,c])=>`<figure onclick="this.classList.toggle('big')">
  <img class="live" data-cam="${esc(n)}" data-view="${CCTVV}" alt="${esc(n)}"><span class="badge" id="pb_${esc(n)}">${c.online?'● ':'○ '}${c.people||0} people</span>
  <figcaption>${esc((S.labels||{})[n]||n)} · ${c.roles.join(', ')} · ${c.fps} fps</figcaption></figure>`).join('')
  :'<div class="empty">No cameras running.</div>'}
function cctvBadges(){for(const [n,c] of Object.entries(S.cams||{})){const b=$('pb_'+n);
 if(b)b.textContent=`${c.online?'● ':'○ '}${c.people||0} ${c.people===1?'person':'people'}`}}

/* ---------- analytics ---------- */
let ADAYS=14,AN=null;
document.querySelectorAll('#arange button').forEach(c=>c.onclick=()=>{ADAYS=+c.dataset.d;
 document.querySelectorAll('#arange button').forEach(x=>x.classList.toggle('on',x===c));anLoad()});
/* live: 1-second samples from the dashboard's own state, seeded with the last hour of per-minute history */
let LIVE={pts:[],hist:[],win:300,seeded:0};
function liveSample(s){const q=Object.values(s.queues||{});
 LIVE.pts.push({t:s.ts,inside:s.footfall.inside,queue:q.reduce((a,x)=>a+x.length,0),ent:s.footfall.in,sales:s.sales_today||0});
 const cut=s.ts-3700;while(LIVE.pts.length&&LIVE.pts[0].t<cut)LIVE.pts.shift();
 if(TABV==='analytics'){liveDraw();anShop(s)}}
async function liveSeed(){if(Date.now()-LIVE.seeded<60000)return;LIVE.seeded=Date.now();
 try{LIVE.hist=(await(await fetch('/api/live?minutes=60')).json()).rows}catch(e){}liveDraw()}
document.querySelectorAll('#lwin button').forEach(c=>c.onclick=()=>{LIVE.win=+c.dataset.w;
 document.querySelectorAll('#lwin button').forEach(x=>x.classList.toggle('on',x===c));liveDraw()});
function liveSeries(key,hkey){const t0=LIVE.pts.length?LIVE.pts[0].t:Infinity;
 return LIVE.hist.filter(r=>r.t<t0&&r[hkey]!=null).map(r=>[r.t,r[hkey]]).concat(LIVE.pts.map(p=>[p.t,p[key]]))}
function livePerMinute(){const by={},live={},pts=LIVE.pts;
 for(const r of LIVE.hist)if(r.entries!=null)by[Math.floor((r.t-60)/60)]=r.entries;   // a history row covers the minute before it
 for(let i=1;i<pts.length;i++){const d=pts[i].ent-pts[i-1].ent;if(d<0)continue;       // counter resets at midnight
  const m=Math.floor(pts[i].t/60);live[m]=(live[m]||0)+d}
 const firstM=pts.length?Math.floor(pts[0].t/60):Infinity;                              // the first watched minute is partial
 for(const k in live)if(+k>firstM||!(k in by))by[k]=live[k];
 return Object.keys(by).map(k=>[+k*60+30,by[k]]).sort((x,y)=>x[0]-y[0])}
function liveSales(){const pts=LIVE.pts,now=pts.length?pts[pts.length-1]:null;if(!now)return [];
 const hist=LIVE.hist.filter(r=>r.t<pts[0].t&&r.revenue!=null);let run=pts[0].sales;const back=[];
 for(let i=hist.length-1;i>=0;i--){back.unshift([hist[i].t,Math.max(0,run)]);run-=hist[i].revenue}
 return back.concat(pts.map(p=>[p.t,p.sales]))}
function liveDraw(){if(!$('l_inside'))return;const now=LIVE.pts.length?LIVE.pts[LIVE.pts.length-1].t:Date.now()/1000;
 const last=a=>a.length?a[a.length-1][1]:null;
 const ins=liveSeries('inside','inside'),que=liveSeries('queue','queue_len'),ent=livePerMinute(),sal=liveSales();
 $('lv_inside').textContent=last(ins)??'—';$('lv_queue').textContent=last(que)??'—';
 $('lv_ent').innerHTML=(ent.length?ent[ent.length-1][1]:0)+'<small>this minute</small>';$('lv_sales').textContent=fmtR(last(sal)||0);
 liveChart($('l_inside'),ins,{now,color:CC.s1,int:true,name:'inside'});
 liveChart($('l_ent'),ent,{now,color:CC.s1,int:true,bars:true,name:'entries'});
 liveChart($('l_queue'),que,{now,color:CC.s2,int:true,name:'in line'});
 liveChart($('l_sales'),sal,{now,color:CC.s2,fmt:fmtR,yfmt:fmtRs,name:'sales today',min:100})}
function liveChart(el,pts,o){const W=Math.max(300,el.clientWidth||500),H=190,m={l:54,r:14,t:10,b:28};
 const win=LIVE.win,t0=o.now-win,iw=W-m.l-m.r,ih=H-m.t-m.b,vis=pts.filter(p=>p[0]>=t0-60);
 const T=ticks(Math.max(o.min||0,...vis.map(p=>p[1]||0))*1.1,o.int),mx=T.mx;
 const X=t=>m.l+(t-t0)/win*iw,Y=v=>m.t+ih-(v/mx)*ih;
 let g=yAxis(m,W,ih,mx,o.yfmt,T.n);
 const stepT=win/5;for(let k=0;k<=5;k++){const t=t0+k*stepT,lab=k===5?'now':'−'+Math.round((win-k*stepT)/60)+'m';
  g+=`<text class="ax" x="${X(t)}" y="${H-8}" text-anchor="${k===0?'start':k===5?'end':'middle'}">${lab}</text>`}
 g+=`<clipPath id="cp${el.id}"><rect x="${m.l}" y="0" width="${iw}" height="${H}"/></clipPath><g clip-path="url(#cp${el.id})">`;
 if(o.bars){const bw=Math.max(2,Math.min(28,iw/(win/60)-3));
  for(const [t,v] of vis)if(v>0)g+=`<path d="${barPath(X(t)-bw/2,Y(v),bw,m.t+ih-Y(v))}" fill="${o.color}"/>`}
 else if(vis.length){const d=vis.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+','+Y(p[1]).toFixed(1)).join('');
  g+=`<path d="${d}L${X(vis[vis.length-1][0]).toFixed(1)},${m.t+ih}L${X(vis[0][0]).toFixed(1)},${m.t+ih}Z" fill="${o.color}" fill-opacity=".12"/>`+
   `<path d="${d}" fill="none" stroke="${o.color}" stroke-width="2" stroke-linejoin="round"/>`;
  const lp=vis[vis.length-1];g+=`<circle cx="${X(lp[0])}" cy="${Y(lp[1])}" r="4.5" fill="${o.color}" stroke="#1b1e23" stroke-width="2"/>`}
 g+='</g>';if(!vis.length)g+=`<text class="ax" x="${m.l+iw/2}" y="${m.t+ih/2}" text-anchor="middle">waiting for data…</text>`;
 g+=`<g class="xh"></g><rect class="hit" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/>`;
 el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${g}</svg>`;
 const svg=el.querySelector('svg'),hit=el.querySelector('.hit'),xh=el.querySelector('.xh');
 hit.onmousemove=e=>{if(!vis.length)return;const r=svg.getBoundingClientRect(),t=t0+((e.clientX-r.left)*W/r.width-m.l)/iw*win;
  let best=vis[0];for(const p of vis)if(Math.abs(p[0]-t)<Math.abs(best[0]-t))best=p;
  xh.innerHTML=`<line x1="${X(best[0])}" x2="${X(best[0])}" y1="${m.t}" y2="${m.t+ih}" stroke="#5d636e"/>`;
  tipAt(e,`<b>${new Date(best[0]*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})}</b>
   <div class="r"><span>${o.name}</span><span>${(o.fmt||fmtN)(best[1])}</span></div>`)};
 hit.onmouseleave=()=>{xh.innerHTML='';untip()}}
/* a tiny trend line under each KPI */
function spark(vals,col){const v=vals.map(x=>x==null?0:x);if(v.length<2||Math.max(...v)===0)return '<svg viewBox="0 0 100 38"></svg>';
 const mx=Math.max(...v),mn=Math.min(...v),sp=(mx-mn)||1,X=i=>i*100/(v.length-1),Y=x=>34-(x-mn)/sp*28;
 const d=v.map((x,i)=>(i?'L':'M')+X(i).toFixed(1)+','+Y(x).toFixed(1)).join('');
 return `<svg viewBox="0 0 100 38" preserveAspectRatio="none"><path d="${d}L100,38L0,38Z" fill="${col}" fill-opacity=".13"/>
  <path d="${d}" fill="none" stroke="${col}" stroke-width="1.8" vector-effect="non-scaling-stroke" stroke-linejoin="round"/></svg>`}
/* change between the first and second half of the range — better to say what it is than to imply a forecast */
function halves(vals,sum){const n=vals.length,h=Math.floor(n/2);if(h<2)return null;
 const f=x=>sum?x.reduce((p,c)=>p+(c||0),0):(x.filter(c=>c!=null).reduce((p,c)=>p+c,0)/Math.max(1,x.filter(c=>c!=null).length));
 const a=f(vals.slice(0,h)),b=f(vals.slice(n-h));return a?{a,b,pct:(b-a)/a}:null}
function pill(ch,goodUp,asPts){if(!ch)return '<span class="muted">no earlier period</span>';
 const d=asPts?(ch.b-ch.a)*100:ch.pct*100,t=asPts?`${Math.abs(d).toFixed(1)} pts`:`${Math.abs(d).toFixed(0)}%`;
 if(Math.abs(d)<(asPts?0.5:1))return '<span class="pill flat">▬ steady</span><span>vs earlier half</span>';
 const up=d>0,good=up===goodUp;return `<span class="pill ${good?'up':'dn'}">${up?'▲':'▼'} ${t}</span><span>vs earlier half</span>`}
function anKpis(a,days){const k=a.kpis,col=CC.s1;
 const ent=days.map(d=>d.entries),bil=days.map(d=>d.bills),rev=days.map(d=>d.revenue),
  bsk=days.map(d=>d.bills?d.revenue/d.bills:null),cv=days.map(d=>d.conversion),
  wait=days.map(d=>null);
 const card=(l,v,sub,ch,goodUp,vals,pts)=>`<div class="hk"><div class="l">${l}</div><div class="v">${v}</div>
  <div class="dl">${ch===undefined?`<span>${sub||''}</span>`:pill(ch,goodUp,pts)}</div>${vals?spark(vals,col):''}</div>`;
 $('akpis').innerHTML=card('Footfall',fmtN(k.footfall),'',halves(ent,true),true,ent)+
  card('Bills',fmtN(k.bills),'',halves(bil,true),true,bil)+
  card('Revenue',fmtRs(k.revenue),'',halves(rev,true),true,rev)+
  card('Average bill',fmtR(k.avg_basket),'',halves(bsk,false),true,bsk)+
  card('Conversion',fmtP(k.conversion),'',halves(cv,false),true,cv,true)+
  card('Avg queue wait',k.avg_wait_min==null?'—':k.avg_wait_min+' min',`${fmtN(k.items)} items sold · ${k.stockouts} stock-outs`,undefined,false,null)}
function anInsights(a,days){const k=a.kpis,out=[];
 if(k.peak_hour){const h=+k.peak_hour.slice(0,2);out.push(['⏱','Busiest hour',`Most people arrive around <em>${k.peak_hour}</em> — about <em>${fmtN(a.hourly_avg[h])}</em> entries an hour on an average day.`])}
 const bd=days.slice().sort((x,y)=>y.revenue-x.revenue)[0];
 if(bd&&bd.revenue)out.push(['₹','Best day',`<em>${dlong(bd.date)}</em> took <em>${fmtR(bd.revenue)}</em> from ${fmtN(bd.bills)} bills.`]);
 const tp=(a.top_products||[])[0];
 if(tp)out.push(['★','Best seller',`<em>${esc(tp.name)}</em> brought in <em>${fmtR(tp.revenue)}</em> (${fmtN(tp.units)} units).`]);
 const so=(a.stockouts||[])[0];
 if(so)out.push(['!','Ran out most',`<em>${esc(so.name)}</em> hit empty <em>${so.count}</em> times — worth a bigger shelf allocation.`]);
 const at=(a.attention||[])[0];
 if(at)out.push(['◉','Most looked at',`Shoppers stopped at <em>${esc(at.name)}</em> <em>${fmtN(at.stops)}</em> times, about ${at.avg_s} s each.`]);
 $('ains').innerHTML=out.slice(0,4).map(([i,t,x])=>`<div class="in"><div class="ic">${i}</div><div><b>${t}</b><span>${x}</span></div></div>`).join('')}
function anShop(s){const sh=s&&s.shoppers;if(!sh||!$('a_shop'))return;
 $('a_shop').innerHTML=`<div class="ashop"><div><b>${sh.issued}</b><span>IDs issued</span></div><div><b>${sh.inside}</b><span>inside now</span></div>
  <div><b>${sh.left_unbilled}</b><span>left without a bill</span></div><div><b style="${sh.unverified?'color:var(--low)':''}">${sh.unverified}</b><span>bills without a verified ID</span></div></div>`}
async function anLoad(){AN=await(await fetch('/api/analytics?days='+ADAYS)).json();anRender()}
const WD=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
const dshort=d=>new Date(d+'T00:00:00').toLocaleDateString('en-IN',{day:'numeric',month:'short'});
const dlong=d=>new Date(d+'T00:00:00').toLocaleDateString('en-IN',{weekday:'short',day:'numeric',month:'short'});
function anRender(){const a=AN;if(!a)return;const k=a.kpis;
 const b=$('demobanner');b.style.display=a.has_demo?'':'none';
 b.innerHTML=a.has_demo?`<b>Includes generated demo history.</b> Past days were filled in with
  <code>--demo-history</code> so the charts have something to show; those rows are tagged in the database and
  real trading is added on top. Remove them with <code>python storesense.py --clear-demo</code>.`:'';
 const t0=new Date(),today0=`${t0.getFullYear()}-${String(t0.getMonth()+1).padStart(2,'0')}-${String(t0.getDate()).padStart(2,'0')}`;
 let full=a.daily.filter(d=>d.date!==today0);if(!full.some(d=>d.entries||d.bills))full=a.daily;
 anKpis(a,full);anInsights(a,full);
 $('asub').textContent=`${dshort(a.from)} – ${dshort(a.to)} · ${a.days} days${a.has_demo?' · includes generated demo history':''}`;
 const t=new Date(),today=`${t.getFullYear()}-${String(t.getMonth()+1).padStart(2,'0')}-${String(t.getDate()).padStart(2,'0')}`;
 let days=a.daily.filter(d=>d.date!==today);if(!days.some(d=>d.entries||d.bills))days=a.daily;
 const dl=days.map(d=>dshort(d.date)),dL=days.map(d=>dlong(d.date));
 lineChart($('c_daily'),{labels:dl,tlabels:dL,series:[{name:'Entries',color:CC.s1,values:days.map(d=>d.entries)},
  {name:'Bills',color:CC.s2,values:days.map(d=>d.bills)}],h:300});
 lineChart($('c_conv'),{labels:dl,tlabels:dL,series:[{name:'Conversion',color:CC.s1,values:days.map(d=>d.conversion)}],
  fmt:fmtP,yfmt:fmtP,max:1,h:300});
 const hrs=[];for(let h=7;h<=22;h++)hrs.push(h);
 heat($('c_heat'),{rows:WD,cols:hrs.map(h=>String(h).padStart(2,'0')),m:a.weekday_hour.map(r=>hrs.map(h=>r[h])),name:'entries (avg)'});
 const hv=hrs.map(h=>a.hourly_avg[h]),pk=hv.indexOf(Math.max(...hv));
 barChart($('c_hour'),{labels:hrs.map(h=>h+''),tlabels:hrs.map(h=>`${h}:00–${h+1}:00`),values:hv,name:'entries a day',xname:'hour',hi:i=>i===pk,h:210});
 barChart($('c_rev'),{labels:dl,tlabels:dL,values:days.map(d=>d.revenue),name:'revenue',fmt:fmtR,yfmt:fmtRs,xname:'day',h:210});
 hbar($('c_top'),{items:a.top_products.map(p=>({label:p.name,value:p.revenue,
  rows:[['revenue',fmtR(p.revenue)],['units',p.units]]})),fmt:fmtR,name:'revenue'});
 hbar($('c_stock'),{items:a.stockouts.map(s=>({label:s.name,value:s.count})),name:'times empty',color:CC.s2});
 hbar($('c_look'),{items:(a.attention||[]).map(x=>({label:x.name,value:x.stops,
  rows:[['stops',fmtN(x.stops)],['average stop',x.avg_s+' s'],['total',x.total_min+' min']]})),name:'stops'});
 hbar($('c_zones'),{items:(a.zones||[]).map(z=>({label:z.zone,value:z.visits,
  rows:[['visits',fmtN(z.visits)],['average dwell',z.avg_dwell_s+' s']]})),name:'visits',color:CC.s2});
 barChart($('c_basket'),{labels:a.basket_hist.map(x=>x.bin),values:a.basket_hist.map(x=>x.n),name:'bills',xname:'items',h:180});
 barChart($('c_wait'),{labels:a.wait_hist.map(x=>x.bin),values:a.wait_hist.map(x=>x.n),name:'observations',xname:'wait',color:CC.s2,h:180});
 const rv=hrs.map(h=>a.revenue_by_hour[h]),rpk=rv.indexOf(Math.max(...rv));
 barChart($('c_revh'),{labels:hrs.map(h=>h+''),tlabels:hrs.map(h=>`${h}:00–${h+1}:00`),values:rv,name:'revenue a day',
  fmt:fmtR,yfmt:fmtRs,xname:'hour',hi:i=>i===rpk,h:180})}

async function intStatus(){try{const j=await(await fetch('/api/integrations')).json();
 $('intstat').textContent=j.webhooks.length?`Webhooks: ${j.webhooks.length} · ${j.sent} sent · ${j.queued} queued`+(j.last_error?` · last problem: ${j.last_error}`:'')
  :'No webhooks set — the calls above work without any setup.'}catch(e){}}
async function ack(id){await fetch('/api/alerts/'+id+'/ack',{method:'POST'})}
async function calib(cam){const j=await(await fetch('/api/calibrate/'+cam,{method:'POST'})).json();alert(j.message)}
async function setCounters(){await fetch('/api/counters?open='+$('ctr').value,{method:'POST'})}
async function loadLay(){const j=await(await fetch('/api/layout')).json();LAY=j.layout;SEL=null;DIRTY=false;
 $('b_save').className='';$('saved').textContent='';side();draw();draw3d()}
async function saveLay(){const j=await(await fetch('/api/layout',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(LAY)})).json();
 if(!j.ok)return alert(j.error||'save failed');
 LAY=j.layout;DIRTY=false;$('b_save').className='';const cs=j.cams||{},er=Object.values(cs.errors||{});
 $('saved').textContent='saved'+((cs.starting||[]).length?' · starting '+cs.starting.join(', '):'')+(er.length?' · '+er.join('; '):'');
 side();draw();draw3d()}
async function report(p){const j=await(await fetch('/api/report?period='+p)).json();
 const rows=[['Footfall',j.footfall_total],['Peak',j.peak||'—'],['Customers billed',j.customers_billed],
  ['Conversion',j.conversion==null?'—':Math.round(j.conversion*100)+'%'],
  ['Avg time in queue',j.avg_time_in_queue_min==null?'—':j.avg_time_in_queue_min+' min'],
  ['Avg staff response',j.avg_alert_response_s==null?'—':j.avg_alert_response_s+' s']];
 $('report').innerHTML=`<div class="rep">${rows.map(r=>`<div><b>${r[1]}</b><span>${r[0]}</span></div>`).join('')}</div>
  <p class="muted" style="margin:12px 0 0">Alerts: ${Object.entries(j.alerts_by_action).map(([k,v])=>k+' ×'+v).join(', ')||'none'} ·
  <a class="lnk" href="/api/report?period=${p}" target="_blank">Download JSON</a></p>`}
loadLay().then(()=>tab(location.hash.slice(1)||'home'));
</script></body></html>"""


_PLACEHOLDERS = {}


def placeholder_jpg(cam, msg):
    """A dark card with the camera name and, in words, why there's no picture."""
    key = (cam, msg)
    if key not in _PLACEHOLDERS:
        if len(_PLACEHOLDERS) > 200:
            _PLACEHOLDERS.clear()
        img = np.full((360, 640, 3), (25, 22, 20), np.uint8)
        cv2.circle(img, (34, 40), 7, (79, 54, 232), -1)
        cv2.putText(img, str(cam), (52, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
        words, lines, cur = str(msg).split(), [], ""
        for wd in words:
            if len(cur) + len(wd) + 1 > 58:
                lines.append(cur)
                cur = wd
            else:
                cur = (cur + " " + wd).strip()
        lines.append(cur)
        for i, ln in enumerate(lines[:9]):
            cv2.putText(img, ln, (26, 92 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 175), 1, cv2.LINE_AA)
        _PLACEHOLDERS[key] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    return _PLACEHOLDERS[key]


def check_roles(name, roles):
    """Role set for a camera, or raises ValueError with a message people can act on."""
    roles = {r.strip() for r in roles if r.strip()}
    if not roles or not roles <= {"entry", "queue", "shelf", "checkout"}:
        raise ValueError(f"{name}: unknown role in {sorted(roles)}")
    for solo in ("shelf", "checkout"):
        if solo in roles and len(roles) > 1:
            raise ValueError(f"{name}: a {solo} camera can't also do other jobs")
    return roles


def start_cam(engine, workers, name, roles, src, from_layout=False):
    roles = check_roles(name, roles)
    cls = ShelfWorker if "shelf" in roles else CheckoutWorker if "checkout" in roles else PeopleWorker
    w = cls(name, roles, src, engine)
    w.from_layout = from_layout
    workers[name] = w
    w.start()
    print(f"[{name}] {','.join(sorted(roles))} <- {w.cap.src}")
    return w


def sync_cams(engine, workers):
    """Make the running cameras match the saved store plan: start new ones, restart ones whose
    source or job changed, stop ones removed. No server restart needed after editing the plan."""
    plan = {c["id"]: c for c in engine.layout.data["cameras"] if str(c.get("source", "")).strip()}
    todo, errors = [], {}
    for name, w in list(workers.items()):
        c = plan.get(name)
        same = c and normalise_source(c["source"]) == normalise_source(w.source) and set(c["roles"]) == set(w.roles)
        if same:
            continue
        if c is None and not getattr(w, "from_layout", False):
            continue                            # started with --cam and no source in the plan: leave it
        w.stop()
        workers.pop(name, None)
        with engine.lock:
            engine.cams.pop(name, None)
        print(f"[{name}] stopped")
    for name, c in plan.items():
        if name in workers:
            continue
        try:
            check_roles(name, c["roles"])
            todo.append((name, c["roles"], c["source"]))
            with engine.lock:
                engine.cams[name] = {"roles": sorted(c["roles"]), "online": False, "fps": 0, "people": 0,
                                     "msg": "starting…"}
        except ValueError as e:
            errors[name] = str(e)

    def go():
        for name, roles, src in todo:
            try:
                start_cam(engine, workers, name, roles, src, from_layout=True)
            except Exception as e:
                with engine.lock:
                    engine.cams[name] = {"roles": sorted(roles), "online": False, "fps": 0, "people": 0,
                                         "msg": f"couldn't start: {type(e).__name__}: {e}"}
    threading.Thread(target=go, daemon=True).start()   # loading models takes a moment
    return {"starting": [t[0] for t in todo], "errors": errors}


def make_app(engine, workers, store):
    import asyncio
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
    from fastapi.responses import HTMLResponse, Response, StreamingResponse

    app = FastAPI(title="StoreSense Edge")
    from inventory.routes import make_router
    app.include_router(make_router(engine.inventory))

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

    @app.post("/api/shelf/{cam}/refilled")
    async def shelf_refilled(cam: str, req: Request):
        """Staff topped a product box up: drop the doubt about how many packs are behind the front one."""
        w = workers.get(cam)
        if not isinstance(w, ShelfWorker):
            raise HTTPException(404, "not a shelf camera")
        w.refilled((await req.json()).get("slot", ""))
        return {"ok": True}

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
        with engine.lock:                       # only a new store size needs a new grid; moving a shelf keeps the day's heat
            if engine.store_heat is None or engine.store_heat.shape != lay.grid_dims():
                engine.store_heat = np.zeros(lay.grid_dims(), np.float32)
                engine.store_live = np.zeros_like(engine.store_heat)
        for w in workers.values():              # cameras may have moved -> rebuild their mappings
            if hasattr(w, "storeH"):
                w.storeH = w._sp_sig = None
                if hasattr(w, "H"):
                    w.H = None
                    w._quad = w._rect = None
        cams = sync_cams(engine, workers)       # new/changed sources start now, no restart
        return {"ok": True, "layout": data, "coverage": coverage(data), "cams": cams}

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
        engine.on_pos(data)
        return {"ok": True}

    @app.get("/api/integrations/stock")
    def int_stock():
        """For an inventory/ERP system to pull: every product, where it sits, what the camera sees,
        what the till count says, and its shelf status."""
        snap = engine.snapshot()
        by = defaultdict(list)
        for cam, cells in snap["shelves"].items():
            for c in cells or []:
                if "slot" in c:
                    by[c.get("sku") or f"{cam}:{c['slot']}"].append(
                        {"cam": cam, "where": f"{snap['labels'].get(cam, cam)}, {c['loc']}", "camera_units": c["est_units"],
                         "full_units": c["full_units"], "method": c.get("method"), "status": c["status"],
                         "wrong_product": c.get("misplaced", False)})
        out = []
        for p in store.products():
            locs = by.pop(p["sku"], [])
            till = engine.pos_units(p["sku"], None) if p["sku"] in engine.stock else None
            out.append({**p, "till_units": till, "shelves": locs})
        for key, locs in by.items():             # boxes not linked to a catalog product
            out.append({"sku": key, "name": None, "till_units": None, "shelves": locs})
        return {"ts": time.time(), "store_id": CONFIG["store_id"], "products": out}

    @app.get("/api/integrations/sales")
    def int_sales(since: float = 0):
        """Bills since a timestamp, for an ERP to pull (real bills only, not demo history)."""
        rows = store.q("SELECT id,ts,items,mrp_total,total,n_items FROM bills WHERE ts>? AND demo=0 ORDER BY ts", since)
        return {"bills": [{"id": i, "ts": t, "items": json.loads(it), "mrp_total": m, "total": tot, "n_items": n}
                          for i, t, it, m, tot, n in rows]}

    @app.post("/api/integrations/products")
    async def int_products(req: Request):
        """Bulk catalog import from an ERP: a JSON list of products, or CSV with a header row
        (sku, name, brand, barcode, mrp, price)."""
        raw = (await req.body()).decode("utf-8", "replace")
        try:
            items = json.loads(raw)
            items = items.get("products", items) if isinstance(items, dict) else items
        except ValueError:
            import csv, io
            items = list(csv.DictReader(io.StringIO(raw)))
        done, errors = 0, []
        for i, it in enumerate(items):
            try:
                it = {k.strip().lower(): v for k, v in it.items() if k}
                for k in ("mrp", "price"):
                    if it.get(k) not in (None, ""):
                        it[k] = float(it[k])
                    else:
                        it.pop(k, None)
                r = store.product_upsert(it)
                if isinstance(r, dict) and r.get("error"):
                    errors.append(f"row {i + 1}: {r['error']}")
                else:
                    done += 1
            except Exception as e:
                errors.append(f"row {i + 1}: {e}")
        return {"ok": not errors, "imported": done, "errors": errors[:20]}

    @app.post("/api/integrations/restock")
    async def int_restock(req: Request):
        """A delivery recorded in the inventory system: {"items": [{"sku": "...", "qty": 12}], "mode": "add"|"set"}."""
        body = await req.json()
        return {"ok": True, "levels": engine.delivered(body.get("items", []), body.get("mode", "add"))}

    @app.get("/api/integrations")
    def int_status():
        pend = store.q("SELECT COUNT(*) FROM hooks WHERE sent=0")[0][0]
        return {"webhooks": CONFIG.get("webhooks") or [], "queued": pend, **engine.hook_state}

    def latest_jpg(cam, view):
        """Newest picture of a camera for the dashboard — or a card saying why there isn't one."""
        w = workers.get(cam)                    # looked up every time: follows restarts from Setup
        st = engine.cams.get(cam) or {}
        if not w:
            return placeholder_jpg(cam, "not running — give it a source in Setup and save the plan")
        if not st.get("online"):
            return placeholder_jpg(cam, st.get("msg") or "connecting…")
        if view == "cctv":
            w.cctv_want = time.time()           # keep the plain view rendered while someone watches
            return w.live_jpg("cctv") or w.cctv_jpg or w.jpg or placeholder_jpg(cam, "starting…")
        if view == "raw":                       # picture only (faces blurred): the Shelves tab draws its own boxes
            return w.live_jpg("raw") or w.jpg or placeholder_jpg(cam, "starting…")
        return w.live_jpg("analytics") or w.jpg or placeholder_jpg(cam, "starting…")

    @app.get("/api/frame/{cam}.jpg")
    def frame_jpg(cam: str, view: str = "analytics"):
        """One frame. The dashboard asks for the next only when the last has arrived, so a slow link
        drops frames instead of falling behind, and no connections are held open."""
        return Response(latest_jpg(cam, view), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/video/{cam}")
    def video(cam: str, view: str = "analytics"):
        """MJPEG stream, for VLC or other viewers outside the dashboard."""
        def gen():
            last = None
            while True:
                f = latest_jpg(cam, view)
                if f is not last:
                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
                    last = f
                time.sleep(0.05)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

    # ── shelves: the still to draw product boxes on, and the boxes themselves ──
    @app.get("/api/shelf/{cam}/still.jpg")
    def shelf_still(cam: str):
        """The calibrated 'full shelf' picture (no shoppers in it) — the right thing to mark products on."""
        w = workers.get(cam)
        img = getattr(w, "ref", None) if w else None
        if img is None:
            p = f"shelf_ref_{cam}.png"
            img = cv2.imread(p) if os.path.exists(p) else None
        if img is None:
            if w and w.jpg:
                return Response(w.jpg, media_type="image/jpeg")
            raise HTTPException(404, "no picture yet")
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), media_type="image/jpeg")

    @app.get("/api/depth/{cam}.jpg")
    def depth_view(cam: str):
        """Latest depth pass of a shelf camera, coloured, with each column's unit count."""
        w = workers.get(cam)
        if not hasattr(w, "depth_want"):
            jpg = placeholder_jpg(cam, "not a shelf camera, or not running")
        else:
            if time.time() - w.depth_want > 15:
                w.depth_t = 0.0                 # someone started watching: measure on the next shelf read
            w.depth_want = time.time()
            jpg = w.depth_jpg or placeholder_jpg(cam, (w.depth_state or {}).get("msg") if not w.depth_state.get("on")
                                                 and DEPTH["err"] else "measuring depth — the first picture takes a few seconds")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/slots/{cam}")
    def get_slots(cam: str):
        lay = engine.layout
        c = lay.cam(cam) if lay else None
        if not c:
            raise HTTPException(404, "camera not in the store plan")
        slots = c.get("slots", [])
        locs = slot_locations(slots)
        prods = {p["sku"]: p for p in store.products()}
        return {"slots": [{**sl, "row": locs.get(sl["id"], (0, 0))[0], "col": locs.get(sl["id"], (0, 0))[1],
                           "product": prods.get(sl.get("sku"))} for sl in slots],
                "calibrated": bool(getattr(workers.get(cam), "ref", None) is not None)}

    @app.post("/api/slots/{cam}")
    async def post_slots(cam: str, req: Request):
        lay = engine.layout
        if not lay or not lay.cam(cam):
            raise HTTPException(404, "camera not in the store plan")
        body = await req.json()
        data = json.loads(json.dumps(lay.data))
        for c in data["cameras"]:
            if c["id"] == cam:
                c["slots"] = body.get("slots", [])
        try:
            lay.save(data)
        except (KeyError, TypeError, ValueError) as e:
            return {"ok": False, "error": f"bad slots: {e}"}
        return {"ok": True, "slots": lay.cam(cam).get("slots", [])}

    # ── product catalog ──
    @app.get("/api/products")
    def list_products():
        return {"products": store.products()}

    @app.post("/api/products")
    async def save_product(req: Request):
        try:
            return {"ok": True, "product": store.product_upsert(await req.json())}
        except (ValueError, TypeError) as e:
            return {"ok": False, "error": str(e)}

    @app.delete("/api/products/{sku}")
    def delete_product(sku: str):
        try:
            store.product_delete(sku)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        return {"ok": True}

    @app.get("/api/products/{sku}/label.png")
    def product_label(sku: str):
        p = store.product_get(sku)
        if not p:
            raise HTTPException(404)
        buf = __import__("io").BytesIO()
        ean13_image(p["barcode"], f"{p['name']}  ₹{p['price']:.2f}").save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png")

    # ── checkout ──
    @app.post("/api/cart")
    def new_cart():
        return engine.cart_view(engine.cart_new())

    @app.get("/api/cart/{cid}")
    def get_cart(cid: str):
        v = engine.cart_view(cid)
        if not v:
            raise HTTPException(404, "no such cart")
        return v

    @app.post("/api/cart/{cid}/scan")
    async def scan(cid: str, req: Request):
        b = await req.json()
        p, err = engine.cart_add(cid, b.get("code", ""), int(b.get("qty", 1)), "screen")
        return {"ok": p is not None, "error": err, "product": p, "cart": engine.cart_view(cid)}

    @app.post("/api/cart/{cid}/set")
    async def set_qty(cid: str, req: Request):
        b = await req.json()
        engine.cart_set(cid, b.get("sku", ""), int(b.get("qty", 0)))
        return engine.cart_view(cid) or {"lines": []}

    @app.post("/api/cart/{cid}/shopper")
    async def cart_shopper(cid: str, req: Request):
        """Staff picks (or types) the shopper this cart belongs to; an empty ID clears it."""
        b = await req.json()
        err = engine.cart_shopper(cid, b.get("shopper", ""))
        return {"ok": err is None, "error": err, "cart": engine.cart_view(cid)}

    @app.post("/api/cart/{cid}/checkout")
    async def do_checkout(cid: str, req: Request):
        try:
            body = await req.json()
        except Exception:
            body = {}
        bill, err = engine.checkout(cid, (body or {}).get("shopper"))
        return {"ok": bill is not None, "error": err, "bill": bill}

    @app.get("/api/shoppers")
    def shoppers():
        """Everyone currently inside, by anonymous ID, and today's entry/billing tally."""
        with engine.lock:
            return engine.shopper_view()

    @app.get("/api/bills/{bid}.{ext}")
    def get_bill(bid: str, ext: str):
        if ext not in ("png", "pdf") or not all(ch.isalnum() or ch == "-" for ch in bid):
            raise HTTPException(404)
        path = os.path.join(CONFIG["bills_dir"], f"{bid}.{ext}")
        if not os.path.exists(path):
            b = store.bill_get(bid)
            if not b:
                raise HTTPException(404)
            png, pdf = render_bill(b)
            data = png if ext == "png" else pdf
        else:
            with open(path, "rb") as fh:
                data = fh.read()
        return Response(data, media_type="image/png" if ext == "png" else "application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{bid}.{ext}"'})

    @app.get("/api/queue")
    def queue_api(days: int = 14):
        return queue_analytics(store, days)

    @app.get("/api/analytics")
    def get_analytics(days: int = 14):
        return analytics(store, days)

    @app.get("/api/live")
    def live(minutes: int = 60):
        """Per-minute history (real trading only) that the live charts start from."""
        since = time.time() - max(1, min(minutes, 24 * 60)) * 60
        keys = ("inside", "entries", "queue_len", "revenue", "bills")
        rows = defaultdict(dict)
        for ts, k, v in store.q(f"SELECT ts,key,value FROM metrics WHERE ts>=? AND demo=0 AND key IN "
                                f"({','.join('?' * len(keys))}) ORDER BY ts", since, *keys):
            rows[round(ts)][k] = v
        return {"rows": [{"t": t, **d} for t, d in sorted(rows.items())], "now": time.time()}

    @app.get("/api/export/{name}.csv")
    def export(name: str, days: int = 30):
        if name == "timeseries_hourly":
            body = to_csv(hourly_timeseries(store, days))
        elif name == "bills":
            since = day_start() - (max(1, days) - 1) * 86400
            body = to_csv([{"bill": i, "time": _local(t).strftime("%Y-%m-%d %H:%M:%S"), "items": n,
                            "mrp_total": m, "total": tot, "demo": dm}
                           for i, t, n, m, tot, dm in store.q(
                               "SELECT id,ts,n_items,mrp_total,total,demo FROM bills WHERE ts>=? ORDER BY ts", since)])
        elif name == "timeseries_minute":
            p = os.path.join(CONFIG["analytics_dir"], "timeseries.csv")
            body = open(p).read() if os.path.exists(p) else ",".join(TS_FIELDS) + "\n"
        elif name == "products":
            body = to_csv(store.products())
        else:
            raise HTTPException(404)
        return Response(body, media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})

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


def depth_test(paths):
    """Run the depth model on your own shelf photos. With two photos taken from the same spot —
    shelf full, then with some front units removed — it shows how far back each part moved (cm)."""
    imgs = [cv2.imread(p) for p in paths[:2]]
    if any(i is None for i in imgs):
        sys.exit("could not read " + ", ".join(p for p, i in zip(paths, imgs) if i is None))
    t = time.time()
    d0 = run_depth(imgs[0])
    if d0 is None:
        sys.exit(f"depth model {DEPTH['err']}")
    print(f"depth pass: {time.time() - t:.2f} s (first pass includes loading)")
    h, w = d0.shape
    print(f"distance at the centre of {paths[0]}: {float(np.median(d0[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5])):.2f} m")
    tiles = [imgs[0], depth_colour(d0)]
    if len(imgs) == 2:
        if imgs[1].shape != imgs[0].shape:
            imgs[1] = cv2.resize(imgs[1], (w, h))
        d1 = run_depth(imgs[1])
        diff = cv2.absdiff(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY))
        stable = cv2.GaussianBlur(diff, (21, 21), 0) < 12      # parts of the picture that did not change
        da, noise = align_depth(d1, d0, stable)
        if noise is None:
            print("the two photos don't line up — take both from exactly the same spot")
        else:
            moved = np.clip((da - d0) * 100, 0, 30)             # cm further away than before
            vis = cv2.applyColorMap((moved / 30 * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            vis[stable] = (vis[stable] * 0.35).astype(np.uint8)
            cv2.putText(vis, "moved back, 0-30 cm", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            print(f"fit error on unchanged parts: {noise * 100:.1f} cm")
            ch = ~stable
            if ch.any():
                print(f"changed area moved back by a median {float(np.median((da - d0)[ch])) * 100:.1f} cm")
            tiles += [imgs[1], vis]
    out = np.hstack([cv2.resize(x, (w // 2, h // 2)) for x in tiles])
    cv2.imwrite("depth_test.png", out)
    print("wrote depth_test.png")


def deep_merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = v


def main():
    ap = argparse.ArgumentParser(description="StoreSense Edge — on-device retail intelligence")
    ap.add_argument("--cam", nargs=3, action="append", metavar=("NAME", "ROLES", "SOURCE"),
                    help="roles: entry, queue (can combine: entry,queue), shelf, or checkout")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", help="optional JSON merged over CONFIG")
    ap.add_argument("--sku-model", help="optional YOLO SKU weights for planogram checks")
    ap.add_argument("--cloud-url")
    ap.add_argument("--layout", help="store layout JSON written by the dashboard editor")
    ap.add_argument("--demo-history", type=int, metavar="DAYS",
                    help="fill the analytics with generated past trading (tagged as demo data)")
    ap.add_argument("--clear-demo", action="store_true", help="delete generated demo data and exit")
    ap.add_argument("--pick", metavar="SOURCE", help="click points on a frame to get geometry coords")
    ap.add_argument("--depth-test", nargs="+", metavar="PHOTO",
                    help="check the depth model: one shelf photo, or two (full, then with units taken) "
                         "to see how far back each spot moved; writes depth_test.png and exits")
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
    if args.depth_test:
        return depth_test(args.depth_test)

    try:                       # nothing leaves the device: kill the model library's usage telemetry
        from ultralytics import settings as ul_settings
        ul_settings.update({"sync": False})
    except Exception:
        pass

    store = Store(CONFIG["db_path"])
    if args.clear_demo:
        store.clear_demo()
        return print("demo data removed")
    if args.demo_history:
        r = seed_demo(store, args.demo_history)
        print(f"generated {r['days']} days of demo history: {r['visits']} visits, {r['bills']} bills "
              f"(tagged demo — remove with --clear-demo)")
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
        try:
            start_cam(engine, workers, name, roles.split(","), src, from_layout=not args.cam)
        except ValueError as e:
            sys.exit(str(e))

    threading.Thread(target=ticker, args=(engine, store), daemon=True).start()
    threading.Thread(target=cloud_sync, args=(engine, store), daemon=True).start()
    threading.Thread(target=webhook_sender, args=(engine, store), daemon=True).start()

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
