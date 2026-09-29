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
        "period_s": 3.0,
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
    "queue": {
        "open_counters": 1, "max_counters": 4,
        "default_service_per_min": 1.5,   # prior per counter, learned online
        "target_wait_min": 3.0, "max_wait_min": 6.0,
        "min_time_in_zone_s": 4.0, "rate_window_min": 5.0,
    },
    "store_name": "StoreSense Mart", # printed on bills
    "bills_dir": "bills",           # PNG + PDF of every bill
    "analytics_dir": "analytics",   # per-minute CSV for forecasting models
    "pos": {"rescan_s": 2.0},       # an item must leave the checkout camera's view this long to count again
    "layout_path": "layout.json",   # the store model: shelves and cameras in metres
    "store_cell_m": 0.25,           # floor-heatmap resolution in the store frame
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
DEFAULT_LAYOUT = {"store": {"w": 6.0, "h": 4.0}, "shelves": [], "cameras": []}


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
                v = face_visibility(c, s, f, self.data["shelves"])
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
    H = 430 + rows * 44
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
                CREATE INDEX IF NOT EXISTS ix_ev ON events(type, ts);
                CREATE INDEX IF NOT EXISTS ix_m ON metrics(key, ts);
                CREATE TABLE IF NOT EXISTS products(sku TEXT PRIMARY KEY, barcode TEXT, name TEXT,
                    brand TEXT, mrp REAL, price REAL, updated REAL);
                CREATE TABLE IF NOT EXISTS bills(id TEXT PRIMARY KEY, ts REAL, items TEXT, mrp_total REAL,
                    total REAL, n_items INTEGER, demo INTEGER DEFAULT 0);
                CREATE INDEX IF NOT EXISTS ix_bill ON bills(ts);
            """)
            for tbl in ("events", "metrics"):           # rows made by --demo-history are tagged, never mixed silently
                cols = [r[1] for r in self.db.execute(f"PRAGMA table_info({tbl})")]
                if "demo" not in cols:
                    self.db.execute(f"ALTER TABLE {tbl} ADD COLUMN demo INTEGER DEFAULT 0")
            self.db.commit()

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
            self.db.execute("DELETE FROM products WHERE sku=?", (sku,))
            self.db.commit()

    # ── bills ────────────────────────────────────────────────────────
    def next_bill_no(self, ts=None):
        day = datetime.fromtimestamp(ts or time.time()).strftime("%Y%m%d")
        n = self.q("SELECT COUNT(*) FROM bills WHERE id LIKE ?", f"SS-{day}-%")[0][0]
        return f"SS-{day}-{n + 1:04d}"

    def bill_save(self, b, demo=0):
        with self.lock:
            self.db.execute("INSERT INTO bills VALUES(?,?,?,?,?,?,?)",
                            (b["id"], b["ts"], json.dumps(b["lines"]), b["mrp_total"], b["total"], b["n_items"], demo))
            self.db.commit()

    def bill_get(self, bid):
        r = self.q("SELECT id,ts,items,mrp_total,total,n_items FROM bills WHERE id=?", bid)
        if not r:
            return None
        i, ts, items, mrp, tot, n = r[0]
        return {"id": i, "ts": ts, "lines": json.loads(items), "mrp_total": mrp, "total": tot, "n_items": n,
                "savings": round(mrp - tot, 2)}

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
        self.cam_face = {}            # camera -> "shelfId:FACE", so the 3D view can colour it
        self.depth_info = {}          # shelf camera -> depth model status
        self.stock, self.sold = {}, {}   # sku -> units at last restock / sold since then (POS)
        self.carts, self.active_cart, self.last_scan = {}, None, None
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
                "total": round(total, 2), "savings": round(mrp_total - total, 2)}

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

    def checkout(self, cid):
        v = self.cart_view(cid)
        if not v or not v["lines"]:
            return None, "Cart is empty"
        ts = time.time()
        bill = {**v, "id": self.store.next_bill_no(ts), "ts": ts}
        self.store.bill_save(bill)
        png, pdf = render_bill(bill)
        os.makedirs(CONFIG["bills_dir"], exist_ok=True)
        for ext, data in (("png", png), ("pdf", pdf)):
            with open(os.path.join(CONFIG["bills_dir"], f"{bill['id']}.{ext}"), "wb") as fh:
                fh.write(data)
        items = [{"sku": ln["sku"], "qty": ln["qty"], "amount": ln["amount"]} for ln in bill["lines"]]
        self.store.event("pos", "pos", {"bill": bill["id"], "items": items, "total": bill["total"],
                                        "n_items": bill["n_items"]})
        self.on_pos({"items": items})
        self.refresh_daily()
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
    def on_entry(self, cam, direction):
        with self.lock:
            self.footfall[direction] += 1
            if direction == "in":
                hh = datetime.now().strftime("%H")
                self.hourly[hh] = self.hourly.get(hh, 0) + 1
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
            if "slot" in c:                          # a marked product, not an anonymous grid cell
                key = f"shelf:{cam}:{c['slot']}"
                loc = f"{c['name']} — {where}, {c.get('loc', '')}"
            else:
                key, loc = f"shelf:{cam}:{c['r']},{c['c']}", f"{where} row {c['r'] + 1} col {c['c'] + 1}"
            if c["status"] == "EMPTY":
                self.fire(key, "stock", 3, f"{loc} is empty", "Critical refill", w)
            elif c["status"] == "LOW":
                left = f" — about {c['est_units']} of {c['full_units']} left" if "slot" in c else \
                       f" ({c['fill'] * 100:.0f}%)"
                self.fire(key, "stock", 2, f"{loc} low{left}", "Refill", w)
            elif c["eta_min"] is not None and c["eta_min"] < 15:
                self.fire(key, "stock", 2, f"{loc} expected empty in ~{c['eta_min']:.0f} min", "Refill soon", w)
            else:
                self.resolve(key)                        # cell restocked
            if c["status"] == "MISPLACED":
                exp = f" (expected {c['expected']}, found {c['found']})" if c.get("expected") else ""
                self.fire(key + ":pg", "planogram", 1, f"{loc} planogram mismatch{exp}", "Correct shelf", w)
            else:
                self.resolve(key + ":pg")

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
        with self.lock:
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
                "labels": dict(self.labels), "cam_face": dict(self.cam_face),
                "depth": dict(self.depth_info),
                "pos_tracked": len(self.stock),
                "pos": {"active_cart": self.active_cart, "last_scan": self.last_scan},
                "sales_today": getattr(self, "sales_today", 0), "bills_today": getattr(self, "bills_today", 0),
                "hour_now": datetime.now().hour,
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
        self.source, self.alive = source, True
        self.cap = Capture(source)
        self.jpg, self.fps = None, 0.0
        self.t_prev = self.t_start = self.t_shown = time.time()
        self.cctv_jpg, self.cctv_want, self.people = None, 0.0, 0

    def cctv(self, frame, boxes):
        """Plain security view: the picture plus people boxes, no analytics overlays. Heads are
        still blurred — privacy is a property of the system, not of which tab you open. Only
        encoded while someone is watching the CCTV tab, to spare the edge CPU."""
        self.people = len(boxes)
        if time.time() - self.cctv_want > 6:
            return
        v = frame.copy()
        if CONFIG["privacy_blur"]:
            blur_heads(v, [b[:4] for b in boxes])
        for b in boxes:
            x1, y1, x2, y2 = map(int, b[:4])
            cv2.rectangle(v, (x1, y1), (x2, y2), (60, 60, 230), 2)
            if len(b) > 4:
                cv2.putText(v, f"#{b[4]}", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 230), 1)
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
        R, C = self.g["floor_grid"]
        self.heat = np.zeros((R, C), np.float32)
        engine.heat[name] = self.heat
        self.H = None
        self.zone_time, self.zone_last = defaultdict(dict), defaultdict(dict)
        self.cand, self.members, self.missing = {}, {}, {}
        self.arrivals, self.departures = deque(), deque()
        self.mu_c = CONFIG["queue"]["default_service_per_min"]
        self.storeH = None            # image -> store metres, built on the first frame
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
        self.g = self.geo()
        r = self.model.track(frame, persist=True, classes=[0], conf=CONFIG["conf"], imgsz=CONFIG["imgsz"],
                             tracker="bytetrack.yaml", verbose=False)[0]
        dets = []
        if r.boxes is not None and r.boxes.id is not None:
            for bb, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
                dets.append((*map(float, bb), tid))
        self.cctv(frame, dets)
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


def shelf_frame(P0, slots, w, h):
    """From the calibrated (full) shelf: the plane the product fronts stand on, and for every column
    the patch of that plane its front unit covers. A unit taken from the front leaves the next one
    further back along the plane's normal — inside the same 'tube', wherever that lands in the picture."""
    per = {}
    for s in slots:
        for box in ShelfWorker.facing_boxes(s, w, h):
            a, b, c2, d2 = depth_region(box, w, h)
            q = P0[b:d2, a:c2].reshape(-1, 3)
            q = q[np.isfinite(q).all(1)]
            if len(q) >= 12:
                per[(s["id"], box[0])] = (q, s)
    if len(per) < 2:
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
    cols = {}
    for k, (q, s) in per.items():
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
        # last: points a unit face must show (live passes use every 2nd pixel each way)
        cols[k] = (a0 + ma, a1 - ma, b0 + mb, b1 - mb, front0, max(6, int(on.sum() / 4 * 0.15)))
    return {"c": c, "n": n, "e1": e1, "e2": e2, "cols": cols}


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
                engine.cam_face[name] = f"{sh['id']}:{fc}"
                n_sl = len((engine.layout.cam(name) or {}).get("slots") or [])
                print(f"[{name}] watching {sh['name']} {fc} face, " +
                      (f"{n_sl} product boxes" if n_sl else f"{self.R}x{self.C} grid (no product boxes yet)"))
        self.clahe = cv2.createCLAHE(2.0, (8, 8))
        self.ref_path = f"shelf_ref_{name}.png"
        self.ref = cv2.imread(self.ref_path) if os.path.exists(self.ref_path) else None
        self.ref_feats = self.features(self.ref) if self.ref is not None else None
        self.recent = defaultdict(lambda: deque(maxlen=3))
        self.history = defaultdict(lambda: deque(maxlen=600))
        self.cells_map, self.calib_request, self.calib_msg = {}, False, None
        self.slot_ref, self.slot_sig = {}, None      # per-facing reference edges, rebuilt when slots change
        # depth: the calibration pass (from the stored reference picture), per-column history, last result
        self.depth_ref, self.depth_t, self.depth_jpg = None, 0.0, None
        self.depth_geo, self.depth_geo_sig, self.depth_stack = None, None, []
        self.depth_back = defaultdict(lambda: deque(maxlen=CONFIG["shelf"]["depth"]["smooth"]))
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
        for s in slots:
            self.slot_ref[s["id"]] = [float(e[y0:y1, x0:x1].mean()) if e[y0:y1, x0:x1].size else 0.0
                                      for _, x0, y0, x1, y1 in self.facing_boxes(s, w, h)]
        self.slot_sig = json.dumps(slots, sort_keys=True)

    def depth_pass(self, frame, slots, persons):
        """Measure how far back the front unit of every depth-enabled column now sits, versus the
        calibration picture. Adds one reading (in units) per column to its short history."""
        want = [s for s in slots if s.get("unit_cm", 0) > 0]
        if not want:
            self.depth_state = {"on": False, "msg": "no product box has a unit depth set", "noise_cm": None}
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
        if len(self.depth_stack) < n_cal:   # the full-shelf depth is the median of a few passes
            self.depth_stack.append(da)
            self.depth_ref = np.median(np.stack(self.depth_stack), 0).astype(np.float32)
            self.depth_geo = None
            self.depth_state = {"on": False, "noise_cm": None,
                                "msg": f"measuring the full shelf ({len(self.depth_stack)}/{n_cal}) — leave it untouched"}
            return
        hfov = self.hfov()
        if self.depth_geo is None or self.depth_geo_sig != (self.slot_sig, hfov):
            self.depth_geo = shelf_frame(backproject(self.depth_ref, hfov), want, w, h)
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
        self.depth_state = {"on": True, "msg": f"{DEPTH['name'] or 'depth model'} · re-anchored every pass",
                            "noise_cm": round(noise * 100, 1)}
        # picture for the dashboard: nearer = brighter red, with each column's count
        fronts = np.concatenate([self.depth_ref[int(s["y"] * h):int((s["y"] + s["h"]) * h),
                                                 int(s["x"] * w):int((s["x"] + s["w"]) * w)].ravel() for s in want])
        fronts = fronts[np.isfinite(fronts)]
        lo = float(np.percentile(fronts, 5)) - 0.05 if fronts.size else 0.5
        reach = max(s["deep"] * s["unit_cm"] / 100 for s in want) + 0.15
        img = depth_colour(da, lo, lo + reach)          # colour range = the shelf's own depth
        for s in want:
            for box in self.facing_boxes(s, w, h):
                i, a, b, c2, d2 = box
                back = self.depth_units_back(s, i)
                label = "?" if self.depth_hidden(s, i) else ("" if back is None else str(max(0, s["deep"] - back)))
                cv2.rectangle(img, (a, b), (c2, d2), (220, 220, 220), 1)
                if label:
                    cv2.putText(img, label, (a + 3, b + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "depth: nearer = brighter, number = units left in that column, ? = hidden", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
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

    def read_slots(self, frame, slots, persons):
        sc = CONFIG["shelf"]
        h, w = frame.shape[:2]
        now = time.time()
        sig = json.dumps(slots, sort_keys=True)
        if sig != self.slot_sig:
            self.build_slot_ref(slots)
            self.depth_back.clear()
        if now - self.depth_t >= sc["depth"]["every_s"]:
            self.depth_t = now
            self.depth_pass(frame, slots, persons)
        e = self.edge_map(frame)
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
            present, fills, cols = 0, [], []
            nf = max(1, s["facings"])
            backs = [self.depth_units_back(s, i) for i in range(nf)] \
                if s.get("unit_cm", 0) > 0 and self.depth_state["on"] else [None] * nf
            for i, a, b, c2, d2 in self.facing_boxes(s, w, h):
                cur = float(e[b:d2, a:c2].mean()) if e[b:d2, a:c2].size else 0.0
                r = ref[i] if i < len(ref) else 0.0
                f = min(1.0, cur / r) if r > 1e-4 else (1.0 if cur > 0.02 else 0.0)
                fills.append(f)
                if backs[i] is not None:     # depth: how many are gone from the front of this column
                    left = max(0, s["deep"] - backs[i])
                else:                        # not measured by depth: front-row rule
                    left = s["deep"] if f >= sc["slot_low"] else 0
                    if self.depth_hidden(s, i):
                        # nothing solid in the column's tube: its front unit is certainly gone,
                        # but neighbours hide how far back the rest goes from this angle
                        left = min(left, s["deep"] - 1)
                cols.append(left)
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
            status = "EMPTY" if est == 0 else ("LOW" if est / full <= 0.5 else "OK")
            p = self.engine.store.product_get(s.get("sku", "")) if s.get("sku") else None
            r_, c_ = locs.get(s["id"], (0, 0))
            cell = {"slot": s["id"], "name": p["name"] if p else s["name"], "sku": s.get("sku", ""),
                    "brand": p["brand"] if p else "", "price": p["price"] if p else None,
                    "row": r_, "col": c_, "loc": f"row {r_} · col {c_}",
                    "facings": n, "deep": s["deep"], "present": present,
                    # "depth": each column counted by how far back its front unit sits (depth model)
                    # "front": facings still visible × stated depth — cannot see behind row 1
                    "est_units": est, "method": method,
                    "columns": cols if method != "front" else None,
                    "hidden": [i for i in range(nf) if self.depth_hidden(s, i)],
                    "full_units": full, "fill": round(fill_s, 2),
                    "status": status, "occluded": False,
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
                self.slot_sig = None                 # remeasure every facing against the new picture
                self.depth_ref, self.depth_t, self.depth_geo = None, 0.0, None   # re-derived from the new picture
                self.depth_back.clear()
                sl = self.slots()
                if sl:
                    self.build_slot_ref(sl)
                    self.engine.restock([s for s in sl if s.get("sku")])
                self.calib_msg = "Calibrated"

        self.cctv(frame, persons)
        vis = frame.copy()
        if CONFIG["privacy_blur"]:
            blur_heads(vis, persons)
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
                tag = "~" if c["method"] == "front" else ""
                cv2.putText(vis, f"{s['name']} {tag}{c['est_units']}/{c['full_units']}", (x0 + 4, max(12, y0 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
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
        self.persons, self.n = [], 0

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
        self.cctv(frame, self.persons)
        vis = frame.copy()
        if CONFIG["privacy_blur"]:
            blur_heads(vis, self.persons)
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
        "zones": [{"zone": z, "visits": len(v), "avg_dwell_s": round(float(np.mean(v)), 1)}
                  for z, v in sorted(zones.items())],
    }


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
</style></head><body>
<header><b>StoreSense Edge</b>
 <nav class="tabs" id="tabs">
  <div class="tab on" data-t="home">Home</div><div class="tab" data-t="shelves">Shelves</div>
  <div class="tab" data-t="checkout">Checkout</div><div class="tab" data-t="cctv">CCTV</div>
  <div class="tab" data-t="analytics">Analytics</div><div class="tab" data-t="setup">Setup</div></nav>
 <div class="right"><span class="pill" id="store"></span><span class="pill" id="conn">connecting…</span>
  <span class="pill" id="cloud"></span></div></header>

<main id="home">
<section class="kpis" id="kpis"></section>
<section class="card s7"><h3>Needs attention <span class="sub" id="acount"></span></h3><div id="alerts"></div></section>
<section class="card s5"><h3>Queue <span class="sub">counters open
 <input id="ctr" type="number" min="1" max="10" style="width:56px"> <button onclick="setCounters()">Set</button></span></h3>
 <div id="queues"></div></section>
<section class="card s5"><h3>Running low <span class="sub"><a href="#" onclick="tab('shelves');return false">all shelves →</a></span></h3>
 <div id="lowstock"></div></section>
<section class="card s7"><h3>Floor heatmap <span class="sub">where shoppers spend time</span></h3>
 <canvas id="mini" class="plan" style="max-width:100%"></canvas><div id="zones"></div></section>
<section class="card s7"><h3>Footfall today <span class="sub">entries per hour</span></h3><div class="chart" id="c_today"></div></section>
<section class="card s5"><h3>Reports <span class="sub"><button onclick="report('day')">Today</button>
 <button onclick="report('week')">7 days</button></span></h3><div id="report" class="muted">Pick a period.</div></section>
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
  <a class="lnk" href="/api/export/products.csv">Export CSV</a></span></h3>
 <div id="prodform"></div><div id="catalog"></div></section>
</main>

<main id="checkout" style="display:none">
<section class="card s7"><h3>Cart <span class="sub" id="cartid"></span></h3>
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

<main id="cctv" style="display:none">
<section class="card s12"><h3>Cameras <span class="sub">
  <span class="chip on" id="cv_plain" onclick="cctvMode('cctv')">Plain</span>
  <span class="chip" id="cv_ana" onclick="cctvMode('analytics')">With analytics</span>
  · faces are blurred on every view · nothing is recorded</span></h3>
 <div class="cctvgrid" id="cctvgrid"></div></section>
</main>

<main id="analytics" style="display:none">
<section class="s12 demobanner" id="demobanner" style="display:none"></section>
<section class="card s12"><h3>Store analytics <span class="sub" id="arange">
  <span class="chip" data-d="7">7 days</span><span class="chip on" data-d="14">14 days</span>
  <span class="chip" data-d="30">30 days</span></span></h3><div class="kpis inner" id="akpis"></div>
 <div class="exports">Download for modelling:
  <a class="lnk" href="/api/export/timeseries_hourly.csv?days=90">hourly time series</a> ·
  <a class="lnk" href="/api/export/timeseries_minute.csv">per-minute log</a> ·
  <a class="lnk" href="/api/export/bills.csv?days=90">bills</a> ·
  <a class="lnk" href="/api/export/products.csv">products</a></div></section>
<section class="card s8"><h3>Footfall and bills per day <span class="sub">complete days</span></h3><div class="chart" id="c_daily"></div></section>
<section class="card s4"><h3>Conversion <span class="sub">bills ÷ entries</span></h3><div class="chart" id="c_conv"></div></section>
<section class="card s12"><h3>When the store is busy <span class="sub">average entries by weekday and hour</span></h3>
 <div class="chart" id="c_heat"></div></section>
<section class="card s6"><h3>Footfall by hour of day <span class="sub">average per day</span></h3><div class="chart" id="c_hour"></div></section>
<section class="card s6"><h3>Revenue per day</h3><div class="chart" id="c_rev"></div></section>
<section class="card s6"><h3>Top products <span class="sub">by revenue</span></h3><div class="chart" id="c_top"></div></section>
<section class="card s6"><h3>Stock-outs <span class="sub">times a product ran empty</span></h3><div class="chart" id="c_stock"></div></section>
<section class="card s4"><h3>Basket size <span class="sub">items per bill</span></h3><div class="chart" id="c_basket"></div></section>
<section class="card s4"><h3>Queue wait</h3><div class="chart" id="c_wait"></div></section>
<section class="card s4"><h3>Revenue by hour <span class="sub">average per day</span></h3><div class="chart" id="c_revh"></div></section>
</main>

<main id="setup" style="display:none">
<section class="card s12"><h3>Store plan <span class="sub">drag to move · scroll a number to nudge · Delete to remove</span></h3>
 <div class="bar2"><button class="pri" onclick="addShelf()">+ Shelf</button>
  <button onclick="addCam()">+ Camera</button><div class="sep"></div>
  <button id="b_save" onclick="saveLay()">Save plan</button><button onclick="loadLay()">Reload</button>
  <div class="sep"></div><span class="muted" id="saved"></span></div>
 <div class="wrap"><canvas id="plan" class="plan"></canvas><div class="side" id="side"></div></div>
 <div class="warnline" id="blind"></div></section>
<section class="card s12" id="camsetup" style="display:none"><h3>Camera view
  <span class="sub">click on the picture to place points</span></h3><div id="camsetupbody"></div></section>
<section class="card s12"><h3>3D view <span class="sub">drag to orbit · scroll to zoom · double-click to reset</span></h3>
 <canvas id="v3d" style="width:100%;height:auto;cursor:grab;border:1px solid var(--line);border-radius:10px;background:#0f1114"></canvas>
 <div class="hint" style="margin-top:8px">Shelf faces are coloured by live stock: green in stock, amber low,
  red empty, violet misplaced. Grey faces are not monitored. Dotted cones are what each camera can see.</div></section>
</main>

<script>
const $=i=>document.getElementById(i);
let LAY={store:{w:6,h:4},shelves:[],cameras:[]},SEL=null,DRAG=null,HEAT=null,HOV=null,DIRTY=false,S=null;
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
function blocked(p,q,skip){for(const s of LAY.shelves){if(s.id===skip)continue;const c=corners(s);
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
 if(HEAT&&HEAT.length){const R=HEAT.length,C=HEAT[0].length,cw=IW/C,ch=H/R;
  for(let r=0;r<R;r++)for(let c=0;c<C;c++){const v=HEAT[r][c];if(v>0.02){
   X.fillStyle=`rgba(${Math.round(200+55*v)},${Math.round(90-60*v)},${Math.round(70-45*v)},${0.30+0.60*v})`;
   X.fillRect(c*cw,r*ch,cw+.6,ch+.6)}}}
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
function drawMini(){planDraw($('mini'),{edit:false,w:900,cones:false})}
function selObj(){if(!SEL)return null;
 if(SEL.t==='s')return LAY.shelves.find(s=>s.id===SEL.id);
 if(SEL.t==='c')return LAY.cameras.find(c=>c.id===SEL.id);
 const c=LAY.cameras.find(c=>c.id===SEL.id);return c&&c.floor_rect}
function handles(o){if(SEL.t==='c')return {head:[o.x+Math.cos(o.heading*Math.PI/180)*Math.min(o.range*.5,1.1),
  o.y+Math.sin(o.heading*Math.PI/180)*Math.min(o.range*.5,1.1)]};
 return {rot:rot(o.x,o.y-o.h/2-0.38,o.x,o.y,o.rot||0),size:corners(o)[2]}}
function pick(mx,my){const o=selObj();
 if(o){const hs=handles(o);for(const k in hs)if(Math.hypot(mx-hs[k][0],my-hs[k][1])<10/SC)return {t:SEL.t,id:SEL.id,mode:k}}
 for(const c of LAY.cameras)if(Math.hypot(mx-c.x,my-c.y)<12/SC)return {t:'c',id:c.id,mode:'move'};
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
  Add a shelf or camera, then drag it into place.<br>Click any item to edit it.</div></div>`}
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
 else h+=`<div class="box"><h4>Floor patch</h4>${f('x (m)',o.x,'selObj().x=+this.value;dirty();draw()')}
  ${f('y (m)',o.y,'selObj().y=+this.value;dirty();draw()')}${f('width (m)',o.w,'selObj().w=+this.value;dirty();draw()')}
  ${f('depth (m)',o.h,'selObj().h=+this.value;dirty();draw()')}${f('rotation °',o.rot||0,'selObj().rot=+this.value;dirty();draw()',5)}
  <div class="hint">The real floor area matching the 4 points marked on this camera's picture.</div></div>`;
 $('side').innerHTML=h;warn();camSetup()}
function camNote(c){const r=c.roles||[];
 if(r.includes('shelf')){let best=null,bv=0;for(const s of LAY.shelves)for(const fc in s.faces){
   const v=vis(c,s,fc);if(v>bv){bv=v;best=s.name+' '+fc}}
  return best?`Watching <b>${best}</b> (${Math.round(bv*100)}% of the face).`:'No shelf face in view yet — move it or widen the lens angle.'}
 return 'Mark the entry line on the picture below.'}
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
 else delete LAY.cameras.find(c=>c.id===SEL.id).floor_rect;
 SEL=null;dirty();side();draw()}
function addShelf(){const id='S'+Date.now().toString(36);
 LAY.shelves.push({id,name:'Shelf '+(LAY.shelves.length+1),x:LAY.store.w/2,y:LAY.store.h/2,
  w:Math.min(2.5,LAY.store.w-1),h:0.5,rot:0,height:1.8,faces:{N:{grid:[4,6]}}});
 SEL={t:'s',id};dirty();side();draw()}
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
let YAW=-0.62,PIT=0.92,ZOOM=1,D3=null;
function raw(p){const ca=Math.cos(YAW),sa=Math.sin(YAW),cp=Math.cos(PIT),sp=Math.sin(PIT);
 const x=p[0]-LAY.store.w/2,y=p[1]-LAY.store.h/2,z=p[2];
 const X=x*ca-y*sa,Y=x*sa+y*ca;
 return [X,-(Y*sp+z*cp)*0.72,Y*cp-z*sp]}
let VX=0,VY=0,VS=60;
function proj(p){const r=raw(p);return [VX+r[0]*VS,VY+r[1]*VS,r[2]]}
function fit3d(W,H){const zs=[0,...LAY.shelves.map(s=>s.height||1.8),...LAY.cameras.map(c=>c.height||2.2)];
 const zmax=Math.max(...zs);let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
 for(const X of [0,LAY.store.w])for(const Y of [0,LAY.store.h])for(const Z of [0,zmax]){
  const r=raw([X,Y,Z]);x0=Math.min(x0,r[0]);x1=Math.max(x1,r[0]);y0=Math.min(y0,r[1]);y1=Math.max(y1,r[1])}
 VS=Math.min(W/(x1-x0+0.6),H/(y1-y0+0.6))*ZOOM;
 VX=W/2-(x0+x1)/2*VS;VY=H/2-(y0+y1)/2*VS}
function draw3d(){const cv=$('v3d');const W=1320,H=620;if(cv.width!==W){cv.width=W;cv.height=H}
 const X=cv.getContext('2d');X.clearRect(0,0,W,H);X.fillStyle='#0f1114';X.fillRect(0,0,W,H);
 fit3d(W,H);
 const polys=[];
 const push=(pts,fill,stroke,lw,dash)=>{const pr=pts.map(p=>proj(p));
  polys.push({pr,fill,stroke,lw:lw||1,dash:dash||null,d:pr.reduce((a,b)=>a+b[2],0)/pr.length})};
 for(let x=0;x<=LAY.store.w+1e-6;x+=1)push([[x,0,0],[x,LAY.store.h,0]],null,'#23272e',1);
 for(let y=0;y<=LAY.store.h+1e-6;y+=1)push([[0,y,0],[LAY.store.w,y,0]],null,'#23272e',1);
 push([[0,0,0],[LAY.store.w,0,0],[LAY.store.w,LAY.store.h,0],[0,LAY.store.h,0]],null,'#454b55',2);
 if(HEAT&&HEAT.length){const R=HEAT.length,C=HEAT[0].length,cw=LAY.store.w/C,ch=LAY.store.h/R;
  for(let r=0;r<R;r++)for(let c=0;c<C;c++){const v=HEAT[r][c];if(v<=0.02)continue;
   push([[c*cw,r*ch,0.002],[(c+1)*cw,r*ch,0.002],[(c+1)*cw,(r+1)*ch,0.002],[c*cw,(r+1)*ch,0.002]],
    `rgba(${Math.round(200+55*v)},${Math.round(90-60*v)},${Math.round(70-45*v)},${0.35+0.6*v})`,null,0)}}
 for(const c of LAY.cameras){const half=c.fov*Math.PI/360,hd=c.heading*Math.PI/180,z=c.height||2.2;
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
 polys.sort((a,b)=>b.d-a.d);
 for(const p of polys){X.beginPath();p.pr.forEach((q,i)=>i?X.lineTo(q[0],q[1]):X.moveTo(q[0],q[1]));
  if(p.fill){X.closePath();X.fillStyle=p.fill;X.fill()}
  if(p.stroke){X.setLineDash(p.dash||[]);X.strokeStyle=p.stroke;X.lineWidth=p.lw;X.stroke();X.setLineDash([])}}
 X.font='600 12px system-ui';X.textAlign='center';
 for(const s of LAY.shelves){const q=proj([s.x,s.y,(s.height||1.8)+0.14]);
  X.strokeStyle='#0f1114';X.lineWidth=3.5;X.strokeText(s.name,q[0],q[1]);
  X.fillStyle='#c3c9d2';X.fillText(s.name,q[0],q[1])}
 for(const c of LAY.cameras){const q=proj([c.x,c.y,c.height||2.2]);
  X.beginPath();X.arc(q[0],q[1],4.5,0,6.3);X.fillStyle='#ff3b52';X.fill();
  X.textAlign='left';X.strokeStyle='#0f1114';X.lineWidth=3.5;X.strokeText(c.name,q[0]+9,q[1]+4);
  X.fillStyle='#ff6b7d';X.fillText(c.name,q[0]+9,q[1]+4);X.textAlign='center'}
 X.textAlign='left'}
function clip(p,q){const W=LAY.store.w,H=LAY.store.h;let t=1;const dx=q[0]-p[0],dy=q[1]-p[1];
 for(const [num,den] of [[p[0],-dx],[W-p[0],dx],[p[1],-dy],[H-p[1],dy]]){
  if(Math.abs(den)<1e-9){if(num<0)return p;continue}
  if(den>0)t=Math.min(t,Math.max(0,num/den))}
 return [p[0]+dx*t,p[1]+dy*t]}
(function(){const cv=$('v3d');let d=null;
 cv.addEventListener('pointerdown',e=>{d=[e.clientX,e.clientY];cv.style.cursor='grabbing';cv.setPointerCapture(e.pointerId)});
 cv.addEventListener('pointermove',e=>{if(!d)return;YAW+=(e.clientX-d[0])*0.01;
  PIT=Math.max(0.12,Math.min(1.48,PIT-(e.clientY-d[1])*0.008));d=[e.clientX,e.clientY];draw3d()});
 addEventListener('pointerup',()=>{d=null;cv.style.cursor='grab'});
 cv.addEventListener('wheel',e=>{e.preventDefault();ZOOM=Math.max(0.4,Math.min(3,ZOOM*(e.deltaY>0?0.92:1.09)));draw3d()},{passive:false});
 cv.addEventListener('dblclick',()=>{YAW=-0.62;PIT=0.92;ZOOM=1;draw3d()})})();
/* ---------- tabs ---------- */
const TABS=['home','shelves','checkout','cctv','analytics','setup'];
let TABV='home';
function tab(t){if(!TABS.includes(t))t='home';TABV=t;
 for(const x of TABS)$(x).style.display=x===t?'grid':'none';
 document.querySelectorAll('#tabs .tab').forEach(el=>el.classList.toggle('on',el.dataset.t===t));
 try{history.replaceState(null,'','#'+t)}catch(e){}
 streams();
 if(t==='setup'){draw();draw3d()}
 if(t==='home'){drawMini();if(S)homeCharts(S)}
 if(t==='shelves')shEnter();
 if(t==='checkout')coEnter();
 if(t==='analytics')anLoad();
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
 return Math.max(1,Math.ceil(labels.length/Math.max(1,Math.floor(iw/(L*6.6+14)))))};
function yAxis(m,W,ih,mx,fmt,n){n=n||4;let g='';for(let k=0;k<=n;k++){const v=mx*k/n,y=m.t+ih-ih*k/n;
 g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y}" y2="${y}" stroke="${k?CC.grid:CC.axis}" stroke-width="1"/>`+
  `<text class="ax" x="${m.l-7}" y="${y+4}" text-anchor="end">${(fmt||fmtN)(v)}</text>`}return g}
function barChart(el,o){chartShell(el,o,(el,o)=>{
 const W=Math.max(260,el.clientWidth||600),H=o.h||200,m={l:46,r:8,t:10,b:24},n=o.values.length;
 const iw=W-m.l-m.r,ih=H-m.t-m.b,intish=o.values.every(v=>Number.isInteger(v||0));
 const T=ticks(Math.max(0,...o.values.map(v=>v||0))*1.02,intish),mx=T.mx;
 const step=iw/Math.max(1,n),bw=Math.max(2,Math.min(40,step-2)),Y=v=>m.t+ih-(v/mx)*ih;
 const every=lblEvery(o.labels,iw);
 let g=yAxis(m,W,ih,mx,o.yfmt,T.n);
 o.values.forEach((v,i)=>{const x=m.l+i*step+(step-bw)/2,top=Y(v||0),hh=m.t+ih-top;
  const col=o.hi?(o.hi(i)?CC.s1:CC.rest):(o.color||CC.s1);
  if(hh>0.5)g+=`<path d="${barPath(x,top,bw,hh)}" fill="${col}"/>`;
  if(i%every===0)g+=`<text class="ax" x="${x+bw/2}" y="${H-6}" text-anchor="middle">${esc(o.labels[i])}</text>`;
  g+=`<rect data-i="${i}" x="${m.l+i*step}" y="${m.t}" width="${step}" height="${ih}" fill="transparent"/>`});
 el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${g}</svg>`;
 el.querySelectorAll('rect[data-i]').forEach(r=>{r.onmousemove=e=>{const i=+r.dataset.i;
  tipAt(e,`<b>${esc(o.tlabels?o.tlabels[i]:o.labels[i])}</b><div class="r"><span>${esc(o.name||'')}</span>
   <span>${(o.fmt||fmtN)(o.values[i])}</span></div>`)};r.onmouseleave=untip})},
 o=>({cols:[o.xname||'',o.name||'value'],rows:o.labels.map((l,i)=>[o.tlabels?o.tlabels[i]:l,(o.fmt||fmtN)(o.values[i])])}))}
function lineChart(el,o){chartShell(el,o,(el,o)=>{
 const multi=o.series.length>1,W=Math.max(260,el.clientWidth||600),H=o.h||220;
 const m={l:46,r:multi?78:14,t:10,b:24},n=o.labels.length,iw=W-m.l-m.r,ih=H-m.t-m.b;
 const all=o.series.flatMap(s=>s.values.filter(v=>v!=null));
 const T=ticks(Math.max(0,...all)*1.04,all.every(v=>Number.isInteger(v)),o.max),mx=T.mx;
 const X=i=>m.l+(n<2?iw/2:i*iw/(n-1)),Y=v=>m.t+ih-(v/mx)*ih;
 const every=lblEvery(o.labels,iw);
 let g=yAxis(m,W,ih,mx,o.yfmt,T.n);
 o.labels.forEach((l,i)=>{if(i%every===0||i===n-1)g+=`<text class="ax" x="${X(i)}" y="${H-6}" text-anchor="middle">${esc(l)}</text>`});
 for(const s of o.series){let d='',pen=false;s.values.forEach((v,i)=>{if(v==null){pen=false;return}
   d+=(pen?'L':'M')+X(i).toFixed(1)+','+Y(v).toFixed(1);pen=true});
  g+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  if(multi){let li=s.values.length-1;while(li>0&&s.values[li]==null)li--;
   if(s.values[li]!=null)g+=`<circle cx="${X(li)}" cy="${Y(s.values[li])}" r="3" fill="${s.color}"/>`+
    `<text class="lab" x="${X(li)+8}" y="${Y(s.values[li])+4}">${esc(s.name)}</text>`}}
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
 const mx=Math.max(...o.m.flat())||1;let g='';
 o.m.forEach((row,r)=>{g+=`<text class="ax" x="${m.l-8}" y="${m.t+r*ch+ch/2+4}" text-anchor="end">${o.rows[r]}</text>`;
  row.forEach((v,c)=>{g+=`<rect data-r="${r}" data-c="${c}" x="${m.l+c*cw+1}" y="${m.t+r*ch+1}" width="${Math.max(1,cw-2)}"
   height="${ch-2}" rx="3" fill="${v>0?ramp(v/mx):'#171a1e'}"/>`})});
 o.cols.forEach((c,i)=>{if(i%2===0)g+=`<text class="ax" x="${m.l+i*cw+cw/2}" y="${m.t+rows*ch+14}" text-anchor="middle">${c}</text>`});
 const lx=W-m.r-180,ly=H-14;
 g+=`<defs><linearGradient id="hg${el.id}">${RAMP.map((c,i)=>`<stop offset="${i/(RAMP.length-1)}" stop-color="rgb(${c})"/>`).join('')}</linearGradient></defs>
  <text class="ax" x="${lx-8}" y="${ly+4}" text-anchor="end">0</text><rect x="${lx}" y="${ly-5}" width="150" height="9" rx="3" fill="url(#hg${el.id})"/>
  <text class="ax" x="${lx+158}" y="${ly+4}">${fmtN(mx)}</text>`;
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
const kpi=(l,v,d)=>`<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div>${d?`<div class="d">${d}</div>`:''}</div>`;
const ago=t=>{const s=Math.round(Date.now()/1000-t);return s<60?s+'s ago':s<3600?Math.round(s/60)+'m ago':Math.round(s/3600)+'h ago'};
function allCells(s){const out=[];for(const cam in s.shelves)for(const c of (s.shelves[cam]||[]))out.push({...c,cam});return out}
function render(s){S=s;
 $('store').textContent=s.store_id;
 $('cloud').textContent=s.cloud_online==null?'edge only':s.cloud_online?'cloud synced':'offline · buffering';
 if(!ctrSet){$('ctr').value=s.open_counters;ctrSet=true}
 if(s.store_heat)HEAT=s.store_heat;
 if(s.layout&&!DIRTY)LAY=s.layout;
 s.faceStatus={};for(const cam in s.shelves){const cells=s.shelves[cam];if(!cells)continue;
  const w=(s.cam_face||{})[cam];if(!w)continue;let st='OK';
  for(const c of cells){if(c.status==='EMPTY'){st='EMPTY';break}if(c.status==='LOW')st='LOW';
   else if(c.status==='MISPLACED'&&st==='OK')st='MISPLACED'}s.faceStatus[w]=st}
 const cells=allCells(s),low=cells.filter(c=>c.status==='LOW').length,emp=cells.filter(c=>c.status==='EMPTY').length;
 const q=Object.values(s.queues),wait=q.length?Math.max(...q.map(x=>x.wait_min)):0;
 $('kpis').innerHTML=kpi('Inside now',s.footfall.inside)+kpi('Entries today',s.footfall.in)+
  kpi('Sales today',fmtR(s.sales_today||0),`${s.bills_today||0} bills`)+
  kpi('Conversion',s.conversion==null?'—':fmtP(s.conversion),'bills ÷ entries')+
  kpi('Queue wait',wait.toFixed(1)+' min',`${q.reduce((a,x)=>a+x.length,0)} in line`)+
  kpi('Low / empty',`${low} / ${emp}`,'products')+
  kpi('Open alerts',s.alerts.length)+kpi('Staff response',s.avg_response_s==null?'—':Math.round(s.avg_response_s)+'s');
 $('acount').textContent=s.alerts.length?s.alerts.length+' open':'';
 $('alerts').innerHTML=s.alerts.length?s.alerts.slice(0,8).map(a=>`<div class="alert ${a.severity}"><div class="m">
  <div class="a">${esc(a.action)}</div><div>${esc(a.message)}</div>
  <small>${a.kind} · ${ago(a.ts)}${a.count>1?' · seen '+a.count+'×':''}</small></div>
  <button onclick="ack(${a.id})">Done</button></div>`).join(''):'<div class="empty">All clear.</div>';
 $('queues').innerHTML=Object.entries(s.queues).map(([cam,x])=>`
  <div class="muted">${x.arrival_per_min}/min arriving · ${x.service_per_counter_min}/min served per counter</div>
  <div class="fc"><div><b>${x.length}</b><small>now · ${x.wait_min}m</small></div>
  ${['5','10','15'].map(t=>`<div><b>${x.forecast[t].len}</b><small>+${t}m · ${x.forecast[t].wait}m</small></div>`).join('')}</div>
  <div>Recommended counters: <b>${x.recommend_counters}</b></div>`).join('')||'<div class="empty">No queue camera.</div>';
 const lows=cells.filter(c=>c.slot&&(c.status==='LOW'||c.status==='EMPTY'))
  .sort((a,b)=>(a.status==='EMPTY'?0:1)-(b.status==='EMPTY'?0:1));
 $('lowstock').innerHTML=lows.length?`<table>${lows.slice(0,8).map(c=>`<tr><td><span class="st ${c.status}">${c.status}</span></td>
  <td>${esc(c.name)}<div class="muted">${esc((s.labels||{})[c.cam]||c.cam)} · ${esc(c.loc)}</div></td>
  <td style="text-align:right">~${c.est_units}<span class="muted"> / ${c.full_units}</span></td></tr>`).join('')}</table>`
  :cells.length?'<div class="empty">Every marked product is stocked.</div>':'<div class="empty">No products marked yet — open Shelves.</div>';
 const z=Object.entries((s.zones||{})[Object.keys(s.zones||{})[0]]||{});
 $('zones').innerHTML=z.length?`<table style="margin-top:8px"><tr><th>Zone</th><th>Visits</th><th>Avg dwell</th></tr>`+
  z.map(([n,v])=>`<tr><td>${esc(n)}</td><td>${v.visits}</td><td>${v.avg_dwell_s}s</td></tr>`).join('')+`</table>`:'';
 if(TABV==='home'){drawMini();homeCharts(s)}
 if(TABV==='setup'&&!DRAG)draw3d();
 if(TABV==='shelves'){shTable();shOv();shDepthLine()}
 if(TABV==='cctv'&&Object.keys(S.cams||{}).join('|')!==CCTVSIG)cctvBuild();   // a camera joined or left
 if(TABV==='setup'&&$('camfeed')&&SEL&&SEL.t==='c')$('camfeed').innerHTML=feedLine(selObj());
 if(TABV==='cctv')cctvBadges();
 coSync(s)}
let HOMEKEY='';
function homeCharts(s){const key=JSON.stringify(s.hourly);if(key===HOMEKEY&&$('c_today')._draw)return;HOMEKEY=key;
 const hrs=[];for(let h=7;h<=22;h++)hrs.push(h);const now=s.hour_now==null?new Date().getHours():s.hour_now;
 barChart($('c_today'),{labels:hrs.map(h=>h+''),tlabels:hrs.map(h=>`${h}:00–${h+1}:00`),xname:'hour',
  values:hrs.map(h=>s.hourly[String(h).padStart(2,'0')]||0),name:'entries',hi:i=>hrs[i]===now,h:170})}

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
async function shLoad(cam){if(!cam)return;SH={cam,slots:[],sel:null,mode:null,drag:null,dirty:false,view:SH.view||'still'};
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
  const tag=`${nm}${lc&&!SH.dirty?`  ${lc.present}/${s.facings}`:''}`;
  X.font='600 12px system-ui';const tw=X.measureText(tag).width+10;
  X.fillStyle='rgba(11,12,14,.85)';X.fillRect(x,Math.max(0,y-19),tw,18);
  X.fillStyle='#e9ebef';X.fillText(tag,x+5,Math.max(13,y-6));
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
   <div class="hint">A full box holds <b id="sh_full">${s.facings*s.deep}</b> units. With the unit size set, the depth model
    measures how far back the front unit of each column sits and counts what's left behind it. Without it, the count is
    an estimate: facings still visible × units deep.</div>
   <button class="danger" style="margin-top:8px" onclick="shDel()">Remove box</button></div>`;
  const lc=liveCell(s.id);
  if(lc&&!SH.dirty)h+=`<div class="box"><h4>Right now</h4><div class="f"><span>status</span><span class="st ${lc.status}">${lc.status}</span></div>
   <div class="f"><span>facings visible</span><b>${lc.present} / ${lc.facings}</b></div>
   <div class="f"><span>camera count</span><b>${lc.method==='front'?'~':''}${lc.est_units} / ${lc.full_units}</b></div>
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
 for(const [id,v] of [['b_vstill','still'],['b_vlive','live'],['b_depth','depth']])$(id).className=SH.view===v?'on':'';
 if(SH.view==='live'){im.classList.add('live');im.dataset.cam=SH.cam;im._busy=false;return}
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
function shTable(){const el=$('shtable');if(!el||!S)return;const cells=(S.shelves[SH.cam]||[]);
 if(!SH.cam){el.innerHTML='';return}
 if(!cells.length){el.innerHTML=`<div class="empty">${SH.calibrated?'Waiting for the camera…':'Not calibrated yet.'}</div>`;return}
 if(!cells[0].slot){el.innerHTML=`<div class="hint" style="margin-bottom:8px">No product boxes yet — showing the fallback grid.
   Draw boxes above for per-product counts.</div>`+gridCells(cells);return}
 el.innerHTML=`<table><tr><th>Where</th><th>Product</th><th>Facings seen</th><th>Camera count</th><th>By the till</th>
  <th>Status</th><th>Runs out</th></tr>`+cells.slice().sort((a,b)=>a.row-b.row||a.col-b.col).map(c=>`<tr>
  <td class="muted">${esc(c.loc)}</td><td>${esc(c.name)}${c.brand?`<div class="muted">${esc(c.brand)}</div>`:''}</td>
  <td><div style="display:flex;align-items:center;gap:8px"><div class="bar-in" style="flex:1"><div style="width:${100*c.present/c.facings}%;
   background:${SCOL[c.status]||'#3fb950'}"></div></div><span>${c.present}/${c.facings}</span></div></td>
  <td>${c.method==='front'?'~':''}${c.est_units} <span class="muted">/ ${c.full_units} · ${{depth:'depth',mixed:'depth, part hidden',front:'front row'}[c.method]||''}</span></td><td>${c.pos_units==null?'<span class="muted">—</span>':c.pos_units}</td>
  <td><span class="st ${c.occluded?'occ':c.status}">${c.occluded?'BLOCKED':c.status}</span></td>
  <td class="muted">${c.eta_min!=null?'~'+Math.round(c.eta_min)+' min':''}</td></tr>`).join('')+`</table>`}
function gridCells(cells){const C=Math.max(...cells.map(c=>c.c))+1;
 return `<div class="grid" style="grid-template-columns:repeat(${C},1fr)">`+cells.map(c=>`<div class="cell ${c.status} ${c.occluded?'occ':''}">
  ${c.fill==null?'?':Math.round(c.fill*100)+'%'}</div>`).join('')+`</div>`}
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
function cartDraw(flash){const v=CARTV;$('cartid').textContent=v?`cart ${v.id} · ${v.n_items} item${v.n_items===1?'':'s'}`:'';
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
  <div class="bar2" style="margin-top:16px"><button class="pri" onclick="window.open('/api/bills/${b.id}.pdf')">Print bill</button>
   <button onclick="cartNew()">Next customer</button></div>`;
 $('billbox').scrollIntoView({behavior:'smooth'})}
function coSync(s){const ls=s.pos&&s.pos.last_scan;if(!ls||ls.ts<=LASTSCAN)return;LASTSCAN=ls.ts;
 if(TABV!=='checkout'||ls.source==='screen')return;const t=$('scantoast');   /* screen scans report inline */
 if(t)t.innerHTML=`<div class="toast ${ls.ok?'ok':'bad'}">${esc(ls.msg)}${ls.ok&&ls.price!=null?` · ₹${ls.price}`:''}</div>`;
 if(ls.ok&&s.pos.active_cart){CART=s.pos.active_cart;cartRefresh()}}

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
document.querySelectorAll('#arange .chip').forEach(c=>c.onclick=()=>{ADAYS=+c.dataset.d;
 document.querySelectorAll('#arange .chip').forEach(x=>x.classList.toggle('on',x===c));anLoad()});
async function anLoad(){AN=await(await fetch('/api/analytics?days='+ADAYS)).json();anRender()}
const WD=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
const dshort=d=>new Date(d+'T00:00:00').toLocaleDateString('en-IN',{day:'numeric',month:'short'});
const dlong=d=>new Date(d+'T00:00:00').toLocaleDateString('en-IN',{weekday:'short',day:'numeric',month:'short'});
function anRender(){const a=AN;if(!a)return;const k=a.kpis;
 const b=$('demobanner');b.style.display=a.has_demo?'':'none';
 b.innerHTML=a.has_demo?`<b>Includes generated demo history.</b> Past days were filled in with
  <code>--demo-history</code> so the charts have something to show; those rows are tagged in the database and
  real trading is added on top. Remove them with <code>python storesense.py --clear-demo</code>.`:'';
 $('akpis').innerHTML=kpi('Footfall',fmtN(k.footfall),`${fmtN(k.footfall_per_day)} a day`)+kpi('Bills',fmtN(k.bills))+
  kpi('Revenue',fmtRs(k.revenue))+kpi('Average bill',fmtR(k.avg_basket))+kpi('Conversion',fmtP(k.conversion))+
  kpi('Items sold',fmtN(k.items))+kpi('Avg queue wait',k.avg_wait_min==null?'—':k.avg_wait_min+' min')+
  kpi('Busiest hour',k.peak_hour||'—')+kpi('Stock-outs',k.stockouts);
 const t=new Date(),today=`${t.getFullYear()}-${String(t.getMonth()+1).padStart(2,'0')}-${String(t.getDate()).padStart(2,'0')}`;
 let days=a.daily.filter(d=>d.date!==today);if(!days.some(d=>d.entries||d.bills))days=a.daily;
 const dl=days.map(d=>dshort(d.date)),dL=days.map(d=>dlong(d.date));
 lineChart($('c_daily'),{labels:dl,tlabels:dL,series:[{name:'Entries',color:CC.s1,values:days.map(d=>d.entries)},
  {name:'Bills',color:CC.s2,values:days.map(d=>d.bills)}],h:240});
 lineChart($('c_conv'),{labels:dl,tlabels:dL,series:[{name:'Conversion',color:CC.s1,values:days.map(d=>d.conversion)}],
  fmt:fmtP,yfmt:fmtP,max:1,h:240});
 const hrs=[];for(let h=7;h<=22;h++)hrs.push(h);
 heat($('c_heat'),{rows:WD,cols:hrs.map(h=>String(h).padStart(2,'0')),m:a.weekday_hour.map(r=>hrs.map(h=>r[h])),name:'entries (avg)'});
 const hv=hrs.map(h=>a.hourly_avg[h]),pk=hv.indexOf(Math.max(...hv));
 barChart($('c_hour'),{labels:hrs.map(h=>h+''),tlabels:hrs.map(h=>`${h}:00–${h+1}:00`),values:hv,name:'entries a day',xname:'hour',hi:i=>i===pk,h:210});
 barChart($('c_rev'),{labels:dl,tlabels:dL,values:days.map(d=>d.revenue),name:'revenue',fmt:fmtR,yfmt:fmtRs,xname:'day',h:210});
 hbar($('c_top'),{items:a.top_products.map(p=>({label:p.name,value:p.revenue,
  rows:[['revenue',fmtR(p.revenue)],['units',p.units]]})),fmt:fmtR,name:'revenue'});
 hbar($('c_stock'),{items:a.stockouts.map(s=>({label:s.name,value:s.count})),name:'times empty',color:CC.s2});
 barChart($('c_basket'),{labels:a.basket_hist.map(x=>x.bin),values:a.basket_hist.map(x=>x.n),name:'bills',xname:'items',h:180});
 barChart($('c_wait'),{labels:a.wait_hist.map(x=>x.bin),values:a.wait_hist.map(x=>x.n),name:'observations',xname:'wait',color:CC.s2,h:180});
 const rv=hrs.map(h=>a.revenue_by_hour[h]),rpk=rv.indexOf(Math.max(...rv));
 barChart($('c_revh'),{labels:hrs.map(h=>h+''),tlabels:hrs.map(h=>`${h}:00–${h+1}:00`),values:rv,name:'revenue a day',
  fmt:fmtR,yfmt:fmtRs,xname:'hour',hi:i=>i===rpk,h:180})}

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
 $('report').innerHTML=`<table>${rows.map(r=>`<tr><td>${r[0]}</td><td><b>${r[1]}</b></td></tr>`).join('')}</table>
  <p class="muted">Alerts: ${Object.entries(j.alerts_by_action).map(([k,v])=>k+' ×'+v).join(', ')||'none'}</p>
  <a href="/api/report?period=${p}" target="_blank" style="color:var(--acc2)">Download JSON</a>`}
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
            if hasattr(w, "storeH"):
                w.storeH = w.H = None
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
            return w.cctv_jpg or w.jpg or placeholder_jpg(cam, "starting…")
        return w.jpg or placeholder_jpg(cam, "starting…")

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
        jpg = getattr(w, "depth_jpg", None) if w else None
        if not jpg:
            raise HTTPException(404, (getattr(w, "depth_state", None) or {}).get("msg", "no depth pass yet"))
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
        store.product_delete(sku)
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

    @app.post("/api/cart/{cid}/checkout")
    def do_checkout(cid: str):
        bill, err = engine.checkout(cid)
        return {"ok": bill is not None, "error": err, "bill": bill}

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

    @app.get("/api/analytics")
    def get_analytics(days: int = 14):
        return analytics(store, days)

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
