"""Cameras start, restart and stop when the store plan is saved — no server restart."""
import os, sys, time, types

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for f in ("cams.db", "cams_layout.json"):
    if os.path.exists(f):
        os.remove(f)

import numpy as np, torch, cv2
import ultralytics


class FakeBoxes:
    def __init__(self):
        self.xyxy = torch.zeros((0, 4))
        self.id = None
        self.cls = torch.zeros(0)
        self.conf = torch.zeros(0)


class FakeYOLO:
    names = {0: "person"}

    def __init__(self, *a, **k):
        pass

    def __call__(self, frame, **kw):
        return [types.SimpleNamespace(boxes=FakeBoxes())]

    def track(self, frame, **kw):
        return self(frame)


ultralytics.YOLO = FakeYOLO
import storesense as ss
from fastapi.testclient import TestClient

fails = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


print("phone URLs")
for raw, want in [("192.168.1.5:8080", "http://192.168.1.5:8080/video"),
                  ("http://172.16.223.121:8080", "http://172.16.223.121:8080/video"),
                  ("http://172.16.223.121:8080/", "http://172.16.223.121:8080/video"),
                  ("http://10.0.0.2:8080/video", "http://10.0.0.2:8080/video"),
                  ("http://10.0.0.2:4747/mjpegfeed", "http://10.0.0.2:4747/mjpegfeed"),
                  ("rtsp://10.0.0.9:554/stream1", "rtsp://10.0.0.9:554/stream1"),
                  ("0", "0"), (" 1 ", "1"), ("../demo/walk.mp4", "../demo/walk.mp4")]:
    got = ss.normalise_source(raw)
    check(f"{raw!r} -> {want}", got == want, got)


def wait(cond, t=8.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


store = ss.Store("cams.db")
lay = ss.Layout("cams_layout.json")
eng = ss.Engine(store, lay)
workers = {}
cli = TestClient(ss.make_app(eng, workers, store))
plan = {"store": {"w": 6, "h": 4}, "shelves": [], "cameras": [
    {"id": "door", "x": 1, "y": 1, "roles": ["entry"], "source": ""}]}

print("\nsaving the plan")
r = cli.post("/api/layout", json=plan).json()
check("no source: nothing started", r["ok"] and not workers and r["cams"]["starting"] == [])

plan["cameras"][0]["source"] = "../demo/walk.mp4"
r = cli.post("/api/layout", json=plan).json()
check("adding a source starts the camera", r["cams"]["starting"] == ["door"], str(r["cams"]))
check("...and it goes live", wait(lambda: eng.snapshot()["cams"].get("door", {}).get("online")),
      str(eng.snapshot()["cams"].get("door")))
first = workers.get("door")

r = cli.post("/api/layout", json=plan).json()
check("saving again doesn't restart it", r["cams"]["starting"] == [] and workers.get("door") is first)

plan["cameras"][0]["source"] = "http://127.0.0.1:9/video"       # nothing listens there
r = cli.post("/api/layout", json=plan).json()
check("changing the source restarts it", r["cams"]["starting"] == ["door"] and wait(lambda: workers.get("door") not in (None, first)))
check("old camera stopped", wait(lambda: not first.is_alive(), 5), f"alive={first.is_alive()}")
check("an unreachable phone says why, in words",
      wait(lambda: "same Wi-Fi" in eng.snapshot()["cams"].get("door", {}).get("msg", ""), 15),
      eng.snapshot()["cams"].get("door", {}).get("msg"))

plan["cameras"].append({"id": "shelfcam", "x": 2, "y": 1, "roles": ["shelf", "entry"], "source": "0"})
r = cli.post("/api/layout", json=plan).json()
check("impossible job mix is refused with a reason", "shelfcam" in r["cams"]["errors"] and "shelfcam" not in workers,
      str(r["cams"]["errors"]))

old = workers.get("door")
plan["cameras"] = []
r = cli.post("/api/layout", json=plan).json()
check("removing a camera stops it", "door" not in workers and wait(lambda: not old.is_alive(), 5))
check("...and drops it from the status list", "door" not in eng.snapshot()["cams"])

print("\nstarted from the command line")
w = ss.start_cam(eng, workers, "cli", ["entry"], "../demo/walk.mp4")
cli.post("/api/layout", json={"store": {"w": 6, "h": 4}, "shelves": [],
                              "cameras": [{"id": "cli", "x": 1, "y": 1, "roles": ["entry"], "source": ""}]})
check("--cam camera not in the plan's sources is left running", workers.get("cli") is w and w.is_alive())
w.stop()

print("\nphone stream lag")
import threading, socketserver, http.server
SENT, STOP = [0], [False]
FRS = []
for i in range(256):
    im = np.random.default_rng(i).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    for bit in range(8):
        im[0:60, bit * 90:bit * 90 + 60] = 255 if (i >> bit) & 1 else 0
    FRS.append(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())


class Phone(http.server.BaseHTTPRequestHandler):      # behaves like IP Webcam's /video
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=Ba4oTvQMY8ew04N8dcnM")
        self.end_headers()
        n, t0 = 0, time.time()
        try:
            while not STOP[0]:
                j = FRS[n % 256]
                self.wfile.write(b"--Ba4oTvQMY8ew04N8dcnM\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j) + j + b"\r\n")
                SENT[0] = n
                n += 1
                time.sleep(max(0, t0 + n / 30 - time.time()))
        except Exception:
            pass


class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


srv = TS(("127.0.0.1", 8766), Phone)
threading.Thread(target=srv.serve_forever, daemon=True).start()
cap = ss.Capture("127.0.0.1:8766")
time.sleep(1.0)
lags = []
for _ in range(20):                              # a slow worker: 5 frames a second from a 30 fps phone
    f = cap.read()
    if f is not None:
        got = sum(1 << b for b in range(8) if f[30, b * 90 + 30].mean() > 127)
        lags.append((SENT[0] - got) % 256)
    time.sleep(0.2)
check("slow reader stays within 3 frames (0.1 s) of the phone", lags and max(lags[3:]) <= 3, str(lags))
seq = cap.seq
STOP[0] = True                                   # phone app closed
srv.shutdown()
time.sleep(0.5)
check("no new frames -> sequence stops, workers don't redo old frames", cap.seq - seq <= 3, f"{cap.seq - seq}")
check("phone gone -> says why", wait(lambda: "same Wi-Fi" in cap.status, 10), cap.status)
cap.stop()

print("\nframe endpoint")
r = cli.get("/api/frame/ghost.jpg")
check("unknown camera -> a picture that says why, not an error",
      r.status_code == 200 and r.headers["content-type"] == "image/jpeg")
w2 = ss.start_cam(eng, workers, "walk", ["entry"], "../demo/walk.mp4")
wait(lambda: eng.snapshot()["cams"].get("walk", {}).get("online"))
wait(lambda: w2.jpg is not None)
r = cli.get("/api/frame/walk.jpg")
check("live camera -> its latest frame", r.content == w2.jpg or len(r.content) > 5000)
src_fps = cv2.VideoCapture("../demo/walk.mp4").get(cv2.CAP_PROP_FPS)
time.sleep(6)
fps = eng.snapshot()["cams"]["walk"]["fps"]
check("fps shown is the camera's real frame rate, not processing speed", 0.6 * src_fps < fps < 1.3 * src_fps,
      f"shown {fps}, video {src_fps:.0f}")
w2.stop()

print("\nlive view")
w3 = ss.start_cam(eng, workers, "walk2", ["entry"], "../demo/walk.mp4")
wait(lambda: w3.jpg is not None and w3.blur_t > 0)
seen = set()
for _ in range(20):
    j = w3.live_jpg("analytics")
    if j:
        seen.add(hash(j))
    time.sleep(0.1)
check("live picture changes with every camera frame", len(seen) >= 8, f"{len(seen)} distinct in 2 s")
w3.blur_t = time.time() - 5
check("detection stalled -> no live frame (falls back to the processed, blurred one)", w3.live_jpg("analytics") is None)
w3.stop()

print("\nalerts")
eng.fire("shelf:camX:0,1", "stock", 3, "grid cell empty", "Critical refill")
eng.fire("shelf:camX:p1", "stock", 3, "Maggi empty", "Critical refill")
cell = {"slot": "p1", "name": "Maggi", "loc": "row 1 · col 1", "occluded": False, "status": "EMPTY",
        "est_units": 0, "full_units": 4, "eta_min": None}
eng.on_shelf("camX", [cell])
msgs = [a["message"] for a in eng.snapshot()["alerts"]]
check("old grid alerts cleared once product boxes report", "grid cell empty" not in msgs and any("Maggi" in m for m in msgs),
      str(msgs))
r = cli.get("/api/live?minutes=60").json()
check("live history endpoint answers", "rows" in r and "now" in r)

page = cli.get("/").text
check("setup shows the feed status and a self-checkout job", "feedLine" in page and "self-checkout" in page)
check("dashboard uses paced frames, no open-ended streams", "liveTick" in page and 'src="/video/' not in page)
store.db.close()  # release the SQLite file before cleanup on Windows
for f in ("cams.db", "cams_layout.json"):
    if os.path.exists(f):
        os.remove(f)
print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
os._exit(1 if fails else 0)
