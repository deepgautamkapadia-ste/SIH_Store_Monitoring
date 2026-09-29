"""Cameras start, restart and stop when the store plan is saved — no server restart."""
import os, sys, time, types

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for f in ("cams.db", "cams_layout.json"):
    if os.path.exists(f):
        os.remove(f)

import numpy as np, torch
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

page = cli.get("/").text
check("setup shows the feed status and a self-checkout job", "feedLine" in page and "self-checkout" in page)
for f in ("cams.db", "cams_layout.json"):
    if os.path.exists(f):
        os.remove(f)
print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
os._exit(1 if fails else 0)
