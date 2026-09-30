"""Depth counting: units behind the front row, from one camera at any angle.

The shelf is ray-traced (tests/shelf3d.py), so the true distance of every pixel is known. The depth
model is replaced by a stand-in that returns that truth corrupted the way a monocular model is:
a different scale and offset on every pass, blurred edges, smooth low-frequency error and pixel noise.
"""
import os, sys, types, json, hashlib

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for f in ("depth.db", "depth_layout.json", "shelf_ref_shelfD.png"):
    if os.path.exists(f):
        os.remove(f)

import numpy as np, cv2, torch
import ultralytics

PERSONS = []


class FakeBoxes:
    def __init__(self, xyxy):
        self.xyxy = torch.tensor(np.array(xyxy, dtype=np.float32).reshape(-1, 4))
        self.id = None
        self.cls = torch.zeros(len(xyxy))


class FakeYOLO:
    names = {0: "person"}

    def __init__(self, *a, **k):
        pass

    def __call__(self, frame, **kw):
        return [types.SimpleNamespace(boxes=FakeBoxes(PERSONS))]


ultralytics.YOLO = FakeYOLO
import storesense as ss
import shelf3d

ss.CONFIG["shelf"]["period_s"] = 0
ss.CONFIG["shelf"]["depth"]["every_s"] = 0
ss.CONFIG["alert_cooldown_s"] = 0

fails = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


# ── stand-in monocular model ─────────────────────────────────────────
TRUTH = {}
rng = np.random.default_rng(3)
CALLS = [0]


def key(img):
    return hashlib.md5(img.tobytes()).hexdigest()


def fake_depth(bgr):
    CALLS[0] += 1
    d = TRUTH[key(bgr)]
    h, w = d.shape
    out = cv2.GaussianBlur(d, (0, 0), 2.0)                          # soft object edges
    lowfreq = cv2.resize(rng.normal(0, 0.012, (6, 8)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    out = out + lowfreq + rng.normal(0, 0.006, d.shape).astype(np.float32)
    s, b = rng.uniform(0.93, 1.07), rng.uniform(-0.06, 0.06)      # scale/offset drift per pass
    return (s * out + b).astype(np.float32)


def frame(removed=None, **view):
    img, d = shelf3d.render(removed, **view)
    TRUTH[key(img)] = d
    return img


# ── align_depth on its own ───────────────────────────────────────────
print("alignment")
d0 = np.random.default_rng(0).uniform(1, 2, (100, 120)).astype(np.float32)
d1 = (d0 - 0.05) / 1.08
d1[40:60, 40:60] += 0.3                        # a changed patch must not bend the fit
m = np.ones_like(d0, bool)
al, noise = ss.align_depth(d1, d0, m)
check("recovers scale and offset despite a changed patch", np.abs(al - d0)[m & (np.abs(al - d0) < 0.1)].max() < 1e-3
      and noise < 1e-3, f"noise {noise}")
_, n2 = ss.align_depth(d1 * 3, d0, m)
check("rejects an implausible fit (view changed)", n2 is None)


# ── one shelf camera, several views ──────────────────────────────────
def setup(view):
    for f in ("depth.db", "depth_layout.json", "shelf_ref_shelfD.png"):
        if os.path.exists(f):
            os.remove(f)
    store = ss.Store("depth.db")
    lay = ss.Layout("depth_layout.json")
    slots = shelf3d.slot_boxes(**view)
    lay.save({"store": {"w": 6, "h": 4}, "shelves": [],
              "cameras": [{"id": "shelfD", "x": 1, "y": 1, "roles": ["shelf"], "slots": slots}]})
    eng = ss.Engine(store, lay)
    sw = ss.ShelfWorker("shelfD", {"shelf"}, "dummy_depth.mp4", eng)
    return eng, sw, slots


def run(sw, eng, img, n=3):
    for _ in range(n):
        sw.step(img)
    return {c["slot"]: c for c in eng.shelves["shelfD"]}


TAKEN = {("cola", 0): 2, ("cola", 1): 1, ("soap", 2): 4, ("tea", 1): 4, ("jam", 3): 1, ("chips", 0): 3}
FULL = {p[0]: p[5] * p[6] for p in shelf3d.PRODUCTS}
DEEP = {p[0]: p[6] for p in shelf3d.PRODUCTS}

VIEWS = {"straight on": dict(), "from the side, 18°": dict(yaw=18, pitch=-4, pos=(-0.35, -0.1, 0.0)),
         "from high up, looking down 14°": dict(yaw=-6, pitch=-14, pos=(0.1, -0.45, 0.05))}
ss.DEPTH.update(fn=fake_depth, err=None, tried=True, name="stand-in")
for vname, view in VIEWS.items():
    print(f"\n{vname}")
    eng, sw, slots = setup(view)
    PERSONS[:] = []
    full_img = frame(None, **view)
    sw.calib_request = True
    sw.step(full_img)
    check("calibrated", sw.calib_msg == "Calibrated")
    cells = run(sw, eng, full_img, 1)
    check("first passes build the full-shelf depth", not sw.depth_state["on"] and "measuring" in sw.depth_state["msg"],
          sw.depth_state["msg"])
    cells = run(sw, eng, full_img, 4)
    check("depth method used on every product", all(c["method"] == "depth" for c in cells.values()),
          str({k: c["method"] for k, c in cells.items()}))
    check("full shelf counts full", all(c["est_units"] == FULL[k] for k, c in cells.items()),
          str({k: (c["est_units"], FULL[k]) for k, c in cells.items()}))
    check("fit error small", sw.depth_state["noise_cm"] is not None and sw.depth_state["noise_cm"] < 3,
          f"{sw.depth_state['noise_cm']} cm")

    after = frame(TAKEN, **view)
    cells = run(sw, eng, after)
    want = {k: [DEEP[k] - TAKEN.get((k, i), 0) for i in range(len(cells[k]["columns"]))] for k in cells}
    got = {k: c["columns"] for k, c in cells.items()}
    hidden = {(k, i) for k, c in cells.items() for i in c["hidden"]}
    wrong = {(k, i): (got[k][i], want[k][i]) for k in cells for i in range(len(want[k]))
             if (k, i) not in hidden and got[k][i] != want[k][i]}
    check("every column the camera can see is counted exactly", not wrong, f"wrong {wrong}")
    check("columns it can't see are reported, not guessed", len(hidden) <= 2,
          f"hidden {sorted(hidden) or 'none'}")
    if not hidden:
        check("...all columns visible from here, all exact", got == want)
    check("cola: 16 -> 13 although all 4 facings still show a unit",
          cells["cola"]["est_units"] == 13 and cells["cola"]["present"] == 4,
          f"{cells['cola']['est_units']} / present {cells['cola']['present']}")
    if ("tea", 1) in hidden:
        check("tea: emptied column hidden from here -> front unit known gone, count capped",
              cells["tea"]["columns"][1] <= DEEP["tea"] - 1, str(cells["tea"]["columns"]))
    else:
        check("tea: an emptied column counts 0", cells["tea"]["columns"][1] == 0)
    check("chips: 3 taken from one column -> 5 of 8, still OK", cells["chips"]["est_units"] == 5 and
          cells["chips"]["status"] == "OK", f"{cells['chips']['est_units']} {cells['chips']['status']}")

    # someone steps in front of the cola: its count must hold, not jump
    x0 = slots[0]["x"] * 640
    PERSONS[:] = [[x0 - 10, 0, x0 + slots[0]["w"] * 640 * 0.6, 479]]
    blocked = frame({**TAKEN, ("cola", 0): 4, ("cola", 1): 4, ("cola", 2): 4, ("cola", 3): 4}, **view)
    cells = run(sw, eng, blocked)
    check("blocked product keeps its last count", cells["cola"]["est_units"] == 13 and cells["cola"]["occluded"],
          f"{cells['cola']['est_units']} occluded={cells['cola']['occluded']}")
    PERSONS[:] = []

# the old way, for comparison: without a unit size the camera can only see the front row
print("\nwithout unit sizes (front row only)")
eng, sw, slots = setup({})
for s in slots:
    s["unit_cm"] = 0
eng.layout.save({"store": {"w": 6, "h": 4}, "shelves": [],
                 "cameras": [{"id": "shelfD", "x": 1, "y": 1, "roles": ["shelf"], "slots": slots}]})
sw.calib_request = True
sw.step(frame())
calls = CALLS[0]
cells = run(sw, eng, frame(TAKEN))
check("falls back to facings x depth", all(c["method"] == "front" for c in cells.values()))
check("...and cannot see the 2 cola taken from behind the front", cells["cola"]["est_units"] == 16,
      str(cells["cola"]["est_units"]))
check("depth model not run when no box needs it", CALLS[0] == calls)

print("\ndepth model missing")
ss.DEPTH.update(fn=None, err="not installed — pip install transformers", tried=True)
eng, sw, slots = setup({})
sw.calib_request = True
sw.step(frame())
cells = run(sw, eng, frame(TAKEN))
check("still counts, front-row method", all(c["method"] == "front" for c in cells.values()))
check("says why depth is off", not sw.depth_state["on"] and "pip install transformers" in sw.depth_state["msg"],
      sw.depth_state["msg"])
check("snapshot carries depth status", eng.snapshot()["depth"]["shelfD"]["on"] is False)

print("\nAPI")
ss.DEPTH.update(fn=fake_depth, err=None, tried=True)
eng, sw, slots = setup({})
sw.calib_request = True
sw.step(frame())
run(sw, eng, frame(TAKEN))
from fastapi.testclient import TestClient
cli = TestClient(ss.make_app(eng, {"shelfD": sw}, eng.store))
r = cli.get("/api/depth/shelfD.jpg")
check("depth picture served", r.status_code == 200 and r.headers["content-type"] == "image/jpeg")
r = cli.get("/api/slots/shelfD")
check("unit size saved on the product box", r.json()["slots"][0]["unit_cm"] == 8.0, str(r.json()["slots"][0]))
page = cli.get("/").text
check("dashboard has depth controls", "b_depth" in page and "one unit, front to back" in page)
with open("depth_view_test.jpg", "wb") as fh:
    fh.write(cli.get("/api/depth/shelfD.jpg").content)

for f in ("depth.db", "depth_layout.json", "shelf_ref_shelfD.png"):
    if os.path.exists(f):
        os.remove(f)
print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
