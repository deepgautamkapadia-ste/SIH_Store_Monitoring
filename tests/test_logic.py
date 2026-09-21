"""Deterministic logic test: stub the detector, drive the workers frame by frame."""
import os, sys, time, json, types
import numpy as np, torch, cv2

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
for f in ("test.db", "shelf_ref_shelfA.png"):
    if os.path.exists(f):
        os.remove(f)

import ultralytics

SCRIPT = {"boxes": [], "ids": []}


class FakeBoxes:
    def __init__(self, xyxy, ids=None, cls=None):
        self.xyxy = torch.tensor(np.array(xyxy, dtype=np.float32).reshape(-1, 4))
        self.id = torch.tensor(np.array(ids, dtype=np.float32)) if ids is not None else None
        self.cls = torch.tensor(np.array(cls, dtype=np.float32)) if cls is not None else None


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    names = {0: "person"}

    def __init__(self, *a, **k):
        pass

    def __call__(self, frame, **kw):
        return [FakeResult(FakeBoxes(SCRIPT["boxes"], cls=[0] * len(SCRIPT["boxes"])))]

    def track(self, frame, **kw):
        return [FakeResult(FakeBoxes(SCRIPT["boxes"], ids=SCRIPT["ids"]))]


ultralytics.YOLO = FakeYOLO

import storesense as ss

ss.CONFIG["db_path"] = "test.db"
ss.CONFIG["alert_cooldown_s"] = 1
ss.CONFIG["queue"]["min_time_in_zone_s"] = 1.0
ss.CONFIG["shelf"]["period_s"] = 0.0
ss.CONFIG["geometry"]["default"]["entry_line"] = [[0.5, 0.0], [0.5, 1.0]]
# queue zone sits low in the frame; the entry walk below stays above it
ss.CONFIG["geometry"]["default"]["queue_zone"] = [[0.0, 0.7], [1.0, 0.7], [1.0, 1.0], [0.0, 1.0]]

# a dummy video file so Capture has something valid to open
vw = cv2.VideoWriter("dummy.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
for _ in range(5):
    vw.write(np.zeros((480, 640, 3), np.uint8))
vw.release()

store = ss.Store("test.db")
engine = ss.Engine(store)
fails = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{extra}]" if extra else ""))
    if not cond:
        fails.append(name)


frame = np.zeros((480, 640, 3), np.uint8)

# ── entry / exit counting ─────────────────────────────────────────────
print("\nentry counting")
pw = ss.PeopleWorker("entry", {"entry", "queue"}, "dummy.mp4", engine)


def walk(tid, xs, y=300):
    for x in xs:
        SCRIPT["boxes"], SCRIPT["ids"] = [[x - 30, y - 120, x + 30, y]], [tid]
        pw.step(frame)
        time.sleep(0.02)


walk(1, [100, 200, 300, 400, 500])          # left -> right = in
walk(2, [500, 400, 300, 200, 100])          # right -> left = out
check("one in, one out", engine.footfall == {"in": 1, "out": 1}, str(engine.footfall))
check("inside = 0", engine.snapshot()["footfall"]["inside"] == 0)
check("heatmap accumulated", float(pw.heat.sum()) > 0)

# a track that never crosses must not count
before = dict(engine.footfall)
walk(3, [100, 110, 120, 110, 100])
check("no phantom count", engine.footfall == before, str(engine.footfall))

# ── queue: build up, then drain ───────────────────────────────────────
print("\nqueue intelligence")
pw.t_start = time.time() - 120                      # pretend the cam has been up 2 min
qbox = lambda i: [60 + i * 40, 340, 100 + i * 40, 460]    # foot y=460, inside the queue polygon

# someone cutting through the zone for under min_time_in_zone_s is not a queuer
SCRIPT["boxes"], SCRIPT["ids"] = [qbox(0)], [99]
pw.step(frame)
check("walk-past not counted yet", engine.queues["entry"]["length"] == 0,
      str(engine.queues["entry"]["length"]))
SCRIPT["boxes"], SCRIPT["ids"] = [], []
pw.step(frame)

for n in range(1, 7):
    SCRIPT["boxes"], SCRIPT["ids"] = [qbox(i) for i in range(n)], list(range(10, 10 + n))
    for _ in range(2):                               # dwell past min_time_in_zone_s
        pw.step(frame)
        time.sleep(0.6)
time.sleep(0.6)                                      # let the last arrival qualify too
pw.step(frame)
q = engine.queues["entry"]
print("   ", json.dumps(q))
check("queue length 6", q["length"] == 6, str(q["length"]))
check("wait estimated", q["wait_min"] > 0)
check("forecast has 5/10/15", set(q["forecast"]) == {"5", "10", "15"})
check("recommends more counters", q["recommend_counters"] > 1, str(q["recommend_counters"]))
check("queue alert fired", any(a["kind"] == "queue" for a in engine.snapshot()["alerts"]))

for _ in range(3):                                   # everyone leaves
    SCRIPT["boxes"], SCRIPT["ids"] = [], []
    pw.step(frame)
    time.sleep(1.1)
check("queue drained", engine.queues["entry"]["length"] == 0, str(engine.queues["entry"]["length"]))
check("served events logged", store.q("SELECT COUNT(*) FROM events WHERE type='served'")[0][0] == 6)

# ── shelf grid ────────────────────────────────────────────────────────
print("\nshelf grid")
ss.CONFIG["shelf"]["grid"] = [2, 3]
sw = ss.ShelfWorker("shelfA", {"shelf"}, "dummy.mp4", engine)


def shelf_frame(empty_cells=()):
    """Textured 'products' in each cell; listed cells are wiped to bare shelf."""
    img = np.full((480, 640, 3), 40, np.uint8)
    rng = np.random.default_rng(0)
    for r in range(2):
        for c in range(3):
            x0, y0 = c * 213, r * 240
            for k in range(6):                        # boxes on the shelf = lots of edges
                x = x0 + 10 + k * 32
                cv2.rectangle(img, (x, y0 + 40), (x + 26, y0 + 200),
                              tuple(int(v) for v in rng.integers(60, 255, 3)), -1)
                cv2.rectangle(img, (x, y0 + 40), (x + 26, y0 + 200), (20, 20, 20), 2)
    for r, c in empty_cells:
        img[r * 240:(r + 1) * 240, c * 213:(c + 1) * 213] = 40
    return img


SCRIPT["boxes"], SCRIPT["ids"] = [], []
full = shelf_frame()
sw.step(full)
check("uncalibrated -> None", engine.shelves["shelfA"] is None)

sw.calib_request = True
sw.step(full)
check("calibrates on clear aisle", sw.calib_msg == "Calibrated", str(sw.calib_msg))

sw.step(full)
cells = engine.shelves["shelfA"]
check("all cells OK when full", all(c["status"] == "OK" for c in cells), str([c["status"] for c in cells]))

for _ in range(3):
    sw.step(shelf_frame(empty_cells=[(0, 0)]))
cells = {(c["r"], c["c"]): c for c in engine.shelves["shelfA"]}
check("cleared cell -> EMPTY", cells[(0, 0)]["status"] == "EMPTY", f"fill={cells[(0,0)]['fill']}")
check("other cells still OK", all(c["status"] == "OK" for k, c in cells.items() if k != (0, 0)))
check("empty alert fired", any("empty" in a["message"] for a in engine.snapshot()["alerts"]))

# occlusion: a person in front of the cleared cell must freeze it, not report a new state
SCRIPT["boxes"] = [[0, 0, 200, 470]]
sw.step(shelf_frame(empty_cells=[(0, 0)]))
cells = {(c["r"], c["c"]): c for c in engine.shelves["shelfA"]}
check("occluded cell flagged", cells[(0, 0)]["occluded"] is True)
check("occluded keeps last state", cells[(0, 0)]["status"] == "EMPTY")
check("aisle occupancy tracked", engine.aisle["shelfA"]["people"] == 1)
SCRIPT["boxes"] = []

# calibration must refuse while someone is in the aisle
SCRIPT["boxes"] = [[0, 0, 200, 470]]
sw.calib_request, sw.calib_msg = True, None
sw.step(shelf_frame())
check("calibration blocked when aisle busy", "not clear" in (sw.calib_msg or ""), str(sw.calib_msg))
SCRIPT["boxes"] = []

# depletion ETA from a falling fill trend
sw.history[(1, 1)].clear()
now = time.time()
for i in range(10):
    sw.history[(1, 1)].append((now - (10 - i) * 60, 1.0 - 0.06 * i))
eta = sw.eta(1, 1, now)
check("ETA computed from trend", eta is not None and 0 < eta < 600, str(eta))

# ── engine: priority, cooldown, ack ───────────────────────────────────
print("\ndecision engine")
snap = engine.snapshot()
pri = [a["priority"] for a in snap["alerts"]]
check("alerts sorted by priority", pri == sorted(pri, reverse=True), str(pri))
n0 = len(engine.snapshot()["alerts"])
engine.fire("dup", "stock", 3, "x", "y")
engine.fire("dup", "stock", 3, "x", "y")
check("cooldown suppresses duplicate", len(engine.snapshot()["alerts"]) == n0 + 1)
time.sleep(1.1)                                      # past the (test) cooldown
engine.fire("dup", "stock", 3, "x", "y")
d = [a for a in engine.snapshot()["alerts"] if a["message"] == "x"]
check("unresolved problem refreshes one row", len(d) == 1 and d[0]["count"] == 2,
      f"rows={len(d)} count={d[0]['count'] if d else '-'}")
aid = engine.snapshot()["alerts"][0]["id"]
check("ack works", engine.ack(aid) is True)
check("acked alert hidden", all(a["id"] != aid for a in engine.snapshot()["alerts"]))
check("ack logged", store.q("SELECT COUNT(*) FROM events WHERE type='ack'")[0][0] == 1)

# acking clears that key's cooldown, so an unresolved problem re-alerts immediately
engine.fire("recur", "stock", 3, "still empty", "Critical refill")
rid = next(a["id"] for a in engine.snapshot()["alerts"] if a["message"] == "still empty")
engine.fire("recur", "stock", 3, "still empty", "Critical refill")
check("cooldown holds before ack",
      sum(a["message"] == "still empty" for a in engine.snapshot()["alerts"]) == 1)
engine.ack(rid)
engine.fire("recur", "stock", 3, "still empty", "Critical refill")
check("re-alerts after ack if unresolved",
      sum(a["message"] == "still empty" for a in engine.snapshot()["alerts"]) == 1,
      "should reappear once acked")
check("unrelated cooldowns untouched", "shelf:shelfA:0,0" in engine.last, str(sorted(engine.last)[:3]))

# a condition that fixes itself clears its own alert
print("\nauto-resolve")
SCRIPT["boxes"] = []
for _ in range(3):
    sw.step(shelf_frame(empty_cells=[(1, 2)]))
check("alert raised for emptied cell",
      any("row 2 col 3" in a["message"] for a in engine.snapshot()["alerts"]))
for _ in range(3):
    sw.step(shelf_frame())                               # restocked
check("alert clears when restocked",
      not any("row 2 col 3" in a["message"] for a in engine.snapshot()["alerts"]))
check("cooldown cleared too", "shelf:shelfA:1,2" not in engine.last)

engine.fire("queue:entry:open", "queue", 3, "long", "Open 1 more counter(s)")
check("queue alert present", any(a["message"] == "long" for a in engine.snapshot()["alerts"]))
engine.on_queue("entry", {"length": 0, "wait_min": 0.0, "recommend_counters": 1,
                          "arrival_per_min": 0, "service_per_counter_min": 1.5,
                          "forecast": {t: {"len": 0, "wait": 0} for t in ("5", "10", "15")}})
check("queue alert clears when drained",
      not any(a["message"] == "long" for a in engine.snapshot()["alerts"]))

# ── api + reports ─────────────────────────────────────────────────────
print("\napi + reports")
from fastapi.testclient import TestClient

app = ss.make_app(engine, {"entry": pw, "shelfA": sw}, store)
cl = TestClient(app)
check("dashboard serves", cl.get("/").status_code == 200 and "StoreSense" in cl.get("/").text)
st = cl.get("/api/state").json()
check("/api/state shape", {"footfall", "queues", "shelves", "alerts", "heat"} <= set(st))
check("POS hook", cl.post("/api/integrations/pos", json={"bill_id": "b1", "items": []}).json()["ok"])
check("counters endpoint", cl.post("/api/counters?open=3").json()["open_counters"] == 3)
check("404 on unknown cam calibrate", cl.post("/api/calibrate/nope").status_code == 404)
rep = cl.get("/api/report?period=day").json()
print("   ", json.dumps({k: rep[k] for k in ("footfall_total", "customers_billed", "conversion",
                                             "avg_time_in_queue_min", "alerts_by_action")}))
check("report footfall", rep["footfall_total"] == 1)
check("report counts POS bill", rep["customers_billed"] == 1)
check("report has alerts", len(rep["alerts_by_action"]) > 0)
check("weekly report runs", cl.get("/api/report?period=week").status_code == 200)

with cl.websocket_connect("/ws") as ws:
    check("websocket pushes state", "footfall" in ws.receive_json())

# ── offline-first sync ────────────────────────────────────────────────
print("\noffline buffering")
ss.CONFIG["cloud_url"] = "http://127.0.0.1:9/nope"
store.outbox_add({"test": 1})
check("payload buffered", len(store.outbox_pending()) == 1)
t = __import__("threading").Thread(target=ss.cloud_sync, args=(engine, store), daemon=True)
t.start()
time.sleep(32)
check("stays buffered while offline", len(store.outbox_pending()) == 1 and engine.online is False,
      f"online={engine.online}")
ss.CONFIG["cloud_url"] = None

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
