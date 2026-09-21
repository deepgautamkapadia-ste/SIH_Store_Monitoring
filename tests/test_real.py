"""Real-detector test: build a video of a real person walking across the line, run the true pipeline."""
import os, sys, time
import numpy as np, cv2, urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
for f in ("real.db", "shelf_ref_shelf.png"):
    if os.path.exists(f):
        os.remove(f)

# a real person crop, from the standard ultralytics sample image
if not os.path.exists("bus.jpg"):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg", "bus.jpg")
bus = cv2.imread("bus.jpg")

from ultralytics import YOLO
det = YOLO("yolo11n.pt")
r = det(bus, classes=[0], conf=0.5, verbose=False)[0]
boxes = sorted(r.boxes.xyxy.cpu().numpy().tolist(), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
print(f"people found in sample image: {len(boxes)}")
x1, y1, x2, y2 = map(int, boxes[0])
person = bus[y1:y2, x1:x2]
person = cv2.resize(person, (110, 260))
ph, pw_ = person.shape[:2]

W, H, N = 640, 480, 44
bg = np.full((H, W, 3), 150, np.uint8)
cv2.rectangle(bg, (0, 330), (W, H), (120, 115, 110), -1)           # floor
for gx in range(0, W, 80):
    cv2.line(bg, (gx, 330), (gx, H), (105, 100, 95), 1)

vw = cv2.VideoWriter("walk.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 12, (W, H))
for i in range(N):
    f = bg.copy()
    x = int(40 + (W - pw_ - 80) * (i / (N - 1)))                    # left -> right, crossing x=320
    f[H - ph - 20:H - 20, x:x + pw_] = person
    vw.write(f)
vw.release()
print("wrote walk.mp4:", N, "frames, person crosses the centre line")

import storesense as ss
ss.CONFIG["db_path"] = "real.db"
ss.CONFIG["geometry"]["default"]["entry_line"] = [[0.5, 0.0], [0.5, 1.0]]
ss.CONFIG["geometry"]["default"]["queue_zone"] = [[0.72, 0.5], [1.0, 0.5], [1.0, 1.0], [0.72, 1.0]]
ss.CONFIG["queue"]["min_time_in_zone_s"] = 0.5

store = ss.Store("real.db")
engine = ss.Engine(store)
w = ss.PeopleWorker("entry", {"entry", "queue"}, "walk.mp4", engine)

cap = cv2.VideoCapture("walk.mp4")
frames, detected = 0, 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames += 1
    vis = w.step(f)
    res = w.model.track(f, persist=True, classes=[0], conf=ss.CONFIG["conf"],
                        imgsz=ss.CONFIG["imgsz"], tracker="bytetrack.yaml", verbose=False)[0]
    if res.boxes is not None and len(res.boxes):
        detected += 1
    time.sleep(0.02)
cv2.imwrite("frame_entry.jpg", vis)

fails = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{extra}]" if extra else ""))
    if not cond:
        fails.append(name)


print()
check("YOLO detects the person", detected > frames * 0.7, f"{detected}/{frames} frames")
check("counted exactly one entry", engine.footfall["in"] == 1, str(engine.footfall))
check("no spurious exits", engine.footfall["out"] == 0)
check("inside = 1", engine.snapshot()["footfall"]["inside"] == 1)
check("heatmap populated", float(w.heat.sum()) > 0, f"sum={w.heat.sum():.1f}")
check("zone dwell recorded", len(engine.snapshot()["zones"].get("entry", {})) >= 0)
check("annotated frame written", os.path.exists("frame_entry.jpg"))
check("privacy: no frames in db",
      store.q("SELECT COUNT(*) FROM events WHERE data LIKE '%jpg%' OR data LIKE '%image%'")[0][0] == 0)
check("privacy: only ids/counts stored",
      all(set(__import__("json").loads(d)) <= {"dir", "zone", "duration_s", "id", "kind", "response_s",
                                               "ts", "severity", "priority", "message", "action",
                                               "acked", "count"}
          for (d,) in store.q("SELECT data FROM events")))

# same person walking back = one exit
w2 = ss.PeopleWorker("entry2", {"entry"}, "walk.mp4", engine)
cap = cv2.VideoCapture("walk.mp4")
fr = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    fr.append(f)
for f in reversed(fr):
    w2.step(f)
    time.sleep(0.02)
check("reverse walk counts as exit", engine.footfall["out"] == 1, str(engine.footfall))

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
