"""Store-layout geometry: coverage from any camera placement, store-frame mapping, 3D render."""
import os, sys, json, math, types

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
for f in ("lay.db", "lay.json", "lay.json.tmp"):
    if os.path.exists(f):
        os.remove(f)

import numpy as np
import storesense as ss

ss.CONFIG["db_path"] = "lay.db"
ss.CONFIG["layout_path"] = "lay.json"

fails = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{extra}]" if extra else ""))
    if not cond:
        fails.append(name)


def near(a, b, tol=1e-6):
    return abs(a - b) < tol


# ── rects and faces ───────────────────────────────────────────────────
print("\ngeometry")
r = {"x": 5, "y": 4, "w": 4, "h": 2, "rot": 0}
c = ss.rect_corners(r)
check("corners unrotated", c[0] == (3, 3) and c[2] == (7, 5), str(c[0]) + str(c[2]))
c90 = ss.rect_corners({**r, "rot": 90})
check("90deg swaps extents", near(c90[0][0], 6) and near(c90[0][1], 2), str(c90[0]))
check("N face is the top edge", ss.face_segment(r, "N") == ((3, 3), (7, 3)))
n = ss.face_normal(r, "N")
check("N normal points up", near(n[0], 0) and near(n[1], -1), str(n))
n90 = ss.face_normal({**r, "rot": 90}, "N")
check("normal rotates with shelf", near(n90[0], 1) and near(n90[1], 0), str(n90))
check("segments cross", ss.seg_cross((0, 0), (2, 2), (0, 2), (2, 0)))
check("parallel segments don't", not ss.seg_cross((0, 0), (2, 0), (0, 1), (2, 1)))

# ── visibility from any placement ─────────────────────────────────────
print("\nvisibility (no assumption about camera arrangement)")
shelf = {"id": "S1", "name": "A", "x": 5, "y": 4, "w": 4, "h": 1, "rot": 0,
         "height": 1.8, "faces": {"N": {"grid": [4, 6]}, "S": {"grid": [4, 6]}}}
front = {"id": "c1", "x": 5, "y": 1.5, "heading": 90, "fov": 90, "range": 8, "roles": ["shelf"]}
check("camera in front sees the N face", ss.face_visibility(front, shelf, "N", [shelf]) == 1.0)
check("...and not the S face behind it", ss.face_visibility(front, shelf, "S", [shelf]) == 0.0)

behind = {**front, "y": 6.5, "heading": -90}
check("camera on the other side sees S", ss.face_visibility(behind, shelf, "S", [shelf]) == 1.0)
check("...and not N", ss.face_visibility(behind, shelf, "N", [shelf]) == 0.0)

far = {**front, "range": 1.0}
check("out of range sees nothing", ss.face_visibility(far, shelf, "N", [shelf]) == 0.0)
narrow = {**front, "fov": 10, "x": 1.0, "heading": 0}
check("outside the FOV cone sees nothing", ss.face_visibility(narrow, shelf, "N", [shelf]) == 0.0)

# a camera off to one side at an angle still sees part of the face
angled = {"id": "c2", "x": 1.0, "y": 1.0, "heading": 45, "fov": 100, "range": 12, "roles": ["shelf"]}
v = ss.face_visibility(angled, shelf, "N", [shelf])
check("angled camera gets partial coverage", 0 < v <= 1.0, f"{v}")

# an obstruction between them blocks the view
wall = {"id": "W1", "name": "wall", "x": 5, "y": 2.6, "w": 6, "h": 0.3, "rot": 0, "height": 2, "faces": {}}
check("a shelf in the way blocks sight", ss.face_visibility(front, shelf, "N", [shelf, wall]) == 0.0)

# rotated shelf: the N face now points up-right, so a camera must stand along that normal
diag = {**shelf, "id": "S2", "rot": 45}
(fx0, fy0), (fx1, fy1) = ss.face_segment(diag, "N")
fcx, fcy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
nx, ny = ss.face_normal(diag, "N")
cam_diag = {"id": "c3", "x": fcx + nx * 2.5, "y": fcy + ny * 2.5, "roles": ["shelf"],
            "heading": math.degrees(math.atan2(-ny, -nx)), "fov": 100, "range": 8}
check("works on a rotated shelf", ss.face_visibility(cam_diag, diag, "N", [diag]) == 1.0,
      str(round(ss.face_visibility(cam_diag, diag, "N", [diag]), 2)))
# and the same camera must NOT see the opposite face through the shelf body
check("rotated shelf blocks its own far face", ss.face_visibility(cam_diag, diag, "S", [diag]) == 0.0)

# ── coverage + blind spots ────────────────────────────────────────────
print("\ncoverage")
lay_data = {"store": {"w": 10, "h": 8}, "shelves": [shelf], "cameras": [front]}
cov = ss.coverage(lay_data)
check("N face covered by c1", cov["S1:N"]["visible"] == 1.0 and cov["S1:N"]["camera"] == "c1")
check("S face is a blind spot", cov["S1:S"]["visible"] == 0.0 and cov["S1:S"]["camera"] is None)
cov2 = ss.coverage({**lay_data, "cameras": [front, behind]})
check("second camera clears the blind spot", cov2["S1:S"]["visible"] == 1.0)
check("only shelf-role cameras count",
      ss.coverage({**lay_data, "cameras": [{**front, "roles": ["entry"]}]})["S1:N"]["camera"] is None)

# ── load / save / auto-binding ────────────────────────────────────────
print("\nlayout file")
lay = ss.Layout("lay.json")
check("defaults when no file", lay.data["shelves"] == [] and lay.data["store"]["w"] == 6)
saved = lay.save({"store": {"w": 10, "h": 8},
                  "shelves": [shelf],
                  "cameras": [front, {**behind, "id": "c2"}]})
check("saved to disk", os.path.exists("lay.json"))
check("normalised shelf keeps faces", set(saved["shelves"][0]["faces"]) == {"N", "S"})
again = ss.Layout("lay.json")
check("reloads identically", again.data == saved)
sh, fc = again.watched("c1")
check("auto-binds c1 to the N face", sh["id"] == "S1" and fc == "N", f"{fc}")
sh2, fc2 = again.watched("c2")
check("auto-binds c2 to the S face", fc2 == "S", f"{fc2}")
explicit = ss.Layout.normalise({**saved, "cameras": [{**front, "watch": {"shelf": "S1", "face": "S"}}]})
l2 = ss.Layout("lay.json")
l2.data = explicit
check("explicit watch wins over geometry", l2.watched("c1")[1] == "S")
check("unplaced camera resolves to nothing", again.watched("nope") == (None, None))
check("grid dims from store size", again.grid_dims() == (32, 40), str(again.grid_dims()))

# ── store-frame heatmap ───────────────────────────────────────────────
print("\nstore-frame mapping")
store = ss.Store("lay.db")
eng = ss.Engine(store, again)
check("store heat allocated", eng.store_heat.shape == (32, 40))
eng.add_store_heat(5.0, 4.0, 2.0)
check("dwell lands in the right cell", eng.store_heat[16, 20] == 2.0)
eng.add_store_heat(-1, 4, 1.0)
eng.add_store_heat(99, 4, 1.0)
check("outside the store is ignored", float(eng.store_heat.sum()) == 2.0)
check("snapshot normalises heat", max(max(r) for r in eng.snapshot()["store_heat"]) == 1.0)
check("snapshot carries the layout", eng.snapshot()["layout"]["store"]["w"] == 10)

# two cameras with different floor patches land in one shared frame
lay.save({"store": {"w": 10, "h": 8}, "shelves": [shelf],
          "cameras": [{**front, "floor_rect": {"x": 2, "y": 2, "w": 4, "h": 4, "rot": 0}},
                      {**behind, "id": "c2", "floor_rect": {"x": 8, "y": 6, "w": 2, "h": 2, "rot": 0}}]})
eng.layout = lay
dummy = types.SimpleNamespace(name="c1", engine=eng, storeH=None,
                              g={"floor_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]})
mid = ss.PeopleWorker.store_point(dummy, 320, 240, 640, 480)
check("image centre maps to patch centre", near(mid[0], 2, 1e-3) and near(mid[1], 2, 1e-3), str(mid))
corner = ss.PeopleWorker.store_point(dummy, 640, 480, 640, 480)
check("image corner maps to patch corner", near(corner[0], 4, 1e-3) and near(corner[1], 4, 1e-3), str(corner))
d2 = types.SimpleNamespace(name="c2", engine=eng, storeH=None,
                           g={"floor_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]})
mid2 = ss.PeopleWorker.store_point(d2, 320, 240, 640, 480)
check("second camera maps into the same frame", near(mid2[0], 8, 1e-3) and near(mid2[1], 6, 1e-3), str(mid2))
none_cam = types.SimpleNamespace(name="nope", engine=eng, storeH=None, g={"floor_quad": [[0, 0]]})
check("unplaced camera contributes nothing", ss.PeopleWorker.store_point(none_cam, 1, 1, 640, 480) is None)

# ── 3D render ─────────────────────────────────────────────────────────
print("\n3D view")
eng.shelves["c1"] = [{"r": 0, "c": 0, "status": "EMPTY", "occluded": False, "fill": 0.0,
                      "corr": 1.0, "eta_min": None, "expected": None, "found": None}]
check("empty face colours red", ss.face_status(eng, "S1", "N") == "EMPTY",
      str(ss.face_status(eng, "S1", "N")))
check("unwatched face has no status", ss.face_status(eng, "S1", "S") is None)
png = ss.render_3d(eng)
check("renders a PNG", png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10000, f"{len(png)} bytes")
open("layout_3d.png", "wb").write(png)

# ── API ───────────────────────────────────────────────────────────────
print("\napi")
from fastapi.testclient import TestClient

# a worker with live mappings must survive a save (numpy attrs must not be truth-tested)
worker = types.SimpleNamespace(storeH=np.eye(3), H=np.eye(3), _quad=[[0, 0]], _rect={"x": 1})
app = ss.make_app(eng, {"entry": worker}, store)
cl = TestClient(app)
g = cl.get("/api/layout").json()
check("GET returns layout + coverage", g["layout"]["store"]["w"] == 10 and "S1:N" in g["coverage"])
new = dict(g["layout"])
new["store"] = {"w": 14, "h": 9}
p = cl.post("/api/layout", json=new).json()
check("POST saves", p["ok"] and p["layout"]["store"]["w"] == 14)
check("worker mappings reset on save", worker.storeH is None and worker._rect is None)
check("heat regridded on save", eng.store_heat.shape == (36, 56), str(eng.store_heat.shape))
check("POST rejects junk", cl.post("/api/layout", json={"shelves": [{"x": 1}]}).json()["ok"] is False)
check("3d endpoint serves PNG", cl.get("/api/layout/3d.png").content[:4] == b"\x89PNG")
check("dashboard has the editor", "id=\"plan\"" in cl.get("/").text)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
