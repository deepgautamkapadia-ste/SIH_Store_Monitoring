"""Product slots, catalog, checkout camera, carts and bills."""
import os, sys, time, json, types, shutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
for f in ("pos.db", "pos_layout.json", "shelf_ref_shelfP.png"):
    if os.path.exists(f):
        os.remove(f)
shutil.rmtree("bills_test", ignore_errors=True)

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

ss.CONFIG["bills_dir"] = "bills_test"
ss.CONFIG["alert_cooldown_s"] = 0
ss.CONFIG["shelf"]["period_s"] = 0

fails = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{extra}]" if extra else ""))
    if not cond:
        fails.append(name)


store = ss.Store("pos.db")

# ── catalog ───────────────────────────────────────────────────────────
print("\ncatalog")
p = store.product_upsert({"sku": "maggi-70", "name": "Maggi Masala 70g", "brand": "Nestle",
                          "mrp": 15, "price": 14, "barcode": "8901058002157"})
check("upsert with real EAN", p["barcode"] == "8901058002157" and p["price"] == 14)
q = store.product_upsert({"sku": "dove-sh", "name": "Dove Shampoo 180ml", "brand": "Dove", "mrp": 199, "price": 169})
check("missing barcode gets an in-store EAN", q["barcode"].startswith("20") and ss.ean13_bits(q["barcode"]) is not None,
      q["barcode"])
check("in-store EAN is stable", ss.internal_ean("dove-sh") == q["barcode"])
r = store.product_upsert({"sku": "parle-g", "name": "Parle-G 250g", "mrp": 25})
check("price defaults to MRP", r["price"] == 25)
try:
    store.product_upsert({"sku": "bad", "name": "x", "mrp": 10, "price": 12})
    check("price above MRP rejected", False)
except ValueError:
    check("price above MRP rejected", True)
try:
    store.product_upsert({"sku": "dupe", "name": "x", "mrp": 10, "barcode": "8901058002157"})
    check("duplicate barcode rejected", False)
except ValueError:
    check("duplicate barcode rejected", True)
check("lookup by barcode", store.product_by_code("8901058002157")["sku"] == "maggi-70")
check("lookup by sku", store.product_by_code("dove-sh")["name"] == "Dove Shampoo 180ml")
check("partial update keeps other fields",
      store.product_upsert({"sku": "maggi-70", "price": 13})["brand"] == "Nestle")
store.product_upsert({"sku": "maggi-70", "price": 14})

# ── cart + checkout ───────────────────────────────────────────────────
print("\ncheckout")
lay = ss.Layout("pos_layout.json")
eng = ss.Engine(store, lay)
cid = eng.cart_new()
eng.cart_add(cid, "8901058002157")
eng.cart_add(cid, "8901058002157")
eng.cart_add(cid, "dove-sh")
_, err = eng.cart_add(cid, "0000000000000")
check("unknown code refused", err is not None and eng.last_scan["ok"] is False)
v = eng.cart_view(cid)
check("cart lines", len(v["lines"]) == 2 and v["n_items"] == 3, str(v["n_items"]))
check("totals", v["total"] == 14 * 2 + 169 and v["mrp_total"] == 15 * 2 + 199, f"{v['total']} / {v['mrp_total']}")
check("savings", v["savings"] == round((30 + 199) - (28 + 169), 2), str(v["savings"]))
eng.cart_set(cid, "dove-sh", 0)
check("qty 0 removes the line", len(eng.cart_view(cid)["lines"]) == 1)
eng.cart_add(cid, "dove-sh")

eng.stock["maggi-70"] = 12                     # as if the shelf had been calibrated/restocked
bill, err = eng.checkout(cid)
check("bill created", bill is not None and err is None)
check("bill number format", bill["id"].startswith("SS-") and bill["id"].endswith("-0001"), bill["id"])
check("PNG + PDF written", os.path.exists(f"bills_test/{bill['id']}.png") and
      open(f"bills_test/{bill['id']}.pdf", "rb").read(4) == b"%PDF")
check("cart closed after checkout", eng.cart_view(cid) is None and eng.active_cart is None)
check("sale decrements tracked stock", eng.pos_units("maggi-70", 12) == 10, str(eng.pos_units("maggi-70", 12)))
check("sale logged for analytics", store.q("SELECT COUNT(*) FROM events WHERE type='pos'")[0][0] == 1)
check("second bill numbers on", store.next_bill_no().endswith("-0002"))
empty, err = eng.checkout(eng.cart_new())
check("empty cart can't be billed", empty is None and err)
png = open(f"bills_test/{bill['id']}.png", "rb").read()
cv2.imwrite("bill_preview.png", cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR))

# ── checkout camera ───────────────────────────────────────────────────
print("\ncheckout camera")
vw = cv2.VideoWriter("dummy_pos.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 480))
vw.write(np.zeros((480, 640, 3), np.uint8))
vw.release()
cw = ss.CheckoutWorker("till", {"checkout"}, "dummy_pos.mp4", eng)


def scene(code=None):
    img = np.full((480, 640, 3), 140, np.uint8)
    if code:
        lab = cv2.cvtColor(np.array(ss.ean13_image(code, "item")), cv2.COLOR_RGB2BGR)
        lab = cv2.resize(lab, None, fx=0.8, fy=0.8)
        img[150:150 + lab.shape[0], 160:160 + lab.shape[1]] = lab
    return img


eng.active_cart = None
for _ in range(4):                               # held in view for several frames
    cw.step(scene("8901058002157"))
till_cart = eng.active_cart
check("camera scan opens a cart", till_cart is not None)
check("item held in view counts once", eng.cart_view(till_cart)["n_items"] == 1,
      str(eng.cart_view(till_cart)["n_items"]))
cw.step(scene())
cw.seen = {k: v - 5 for k, v in cw.seen.items()}  # time passes with it out of view
cw.step(scene("8901058002157"))
check("shown again later counts again", eng.cart_view(till_cart)["n_items"] == 2)
cw.step(scene(ss.internal_ean("dove-sh")))
check("in-store label scans too", any(l["sku"] == "dove-sh" for l in eng.cart_view(till_cart)["lines"]))

# ── product slots on the shelf ────────────────────────────────────────
print("\nshelf slots")
slots = [
    {"id": "a", "name": "A", "sku": "maggi-70", "x": 0.05, "y": 0.08, "w": 0.40, "h": 0.36, "facings": 4, "deep": 3},
    {"id": "b", "name": "B", "sku": "dove-sh", "x": 0.55, "y": 0.08, "w": 0.40, "h": 0.36, "facings": 2, "deep": 2},
    {"id": "c", "name": "C", "sku": "parle-g", "x": 0.05, "y": 0.56, "w": 0.90, "h": 0.36, "facings": 6, "deep": 1},
]
locs = ss.slot_locations(slots)
check("row/col read from box positions", locs == {"a": (1, 1), "b": (1, 2), "c": (2, 1)}, str(locs))
lay.save({"store": {"w": 6, "h": 4}, "shelves": [],
          "cameras": [{"id": "shelfP", "x": 1, "y": 1, "roles": ["shelf"], "slots": slots}]})
check("slots survive save/normalise", len(lay.cam("shelfP")["slots"]) == 3 and
      lay.cam("shelfP")["slots"][0]["deep"] == 3)
sw = ss.ShelfWorker("shelfP", {"shelf"}, "dummy_pos.mp4", eng)

# someone browsing in front of a shelf camera appears on the floor heatmap
PERSONS[:] = [[300, 120, 360, 470]]
sw.step(np.zeros((480, 640, 3), np.uint8))
time.sleep(0.3)
total0 = float(eng.store_heat.sum())
sw.step(np.zeros((480, 640, 3), np.uint8))
check("a shopper at the shelf warms the floor map", float(eng.store_heat.sum()) > total0 and
      float(eng.store_live.max()) > 0, f"{total0} -> {float(eng.store_heat.sum())}")
check("...and the dashboard is told the positions are estimates", eng.snapshot()["heat_src"].get("shelfP") == "approx")
PERSONS[:] = []


def shelf(missing=(), swapped=()):
    """Each facing is a boxy product with strong edges; missing ones are bare shelf; swapped ones hold a
    different product (other colours, same shape)."""
    img = np.full((480, 640, 3), 50, np.uint8)
    rng = np.random.default_rng(1)
    H, W = img.shape[:2]
    for s in slots:
        for i, a, b, c, d in ss.ShelfWorker.facing_boxes(s, W, H):
            col = tuple(int(v) for v in rng.integers(80, 255, 3))     # same colours whatever is missing
            if (s["id"], i) in missing:
                continue
            if (s["id"], i) in swapped:
                col = (int(255 - col[0] * 0.3), int(col[2] * 0.2), int(255 - col[1] * 0.5))
            cv2.rectangle(img, (a + 4, b + 6), (c - 4, d - 4), col, -1)
            cv2.rectangle(img, (a + 4, b + 6), (c - 4, d - 4), (15, 15, 15), 2)
            cv2.putText(img, "SS", (a + 8, b + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


PERSONS[:] = []
sw.calib_request = True
sw.step(shelf())
check("calibrates with slots", sw.calib_msg == "Calibrated")
check("calibration restocks POS levels", eng.stock.get("dove-sh") == 4 and eng.stock.get("parle-g") == 6,
      str(eng.stock))
for _ in range(3):
    sw.step(shelf())
cells = {c["slot"]: c for c in eng.shelves["shelfP"]}
check("all facings present when full",
      cells["a"]["present"] == 4 and cells["b"]["present"] == 2 and cells["c"]["present"] == 6,
      str({k: v["present"] for k, v in cells.items()}))
check("estimate = facings seen x depth", cells["a"]["est_units"] == 12 and cells["a"]["full_units"] == 12)
check("catalog name + location on the slot", cells["a"]["name"] == "Maggi Masala 70g" and cells["c"]["loc"] == "row 2 · col 1")

for _ in range(3):
    sw.step(shelf(missing={("a", 0), ("a", 1)}))
cells = {c["slot"]: c for c in eng.shelves["shelfP"]}
check("taking 2 facings is noticed", cells["a"]["present"] == 2, str(cells["a"]["present"]))
check("...estimate drops by 2 x depth", cells["a"]["est_units"] == 6, str(cells["a"]["est_units"]))
check("half gone -> LOW", cells["a"]["status"] == "LOW", cells["a"]["status"])
check("neighbouring slot unaffected", cells["b"]["status"] == "OK" and cells["b"]["present"] == 2)
al = [a for a in eng.snapshot()["alerts"] if "Maggi" in a["message"]]
check("alert names the product and its place", al and "row 1 · col 1" in al[0]["message"] and "left" in al[0]["message"],
      al[0]["message"] if al else "no alert")

for _ in range(3):
    sw.step(shelf(missing={("b", 0), ("b", 1)}))
cells = {c["slot"]: c for c in eng.shelves["shelfP"]}
check("all facings gone -> EMPTY", cells["b"]["status"] == "EMPTY" and cells["b"]["est_units"] == 0)

# a single narrow product in a 6-wide slot: the old uniform grid missed this, slots don't
for _ in range(3):
    sw.step(shelf(missing={("c", 5)}))
cells = {c["slot"]: c for c in eng.shelves["shelfP"]}
check("one of six small items taken is seen", cells["c"]["present"] == 5, str(cells["c"]["present"]))

PERSONS[:] = [[0, 0, 300, 480]]                  # shopper blocks the left half
sw.step(shelf(missing={("a", 0), ("a", 1), ("a", 2), ("a", 3)}))
cells = {c["slot"]: c for c in eng.shelves["shelfP"]}
check("occluded slot keeps its last reading, not a false EMPTY",
      cells["a"]["occluded"] and cells["a"]["present"] == 4 and cells["a"]["status"] == "OK",
      f"{cells['a']['occluded']} {cells['a']['present']} {cells['a']['status']}")
PERSONS[:] = []

eng.on_pos({"items": [{"sku": "dove-sh", "qty": 1}]})
cells_now = sw.read_slots(shelf(), slots, [])
b = {c["slot"]: c for c in cells_now}["b"]
check("POS count alongside the camera estimate", b["pos_units"] == 3, str(b["pos_units"]))

# editing slots re-derives references from the stored picture, no re-shoot
moved = [dict(slots[0], facings=2)] + slots[1:]
cells2 = {c["slot"]: c for c in sw.read_slots(shelf(), moved, [])}
check("changing facings needs no recalibration", cells2["a"]["facings"] == 2 and cells2["a"]["present"] == 2)

# ── planogram: a different product in a box ──────────────────────────
print("\nplanogram")
sw.slot_sig = None
for _ in range(3):
    sw.step(shelf())
cells = {c["slot"]: c for c in eng.shelves["shelfP"]}
check("right products -> no wrong-product flag", not any(c["misplaced"] for c in cells.values()),
      str({k: c["match"] for k, c in cells.items()}))
for _ in range(3):
    sw.step(shelf(missing={("a", 0), ("a", 1)}))
cells = {c["slot"]: c for c in eng.shelves["shelfP"]}
check("taking products out is not 'wrong product'", not cells["a"]["misplaced"], str(cells["a"]["match"]))
for _ in range(3):
    sw.step(shelf(swapped={("c", i) for i in range(6)}))
cells = {c["slot"]: c for c in eng.shelves["shelfP"]}
check("a different product in a box is flagged", cells["c"]["misplaced"] and not cells["a"]["misplaced"],
      f"match {cells['c']['match']}")
al = [a for a in eng.snapshot()["alerts"] if "different product" in a["message"]]
check("...with an alert naming the product and spot", al and "row 2 · col 1" in al[0]["message"], al[0]["message"] if al else "none")
for _ in range(4):
    sw.step(shelf())
check("put back -> flag and alert clear", not {c["slot"]: c for c in eng.shelves["shelfP"]}["c"]["misplaced"]
      and not [a for a in eng.snapshot()["alerts"] if "different product" in a["message"]])

# ── shopper attention per product ─────────────────────────────────────
print("\nattention")
fr = shelf()
bx = next(s for s in slots if s["id"] == "b")
over = [[bx["x"] * 640, bx["y"] * 480 - 60, (bx["x"] + bx["w"]) * 640, 479]]
t0 = time.time()
for k in range(13):                              # 3 s in front of product b
    sw.track_attention(fr, over, t0 + k * 0.25)
for k in range(8):                               # then gone
    sw.track_attention(fr, [], t0 + 3.25 + k * 0.25)
for k in range(4):                               # someone walking past: 0.75 s
    sw.track_attention(fr, over, t0 + 6 + k * 0.25)
for k in range(8):
    sw.track_attention(fr, [], t0 + 7 + k * 0.25)
att = eng.attention_today("shelfP", "b")
check("a 3 s stop is counted, a walk-past isn't", att["visits"] == 1 and 2.5 <= att["avg_s"] <= 3.5, str(att))
last = [json.loads(d) for (d,) in store.q("SELECT data FROM events WHERE type='product_dwell' ORDER BY id")]
last = [d for d in last if d["slot"] == "b"][-1]
check("stop logged for analytics", last["slot"] == "b" and 2.5 <= last["s"] <= 3.5, str(last))

# ── shoppers: an ID at the door, checked again at the till ───────────
print("\nshoppers")
import re


def person(shirt, trousers, box=(250, 60, 390, 440)):
    """A flat picture of someone: shirt colour on top, trousers below (BGR)."""
    img = np.full((480, 640, 3), 140, np.uint8)
    x1, y1, x2, y2 = box
    img[y1:y1 + 190, x1:x2] = shirt
    img[y1 + 190:y2, x1:x2] = trousers
    return img, list(box)


RED, NAVY, BLUE, TAN, GREEN, WHITE = (40, 40, 200), (90, 40, 30), (200, 90, 30), (110, 160, 200), (60, 170, 60), (235, 235, 235)
(imA, bA), (imB, bB), (imC, bC) = person(RED, NAVY), person(BLUE, TAN), person(GREEN, WHITE)
sA, sB, sC = (ss.body_signature(im, b) for im, b in ((imA, bA), (imB, bB), (imC, bC)))
check("signature is numbers only", sA.shape == (3, 15) and sA.dtype == np.float32 and abs(float(sA.sum()) - 3) < 1e-3)
check("same clothes match, different clothes don't", ss.sig_similarity(sA, sA) > 0.99 and ss.sig_similarity(sA, sB) < 0.5,
      f"{ss.sig_similarity(sA, sA):.2f} vs {ss.sig_similarity(sA, sB):.2f}")
check("the head is not part of the signature", np.allclose(
    ss.body_signature(imA, bA), ss.body_signature(np.where(np.arange(480)[:, None, None] < 100, 20, imA).astype(np.uint8), bA)))
check("a tiny or cut-off person gives no signature", ss.body_signature(imA, [10, 10, 20, 30]) is None)

ss.CONFIG["shopper"]["require_id"] = False
ida = eng.on_entry("entry", "in", sig=sA, tid=1)
idb = eng.on_entry("entry", "in", sig=sB, tid=2)
check("everyone who comes in gets a unique ID", re.fullmatch(r"SH-\d{6}-\d{3}", ida or "") and idb and ida != idb, f"{ida} {idb}")
check("a repeat crossing doesn't issue a second ID", eng.on_entry("entry", "in", sig=sA, tid=1) == ida)
sv = eng.shopper_view()
check("both are inside", sv["inside"] == 2 and {x["id"] for x in sv["list"]} == {ida, idb}, str(sv))
check("only numbers are kept: no picture, JSON-safe", json.dumps(sv) and all(
    not isinstance(v, np.ndarray) or v.shape == (3, 15) for sh in eng.shoppers.values() for v in sh.values()))
dim = np.clip(imA.astype(int) * 0.85, 0, 255).astype(np.uint8)          # same person, other camera, dimmer
check("the till finds the same shopper again", eng.identify(ss.body_signature(dim, bA))[0] == ida)
check("a stranger matches nobody", eng.identify(sC)[0] is None)
twin = eng.on_entry("entry", "in", sig=sA, tid=3)
check("two people dressed alike are never guessed between", eng.identify(sA)[0] is None and twin != ida)
eng.on_entry("entry", "out", sig=None, tid=3)                              # the twin goes back out
eng.shoppers.pop(twin, None)

# the till camera: needs the same match twice in a row before it is trusted
PERSONS[:] = [bB]
for _ in range(2):
    cw.step(imB)
check("one sighting at the till isn't enough", eng.till_current() is None, str(eng.till))
for _ in range(6):
    cw.step(imB)
t = eng.till_current()
check("till camera identifies the shopper", t and t["id"] == idb and t["score"] > 0.9, str(t))
check("...and the snapshot shows it", eng.snapshot()["shoppers"]["till"]["id"] == idb)
PERSONS[:] = []

cid = eng.cart_new()
eng.cart_add(cid, "dove-sh")
bill, err = eng.checkout(cid)
check("bill is made for the shopper the camera found", bill["shopper"]["id"] == idb and bill["shopper"]["how"] == "camera"
      and bill["shopper"]["check"] == "ok", str(bill["shopper"]))
check("the shopper is marked billed", eng.shoppers[idb]["status"] == "billed" and eng.shoppers[idb]["bills"] == [bill["id"]])
check("bill remembers who it was for", store.bill_get(bill["id"])["shopper"]["id"] == idb)
check("ID is printed on the bill", cv2.imdecode(np.frombuffer(ss.render_bill(bill)[0], np.uint8), 1) is not None)

eng.till = None
cid = eng.cart_new()
eng.cart_add(cid, "dove-sh")
check("a made-up ID is refused", "No shopper" in (eng.cart_shopper(cid, "SH-000000-999") or ""))
check("staff can pick the shopper", eng.cart_shopper(cid, ida.lower()) is None and eng.cart_view(cid)["shopper"] == ida)
bill, err = eng.checkout(cid)
check("manual pick is recorded as such", bill["shopper"] == {"id": ida, "how": "manual", "check": "ok"}, str(bill["shopper"]))

cid = eng.cart_new()
eng.cart_add(cid, "dove-sh")
n_alerts = len([a for a in eng.snapshot()["alerts"] if a["action"] == "Check the shopper ID"])
bill, err = eng.checkout(cid)
check("no ID identified: bill still prints but is flagged", bill and bill["shopper"] is None and eng.sh_stats["unverified"] == 1)
check("...and staff are told", any(a["action"] == "Check the shopper ID" for a in eng.snapshot()["alerts"]))

ss.CONFIG["shopper"]["require_id"] = True
cid = eng.cart_new()
eng.cart_add(cid, "dove-sh")
bill, err = eng.checkout(cid)
check("with require_id, no verified ID = no bill", bill is None and "Shopper ID not verified" in err, str(err))
check("...cart is kept so staff can pick one", eng.cart_view(cid) is not None)
check("naming an ID at checkout verifies it", eng.checkout(cid, idb)[0] is not None)
ss.CONFIG["shopper"]["require_id"] = False

check("leaving by track gives back the same ID", eng.on_entry("entry", "out", sig=None, tid=1) == ida)
check("a shopper who has left can't be picked", "already left" in (eng.cart_shopper(eng.cart_new(), ida) or ""))
unbilled0 = eng.sh_stats["left_unbilled"]       # the dressed-alike twin walked out above, with no bill
check("a shopper who was billed leaving isn't counted as unbilled", True)
check("leaving by clothing finds the right person", eng.on_entry("entry", "out", sig=ss.body_signature(imB, bB), tid=77) == idb)
check("signature dropped once they leave", eng.shoppers[idb]["sig"] is None and eng.shopper_view()["inside"] == 0)
idc = eng.on_entry("entry", "in", sig=sC, tid=5)
eng.on_entry("entry", "out", sig=None, tid=5)
check("walking out without a bill is counted", eng.sh_stats["left_unbilled"] == unbilled0 + 1 and idc not in {x["id"] for x in eng.shopper_view()["list"]})
idd = eng.on_entry("entry", "in", sig=sC, tid=5)
check("stepping out and straight back in keeps the ID", idd == idc and eng.shoppers[idc]["status"] == "inside")
check("IDs survive in the event log", store.q("SELECT COUNT(*) FROM events WHERE type='shopper_in'")[0][0] >= 3)
eng.shoppers.clear()
eng.sh_stats = {"issued": 0, "left_unbilled": 0, "unverified": 0}
eng.resolve("checkout:unverified")

# ── API ───────────────────────────────────────────────────────────────
print("\napi")
from fastapi.testclient import TestClient

cl = TestClient(ss.make_app(eng, {"shelfP": sw, "till": cw}, store))
check("products list", len(cl.get("/api/products").json()["products"]) == 3)
check("save product via API", cl.post("/api/products", json={"sku": "coke", "name": "Coke 750ml", "mrp": 40}).json()["ok"])
check("bad product reports error", not cl.post("/api/products", json={"name": "no sku"}).json()["ok"])
lab = cl.get("/api/products/coke/label.png")
img = cv2.imdecode(np.frombuffer(lab.content, np.uint8), cv2.IMREAD_COLOR)
check("printed label scans back", store.product_get("coke")["barcode"] in [c for c, _ in cw.decode(img)])
c = cl.post("/api/cart").json()
r = cl.post(f"/api/cart/{c['id']}/scan", json={"code": "coke"}).json()
check("scan via screen", r["ok"] and r["cart"]["n_items"] == 1)
check("scan unknown via screen", not cl.post(f"/api/cart/{c['id']}/scan", json={"code": "nope"}).json()["ok"])
check("set qty via API", cl.post(f"/api/cart/{c['id']}/set", json={"sku": "coke", "qty": 3}).json()["n_items"] == 3)
bj = cl.post(f"/api/cart/{c['id']}/checkout").json()
check("checkout via API", bj["ok"] and bj["bill"]["total"] == 120)
check("bill PDF served", cl.get(f"/api/bills/{bj['bill']['id']}.pdf").content[:4] == b"%PDF")
check("bill PNG served", cl.get(f"/api/bills/{bj['bill']['id']}.png").content[:4] == b"\x89PNG")
check("bill path traversal blocked", cl.get("/api/bills/..%2Fpos.png").status_code == 404)
g = cl.get("/api/slots/shelfP").json()
check("slots API returns rows/cols + product", g["slots"][2]["row"] == 2 and g["slots"][0]["product"]["sku"] == "maggi-70")
ok = cl.post("/api/slots/shelfP", json={"slots": slots[:2]}).json()
check("slots API saves", ok["ok"] and len(lay.cam("shelfP")["slots"]) == 2)
check("reference still served", cl.get("/api/shelf/shelfP/still.jpg").content[:2] == b"\xff\xd8")
check("snapshot carries POS state", "pos" in eng.snapshot())
ids = eng.on_entry("entry", "in", sig=sA, tid=41)
check("shoppers API lists who is inside", [x["id"] for x in cl.get("/api/shoppers").json()["list"]] == [ids])
c = cl.post("/api/cart").json()
cl.post(f"/api/cart/{c['id']}/scan", json={"code": "coke"})
check("shopper API refuses an unknown ID", not cl.post(f"/api/cart/{c['id']}/shopper", json={"shopper": "SH-1"}).json()["ok"])
check("shopper API attaches an ID", cl.post(f"/api/cart/{c['id']}/shopper", json={"shopper": ids}).json()["cart"]["shopper"] == ids)
bj = cl.post(f"/api/cart/{c['id']}/checkout", json={}).json()
check("checkout via API carries the shopper", bj["ok"] and bj["bill"]["shopper"]["id"] == ids)
eng.shoppers.clear()
eng.sh_stats = {"issued": 0, "left_unbilled": 0, "unverified": 0}

# ── integrations ─────────────────────────────────────────────────────
print("\nintegrations")
r = cl.post("/api/integrations/products", content="sku,name,brand,mrp,price\nbhujia-200,Haldiram Bhujia 200g,Haldiram,60,55\n"
            "bad-price,Bad,X,10,20\n", headers={"Content-Type": "text/csv"}).json()
check("catalog CSV import, bad rows reported", r["imported"] == 1 and len(r["errors"]) == 1 and store.product_get("bhujia-200"),
      str(r))
r = cl.post("/api/integrations/products", json=[{"sku": "tea-250", "name": "Tea 250g", "mrp": 120}]).json()
check("catalog JSON import", r["imported"] == 1)
before = eng.pos_units("maggi-70", None)
r = cl.post("/api/integrations/restock", json={"items": [{"sku": "maggi-70", "qty": 10}]}).json()
check("delivery adds to the till stock", r["levels"]["maggi-70"] == (before or 0) + 10, f"{before} -> {r['levels']}")
st = cl.get("/api/integrations/stock").json()
m = next(p for p in st["products"] if p["sku"] == "maggi-70")
check("stock pull: camera count, till count, place", m["till_units"] == (before or 0) + 10 and m["shelves"]
      and "row" in m["shelves"][0]["where"], str(m)[:200])
check("sales pull", any(b["id"] == bj["bill"]["id"] for b in cl.get("/api/integrations/sales?since=0").json()["bills"]))

import http.server, socketserver, threading
GOT = []


class Hook(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        GOT.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.end_headers()


srv = socketserver.TCPServer(("127.0.0.1", 8767), Hook)
threading.Thread(target=srv.serve_forever, daemon=True).start()
ss.CONFIG["webhooks"] = ["http://127.0.0.1:8767/hook", {"url": "http://127.0.0.1:9/down", "events": ["bill"]}]
threading.Thread(target=ss.webhook_sender, args=(eng, store), daemon=True).start()
eng.fire("test:hook", "stock", 3, "webhook test alert", "Refill")
c = cl.post("/api/cart").json()
cl.post(f"/api/cart/{c['id']}/scan", json={"code": "tea-250"})
cl.post(f"/api/cart/{c['id']}/checkout")
deadline = time.time() + 15
while time.time() < deadline and not {"alert", "bill"} <= {g["type"] for g in GOT}:
    time.sleep(0.2)
check("webhooks deliver alert and bill events", {"alert", "bill"} <= {g["type"] for g in GOT},
      str([g["type"] for g in GOT]))
check("event names the store", all(g["store_id"] == ss.CONFIG["store_id"] for g in GOT))
time.sleep(4)
check("unreachable system: its events stay queued for retry", cl.get("/api/integrations").json()["queued"] >= 1)
ss.CONFIG["webhooks"] = []
srv.shutdown()

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
