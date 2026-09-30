# StoreSense Edge: team handoff

Everything we have built for **SIH 2026, PS SIH26179** (AI-powered retail intelligence on edge devices, set by Qualcomm). It covers what the product does, how to run it, how the code is laid out, what is tested and what isn't, and what's left to do.

Repo: `github.com/deepgautamkapadia-ste/SIH_Store_Monitoring` · one Python file: `storesense.py` (~4,100 lines)

---

## 1. The pitch in one paragraph

Ordinary cameras (old Android phones, CCTV, webcams) plus one edge box in the store give a kirana or supermarket four things: footfall and floor heatmaps, **per-product shelf stock** (including units hidden behind the front row), queue monitoring, and a self-checkout/billing counter. Everything runs offline on the device, and faces are blurred on every view. The target hardware is Qualcomm: Dragonwing RB3 Gen 2 (QCS6490) or a Snapdragon phone, with models on the Hexagon NPU.

### How it maps to the problem statement

| PS asks for | What we built |
|---|---|
| Shopper analytics | Entry/exit counting, live occupancy, hourly footfall, floor heatmap in store coordinates, zone dwell time, conversion (footfall → bills) |
| Inventory visibility | Product boxes drawn on the shelf picture, live count per product, **depth counting behind the front row**, alerts with the exact spot ("Aisle A, row 1 · col 1"), time-to-empty, and a till-based exact count next to the camera count |
| Queue management | Queue length, time in queue, +5/10/15 min forecast, "open another counter" recommendation |
| Edge / on-device | Everything local (SQLite, FastAPI dashboard), offline sync outbox, telemetry off, Qualcomm export path |
| (extra) | Self-checkout with barcode scanning and PNG/PDF bills, CCTV tab, analytics tab with CSV exports for forecasting |

### USPs (the ones we can defend)

1. **Counts per product, not per grid cell.** The storekeeper draws a box around each product and links it to its SKU. Alerts name the product and where it sits.
2. **Counts units behind the front row with one ordinary camera, at any angle.** Uses a monocular depth model; see §5. No depth camera.
3. **Two counts that check each other.** The camera count (what's on the shelf) and the till count (restocked − sold). A gap between them points to misplaced stock, theft or an unlogged restock.
4. **Fits any store layout.** Drag shelves and cameras on a plan. It computes which shelf face each camera covers and shows blind spots before anything is mounted.
5. **Occlusion-aware.** A shopper standing in front of a product doesn't trigger a false "empty"; the last reading holds.
6. **Uses hardware the shop already has.** Phones as cameras, a basic USB barcode scanner or the checkout camera, and in-store EAN-13 labels for loose items.
7. **Private by design.** Heads are blurred on every view, including CCTV. Only anonymous IDs and numbers are stored; no frames, no faces.

> **Honesty rule for the PPT and demo:** we do **not** use a depth camera and must never say we do. Depth comes from a monocular model on a normal camera. Say "tested on a simulated shelf" until we've run it on a real one (see §8).

---

## 2. Run it

```bash
pip install -r requirements.txt           # ultralytics, fastapi, uvicorn[standard], opencv, numpy, transformers
python storesense.py --config demo/demo_config.json --demo-history 14
# open http://localhost:8000
```

- The demo runs looping videos for an entry camera, a shelf camera with 9 products marked on it, and a self-checkout camera. Nothing needs to be plugged in.
- In **Shelves**, press **Calibrate** once. The shelf video then alternates between full and partly emptied, so you can watch products go LOW.
- `--demo-history 14` fills Analytics with 14 days of **generated** trading. It's tagged in the database, and the UI shows a banner while it's present. Remove it with `--clear-demo`.
- YOLO weights (~5 MB) download on the first run. The depth model downloads the first time a product box has a unit size set.

### Real cameras (phones)

1. Install **IP Webcam** (Android) and press *Start server*. It shows an address like `192.168.1.24:8080`.
2. Put the laptop and phones on the **same phone hotspot**. Campus Wi-Fi (TIET's 172.16.x.x) usually blocks devices from reaching each other.
3. Check the address opens in the laptop's browser first.
4. Dashboard → **Setup** → select or add a camera → type the address (e.g. `192.168.1.24:8080`; `http://` and `/video` are added automatically) → choose its job → **Save plan**. It connects immediately, and the panel shows **● live** or why it isn't connecting.
5. In the IP Webcam app, set 1280×720 and about 50% quality for less load.

Command-line alternative: `python storesense.py --cam entry entry,queue 192.168.1.23:8080 --cam shelfA shelf 192.168.1.24:8080 --cam till checkout 192.168.1.25:8080`

---

## 3. The dashboard (six tabs)

| Tab | What's there |
|---|---|
| **Home** | Live KPIs (inside now, entries, sales today, conversion, queue wait), prioritised alerts with actions, products running low, floor heatmap, footfall by hour |
| **Shelves** | Pick a shelf camera → **Calibrate** (shelf full, aisle clear) → **Draw product box** → choose or create the product (SKU, barcode, brand, MRP, price) → set facings (side by side), units deep and **unit size in cm** (turns on depth counting) → **Save products**. **Still / Live / Depth** picture views. Live stock table (camera count, till count, status, time to empty). Product catalog with printable barcode labels |
| **Checkout** | Cart built from the checkout camera, a USB scanner (types into the box), or typing a SKU or name. Quantities, savings vs MRP. **Make bill** → PNG and PDF: "please move to the payment counter" |
| **CCTV** | Every camera as a plain security view (people boxes, count, timestamp, blurred heads) or with analytics overlays |
| **Analytics** | Footfall and bills per day, conversion, weekday × hour heatmap, revenue by hour, top products, stock-outs, basket sizes, queue waits. Every chart has a table view. CSV downloads |
| **Setup** | Store plan in metres: drag shelves and cameras, set lens angle and range, see coverage and blind spots, 3D view. Per camera: source, job, live status, and click-to-place the entry line, queue area and 4 floor points |

Theme: black / red / grey (Snapdragon red; Qualcomm's corporate colour is blue).

---

## 4. Code map (`storesense.py`)

| Section (search for the banner) | What it does |
|---|---|
| `CONFIG` | Every setting. A JSON passed with `--config` is merged over it (see `demo/demo_config.json`) |
| `STORE LAYOUT` | Plan model (shelves, faces, cameras, product slots), coverage solver (FOV, range, line of sight, occlusion by other shelves), slot row/col |
| `CAPTURE` | `Capture`: latest-frame reader. Phones: our own MJPEG parser that keeps only the newest JPEG (no lag build-up). Files loop; streams reconnect with a plain-words status. `normalise_source` fixes phone URLs |
| `BARCODES & BILLS` | EAN-13 check digit, in-store codes (GS1 prefix 20–29 from the SKU), label PNGs, bill rendering to PNG + PDF |
| `STORAGE` | SQLite: events, metrics, products, bills, cloud outbox. `demo` column tags generated rows |
| `DECISION ENGINE` | `Engine`: alerts (priority, cooldown, auto-resolve, ack time), carts and checkout, stock levels (till count), live snapshot for the dashboard |
| `CAMERA WORKERS` | `PeopleWorker` (YOLO11n + ByteTrack → entry line, heatmap, zones, queue), `ShelfWorker` (edge comparison per facing vs calibration, occlusion, depth pass), `CheckoutWorker` (multi-scale barcode decoding with debounce) |
| `MONOCULAR DEPTH` | Depth model loader, alignment, back-projection, shelf plane and column tubes, nearest-surface picking (§5) |
| `BACKGROUND JOBS` | Per-minute metrics and CSV log, cloud sync |
| `ANALYTICS` | Aggregations for the Analytics tab, hourly training table, demo-history generator |
| `3D STORE VIEW` | Matplotlib render of the plan |
| `API + DASHBOARD` | FastAPI routes; the whole dashboard is one HTML/JS string (`DASHBOARD`) with hand-rolled SVG charts |
| `SETUP TOOL + MAIN` | CLI, `--depth-test`, camera start/stop (`start_cam`, `sync_cams`) |

Other files: `demo/` (videos, demo layout and config), `docs/` (screenshots), `tests/` (§7), `README.md` (user-facing docs and API table).

---

## 5. How depth counting works (the main technical story)

The problem: one camera sees only the front unit of each column. Take that unit and the next one is still there, just further back, so the shelf still looks full.

1. **Calibrate** with the shelf full. The depth model (Depth Anything V2, metric indoor, small) runs on the full-shelf picture; the median of 3 passes becomes the reference.
2. The reference is back-projected to 3D points using the camera's lens angle from the plan. A plane is fitted through the product fronts, and each column's front face becomes a patch on it: the mouth of that column's **tube**.
3. Every few seconds a new depth pass is **re-anchored** to the reference, using a trimmed least-squares fit on the parts that shouldn't change (shelf frame, walls). A monocular model's scale drifts a few percent between frames, which at 1.5 m is a whole unit, so this step is essential.
4. For each column: find the **nearest dense surface** inside its tube, ignoring the smeared "flying pixels" at the rim of a gap. Distance behind the full front ÷ unit size = units gone. Median of the last 3 passes.
5. **Hidden columns:** at an angle, the neighbours can hide a deep gap. Then the column is reported hidden: the front unit is known to be gone, and the rest isn't guessed. On the Depth view these show as "?".

Why 3D and not just "how far back is the middle of the box": parallax. A unit that's pushed back slides sideways in the picture into the neighbouring column's area. The first version got this wrong, and the 3D-tube version fixed it.

Check on a real shelf: `python storesense.py --depth-test full.jpg taken.jpg` (two photos from the same spot). It prints the fit error and how far the changed area moved, and writes `depth_test.png`.

On Qualcomm hardware the same model family is on Qualcomm AI Hub (aihub.qualcomm.com/models/depth_anything_v2).

---

## 6. Data for the forecasting model later

- **Downloads** (Analytics tab, or `/api/export/{name}.csv`):
  - `timeseries_hourly.csv`: one row per hour with entries, exits, bills, revenue, items, occupancy, queue wait, stock alerts, weekday and a `demo` flag. This is the training table for a recurrent/sequence model.
  - `timeseries_minute.csv`: a finer version of the same.
  - `bills.csv`
  - `products.csv`
- A per-minute log is also appended to `analytics/timeseries.csv` while the app runs.
- **Filter `demo == 0`** so the generated history never ends up in a model.

---

## 7. Tests

```bash
python tests/test_logic.py    # counting, queue, shelf grid, alerts, API, offline sync
python tests/test_layout.py   # plan geometry, coverage, store-frame mapping, 3D render
python tests/test_pos.py      # catalog, carts, bills, checkout camera, product boxes, stock counts
python tests/test_depth.py    # depth counting on a ray-traced shelf from 3 camera angles
python tests/test_cams.py     # cameras start/stop on save, phone URL fixing, stream lag, frame endpoint
python tests/test_real.py     # real YOLO on a generated walk-through video
```

All six pass as of the last commit.

- `tests/shelf3d.py` is a small ray-tracer for a shelf with known true depth.
- `test_depth.py` swaps the depth model for true depth plus scale drift, blur, low-frequency error and noise.
- `test_cams.py` runs a fake IP Webcam server at 30 fps. It checks that a slow reader stays within 3 frames (0.1 s) of the phone.

---

## 8. What's proven and what isn't (say this straight to judges)

**Working and tested:**
- Everything in §3, on generated video and with the real person detector.
- Barcode reading on generated labels under blur, rotation and noise.
- Depth counting on the simulated shelf: every visible column counted exactly from straight on, 18° to the side and 14° from above.
- Live phone-feed handling against a simulated phone.

**Not yet validated in a real store:**
- The **real depth model on a real shelf**. It has never been run from this workspace: model downloads are blocked here, so the first real run happens on a teammate's laptop. The likely weak spots are thin products (< ~3 cm deep) and shiny or transparent packs.
- Product-box detection under real shelf lighting, floor mapping on a real floor, queue forecasts against real queues, and barcode reading on real packaging at a real counter.

**Known limits:**
- If staff pull stock forward, the camera sees a full front row again; the till count catches that.
- The till count is exact only for products sold through our checkout.
- Deep gaps can be hidden from an angled camera; these are reported, not guessed.
- Qualcomm hardware porting (QNN export, NPU benchmarks) is **not done**. The app runs on a laptop today.

---

## 9. Next steps (suggested owners in brackets, fill in)

1. **[ ]** Run `--depth-test` on a real shelf with 2–3 product types; record the results and a screenshot for the PPT.
2. **[ ]** Full rehearsal with 3 phones on a hotspot: entry, shelf and till.
3. **[ ]** Measure the laptop's CPU/fps with 3 cameras running, to put real numbers in the PPT.
4. **[ ]** Qualcomm port: export YOLO11n with `model.export(format="qnn")` and the depth model through AI Hub, then benchmark on RB3 Gen 2 or a Snapdragon phone once we have hardware.
5. **[ ]** Train the footfall/sales forecaster on `timeseries_hourly.csv` (real rows only).
6. Roadmap ideas: two cameras cross-checking one shelf face, a trained SKU model for planogram checks (SKU-110K fine-tune), a multi-store HQ view.

---

## 10. Working on the repo

- Unzip a new build over the repo folder, then `git add -A && git commit -m "…" && git push`.
- Don't commit `storesense.db`, `layout.json`, `shelf_ref_*.png`, `bills/` or `analytics/`. They're per-store runtime files and already in `.gitignore`.
- After changing anything, run the six tests in §7.
- Settings go in `CONFIG` (top of the file) or in a JSON passed with `--config`; don't add new config files.

Commit history (newest first):

| Commit | What it added |
|---|---|
| `9cc153f` | Fix laggy and missing phone feeds |
| `52d35b1` | Cameras connect when the plan is saved; phone URLs fixed; feed status in Setup |
| `3b5cf66` | Depth counting behind the front row |
| `3f0f566` | Products, checkout, CCTV and analytics: the six-tab product |
| `c7c0965` | Dark Snapdragon-red UI, tabs, on-picture camera setup |
| `e45c476` | Drag-and-drop store plan, coverage, 3D view |
| `d173f09` | Retargeted to Qualcomm edge hardware |
| `bd67ef3` | First version |
