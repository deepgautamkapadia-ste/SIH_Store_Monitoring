# CLAUDE.md: StoreSense Edge

Context for Claude Code when it works in this repo. Read this first. `HANDOFF.md` is the team-facing doc (pitch, features, how to run); `README.md` is the public one.

## What this is

This repo is an SIH 2026 entry for **PS SIH26179** (set by Qualcomm, hardware category).
- **Team:** CodeDarbar, Team ID 151112. The owner is Rudy (Rudhratej Singh).
- **Product:** an on-device retail intelligence system. Ordinary cameras (old Android phones over IP Webcam, CCTV, webcams) feed one edge box. The box produces:
  - footfall and heatmaps
  - per-product shelf stock, including units behind the front row via a monocular depth model
  - queue length, a forecast and counter advice
  - self-checkout billing
  - alerts
  - a 6-tab web dashboard

It all runs offline, and faces are blurred on every view. The target hardware is a Qualcomm Dragonwing RB3 Gen 2 (QCS6490). **The Qualcomm port is not done yet; today it runs on a laptop CPU.**

## Hard rules (don't break these)

1. **One file.** All app code lives in `storesense.py`, including the dashboard HTML/CSS/JS, which is the `DASHBOARD` string. Don't split it into modules, and don't add config files when `CONFIG` or an existing JSON will do. Rudy wants a single runnable file.
2. **Never claim a depth camera.** Depth comes from Depth Anything V2 on a normal RGB camera. This applies to code comments, the README and slides.
3. **Faces are always blurred**, on every view and endpoint, including the live, CCTV and depth views. Store only numbers, never frames.
4. **Report, don't guess.** When a shelf column's back isn't visible, report it as `hidden` (NaN) and never invent a count. The same honesty applies to docs: demo analytics are generated data, the ERP side is a generic REST and webhook API, and there's no multi-store or chain monitoring.
5. **Tests must pass** before any commit. Add a test for every new behaviour.
6. **Commits:** author `Rudhratej Singh <rudhratejsingh6@gmail.com>`. Ask before pushing; Rudy usually pushes himself.

## Run

```bash
pip install -r requirements.txt
python storesense.py                                                   # webcam as entry + queue camera
python storesense.py --config demo/demo_config.json --demo-history 14   # full demo with demo videos + 14 days of fake analytics
python storesense.py --config demo/demo_config.json --clear-demo        # wipe generated demo data
python storesense.py --cam shelfA shelf http://192.168.1.5:8080/video   # add a camera: NAME ROLES SOURCE (repeatable)
python storesense.py --depth-test full.jpg taken.jpg                    # check depth counting on two photos
```

The dashboard is at `http://localhost:8000`. Other flags are `--layout`, `--sku-model`, `--cloud-url`, `--pick` and `--port`.

Phone sources are normalised automatically: `192.168.1.5:8080` becomes `http://192.168.1.5:8080/video`.

## Tests

The tests are plain scripts, not pytest. Run each one from the repo root:

```bash
for t in logic layout pos depth cams real; do python tests/test_$t.py || break; done
```

| Suite | Covers |
|---|---|
| `test_logic.py` | counting, queue, shelf grid, alerts, API, offline sync |
| `test_layout.py` | store plan geometry, coverage, fixtures (doors, counters), 3D render |
| `test_pos.py` | catalog, carts, bills, product boxes, planogram (wrong product), dwell per product, integrations API and webhooks |
| `test_depth.py` | depth counting on a ray-traced shelf (`tests/shelf3d.py`) from 3 camera angles |
| `test_cams.py` | camera start/stop on plan save, URL fixing, phone-stream lag (fake MJPEG server), live view, alerts |
| `test_real.py` | real YOLO on a generated walk-through video (slowest) |

There are about 260 checks in total, and all of them passed at commit `73185c1`. `test_cams.py` and `test_pos.py` swap in a `FakeYOLO`, so they don't need weights. `yolo11n.pt` downloads automatically, and the depth model downloads from Hugging Face on first use.

## Code map: `storesense.py` (~4,700 lines)

Line numbers are approximate; use grep.

| ~Line | What |
|---|---|
| 49 | `CONFIG`: all tunables. The shelf depth block is `{enabled, model, every_s 6, smooth 3, hfov_deg 65, calib_passes 3}`, plus `detect_s 0.25`, `period_s 3.0`, `max_width 1280` and `webhooks` |
| 150–270 | Store-plan geometry: `DEFAULT_LAYOUT`, `FIXTURE_KINDS` (door, counter, fixture), `blockers()` (everything except doors blocks camera sight lines), `face_visibility`, `coverage` |
| 270 | `Layout`: loads, normalises and saves the store plan JSON |
| 380–545 | `normalise_source`, `Capture`: keeps only the newest frame. `_run_http` is a hand-rolled MJPEG reader with boundary parsing, lazy decode and a `seq` counter. `why()` gives a human status text |
| 545–675 | EAN-13 generation, barcode images, `render_bill` (PNG/PDF) |
| 675 | `Store`: the SQLite layer, including the webhook queue (`hook_add`, `hook_pending`, `hook_done`) |
| 865 | `Engine`: the brain. It holds state, fires and resolves alerts, takes `on_shelf` and `on_product_dwell` events, tracks `attention_today`, runs `emit` (webhooks) and exposes `snapshot()` |
| 1264 | `CamWorker` base thread: `blur()` and `live_jpg(view)`. The live view lays the last analysis overlay over the newest raw frame and falls back if detection is more than 2 s stale |
| 1395 | `PeopleWorker`: YOLO11n + ByteTrack, entry line, zones, queue |
| 1602–1780 | Depth helpers: `depth_model`, `run_depth`, `align_depth`, `backproject`, `shelf_frame`, `nearest_surface`, `hs_hist`, `count_badge` |
| 1781 | `ShelfWorker`: product boxes. `read_slots` sets the method (`depth`, `mixed` or `front`), `columns`, `hidden`, `misplaced` and `attention`. Also `depth_pass`, `depth_picture` and `track_attention` |
| 2248 | `CheckoutWorker`: barcode reading into a cart |
| 2300–2660 | Background threads (`ticker`, `cloud_sync`, `webhook_sender`), `analytics()`, CSV export, `seed_demo` |
| 2692 | `render_3d`: a 3D picture of the store plan |
| 2780–4097 | `DASHBOARD`: the entire front end as one HTML string. It has six tabs: Home, Shelves (Still, Live and Depth views), CCTV, Checkout, Analytics (live strip) and Setup (plan editor plus fixtures and integrations). Frames come from polling `/api/frame/{cam}.jpg` paced by `liveTick`. It deliberately avoids MJPEG `<img>` streams because browsers allow only 6 connections per host |
| 4124–4186 | `check_roles`, `start_cam`, `sync_cams`: cameras hot-start and stop when the plan is saved |
| 4187 | `make_app`: FastAPI routes. These include `/api/frame/{cam}.jpg`, `/api/depth/{cam}.jpg`, `/api/live`, `/api/layout`, the `/api/integrations/{stock,sales,products,restock}` endpoints and the WebSocket |
| 4584 / 4630 | `depth_test` CLI; `main()` with argparse |

## How depth counting works

This is the main differentiator, so be careful when changing it.

1. **Calibration.** With the shelf full, take 3 depth passes and use their median. Back-project to 3D using the camera's hfov, fit the plane of the product fronts (`shelf_frame`) and build one "tube" per facing column.
2. **Each pass.** `align_depth` re-anchors the new depth map to the calibration with a trimmed least-squares affine fit on static regions. This cancels the monocular model's scale and shift drift.
3. **Per column.** `nearest_surface` takes the densest near bin, ignoring flying pixels at edges. The result is distance behind the front plane ÷ the unit's depth (the slot's `unit_cm`), which gives the number of units missing.
4. Columns whose back isn't visible are reported as hidden, capped at deep−1. The dashboard draws badges such as `4/5` on each box.

Approaches that failed and shouldn't be revived:
- A per-box depth median: parallax broke it.
- A single calibration pass: it caused a one-unit bias.
- Guessing hidden columns from the smear.

## Feature status vs the PS

**Done:** entry/exit counting, footfall by hour/day/zone, heatmap, zone dwell, **dwell per product**, per-product stock, depth behind the front row, **wrong-product (planogram) check** using an HS-histogram fingerprint per facing, prioritised alerts, queue length, forecast, counter advice, learned service time, offline operation with a sync outbox, blurring, KPIs and reports, store plan with **doors, counters and fixtures**, and a **REST + webhook integration API** (queued while offline).

**Not done / future:**
- Qualcomm NPU port. The plan is a YOLO QNN export plus Depth Anything V2 from Qualcomm AI Hub.
- Multi-store HQ dashboard and chains.
- Named ERP connectors (SAP, Tally).
- SMS or push alerts (alerts appear on the dashboard only).
- A trained SKU recogniser.
- A real-store trial. The team planned to record a demo at the college grocery store.

## Known gotchas

- Run the server from the repo root. Tests `chdir` into `tests/` and write their scratch files there; these are gitignored.
- `*.db`, `layout.json`, `*.pt`, `bills/` and `shelf_ref_*.png` are runtime state and gitignored. Never commit them.
- IP Webcam lag was caused by OpenCV buffering. `_run_http` exists to always serve the newest frame, so don't swap it back to `cv2.VideoCapture` for HTTP sources.
- The dashboard's fps readout is the camera's real frame rate, not the processing rate. A test checks this.
- Old grid-mode shelf alerts are cleared once product boxes report for that camera.

## Other deliverables (not in the repo)

- The PPT and its per-slide brief live in a claude.ai Docs artifact, "StoreSense Edge — PPT brief (SIH26179)", which has an "All features" tab.
- The deck itself is `StoreSense_Edge_SIH26179.pptx`, built from the team's Canva export.
- The SIH idea template allows 6 slides and must be submitted as a PDF.
