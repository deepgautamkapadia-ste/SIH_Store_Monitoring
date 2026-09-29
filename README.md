# StoreSense Edge

On-device retail intelligence for Indian stores. Ordinary cameras (old phones, CCTV, webcams) watch footfall, shelf stock, queues and the checkout — all on one edge box in the store, with no cloud dependency and no stored personal data.

Built for **Smart India Hackathon 2026 — PS SIH26179** (AI-powered retail intelligence with edge AI).

![Analytics](docs/analytics.png)

## What's in it

The dashboard has six tabs:

| Tab | What you do there |
|---|---|
| **Home** | Live KPIs (inside now, entries, sales today, conversion, queue wait), alerts that say what to do, products running low, floor heatmap, footfall by hour |
| **Shelves** | Draw a box around each product on the shelf picture, attach the product (SKU, barcode, brand, MRP, price), say how many sit side by side and how deep they stack. Live per-product stock table. Product catalog with printable barcode labels |
| **Checkout** | Self-checkout / billing counter: scan with a USB barcode scanner, a checkout camera, or type a SKU or name. Cart with quantities, savings vs MRP, and a bill as PNG and PDF — "please move to the payment counter" |
| **CCTV** | Every camera as a plain security view with people boxes and counts (faces blurred here too), or with analytics overlays |
| **Analytics** | Footfall and bills per day, conversion, a weekday × hour busyness heatmap, footfall and revenue by hour, top products, stock-outs, basket sizes, queue waits — plus CSV exports for forecasting models |
| **Setup** | Top-down store plan: drag shelves and cameras, see which shelf faces each camera covers and where the blind spots are; set a camera's entry line, queue area and floor points by clicking on its picture; orbitable 3D view |

## Why it's different

- **Products, not grid cells.** You mark each product block on the shelf picture — a box can be as narrow as a single item, so taking one out registers. Each box is split into its facings (units side by side), and each knows its row and column on the shelf, so an alert reads *"Maggi Masala Noodles 70g — Aisle A, row 1 · col 1 low — about 12 of 24 left."*
- **Honest stock counts.** One ordinary camera sees the front row of a shelf, not what's behind it. So the camera count is an **estimate** — facings still visible × how deep the storekeeper says they stack — and it's labelled as one. Where the till is used, stock is also tracked exactly: calibrating sets the level, each sale decrements it. Both numbers are shown side by side.
- **Model your store, don't match a template.** Shelves and cameras go anywhere on the plan. Which shelf face a camera watches is computed from its position, lens angle, range and line of sight, so blind spots show up before you mount anything.
- **Occlusion-aware.** When a shopper stands in front of a product, that box keeps its last reading instead of raising a false "empty".
- **Checkout that works with what a shop has.** A basic USB barcode scanner, a phone pointed at the counter, or typing. Products without a barcode get an in-store EAN-13 from the GS1 20–29 range reserved for exactly this, with a printable label.
- **Runs offline, private by design.** Detection, storage, billing and the dashboard all run on the edge box. Only anonymous track IDs and numbers are stored — no frames, no faces. Heads are blurred on every view, including CCTV, and model telemetry is disabled.

## Quick start

```bash
pip install -r requirements.txt
python storesense.py              # laptop webcam as entry + queue camera
```

Open http://localhost:8000. The YOLO weights (~5 MB) download on the first run.

### Full demo, no cameras needed

```bash
python storesense.py --config demo/demo_config.json --demo-history 14
```

This runs the demo store plan (`demo/demo_layout.json`) with looping videos for the entry camera, a shelf camera with nine products marked on it, and a self-checkout camera that holds up three barcoded products. Open **Shelves** and press **Calibrate** once — the shelf video then alternates between full and two partly-emptied products, so you can watch them go LOW and recover.

`--demo-history 14` fills the Analytics tab with 14 days of **generated** past trading so the charts have something to show. Every generated row is tagged in the database, the Analytics tab shows a banner while any are present, and real trading is added on top. Remove them with:

```bash
python storesense.py --config demo/demo_config.json --clear-demo
```

### Real cameras (phones)

Install **IP Webcam** (Android) on the phones, start its server, and put the laptop on the same Wi-Fi or hotspot. Then either pass cameras on the command line:

```bash
python storesense.py \
  --cam entry  entry,queue  http://192.168.1.23:8080/video \
  --cam shelfA shelf        http://192.168.1.24:8080/video \
  --cam till   checkout     http://192.168.1.25:8080/video
```

or add them in **Setup** with their `source` and just run `python storesense.py`. A source can be a webcam index, an HTTP/RTSP stream or a video file. Camera roles: `entry`, `queue` (these two can share a camera), `shelf`, `checkout`.

## Setting up a store

1. **Plan** (Setup tab): set the store size, drag in shelves and cameras, tick which shelf faces to monitor, give each camera its source, **Save plan**. Restart to start new cameras.
2. **Point the cameras** (Setup tab, select a camera): click on its picture to place the **entry line** (flip the in/out direction with one button), the **queue area**, and four **floor points** that tie it into the store heatmap.
3. **Calibrate shelves** (Shelves tab): fill the shelf, clear the aisle, press **Calibrate**.
4. **Mark products** (Shelves tab): **Draw product box** around each product block, choose the product or create it, set facings and depth, **Save products**. Editing boxes later needs no re-calibration.
5. **Products and labels** (Shelves tab, catalog): add products, print barcode labels for anything without one.

All settings live in `CONFIG` at the top of `storesense.py`; a JSON file passed with `--config` is merged over it.

## Data for forecasting

The Analytics tab links these downloads (also at `/api/export/{name}.csv`):

| File | Rows | Use |
|---|---|---|
| `timeseries_hourly.csv` | one per hour: entries, exits, bills, revenue, items, average occupancy, average queue wait, stock alerts, weekday, `demo` flag | training table for a recurrent/sequence model of footfall and sales |
| `timeseries_minute.csv` | one per minute while running (also written to `analytics/timeseries.csv`) | fine-grained live log |
| `bills.csv` | one per bill | basket and revenue analysis |
| `products.csv` | the catalog | |

Filter on the `demo` column to keep generated history out of a model.

## Architecture

```
store plan (metres) ─► coverage solver: which camera sees which shelf face
        │
cameras ─► capture threads (latest frame only)
        ├─ PeopleWorker   : YOLO + ByteTrack ─► entry/exit line, store-frame heatmap, queue
        ├─ ShelfWorker    : person mask + per-facing edge compare vs calibrated picture ─► product boxes
        └─ CheckoutWorker : multi-scale barcode decoding ─► open cart
                  │
                  ▼
   Engine: priority alerts (dedupe, auto-resolve) · carts & bills · stock levels
                  │
   SQLite (events, metrics, products, bills, outbox) · bills/ (PNG+PDF) · analytics/ (CSV)
                  │
   FastAPI ─► six-tab dashboard over WebSocket, MJPEG feeds, REST, CSV exports
```

One Python file. It runs on any laptop for development. The deployment target is Qualcomm edge hardware: the Dragonwing RB3 Gen 2 (QCS6490) Vision Kit, or a Snapdragon phone for small stores. The detector is exported with `model.export(format="qnn")` to run on the Hexagon NPU.

## Tests

```bash
python tests/test_logic.py    # counting, queue, shelf grid, alerts, API, offline sync
python tests/test_layout.py   # store plan geometry, coverage, store-frame mapping, 3D render
python tests/test_pos.py      # catalog, carts, bills, checkout camera, product boxes, stock counts
python tests/test_real.py     # real YOLO on a generated walk-through video
```

## Status

**Working:** everything above, tested on generated video and with the real detector; barcode reading tested on generated labels under blur, rotation and noise.

**Not yet validated in a real store:** product-box detection under real shelf lighting, the floor mapping on a real floor, queue forecasts against real queues, and barcode reading on real packaging at a real counter.

**Limits to be upfront about:** the camera stock count is an estimate (it cannot see behind the front row); the till count is exact only for products sold through this checkout.

**Roadmap:** cross-checking a shelf face seen by two cameras; a trained SKU model for planogram checks (SKU-110K fine-tune); a multi-store HQ view; porting to Qualcomm hardware and benchmarking on RB3 Gen 2 / Snapdragon.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/api/state` · WS `/ws` | Full live state (pushed once a second) |
| GET/POST | `/api/layout` | Store plan, with per-face coverage |
| GET/POST | `/api/slots/{cam}` | Product boxes on a shelf camera |
| GET | `/api/shelf/{cam}/still.jpg` | Calibrated shelf picture to mark products on |
| POST | `/api/calibrate/{cam}` | Capture the shelf reference |
| GET/POST/DELETE | `/api/products` · `/api/products/{sku}` | Product catalog |
| GET | `/api/products/{sku}/label.png` | Printable EAN-13 label |
| POST | `/api/cart` · `/api/cart/{id}/scan` · `/set` · `/checkout` | Carts and billing |
| GET | `/api/bills/{id}.png` · `.pdf` | Bill |
| GET | `/api/analytics?days=` | Everything the Analytics tab draws |
| GET | `/api/export/{name}.csv` | CSV exports |
| GET | `/video/{cam}?view=cctv` | Live feed (plain CCTV or analytics overlay) |
| POST | `/api/integrations/pos` | Record a sale from an external POS |
| GET | `/api/report?period=day\|week` | Summary report |
| POST | `/api/alerts/{id}/ack` · `/api/counters?open=N` | Staff actions |
