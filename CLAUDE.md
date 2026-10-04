# CLAUDE.md: StoreSense Edge

SIH 2026 entry for **PS SIH26179** (Qualcomm, hardware category). Team CodeDarbar (ID 151112); owner Rudy (Rudhratej Singh). On-device retail analytics from ordinary cameras, running offline on one edge box. Target hardware is a Qualcomm Dragonwing RB3 Gen 2 (QCS6490), but **the Qualcomm port is not done; today it runs on a laptop CPU.** `HANDOFF.md` is the team doc, `README.md` the public one.

## Hard rules

1. **One file.** All app code, including the dashboard (the `DASHBOARD` string), lives in `storesense.py`. Don't split it into modules or add config files when `CONFIG` or an existing JSON will do.
2. **Never claim a depth camera.** Depth comes from Depth Anything V2 on a normal RGB camera. This applies to comments, the README and slides.
3. **Faces are always blurred** on every view and endpoint (live, CCTV, depth included). Store numbers, never frames.
4. **Report, don't guess.** A shelf column whose back isn't visible is `hidden` (NaN), never an invented count. Docs follow the same rule: demo analytics are generated data, the ERP side is a generic REST + webhook API (no SAP/Tally connectors), and there is no multi-store or chain monitoring, no SMS/push alerts and no trained SKU recogniser.
5. **Tests must pass** before any commit, and every new behaviour gets a test.
6. **Commits:** author `Rudhratej Singh <rudhratejsingh6@gmail.com>`. Ask before pushing; Rudy usually pushes himself.

## Tests

The tests are plain scripts, not pytest. Run them from the repo root:

```bash
for t in logic layout pos depth cams real; do python tests/test_$t.py || break; done
```

`test_real.py` is the slowest. The depth model downloads from Hugging Face on first use.

## Depth counting

This is the main differentiator, so change it carefully. These approaches were tried and failed; don't revive them:
- A per-box depth median: parallax broke it.
- A single calibration pass: it caused a one-unit bias (calibration uses the median of several passes).
- Guessing hidden columns from the smear.

## Gotchas

- IP Webcam lag came from OpenCV buffering. `Capture._run_http` exists to always serve the newest frame; don't swap HTTP sources back to `cv2.VideoCapture`.
- The dashboard polls `/api/frame/{cam}.jpg` on purpose instead of using MJPEG `<img>` streams, because browsers allow only 6 connections per host.
- The dashboard's fps readout is the camera's real frame rate, not the processing rate.

## Deliverables outside the repo

- The PPT brief is a claude.ai Docs artifact, "StoreSense Edge — PPT brief (SIH26179)", with an "All features" tab.
- The deck is `StoreSense_Edge_SIH26179.pptx`, built from the team's Canva export.
- The SIH idea template allows 6 slides and is submitted as a PDF.
