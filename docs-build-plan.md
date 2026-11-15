# Sanjay: Emergent build plan v3 (scoped proof of concept)

**Status (15 Nov, 23:55):** Steps A and C-prep are DONE. Data downloaded, both models trained and tested on real data (27/27 tests pass), figures rendered. Everything is in `Downloads/sanjay/` (`data/figures/` = 12:00 fallback). Remaining: push to GitHub → Message 1.

**Held-out India results (68 chips, never seen in training or tuning):** ground IoU 0.53, precision 0.74, recall 0.66, 77% of flood chips alerted, 1/6 dry chips false-alarmed, 21 MB, ~1.1 s/chip. Onboard IoU 0.50, precision 0.85, recall 0.55, 51% of flood chips alerted, 0/6 false alarms, 44 KB, ~39 ms/chip. Interval coverage 90% (ground) / 85% (onboard) against a 90% target. Main error source: hill shadow reads as water (no terrain mask yet).

**What changed from v2:** the science is no longer left to Emergent. `sanjay_core.py` (in `sanjay-poc.zip`) already does detection, severity, 90% intervals, packets, crops and the downlink queue, with 25 passing tests. Emergent only wraps it in an API and builds the screens. That cuts the time, the credits and the risk of a wrong algorithm.

**Cut for scope:** relay node, multi-satellite fleet, Satellite B's accept/reject simulation, Gantt chart, and the Monte Carlo and latency comparison. The follow-up request is still generated and sent as a packet.

## Timeline (IST)
| When | What | Who |
|---|---|---|
| Tonight, ~45 min | **Step A:** Colab data prep and training. **Step B:** Assam export | Team |
| Tonight, ~15 min | **Step C:** push the repo to GitHub | Team |
| Tonight → morning | **Message 1** to Emergent, then deploy | Emergent |
| Before 16 Nov 12:00 | Submit the deployed link. If Emergent isn't ready, submit the `render_demo.py` figures instead | Team |
| After mentorship | **Message 2** (polish) | Emergent |

---

## Step A: data and training in Google Colab (free)
1. Go to colab.research.google.com and start a new notebook. Upload `backend/sanjay_core.py` and `prep/prepare_data.py` from the zip using the Files panel.
2. Run these cells:
   ```
   from google.colab import auth; auth.authenticate_user()
   !pip -q install rasterio
   !python prepare_data.py --download
   !python render_demo.py --data sanjay_data      # upload prep/render_demo.py first
   from google.colab import files; files.download("sanjay_data.zip")
   ```
3. The script downloads Sen1Floods11's hand-labelled set: the Sentinel-1 VV/VH chips, the labels and the JRC permanent-water layer. It then:
   - trains the **ground** and **onboard** models on non-India chips;
   - calibrates the 90% interval on held-out chips;
   - tests both models on the **India chips**, a flood neither model has seen;
   - picks one real chip for each test case (see the table at the end).
4. Check the printed `roles:` line. A role with no matching chip is reported, and its test is skipped instead of being faked.
5. `sanjay_data/figures/` now holds submission-ready images: severity maps for onboard vs ground, the alert JSON, the cue JSON, the crops and the downlink order. That is your fallback prototype proof for 12:00.

## Step B: upload real Sentinel-1 data (Assam)
**Route 1, recommended: Google Earth Engine.** It uses the same preprocessing as the training data.
1. Register at code.earthengine.google.com (free for non-commercial use; you'll need a Cloud project, a few minutes to set up).
2. Paste `prep/assam_gee.js` and click Run. The console lists the acquisition dates, and the map shows VV, where dark means water. If the tile looks dry, move `centre` or pick the next date.
3. Open the Tasks tab and run both exports. They land in Google Drive as `assam_silchar_2022_s1.tif` (2 bands: VV, VH) and `assam_silchar_2022_perm.tif`.
4. In the Sanjay app, click **Upload scene**, add the S1 file, optionally add the perm file, and enter the acquisition date. No accuracy is shown for this scene because it has no labels. That's correct (test U6).

**Route 2, no code: Copernicus Browser** (dataspace.copernicus.eu/browser; free account).
1. Search "Silchar". Pick Sentinel-1 GRD, IW mode, VV+VH, on a date between 17 June and 5 July 2022.
2. Zoom to about a 5 km area. Use **Download → Analytical**: TIFF 32-bit float, WGS 84 (EPSG:4326), the highest resolution, and layers **VV and VH**. If it asks for a backscatter coefficient, choose **sigma0**, which matches the training data.
3. Upload the VV and VH files together. Sanjay converts linear values to dB automatically. Without a perm file, permanent rivers will count as flood; the UI says so.

Full Sentinel-1 scenes (the ~1 GB SAFE downloads) are not needed and won't upload. Use small tiles only.

## Step C: give Emergent the repo
Unzip `sanjay-poc.zip`, put `sanjay_data/` inside it as `data/` (keep `data/figures/` out of it), run `pytest -q tests`, then push to a **public** GitHub repo. The models are a few MB, so a normal push works. If `pytest` fails on real data (most likely a false alarm on the dry chip), raise `min_event_km2` in the config and note it in the deck as a known limit.

---

## Message 1 (paste into Emergent)
```
Build "Sanjay — onboard flood triage (proof of concept)" for an Ashoka University deep-tech sprint (challenge: satellite capture to actionable intelligence in minutes, without expensive infrastructure). Thesis: the satellite sends the warning before the picture.

SOURCE: clone https://github.com/<YOU>/sanjay-poc into the project.
- backend/sanjay_core.py is the tested science core. Use sc.analyse(bundle, scene, perm) for detection + severity (it applies the calibrated settings stored in each model bundle). Import it; do NOT rewrite, simplify or "improve" its algorithms. If you think something in it is wrong, tell me instead of changing it.
- data/ holds real Sentinel-1 chips (Sen1Floods11, CC BY 4.0), trained models (joblib), catalog.json (one chip per demo role), metrics.json (accuracy on held-out India chips) and model_cards.json.
- Pin scikit-learn to the version in data/model_cards.json ("sklearn_version") so the models load. Other deps: numpy, scipy, rasterio, pillow, joblib.
STACK: your standard React + Tailwind + FastAPI (Pydantic v2). The database is only for uploaded-scene records. No auth.

RULES
1. Every number on screen comes from sanjay_core output or data/*.json. No hard-coded or placeholder values.
2. Tag panels: REAL DATA (Sentinel-1 pixels, labels) · MODEL OUTPUT · SIMULATED (satellite roles, link, bitrate, contact window). SAT-A and SAT-B are hypothetical satellites.
3. Accuracy appears only for scenes that have a *_label.tif. Uploaded scenes have none, so hide the accuracy UI (do not show "N/A" figures).
4. Every control must do something. No decorative controls.

API (FastAPI, cache results by (scene_id, profile, params))
GET  /api/health
GET  /api/scenes                      -> catalog roles + uploaded scenes
POST /api/scenes/upload               -> multipart: s1 (2-band) OR vv + vh; optional perm; acquired_at. Use sc.load_scene. Return SceneError messages as 422 with the text.
POST /api/analyse {scene_id, profile: onboard|ground}
     -> pred, grid = sc.analyse(bundle, scene, perm)
        sc.build_alert ; sc.build_cue ; sc.make_crops ; sc.evaluate (labels only)
     -> return: cells, flood area + interval, interval_method, alert payload + size_bytes (or null), cue (or null),
        crops (PNG URLs + size_bytes), runtime_ms, peak_mem_mb, out_of_range flag, model card, accuracy|null,
        PNG URLs for the SAR image (sc.render_png(vv)) and flood overlay (sc.render_png(vv, flood_full))
POST /api/compare {scene_id}          -> both profiles + agreement = IoU between the onboard and ground water masks
POST /api/downlink {scene_id, profile, bitrate_mbps, window_start_s, window_end_s, order: priority|raw_first}
     -> items: alert (tier 1, prio 1), cue (tier 1, prio 1, ready 0.001 s), crops (tier 2, prio 2),
        raw scene (tier 3, prio 3, sc.raw_scene_bytes); sc.schedule_downlink
created_at = acquired_at (or "replay") so results stay reproducible.

SCREENS (two, 1440x900, light theme)
Tokens: white #FFFFFF, charcoal #17212B text, grey #64707C secondary, light grey #E8EBEF surfaces, steel blue #315B78 active / Sanjay. Severity scale (from sc.SEVERITY_COLOURS): none #E8EBEF, low #315B78, moderate #B7791F, severe #B42318, no data white with a hatched border. Inter font; 30/22/16/12 px; key figures 34 px tabular. No gradients, no glow, lucide icons.

1) Mission (default)
- Left rail (260 px): scene list with plain names ("Severe flood — India chip", "Moderate flood", "Flat dry land (no flood expected)", "Hilly dry land (hill-shadow test)", "River/sea only (permanent water)", "Partial no-data", uploaded scenes with their date); Upload scene button; model toggle ONBOARD | GROUND (default ONBOARD); "Analyse" button with a loading state.
- Centre: SAR image (REAL DATA) with toggles for the flood mask and the severity grid; a severity legend; hover/click a cell -> flood fraction, 90% interval, area km2, lon/lat, severity. Status banner: "ALERT generated" (red) or "No qualifying flood — nothing sent" (grey).
- Right (360 px), "What SAT-A sends" (SIMULATED link): bitrate slider (Mbps, 0.05–10, default 0.2), contact window start/end (s), order toggle "Sanjay: alert first" | "Image first". Show a horizontal timeline of items in send order with sizes in bytes, then a "Ground inbox" listing items in arrival order: alert card (expand JSON, copy), cue card, crop thumbnails, raw scene last.
- "Compare onboard vs ground" button -> side-by-side overlays plus a table: trees, features, input bits, downsample, model size, runtime, peak memory, India held-out IoU / precision / recall / chip detection rate / chip false-alarm rate (metrics.json), agreement on this scene.

2) Evidence
Model cards; metrics.json for both profiles with the interval coverage and method; data provenance (Sen1Floods11 citation, uploaded-scene source); known limits from the README; full JSON inspectors for the alert and the cue; export the current analysis as JSON.

TESTS
- Run the repo's pytest suite unchanged; it must stay green.
- Add API tests: upload a 1-band file -> 422 mentioning VH; upload a UTM file -> 422 mentioning EPSG:4326; analyse the dry_flat scene -> alert null, cue null; analyse flood_severe -> alert size_bytes equals the serialised length; downlink with the window ending before the alert is ready -> status "queued"; halving the bitrate doubles duration_s; the same request twice gives identical JSON.
- Playwright golden path: open -> pick Severe flood -> Analyse -> severity grid visible -> alert card in the Ground inbox before the raw scene -> switch to Image first -> raw scene arrives first -> Compare -> both overlays and the table visible.

STOP when this works. Reply with a pass/fail table and anything you could not do. Then I'll ask you to deploy.
```

## Message 2 (after mentorship, polish only)
```
Polish pass, no new science. (1) Upload flow: drag-and-drop, file checks before upload, readable errors. (2) Usable at 1280 px wide. (3) Empty, loading and error states on every panel. (4) A short "How Sanjay works" strip on Mission: capture -> onboard model -> severity + interval -> alert / crops / cue first -> raw later. (5) Rerun all tests and the pre-deployment check, then redeploy.
```

---

## Every test case has real data behind it
| Test | Data that makes it plausible |
|---|---|
| Flood detected; severity shows severe/moderate cells | `flood_severe`, `flood_moderate`: India chips with the most labelled flood water |
| No flood → no alert, no cue | `dry_flat` (India_631692) and `dry_hilly` (India_244057): labelled-dry chips picked by terrain roughness of the input, not by model output |
| Permanent water is not flood | `permanent_water_only` (Spain_7604243; no India chip qualified): JRC water, ~0 flood |
| No-data handling | `partial_nodata`: a chip with >5% invalid pixels |
| Accuracy shown only with labels; interval coverage | All India chips (hand labels) vs the Assam upload (no labels) |
| Onboard reduced vs ground | Both models trained on the same chips; size, features, bits, runtime and memory measured |
| Real upload, linear→dB, CRS check | Assam GEE export (dB) or Copernicus Browser export (linear); unit tests cover the bad-file cases |
| Packet size, priority order, no window, bandwidth, resume | Generated packets plus downlink config (unit-tested) |
| Reproducible / replay | Fixed seeds; created_at = acquisition time |
