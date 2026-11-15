# Sanjay: onboard flood triage (proof of concept)

Sanjay is a satellite that sends the warning before the picture. It reads a Sentinel-1 radar scene and grades flood severity on a colour scale, with a 90% interval for each grid cell. It then sends a few-KB alert, image crops of the worst cells and a follow-up request ahead of the raw scene. A reduced **onboard** model is compared with a full **ground** model.

| Folder | What it is |
|---|---|
| `backend/sanjay_core.py` | Science core: features, models, severity, intervals, packets, crops, downlink queue. Tested. |
| `tests/` | `test_core.py` (synthetic fixtures, always runs), `test_real_data.py` (runs once `data/` exists) |
| `prep/prepare_data.py` | Colab: download Sen1Floods11 → train both models → calibrate intervals → test on India chips → pick demo scenes |
| `prep/render_demo.py` | Makes submission figures from real data without the web app |
| `prep/assam_gee.js` | Earth Engine export of a real Assam (Silchar, June 2022) tile to upload |
| `data/` | Output of `prepare_data.py` (unzip `sanjay_data.zip` here) |

## Run order
1. **Colab:** `prepare_data.py --download`, then `render_demo.py --data sanjay_data`. Download the zip and the figures.
2. Unzip into `data/`, then run `pytest -q tests`. The real-data tests now run too.
3. Push this folder to a **public** GitHub repo and give Emergent the build prompt.

## What is real, what is simulated
- **Real data:** Sentinel-1 pixels (Sen1Floods11, CC BY 4.0; your Assam export), hand labels, JRC permanent water.
- **Model output:** water/flood masks, severity, intervals. Accuracy is shown only where hand labels exist.
- **Simulated:** satellite roles (SAT-A, SAT-B), contact windows, bitrate, the downlink order. These are assumptions in config.
- **Known limits:** detection uses a single image, so seasonal water and paddy can read as flood. The models are random-forest ensembles (machine learning, not deep learning). The onboard model is emulated on a laptop CPU, not run on flight hardware.
