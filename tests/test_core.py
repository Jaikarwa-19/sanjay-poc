"""Unit tests for sanjay_core on SYNTHETIC test fixtures (never shown in the UI as real data).

Real-data tests (Sen1Floods11 India chips, Assam scene) run in tests/test_real_data.py once
data/ is present; they are skipped otherwise.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import sanjay_core as sc  # noqa: E402

TRANSFORM = (0.0001, 0.0, 92.7, 0.0, -0.0001, 24.9, )  # ~10 m pixels near Silchar, EPSG:4326


def synth(seed, water_target=0.3, size=128, perm_band=False):
    """Water blobs: VV ~ -20 dB, VH ~ -27 dB; land: VV ~ -9, VH ~ -16."""
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.normal(size=(size, size)), 8)
    water = field > np.quantile(field, 1 - water_target) if water_target > 0 else np.zeros((size, size), bool)
    perm = np.zeros((size, size), bool)
    if perm_band:
        perm[:, 10:22] = True
        water = water | perm
    vv = np.where(water, rng.normal(-20, 1.5, water.shape), rng.normal(-9, 2.5, water.shape)).astype("float32")
    vh = np.where(water, rng.normal(-27, 1.5, water.shape), rng.normal(-16, 2.5, water.shape)).astype("float32")
    return {"id": f"synth{seed}", "vv": vv, "vh": vh, "label": water.astype("int16"), "perm": perm}


@pytest.fixture(scope="module")
def bundles():
    chips = [synth(s, w) for s, w in [(1, 0.3), (2, 0.5), (3, 0.1), (4, 0.4), (5, 0.2)]]
    return {name: sc.train(chips, p, px_per_chip=2000, seed=0) for name, p in sc.PROFILES.items()}


def run(bundle, chip, perm=None, cell_px=32):
    pred = sc.predict(bundle, chip["vv"], chip["vh"], perm)
    grid = sc.severity_grid(pred, TRANSFORM, cell_px=cell_px, min_event_km2=0.01)
    alert = sc.build_alert(chip["id"], grid, pred, bundle, created_at="2022-06-20T00:00:00Z")
    return pred, grid, alert


# S1 reproducible
def test_reproducible(bundles):
    chip = synth(10, 0.35)
    a = run(bundles["ground"], chip)
    b = run(bundles["ground"], chip)
    assert np.array_equal(a[0]["flood_full"], b[0]["flood_full"])
    assert a[2]["bytes"] == b[2]["bytes"]
    retrained = sc.train([synth(1, 0.3), synth(2, 0.5)], sc.ONBOARD, px_per_chip=2000, seed=0)
    again = sc.train([synth(1, 0.3), synth(2, 0.5)], sc.ONBOARD, px_per_chip=2000, seed=0)
    assert sc.model_card(retrained)["model_size_bytes"] == sc.model_card(again)["model_size_bytes"]


# S2 flood detected, severity reflects extent
def test_flood_detected_and_severity(bundles):
    for name in ("ground", "onboard"):
        pred, grid, alert = run(bundles[name], synth(11, 0.45))
        assert alert is not None and grid["qualifies"]
        assert any(c["severity"] == "severe" for c in grid["cells"])
        m = sc.evaluate(pred["water_full"], synth(11, 0.45)["label"])
        assert m["iou"] > 0.7, (name, m)


# S3 dry scene -> no alert, no cue
def test_dry_scene_no_alert(bundles):
    for name in ("ground", "onboard"):
        _, grid, alert = run(bundles[name], synth(12, 0.0))
        assert alert is None and not grid["qualifies"]
        assert sc.build_cue(alert, "t") is None


# S4 permanent water only -> no flood
def test_permanent_water_excluded(bundles):
    chip = synth(13, 0.0, perm_band=True)
    pred, grid, alert = run(bundles["ground"], chip, perm=chip["perm"])
    assert pred["water_full"][:, 12:20].mean() > 0.8   # water is seen...
    assert alert is None                                # ...but it is not flood


# S5 packet size is the real serialised size
def test_packet_size(bundles):
    _, _, alert = run(bundles["ground"], synth(14, 0.4))
    assert alert["size_bytes"] == len(json.dumps(alert["payload"], separators=(",", ":"), sort_keys=True).encode())
    cue = sc.build_cue(alert, "2022-06-20T00:00:05Z")
    assert cue["size_bytes"] == len(cue["bytes"])


# S6 interval contains the point estimate and is ordered
def test_interval_well_formed(bundles):
    for name in ("ground", "onboard"):
        _, grid, _ = run(bundles[name], synth(15, 0.3))
        for c in grid["cells"]:
            if c["ci90"]:
                assert 0 <= c["ci90"][0] <= c["flood_fraction"] <= c["ci90"][1] <= 1
        lo, hi = grid["flood_area_ci90_km2"]
        assert lo <= grid["flood_area_km2"] <= hi


# S7 no-data handling
def test_nodata_cells(bundles):
    chip = synth(16, 0.3)
    chip["vv"][:32, :32] = np.nan
    _, grid, _ = run(bundles["ground"], chip)
    assert grid["cells"][0]["severity"] == "no_data" and grid["cells"][0]["flood_fraction"] is None


# S8 onboard model is genuinely reduced
def test_onboard_is_reduced(bundles):
    g, o = sc.model_card(bundles["ground"]), sc.model_card(bundles["onboard"])
    assert o["model_size_bytes"] < g["model_size_bytes"]
    assert o["n_features"] < g["n_features"] and o["trees"] < g["trees"]
    assert o["input_bits"] == 8 and o["input_downsample"] == 2


# S9 linear input is converted to dB
def test_linear_input():
    db = np.array([[-20.0, -10.0], [0.0, np.nan]], dtype="float32")
    lin = np.power(10, db / 10)
    np.testing.assert_allclose(sc.to_db(lin), db, rtol=1e-5)
    np.testing.assert_allclose(sc.to_db(db), db)


# S10 tiered priority: alert -> crops -> raw; raw-first reverses it
def test_priority_order(bundles):
    chip = synth(17, 0.4)
    pred, grid, alert = run(bundles["onboard"], chip)
    crops = sc.make_crops(chip["vv"], pred, grid, alert)
    assert crops and all(c["png"][:8] == b"\x89PNG\r\n\x1a\n" for c in crops)
    items = [{"id": "alert", "tier": 1, "priority": 1, "size_bytes": alert["size_bytes"], "ready_at_s": 0}]
    items += [{"id": f"crop{i}", "tier": 2, "priority": 2, "size_bytes": c["size_bytes"], "ready_at_s": 0}
              for i, c in enumerate(crops)]
    items += [{"id": "raw", "tier": 3, "priority": 3, "size_bytes": sc.raw_scene_bytes(chip["vv"]), "ready_at_s": 0}]
    s = sc.schedule_downlink(items, [(0, 1e6)], 1e6)
    assert [x["id"] for x in s][0] == "alert" and s[-1]["id"] == "raw"
    conv = sc.schedule_downlink(items, [(0, 1e6)], 1e6, order="raw_first")
    assert conv[0]["id"] == "raw"
    assert {x["id"]: x["end_s"] for x in s}["alert"] < {x["id"]: x["end_s"] for x in conv}["alert"]


# S11 no contact window -> stays queued
def test_no_window_queued():
    s = sc.schedule_downlink([{"id": "a", "tier": 1, "priority": 1, "size_bytes": 500, "ready_at_s": 100}],
                             [(0, 50)], 1e6)
    assert s[0]["status"] == "queued" and s[0]["end_s"] is None


# S12 bandwidth effect, and resume across windows
def test_bandwidth_and_resume():
    it = [{"id": "a", "tier": 3, "priority": 3, "size_bytes": 125_000, "ready_at_s": 0}]  # 1,000,000 bits
    assert sc.schedule_downlink(it, [(0, 100)], 10e6)[0]["duration_s"] == pytest.approx(0.1)
    assert sc.schedule_downlink(it, [(0, 100)], 5e6)[0]["duration_s"] == pytest.approx(0.2)
    r = sc.schedule_downlink(it, [(0, 0.05), (10, 20)], 10e6)[0]
    assert r["status"] == "delivered" and r["end_s"] == pytest.approx(10.05)


# S13 no labels -> no accuracy
def test_no_labels_no_accuracy(bundles):
    pred, _, _ = run(bundles["ground"], synth(18, 0.3))
    assert sc.evaluate(pred["water_full"], None) is None
    assert sc.interval_coverage(pred, None) is None


# S14 GeoTIFF loading: 2-band OK, wrong CRS and single band rejected, linear converted
def test_load_scene(tmp_path):
    import rasterio
    from rasterio.transform import Affine

    chip = synth(19, 0.3)
    tf = Affine(*TRANSFORM)

    def write(path, bands, crs="EPSG:4326"):
        with rasterio.open(path, "w", driver="GTiff", height=128, width=128, count=len(bands),
                           dtype="float32", crs=crs, transform=tf) as ds:
            for i, b in enumerate(bands, 1):
                ds.write(b, i)

    write(tmp_path / "ok.tif", [chip["vv"], chip["vh"]])
    s = sc.load_scene(str(tmp_path / "ok.tif"))
    np.testing.assert_allclose(s["vv"], chip["vv"])
    write(tmp_path / "lin.tif", [10 ** (chip["vv"] / 10), 10 ** (chip["vh"] / 10)])
    np.testing.assert_allclose(sc.load_scene(str(tmp_path / "lin.tif"))["vv"], chip["vv"], atol=1e-3)
    write(tmp_path / "utm.tif", [chip["vv"], chip["vh"]], crs="EPSG:32646")
    with pytest.raises(sc.SceneError, match="EPSG:4326"):
        sc.load_scene(str(tmp_path / "utm.tif"))
    write(tmp_path / "vv.tif", [chip["vv"]])
    with pytest.raises(sc.SceneError, match="VH"):
        sc.load_scene(str(tmp_path / "vv.tif"))
    write(tmp_path / "vh.tif", [chip["vh"]])
    two = sc.load_scene(str(tmp_path / "vv.tif"), str(tmp_path / "vh.tif"))
    np.testing.assert_allclose(two["vh"], chip["vh"])


# S15 interval coverage is computable on labelled data (diagnostic, reported not asserted high)
def test_interval_coverage_runs(bundles):
    chip = synth(20, 0.3)
    pred = sc.predict(bundles["ground"], chip["vv"], chip["vh"])
    cov = sc.interval_coverage(pred, chip["label"], cell_px=32)
    assert cov["cells"] > 0 and 0 <= cov["coverage"] <= 1


# S16 conformal interval reaches its target coverage on held-out labelled data
def test_conformal_coverage(bundles):
    def noisy(seed, w):
        c = synth(seed, w, size=256)
        r = np.random.default_rng(seed)
        c["vv"] += r.normal(0, 3, c["vv"].shape).astype("float32")
        c["vh"] += r.normal(0, 3, c["vh"].shape).astype("float32")
        return c
    b = bundles["ground"]
    cal = [noisy(s, w) for s, w in [(30, .3), (31, .2), (32, .4), (33, .1), (34, .25), (35, .35)]]
    conf = sc.calibrate_interval([(sc.predict(b, c["vv"], c["vh"]), c["label"]) for c in cal], cell_px=32)
    hits = cells = 0
    for s, w in [(40, .3), (41, .2), (42, .4), (43, .15)]:
        c = noisy(s, w)
        cov = sc.interval_coverage(sc.predict(b, c["vv"], c["vh"]), c["label"], cell_px=32, conformal=conf)
        hits += cov["coverage"] * cov["cells"]
        cells += cov["cells"]
    assert hits / cells >= 0.8
    pred = sc.predict(b, cal[0]["vv"], cal[0]["vh"])
    grid = sc.severity_grid(pred, TRANSFORM, cell_px=32, conformal=conf)
    assert "split-conformal" in grid["interval_method"]
