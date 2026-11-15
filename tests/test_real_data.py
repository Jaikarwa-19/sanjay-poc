"""Tests on the REAL Sen1Floods11 demo scenes written by prep/prepare_data.py.
Skipped until data/ (unzipped sanjay_data) exists. Set SANJAY_DATA to override the path."""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import sanjay_core as sc  # noqa: E402

DATA = Path(os.environ.get("SANJAY_DATA", Path(__file__).resolve().parents[1] / "data"))
pytestmark = pytest.mark.skipif(not (DATA / "catalog.json").exists(), reason="real data not present")


def scene(role):
    cat = json.loads((DATA / "catalog.json").read_text())
    if role not in cat["scenes"]:
        pytest.skip(f"no chip for role {role}")
    cid = cat["scenes"][role]["chip"]
    s = sc.load_scene(str(DATA / "scenes" / f"{cid}.tif"))
    perm_p = DATA / "scenes" / f"{cid}_perm.tif"
    perm = rasterio.open(perm_p).read(1) > 0 if perm_p.exists() else None
    label = rasterio.open(DATA / "scenes" / f"{cid}_label.tif").read(1)
    return cid, s, perm, label


def outcome(role, profile):
    b = joblib.load(DATA / "models" / f"{profile}.joblib")
    cid, s, perm, label = scene(role)
    pred, grid = sc.analyse(b, s, perm)
    return pred, grid, sc.build_alert(cid, grid, pred, b, created_at="fixed"), label


@pytest.mark.parametrize("profile", ["ground", "onboard"])
def test_severe_flood_alerts(profile):
    _, grid, alert, _ = outcome("flood_severe", profile)
    assert alert is not None and alert["size_bytes"] == len(alert["bytes"])


@pytest.mark.parametrize("profile", ["ground", "onboard"])
@pytest.mark.parametrize("role", ["dry_flat", "permanent_water_only"])
def test_no_flood_no_alert(role, profile):
    _, grid, alert, _ = outcome(role, profile)
    assert alert is None, f"{role}/{profile}: false alarm, flood area {grid['flood_area_km2']} km2"


def test_hilly_dry_land_onboard():
    """Hill shadow is the main false-alarm source; the rate across all dry chips is in metrics.json."""
    _, grid, alert, _ = outcome("dry_hilly", "onboard")
    assert alert is None


def test_metrics_are_reported_honestly():
    m = json.loads((DATA / "metrics.json").read_text())
    for p in ("ground", "onboard"):
        assert m["profiles"][p]["chip_false_alarm_rate"] is not None


def test_nodata_cells_marked():
    _, grid, _, _ = outcome("partial_nodata", "ground")
    assert any(c["severity"] == "no_data" for c in grid["cells"])


def test_onboard_smaller_than_ground():
    cards = json.loads((DATA / "model_cards.json").read_text())
    assert cards["onboard"]["model_size_bytes"] < cards["ground"]["model_size_bytes"]


def test_replay_identical():
    a, b = outcome("flood_moderate", "onboard"), outcome("flood_moderate", "onboard")
    assert (a[2] is None and b[2] is None) or a[2]["bytes"] == b[2]["bytes"]
