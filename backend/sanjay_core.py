"""Sanjay core: onboard flood triage for Sentinel-1 SAR (proof of concept).

Pure functions, no web code. The FastAPI layer wraps these; it must not reimplement them.

Pipeline: load VV/VH -> features -> per-pixel water (random-forest ensemble)
-> remove permanent water -> severity grid with 90% interval per cell
-> Tier 1 alert JSON, Tier 2 image crops, Tier 3 raw scene -> priority downlink queue.

Two model profiles show "reduced onboard AI vs regular ground AI":
  GROUND : 60 trees, depth 12, 10 features, full resolution, float32
  ONBOARD: 8 trees, depth 5, 4 features, 2x downsampled, 8-bit quantised input
Both are trained on the same Sen1Floods11 hand-labelled chips.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
import time
import tracemalloc
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier

VERSION = "sanjay-core-0.1"
SEVERITY_LEVELS = ("none", "low", "moderate", "severe")
SEVERITY_COLOURS = {  # light grey -> steel blue -> amber -> red; no-data = white
    "no_data": "#FFFFFF", "none": "#E8EBEF", "low": "#315B78",
    "moderate": "#B7791F", "severe": "#B42318",
}
DB_MIN, DB_MAX = -30.0, 5.0  # clip range for 8-bit quantisation


@dataclass(frozen=True)
class Profile:
    name: str
    features: tuple
    n_estimators: int
    max_depth: int
    downsample: int
    quantize: bool
    min_samples_leaf: int = 20


GROUND = Profile("ground", ("vv", "vh", "diff", "vv_mean5", "vh_mean5", "vv_std5",
                            "vv_mean21", "vh_mean21", "vv_std21", "vh_std21"), 60, 12, 1, False)
ONBOARD = Profile("onboard", ("vv", "vh", "diff", "vv_std9"), 8, 5, 2, True)
# Post-processing and alert rule, chosen on the calibration chips (see prep/prepare_data.py).
DEFAULT_POST = {"open": True, "min_blob_px": 1000}
DEFAULT_MIN_EVENT_KM2 = 0.25
PROFILES = {"ground": GROUND, "onboard": ONBOARD}


class SceneError(ValueError):
    """Input that Sanjay cannot process; message is shown to the user."""


# ---------------------------------------------------------------- loading
def to_db(arr: np.ndarray) -> np.ndarray:
    """Return backscatter in dB. Linear input (median > 0) is converted; <=0 becomes NaN."""
    a = np.asarray(arr, dtype="float32").copy()
    finite = np.isfinite(a)
    if finite.any() and float(np.median(a[finite])) > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(a > 0, 10.0 * np.log10(a), np.nan).astype("float32")
    return a


def load_scene(path_or_bytes, vh_path=None) -> dict:
    """Read a 2-band (VV, VH) GeoTIFF, or VV + VH as two single-band files. Must be EPSG:4326."""
    import rasterio

    def _read(src):
        with rasterio.open(src) as ds:
            if ds.crs is None or ds.crs.to_epsg() != 4326:
                raise SceneError(f"Scene must be in WGS84 (EPSG:4326); got {ds.crs}. Re-export in WGS84.")
            data = ds.read().astype("float32")
            if ds.nodata is not None:
                data[data == ds.nodata] = np.nan
            return data, tuple(ds.transform)[:6], ds.tags()

    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    data, transform, tags = _read(src)
    if vh_path is not None:
        vh_src = io.BytesIO(vh_path) if isinstance(vh_path, (bytes, bytearray)) else vh_path
        vh_data, t2, _ = _read(vh_src)
        if vh_data.shape[1:] != data.shape[1:] or not np.allclose(t2, transform):
            raise SceneError("VV and VH files do not have the same grid.")
        data = np.concatenate([data[:1], vh_data[:1]])
    if data.shape[0] < 2:
        raise SceneError("Sanjay needs both VV and VH bands (a 2-band file, or VV and VH as two files).")
    vv, vh = to_db(data[0]), to_db(data[1])
    if not (np.isfinite(vv) & np.isfinite(vh)).any():
        raise SceneError("Scene contains no valid pixels.")
    return {"vv": vv, "vh": vh, "transform": transform, "tags": tags}


def pixel_size_m(transform, lat: float) -> tuple[float, float]:
    a, _, _, _, e, _ = transform
    return abs(a) * 111_320.0 * math.cos(math.radians(lat)), abs(e) * 110_574.0


def pix_to_lonlat(transform, row: float, col: float) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    return a * col + b * row + c, d * col + e * row + f


# ---------------------------------------------------------------- features
def _crop(a, k):
    h, w = a.shape[-2:]
    return a[..., : h - h % k, : w - w % k]


def _block_mean(a, k):
    a = _crop(a, k)
    h, w = a.shape
    b = a.reshape(h // k, k, w // k, k)
    fin = np.isfinite(b)
    valid = fin.mean(axis=(1, 3)) >= 0.5
    s = np.where(fin, b, 0).sum(axis=(1, 3))
    n = fin.sum(axis=(1, 3))
    m = s / np.maximum(n, 1)
    return np.where(valid, m, np.nan).astype("float32")


def _block_mask(mask, k, valid=None):
    """Majority of True among valid pixels in each k x k block."""
    m = _crop(mask, k).astype("float32")
    v = np.ones_like(m) if valid is None else _crop(valid, k).astype("float32")
    h, w = m.shape
    num = (m * v).reshape(h // k, k, w // k, k).sum(axis=(1, 3))
    den = v.reshape(h // k, k, w // k, k).sum(axis=(1, 3))
    return np.where(den > 0, num / np.maximum(den, 1) >= 0.5, False)


def _block_label(label, k):
    lab = _crop(label, k)
    valid = lab >= 0
    water = _block_mask(lab == 1, k, valid)
    has = _block_mask(valid, k)
    return np.where(has, water.astype("int16"), -1).astype("int16")


def _quantize(a):
    q = np.round((np.clip(a, DB_MIN, DB_MAX) - DB_MIN) * 255.0 / (DB_MAX - DB_MIN))
    return np.where(np.isfinite(a), q * (DB_MAX - DB_MIN) / 255.0 + DB_MIN, np.nan).astype("float32")


def features(vv, vh, profile: Profile):
    """Return (X[h, w, n_features], valid[h, w]) on the profile's model grid."""
    k = profile.downsample
    if k > 1:
        vv, vh = _block_mean(vv, k), _block_mean(vh, k)
    valid = np.isfinite(vv) & np.isfinite(vh)
    if profile.quantize:
        vv, vh = _quantize(vv), _quantize(vh)
    vvf = np.where(valid, vv, np.nanmedian(vv[valid]) if valid.any() else -15.0)
    vhf = np.where(valid, vh, np.nanmedian(vh[valid]) if valid.any() else -22.0)
    base = {"vv": vvf, "vh": vhf}
    f = {"vv": vvf, "vh": vhf, "diff": vvf - vhf}
    for name in profile.features:  # context features: <band>_<mean|std><window px>, e.g. vv_std15
        if name in f:
            continue
        band, stat = name.split("_")
        win = int("".join(ch for ch in stat if ch.isdigit()))
        m = ndimage.uniform_filter(base[band], win)
        f[name] = m if stat.startswith("mean") else np.sqrt(
            np.maximum(ndimage.uniform_filter(base[band] ** 2, win) - m * m, 0))
    X = np.stack([f[n] for n in profile.features], axis=-1).astype("float32")
    return X, valid


# ---------------------------------------------------------------- training
def train(chips: list[dict], profile: Profile, px_per_chip: int = 4000, seed: int = 0) -> dict:
    """chips: [{'id', 'vv', 'vh', 'label' (-1 nodata, 0 dry, 1 water)}]. Balanced pixel sampling."""
    rng = np.random.default_rng(seed)
    Xs, ys, ids = [], [], []
    vv_all = []
    for c in chips:
        X, valid = features(c["vv"], c["vh"], profile)
        lab = c["label"] if profile.downsample == 1 else _block_label(c["label"], profile.downsample)
        lab = lab[: valid.shape[0], : valid.shape[1]]
        m = (valid & (lab >= 0)).ravel()
        flat_lab = lab.ravel()
        picks = []
        for cls in (0, 1):
            idx = np.flatnonzero(m & (flat_lab == cls))
            if idx.size:
                picks.append(rng.choice(idx, min(px_per_chip // 2, idx.size), replace=False))
        if not picks:
            continue
        sel = np.sort(np.concatenate(picks))
        Xs.append(X.reshape(-1, X.shape[-1])[sel])
        ys.append(flat_lab[sel])
        ids.append(c.get("id", f"chip{len(ids)}"))
        vv_all.append(c["vv"][np.isfinite(c["vv"])])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training data must contain both water and non-water pixels.")
    model = RandomForestClassifier(
        n_estimators=profile.n_estimators, max_depth=profile.max_depth,
        min_samples_leaf=profile.min_samples_leaf, random_state=seed, n_jobs=1,
    ).fit(X, y)
    vv_cat = np.concatenate(vv_all)
    return {
        "profile": asdict(profile), "model": model, "version": VERSION, "seed": seed,
        "trained_on": ids, "n_samples": int(y.size),
        "vv_range_db": [float(np.percentile(vv_cat, 2)), float(np.percentile(vv_cat, 98))],
    }


def model_card(bundle: dict) -> dict:
    p = bundle["profile"]
    return {
        "profile": p["name"], "trees": p["n_estimators"], "max_depth": p["max_depth"],
        "n_features": len(p["features"]), "features": list(p["features"]),
        "input_downsample": p["downsample"], "input_bits": 8 if p["quantize"] else 32,
        "model_size_bytes": len(pickle.dumps(bundle["model"], protocol=4)),
        "trained_on_chips": len(bundle["trained_on"]), "training_pixels": bundle["n_samples"],
        "version": bundle["version"],
    }


# ---------------------------------------------------------------- inference
def predict(bundle: dict, vv, vh, perm=None) -> dict:
    """Per-tree water votes on the model grid, plus full-resolution masks for display."""
    profile = Profile(**{**bundle["profile"], "features": tuple(bundle["profile"]["features"])})
    model, k = bundle["model"], profile.downsample
    tracemalloc.start()
    t0 = time.perf_counter()
    X, valid = features(vv, vh, profile)
    h, w = valid.shape
    flat, vflat = X.reshape(-1, X.shape[-1]), valid.ravel()
    votes = np.zeros((len(model.estimators_), h * w), dtype=bool)
    if vflat.any():
        sub = flat[vflat]
        for i, est in enumerate(model.estimators_):
            votes[i, vflat] = est.predict(sub) == 1  # classes_ are [0, 1]
    votes = votes.reshape(-1, h, w)
    runtime_ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    perm_m = np.zeros((h, w), bool)
    if perm is not None:
        perm_m = (perm.astype(bool) if k == 1 else _block_mask(perm.astype(bool), k))[:h, :w]
    share = votes.mean(axis=0)
    water = clean_mask(valid & (share >= 0.5), bundle.get("post", {}), k)
    flood_votes = votes & valid & ~perm_m & water[None]
    flood = water & ~perm_m

    def up(a):  # model grid -> full grid (cropped edge stays False)
        full = np.zeros(vv.shape, dtype=a.dtype)
        r = np.repeat(np.repeat(a, k, 0), k, 1) if k > 1 else a
        full[: r.shape[0], : r.shape[1]] = r
        return full

    vv_med = float(np.nanmedian(vv)) if np.isfinite(vv).any() else float("nan")
    lo, hi = bundle.get("vv_range_db", [-40, 10])
    return {
        "k": k, "profile": profile.name, "valid": valid, "water": water, "flood": flood,
        "flood_votes": flood_votes, "perm": perm_m, "water_share": share,
        "water_full": up(water), "flood_full": up(flood), "valid_full": up(valid),
        "runtime_ms": runtime_ms, "peak_mem_mb": peak / 1e6,
        "out_of_range": bool(not (lo - 3 <= vv_med <= hi + 3)) if np.isfinite(vv_med) else True,
    }


def clean_mask(mask, post: dict, k: int = 1):
    """Drop speckle and small blobs (radar shadow on slopes looks like water; floods are contiguous).
    post = {"open": bool, "min_blob_px": full-resolution pixels}. Empty dict = no cleaning."""
    if post.get("open"):
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), bool))
    min_px = int(post.get("min_blob_px", 0)) // (k * k)
    if min_px > 1:
        lab, n = ndimage.label(mask)
        if n:
            sizes = np.bincount(lab.ravel())
            keep = sizes >= min_px
            keep[0] = False
            mask = keep[lab]
    return mask


def severity_grid(pred: dict, transform, cell_px: int = 64, thresholds=(0.05, 0.20, 0.50),
                  min_valid: float = 0.5, ci=(5, 95), min_event_km2: float = 0.1,
                  min_class: str = "moderate", conformal: dict | None = None) -> dict:
    """Cells of cell_px x cell_px full-res pixels.

    Interval: split-conformal (point +/- q) when `conformal` (from calibrate_interval) is given,
    otherwise the 5th-95th percentile spread of per-tree estimates (uncalibrated).
    """
    k = pred["k"]
    if cell_px % k:
        raise ValueError("cell_px must be divisible by the model downsample factor")
    c = cell_px // k
    valid, flood, votes = _crop(pred["valid"], c), _crop(pred["flood"], c), _crop(pred["flood_votes"], c)
    h, w = valid.shape
    nr, nc = h // c, w // c
    if nr == 0 or nc == 0:
        raise SceneError(f"Scene is smaller than one {cell_px}px cell.")
    v = valid.reshape(nr, c, nc, c).sum(axis=(1, 3)).astype(float)
    f = flood.reshape(nr, c, nc, c).sum(axis=(1, 3)).astype(float)
    t = votes.reshape(votes.shape[0], nr, c, nc, c).sum(axis=(2, 4)).astype(float)
    lat0 = pix_to_lonlat(transform, 0, 0)[1]
    dx, dy = pixel_size_m(transform, lat0)
    px_area_model = dx * dy * k * k
    cells, rank = [], {s: i for i, s in enumerate(SEVERITY_LEVELS)}
    for r in range(nr):
        for q in range(nc):
            n = v[r, q]
            r0, c0 = r * cell_px, q * cell_px
            lon, lat = pix_to_lonlat(transform, r0 + cell_px / 2, c0 + cell_px / 2)
            base = {"id": f"r{r}c{q}", "row": r, "col": q, "px_bounds": [r0, c0, r0 + cell_px, c0 + cell_px],
                    "centroid": [round(lon, 6), round(lat, 6)]}
            if n < min_valid * c * c:
                cells.append({**base, "severity": "no_data", "flood_fraction": None, "ci90": None,
                              "area_km2": None})
                continue
            point = f[r, q] / n
            if conformal:
                lo, hi = max(point - conformal["q"], 0.0), min(point + conformal["q"], 1.0)
            else:
                lo, hi = np.percentile(t[:, r, q] / n, ci)
            sev = SEVERITY_LEVELS[int(np.searchsorted(thresholds, point, side="right"))]
            cells.append({**base, "severity": sev, "flood_fraction": round(float(point), 4),
                          "ci90": [round(float(min(lo, point)), 4), round(float(max(hi, point)), 4)],
                          "area_km2": round(float(f[r, q] * px_area_model / 1e6), 4)})
    tot = votes.reshape(votes.shape[0], -1).sum(axis=1) * px_area_model / 1e6
    area = float(flood.sum() * px_area_model / 1e6)
    flagged = [x for x in cells if x["severity"] in SEVERITY_LEVELS and rank[x["severity"]] >= rank[min_class]]
    return {
        "cell_px": cell_px, "thresholds": list(thresholds), "cells": cells, "shape": [nr, nc],
        "flood_area_km2": round(area, 4),
        "flood_area_ci90_km2": [round(float(min(np.percentile(tot, ci[0]), area)), 4),
                                round(float(max(np.percentile(tot, ci[1]), area)), 4)],
        "qualifies": bool(flagged) and area >= min_event_km2,
        "interval_method": (f"split-conformal {int(conformal['level']*100)}% (q={conformal['q']:.3f}, "
                            f"calibrated on {conformal['cells']} labelled cells)") if conformal else
                           f"{ci[0]}th-{ci[1]}th percentile of per-tree estimates (uncalibrated ensemble spread)",
    }


def analyse(bundle: dict, scene: dict, perm=None, cell_px: int = 64) -> tuple[dict, dict]:
    """predict + severity_grid with the bundle's calibrated settings. Use this from the API."""
    pred = predict(bundle, scene["vv"], scene["vh"], perm)
    grid = severity_grid(pred, scene["transform"], cell_px=cell_px, conformal=bundle.get("conformal"),
                         min_event_km2=bundle.get("min_event_km2", DEFAULT_MIN_EVENT_KM2))
    return pred, grid


# ---------------------------------------------------------------- outputs
def build_alert(scene_id: str, grid: dict, pred: dict, bundle: dict, created_at: str,
                acquired_at: str | None = None, max_cells: int = 20) -> dict | None:
    """Tier 1 packet. None when no qualifying event (no alert, no cue)."""
    if not grid["qualifies"]:
        return None
    rank = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
    top = sorted((x for x in grid["cells"] if x["severity"] in ("moderate", "severe")),
                 key=lambda x: (-rank[x["severity"]], -x["flood_fraction"], x["id"]))[:max_cells]
    eid = hashlib.sha256(json.dumps([scene_id, bundle["version"], bundle["profile"]["name"],
                                     grid["cell_px"], grid["thresholds"]]).encode()).hexdigest()[:12]
    payload = {
        "event_id": f"SNJ-{eid}", "disaster_type": "flood", "source": "SAT-A (simulated)",
        "scene_id": scene_id, "observation_timestamp": acquired_at, "packet_created_at": created_at,
        "model": {"profile": bundle["profile"]["name"], "version": bundle["version"]},
        "flood_area_km2": grid["flood_area_km2"], "flood_area_ci90_km2": grid["flood_area_ci90_km2"],
        "interval_method": grid["interval_method"], "input_out_of_range": pred["out_of_range"],
        "cells": [{k2: x[k2] for k2 in ("id", "centroid", "severity", "flood_fraction", "ci90", "area_km2")}
                  for x in top],
        "priority": "P1",
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {"payload": payload, "bytes": raw, "size_bytes": len(raw)}


def build_cue(alert: dict, created_at: str) -> dict | None:
    """Follow-up observation request for the most severe cell (sent, not simulated further)."""
    if alert is None:
        return None
    cell = alert["payload"]["cells"][0]
    payload = {"request_id": alert["payload"]["event_id"].replace("SNJ", "CUE"),
               "source": "SAT-A (simulated)", "destination": "SAT-B (simulated)",
               "target_centroid": cell["centroid"], "target_cell": cell["id"], "sensor": "SAR",
               "priority": "P1", "request_creation_time": created_at}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {"payload": payload, "bytes": raw, "size_bytes": len(raw)}


def _to_u8(a, lo=-25.0, hi=0.0):
    return np.nan_to_num(np.clip((a - lo) / (hi - lo) * 255, 0, 255), nan=0).astype("uint8")


def render_png(vv, flood_full=None, max_side=None) -> bytes:
    """Grey VV image with flood pixels tinted steel blue."""
    g = _to_u8(vv)
    rgb = np.stack([g, g, g], -1)
    if flood_full is not None:
        m = flood_full[: g.shape[0], : g.shape[1]]
        rgb[m] = (0.35 * rgb[m] + 0.65 * np.array([49, 91, 120])).astype("uint8")
    im = Image.fromarray(rgb)
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def make_crops(vv, pred: dict, grid: dict, alert: dict | None, top_k: int = 3, pad_px: int = 16) -> list[dict]:
    """Tier 2: PNG crops around the top-k alert cells."""
    if alert is None:
        return []
    by_id = {x["id"]: x for x in grid["cells"]}
    out = []
    for cell in alert["payload"]["cells"][:top_k]:
        r0, c0, r1, c1 = by_id[cell["id"]]["px_bounds"]
        r0, c0 = max(r0 - pad_px, 0), max(c0 - pad_px, 0)
        r1, c1 = min(r1 + pad_px, vv.shape[0]), min(c1 + pad_px, vv.shape[1])
        png = render_png(vv[r0:r1, c0:c1], pred["flood_full"][r0:r1, c0:c1])
        out.append({"cell_id": cell["id"], "severity": cell["severity"], "px_bounds": [r0, c0, r1, c1],
                    "png": png, "size_bytes": len(png)})
    return out


def raw_scene_bytes(vv) -> int:
    """Tier 3 size: VV + VH as float32."""
    return int(vv.size * 2 * 4)


# ---------------------------------------------------------------- downlink
def schedule_downlink(items: list[dict], windows: list[tuple[float, float]], bitrate_bps: float,
                      order: str = "priority") -> list[dict]:
    """Single link, non-preemptive between items; an item that overruns a window resumes in the next.

    items: {id, tier, priority (0 = highest), size_bytes, ready_at_s}
    order: 'priority' (Sanjay: alert -> crops -> raw) or 'raw_first' (conventional image-first).
    """
    if bitrate_bps <= 0:
        raise ValueError("bitrate must be positive")
    if order == "priority":
        seq = sorted(items, key=lambda x: (x["priority"], x["ready_at_s"], x["id"]))
    elif order == "raw_first":
        seq = sorted(items, key=lambda x: (x["tier"] != 3, x["priority"], x["ready_at_s"], x["id"]))
    else:
        raise ValueError(order)
    wins = sorted(windows)
    t, out = 0.0, []
    for it in seq:
        remaining = it["size_bytes"] * 8 / bitrate_bps
        t = max(t, it["ready_at_s"])
        start = end = None
        for ws, we in wins:
            if we <= t or remaining <= 0:
                continue
            s = max(t, ws)
            if start is None:
                start = s
            used = min(remaining, we - s)
            remaining -= used
            t = end = s + used
        status = "delivered" if remaining <= 1e-9 else "queued"
        out.append({**it, "start_s": start if status == "delivered" else None,
                    "end_s": end if status == "delivered" else None,
                    "duration_s": it["size_bytes"] * 8 / bitrate_bps, "status": status})
        if status == "queued":
            t = float("inf")
    return out


# ---------------------------------------------------------------- evaluation
def evaluate(water_full, label) -> dict | None:
    """IoU/precision/recall/F1 of water vs hand labels. None when there are no labels."""
    if label is None:
        return None
    m = label >= 0
    if not m.any():
        return None
    p, g = water_full[m].astype(bool), label[m] == 1
    tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    return {"iou": tp / (tp + fp + fn) if tp + fp + fn else None, "precision": prec, "recall": rec,
            "f1": f1, "pixels": int(m.sum())}


def _cell_pairs(pred: dict, label, cell_px: int = 64, min_valid: float = 0.5):
    """Yield (true water fraction, point estimate, per-tree estimates) per labelled cell.
    Use a prediction made with perm=None so estimates are of all water, like the labels."""
    k = pred["k"]
    lab = label if k == 1 else _block_label(label, k)
    c = cell_px // k
    lab = _crop(lab[: pred["valid"].shape[0], : pred["valid"].shape[1]], c)
    votes, valid, flood = _crop(pred["flood_votes"], c), _crop(pred["valid"], c), _crop(pred["flood"], c)
    h, w = lab.shape
    for r in range(h // c):
        for q in range(w // c):
            sl = (slice(r * c, (r + 1) * c), slice(q * c, (q + 1) * c))
            m = valid[sl] & (lab[sl] >= 0)
            if m.sum() < min_valid * c * c:
                continue
            yield (lab[sl][m] == 1).mean(), flood[sl][m].mean(), votes[(slice(None),) + sl][:, m].mean(axis=1)


def calibrate_interval(pairs: list, cell_px: int = 64, level: float = 0.9) -> dict:
    """Split-conformal half-width from labelled calibration chips: pairs = [(pred, label), ...]."""
    errs = sorted(abs(t - p) for pred, lab in pairs for t, p, _ in _cell_pairs(pred, lab, cell_px))
    if not errs:
        raise ValueError("No labelled cells to calibrate on.")
    n = len(errs)
    q = float(errs[min(math.ceil((n + 1) * level), n) - 1])
    return {"q": q, "cells": n, "level": level, "cell_px": cell_px}


def interval_coverage(pred: dict, label, cell_px: int = 64, conformal: dict | None = None) -> dict | None:
    """Share of labelled cells whose true water fraction lies inside the interval."""
    if label is None:
        return None
    hits = n = 0
    for truth, point, est in _cell_pairs(pred, label, cell_px):
        if conformal:
            lo, hi = point - conformal["q"], point + conformal["q"]
        else:
            lo, hi = np.percentile(est, (5, 95))
        hits += lo - 1e-9 <= truth <= hi + 1e-9
        n += 1
    return {"cells": n, "coverage": hits / n if n else None}
