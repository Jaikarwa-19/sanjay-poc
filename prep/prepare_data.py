"""Sanjay data prep: run in Google Colab (free). About 15-20 min end to end.

What it does
 1. Downloads the Sen1Floods11 hand-labelled set (Sentinel-1 VV/VH chips, water labels, JRC permanent water).
 2. Trains the GROUND and ONBOARD models on non-India chips; calibrates the 90% interval on held-out chips.
 3. Tests both on the India chips (a flood event neither model saw) -> real IoU / coverage.
 4. Picks demo scenes from the India chips so every app test case has real data behind it.
 5. Writes sanjay_data.zip -> download it and give it to Emergent.

Colab usage (each block is one cell):
    from google.colab import auth; auth.authenticate_user()     # any Google account
    !pip -q install rasterio
    # upload sanjay_core.py and this file via the Files panel, then:
    !python prepare_data.py --download
    from google.colab import files; files.download("sanjay_data.zip")

Local test without downloading:  python prepare_data.py --root <folder with S1Hand/LabelHand/JRCWaterHand>
"""
import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import sanjay_core as sc  # noqa: E402

BUCKET = "gs://sen1floods11/v1.1"
LAYERS = ("S1Hand", "LabelHand", "JRCWaterHand")
CELL_PX = 64


def download(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    print(subprocess.run(["gsutil", "ls", f"{BUCKET}/"], capture_output=True, text=True).stdout)
    for layer in LAYERS:
        subprocess.run(["gsutil", "-m", "-q", "cp", "-r",
                        f"{BUCKET}/data/flood_events/HandLabeled/{layer}", str(root)], check=True)
    subprocess.run(["gsutil", "-m", "-q", "cp", "-r", f"{BUCKET}/splits/flood_handlabeled", str(root)],
                   check=False)  # optional; a seeded split is used if missing


def download_http(root: Path, workers: int = 16):
    """Same files as download(), over public HTTPS (no gsutil or Google sign-in needed)."""
    import concurrent.futures as cf
    import urllib.parse
    import urllib.request
    api = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"
    prefixes = [f"v1.1/data/flood_events/HandLabeled/{l}/" for l in LAYERS] + ["v1.1/splits/flood_handlabeled/"]
    names = []
    for pre in prefixes:
        token = None
        while True:
            q = {"prefix": pre, "fields": "items(name),nextPageToken", "maxResults": "1000"}
            if token:
                q["pageToken"] = token
            with urllib.request.urlopen(f"{api}?{urllib.parse.urlencode(q)}", timeout=60) as r:
                page = json.load(r)
            names += [i["name"] for i in page.get("items", [])]
            token = page.get("nextPageToken")
            if not token:
                break

    def fetch(name):
        rel = name.split("HandLabeled/")[-1] if "HandLabeled/" in name else "flood_handlabeled/" + name.rsplit("/", 1)[-1]
        dst = root / rel
        if dst.exists() and dst.stat().st_size > 0:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/sen1floods11/" + urllib.parse.quote(name)
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)

    names = [n for n in names if not n.endswith("/")]
    print(f"downloading {len(names)} files")
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, names))


def chip_ids(root: Path):
    return sorted(p.name.replace("_S1Hand.tif", "") for p in (root / "S1Hand").glob("*_S1Hand.tif"))


def read_chip(root: Path, cid: str) -> dict:
    with rasterio.open(root / "S1Hand" / f"{cid}_S1Hand.tif") as ds:
        s1, transform, profile = ds.read().astype("float32"), tuple(ds.transform)[:6], ds.profile
    with rasterio.open(root / "LabelHand" / f"{cid}_LabelHand.tif") as ds:
        label = ds.read(1).astype("int16")
    perm = None
    jp = root / "JRCWaterHand" / f"{cid}_JRCWaterHand.tif"
    if jp.exists():
        with rasterio.open(jp) as ds:
            perm = ds.read(1) > 0
    vv, vh = sc.to_db(s1[0]), sc.to_db(s1[1])
    return {"id": cid, "vv": vv, "vh": vh, "label": label, "perm": perm,
            "transform": transform, "profile": profile}


def split(root: Path, ids: list[str]):
    """Train/calibrate on non-India chips; India chips are the held-out test event."""
    india = [i for i in ids if i.startswith("India_")]
    rest = [i for i in ids if not i.startswith("India_")]
    sp = root / "flood_handlabeled"
    valid_csv = sp / "flood_valid_data.csv"
    if valid_csv.exists():
        with open(valid_csv) as f:
            val = {r[0].replace("_S1Hand.tif", "") for r in csv.reader(f) if r}
        cal = [i for i in rest if i in val]
        train = [i for i in rest if i not in val]
    else:
        rnd = random.Random(0)
        rest = rest[:]
        rnd.shuffle(rest)
        cut = max(1, len(rest) // 5)
        cal, train = sorted(rest[:cut]), sorted(rest[cut:])
    return train, cal, india


def stats(c):
    lab = c["label"]
    labelled = lab >= 0
    water = (lab == 1) & labelled
    perm = c["perm"] if c["perm"] is not None else np.zeros_like(water)
    vv = np.where(np.isfinite(c["vv"]), c["vv"], np.nanmedian(c["vv"]))
    rough = float(np.std(ndimage.uniform_filter(vv, 21)))  # terrain roughness proxy (input only, not model)
    return {"roughness_db": rough, "water_frac": float(water.sum() / max(labelled.sum(), 1)),
            "flood_frac": float((water & ~perm).sum() / max(labelled.sum(), 1)),
            "perm_frac": float(perm.mean()),
            "nodata_frac": float(1 - (np.isfinite(c["vv"]) & np.isfinite(c["vh"])).mean())}


def pick_demo(india: list[dict], fallback: list[dict]) -> dict:
    """One real chip per test case, chosen by label statistics, not by hand.
    India chips first; a role no India chip fits is filled from the held-out non-India chips."""
    roles = _pick(india)
    extra = _pick(fallback, taken=set(d["chip"] for d in roles.values()))
    for r, d in extra.items():
        roles.setdefault(r, d)
    return roles


def _pick(chips: list[dict], taken=frozenset()) -> dict:
    s = {c["id"]: stats(c) for c in chips if c["id"] not in taken}
    ids = list(s)
    if not ids:
        return {}
    roles = {}
    roles["flood_severe"] = max(ids, key=lambda i: s[i]["flood_frac"])
    rest = [i for i in ids if i not in roles.values()]
    mid = sorted(rest, key=lambda i: abs(s[i]["flood_frac"] - 0.15))
    if mid:
        roles["flood_moderate"] = mid[0]
    dry = [i for i in ids if i not in roles.values() and s[i]["water_frac"] < 0.005 and s[i]["nodata_frac"] < 0.2]
    if dry:  # chosen by terrain roughness of the input, never by model output
        roles["dry_flat"] = min(dry, key=lambda i: s[i]["roughness_db"])
        if len(dry) > 1:
            roles["dry_hilly"] = max(dry, key=lambda i: s[i]["roughness_db"])
    river = [i for i in ids if i not in roles.values() and s[i]["perm_frac"] > 0.02 and s[i]["flood_frac"] < 0.01]
    if river:
        roles["permanent_water_only"] = max(river, key=lambda i: s[i]["perm_frac"])
    nod = [i for i in ids if i not in roles.values() and s[i]["nodata_frac"] > 0.05]
    if nod:
        roles["partial_nodata"] = max(nod, key=lambda i: s[i]["nodata_frac"])
    return {r: {"chip": i, **{k: round(v, 4) for k, v in s[i].items()}} for r, i in roles.items()}


def write_scene(out: Path, c: dict):
    prof = {**c["profile"], "count": 2, "dtype": "float32", "compress": "deflate", "nodata": None}
    with rasterio.open(out / f"{c['id']}.tif", "w", **prof) as ds:
        ds.write(np.stack([c["vv"], c["vh"]]))
    lp = {**c["profile"], "count": 1, "dtype": "int16", "compress": "deflate", "nodata": None}
    with rasterio.open(out / f"{c['id']}_label.tif", "w", **lp) as ds:
        ds.write(c["label"][None])
    if c["perm"] is not None:
        pp = {**c["profile"], "count": 1, "dtype": "uint8", "compress": "deflate", "nodata": None}
        with rasterio.open(out / f"{c['id']}_perm.tif", "w", **pp) as ds:
            ds.write(c["perm"].astype("uint8")[None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="sen1floods11")
    ap.add_argument("--out", default="sanjay_data")
    ap.add_argument("--download", action="store_true", help="gsutil (Colab)")
    ap.add_argument("--download-http", action="store_true", help="plain HTTPS, no sign-in")
    ap.add_argument("--px-per-chip", type=int, default=3000)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    if a.download:
        download(root)
    if a.download_http:
        download_http(root)
    ids = chip_ids(root)
    train_ids, cal_ids, india_ids = split(root, ids)
    print(f"chips: train {len(train_ids)}, calibrate {len(cal_ids)}, India test {len(india_ids)}")
    if not india_ids:
        sys.exit("No India chips found - check the bucket listing printed above.")

    load = lambda L: [read_chip(root, i) for i in L]  # noqa: E731
    train, cal, india = load(train_ids), load(cal_ids), load(india_ids)

    if out.exists():
        shutil.rmtree(out)
    (out / "models").mkdir(parents=True)
    (out / "scenes").mkdir()
    cards, metrics = {}, {"test_event": "Sen1Floods11 India hand-labelled chips (held out from training)",
                          "chips": len(india), "profiles": {}}
    for name, prof in sc.PROFILES.items():
        b = sc.train(train, prof, px_per_chip=a.px_per_chip, seed=0)
        b["post"], b["min_event_km2"] = dict(sc.DEFAULT_POST), sc.DEFAULT_MIN_EVENT_KM2
        b["conformal"] = sc.calibrate_interval(
            [(sc.predict(b, c["vv"], c["vh"]), c["label"]) for c in cal], cell_px=CELL_PX)
        joblib.dump(b, out / "models" / f"{name}.joblib", compress=3)
        import sklearn
        cards[name] = {**sc.model_card(b), "conformal": b["conformal"], "sklearn_version": sklearn.__version__,
                       "file_bytes": (out / "models" / f"{name}.joblib").stat().st_size}
        tp = fp = fn = hits = cells = 0
        runtimes, ev_hit, dry_alarm = [], [], []
        for c in india:
            p = sc.predict(b, c["vv"], c["vh"])
            runtimes.append(p["runtime_ms"])
            st = stats(c)
            _, g2 = sc.analyse(b, {"vv": c["vv"], "vh": c["vh"], "transform": c["transform"]}, c["perm"])
            if st["flood_frac"] >= 0.02:
                ev_hit.append(g2["qualifies"])
            elif st["flood_frac"] < 0.005:
                dry_alarm.append(g2["qualifies"])
            m = c["label"] >= 0
            pw, g = p["water_full"][m], c["label"][m] == 1
            tp, fp, fn = tp + int((pw & g).sum()), fp + int((pw & ~g).sum()), fn + int((~pw & g).sum())
            cov = sc.interval_coverage(p, c["label"], CELL_PX, conformal=b["conformal"])
            hits += (cov["coverage"] or 0) * cov["cells"]
            cells += cov["cells"]
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        metrics["profiles"][name] = {
            "iou": round(tp / max(tp + fp + fn, 1), 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "interval_coverage": round(hits / max(cells, 1), 4), "cells": cells,
            "median_runtime_ms": round(float(np.median(runtimes)), 1),
            "chip_detection_rate": round(float(np.mean(ev_hit)), 3) if ev_hit else None,
            "chip_false_alarm_rate": round(float(np.mean(dry_alarm)), 3) if dry_alarm else None,
            "flood_chips": len(ev_hit), "dry_chips": len(dry_alarm),
            "chip_rule": "flood chip = >=2% labelled flood water; dry chip = <0.5%"}
        print(name, cards[name]["model_size_bytes"], metrics["profiles"][name])

    demo = pick_demo(india, cal)
    for role, d in demo.items():
        write_scene(out / "scenes", next(c for c in india + cal if c["id"] == d["chip"]))
    catalog = {"source": "Sen1Floods11 v1.1 hand-labelled (Bonafilia et al., CVPRW 2020), CC BY 4.0",
               "cell_px": CELL_PX, "scenes": demo}
    (out / "catalog.json").write_text(json.dumps(catalog, indent=2))
    (out / "model_cards.json").write_text(json.dumps(cards, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    shutil.make_archive(str(out), "zip", out)
    print("roles:", {r: d["chip"] for r, d in demo.items()})
    missing = {"flood_severe", "flood_moderate", "dry_flat", "permanent_water_only", "partial_nodata"} - set(demo)
    if missing:
        print("WARNING: no India chip matched", missing, "- those tests will be skipped in the app.")
    print(f"wrote {out}.zip")


if __name__ == "__main__":
    main()
