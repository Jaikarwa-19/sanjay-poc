"""Render submission-ready figures from real data, without the web app (safety net for 16 Nov 12:00).

    python render_demo.py --data sanjay_data [--scene path/to/upload.tif --perm path/to/perm.tif]

Writes to <data>/figures/: one PNG per scene (SAR | onboard severity | ground severity, with 90% intervals
on the flagged cells), the onboard alert JSON, the cue JSON, the Tier 2 crops and a downlink timeline.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import sanjay_core as sc  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#64707C"})
ORDER = ["no_data", "none", "low", "moderate", "severe"]
CMAP = ListedColormap([sc.SEVERITY_COLOURS[k] for k in ORDER])


def sev_raster(grid):
    nr, nc = grid["shape"]
    a = np.zeros((nr, nc))
    for c in grid["cells"]:
        a[c["row"], c["col"]] = ORDER.index(c["severity"])
    return a


def figure(title, s, perm, bundles, out: Path, label=None):
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(14, 5.4))
    ax[0].imshow(np.clip(s["vv"], -25, 0), cmap="gray")
    ax[0].set_title("Sentinel-1 VV (REAL DATA)\ndark = smooth surface / water", color="#17212B")
    alerts = {}
    for i, name in enumerate(["onboard", "ground"], start=1):
        b = bundles[name]
        pred, grid = sc.analyse(b, s, perm)
        alert = sc.build_alert(title, grid, pred, b, created_at="replay")
        alerts[name] = (pred, grid, alert)
        cp = grid["cell_px"]
        ax[i].imshow(np.clip(s["vv"], -25, 0), cmap="gray")
        ax[i].imshow(sev_raster(grid), cmap=CMAP, vmin=-0.5, vmax=4.5, alpha=0.55, interpolation="nearest",
                     extent=(0, grid["shape"][1] * cp, grid["shape"][0] * cp, 0))
        for c in grid["cells"]:
            if c["severity"] in ("moderate", "severe"):
                ax[i].text(c["px_bounds"][1] + 3, c["px_bounds"][0] + cp / 2,
                           f"{c['flood_fraction']:.0%}\n[{c['ci90'][0]:.0%}-{c['ci90'][1]:.0%}]",
                           fontsize=6, color="white", va="center")
        card = sc.model_card(b)
        acc = sc.evaluate(pred["water_full"], label)
        acc_s = ""
        if acc and label is not None and (label == 1).any():
            acc_s = f" | IoU vs hand labels {acc['iou']:.2f}"
        elif acc:  # labelled dry: IoU is undefined, report wrongly-flagged share instead
            m = label >= 0
            acc_s = f" | labelled dry: {pred['water_full'][m].mean():.1%} px marked water"
        ax[i].set_title(f"{name.upper()} model (MODEL OUTPUT)\n{card['trees']} trees · {card['n_features']} features · "
                        f"{card['input_bits']}-bit input · {card['model_size_bytes']/1e3:.0f} KB\n"
                        f"{pred['runtime_ms']:.0f} ms{acc_s}", color="#17212B", fontsize=8.5)
    for a in ax:
        a.set_xticks([]), a.set_yticks([])
    fig.legend(handles=[Patch(facecolor=sc.SEVERITY_COLOURS[k], edgecolor="#64707C", label=k.replace("_", " "))
                        for k in ORDER],
               loc="lower center", ncol=5, frameon=False)
    a = alerts["onboard"][2]
    fig.suptitle(f"Sanjay — {title}: " + ("ALERT generated" if a else "no qualifying flood, no alert"),
                 color="#315B78", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(out / f"{title}_severity.png", dpi=160)
    plt.close(fig)

    pred, grid, alert = alerts["onboard"]
    if alert:
        (out / f"{title}_alert.json").write_text(json.dumps(alert["payload"], indent=2))
        cue = sc.build_cue(alert, "replay")
        (out / f"{title}_cue.json").write_text(json.dumps(cue["payload"], indent=2))
        crops = sc.make_crops(s["vv"], pred, grid, alert)
        for c in crops:
            (out / f"{title}_crop_{c['cell_id']}.png").write_bytes(c["png"])
        items = [{"id": "Tier 1 alert", "tier": 1, "priority": 1, "size_bytes": alert["size_bytes"], "ready_at_s": 0},
                 {"id": "Cue request", "tier": 1, "priority": 1, "size_bytes": cue["size_bytes"], "ready_at_s": 0.001}]
        items += [{"id": f"Tier 2 crop {c['cell_id']}", "tier": 2, "priority": 2, "size_bytes": c["size_bytes"],
                   "ready_at_s": 0} for c in crops]
        items += [{"id": "Tier 3 raw scene", "tier": 3, "priority": 3, "size_bytes": sc.raw_scene_bytes(s["vv"]),
                   "ready_at_s": 0}]
        timeline(title, items, out)


def timeline(title, items, out, bitrate_bps=2e5, window=(0.0, 600.0)):
    """Illustrative only: 0.2 Mbps link, one 10-minute window (ASSUMPTIONS)."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 3.8), sharex=True)
    for ax, order, lab in zip(axes, ["priority", "raw_first"], ["Sanjay: alert first", "Image first"]):
        sched = sc.schedule_downlink(items, [window], bitrate_bps, order=order)
        span = max(x["end_s"] or 0 for x in sched) or 1
        for y, it in enumerate(sched):
            col = "#315B78" if it["tier"] == 1 else "#64707C" if it["tier"] == 3 else "#9DB3C4"
            if it["status"] == "delivered":
                ax.barh(y, max(it["end_s"] - it["start_s"], span * 0.006), left=it["start_s"], color=col)
                ax.text(it["end_s"] + span * 0.01, y, f"received at {it['end_s']:.2f} s", va="center",
                        fontsize=6.5, color=col)
            else:
                ax.text(span * 0.5, y, "queued (no window)", va="center", fontsize=6.5, color="#B42318")
            ax.text(-span * 0.01, y, f"{it['id']} ({it['size_bytes']:,} B)", va="center", ha="right", fontsize=6.5)
        ax.set_xlim(0, span * 1.18)
        ax.set_yticks([]), ax.set_title(lab, loc="left", fontsize=9, color="#17212B")
    axes[-1].set_xlabel("seconds into contact window (SIMULATED: 0.2 Mbps, 10-min window — assumptions)")
    fig.tight_layout()
    fig.subplots_adjust(left=0.24)
    fig.savefig(out / f"{title}_downlink.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sanjay_data")
    ap.add_argument("--scene")
    ap.add_argument("--vh")
    ap.add_argument("--perm")
    a = ap.parse_args()
    data = Path(a.data)
    bundles = {n: joblib.load(data / "models" / f"{n}.joblib") for n in ("onboard", "ground")}
    out = data / "figures"
    if a.scene:
        s = sc.load_scene(a.scene, a.vh)
        perm = rasterio.open(a.perm).read(1) > 0 if a.perm else None
        figure(Path(a.scene).stem, s, perm, bundles, out)
    else:
        cat = json.loads((data / "catalog.json").read_text())
        for role, d in cat["scenes"].items():
            cid = d["chip"]
            s = sc.load_scene(str(data / "scenes" / f"{cid}.tif"))
            pp = data / "scenes" / f"{cid}_perm.tif"
            perm = rasterio.open(pp).read(1) > 0 if pp.exists() else None
            label = rasterio.open(data / "scenes" / f"{cid}_label.tif").read(1)
            figure(f"{role}_{cid}", s, perm, bundles, out, label)
    print("figures in", out)


if __name__ == "__main__":
    main()
