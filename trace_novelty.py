#!/usr/bin/env python3
"""
trace_novelty.py — experiments that turn TRACE into a contribution-bearing (Q1/Q2) manuscript.

PATH B — Silent label corruption in mixed bbox/polygon YOLO datasets
  scan      : audit any number of YOLO datasets for mixed-format label files
  download  : (optional) fetch public Roboflow Universe datasets listed in a CSV
  impact    : train original vs cleaned labels with identical settings -> ΔmAP

PATH A — Surabaya traffic dataset + cross-domain study
  frames    : extract frames from your own videos, drop near-duplicates (dHash)
  prelabel  : pre-annotate frames with the TRACE model for human correction (CVAT/Roboflow/Label Studio)
  split     : leakage-free split grouped by video/location (70/15/15)
  crossdomain : PH->SBY, SBY->PH, joint training, all on a shared 3-class schema

report      : results/novelty_summary.md

Examples
  python trace_novelty.py scan --datasets ./data ./rf_datasets/* --out results_novelty
  python trace_novelty.py impact --datasets ./data --model yolo26m.pt --epochs 50 --seeds 0 1 2
  python trace_novelty.py frames --videos ./videos_sby --dst ./sby_raw --every 2
  python trace_novelty.py prelabel --images ./sby_raw/images --weights app/models/best.pt
  python trace_novelty.py split --src ./sby_labeled --dst ./sby_dataset --names bus car van angkot truck
  python trace_novelty.py crossdomain --ph ./data --sby ./sby_dataset --sby-names bus car van angkot truck \
         --map angkot:van truck:drop --seeds 0 1 2
  python trace_novelty.py report

Every reported number comes from a run of this script. Nothing is estimated or filled in.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import time
from datetime import datetime
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "valid", "test"]
SHARED = ["bus", "car", "van"]  # schema shared by the Philippine (TRACE) and Surabaya datasets
OUT = Path("results_novelty")

PHOTOMETRIC = dict(hsv_h=0.015, hsv_s=0.3, hsv_v=0.3, degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
                   perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0, cutmix=0.0,
                   copy_paste=0.0, erasing=0.0)
TRAIN = dict(imgsz=640, optimizer="auto", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
             warmup_epochs=3.0, patience=100, amp=True, deterministic=True, workers=8, plots=True,
             verbose=False)


# ───────────────────────────── helpers ─────────────────────────────
DEVICE = "cpu"

def auto_device(requested: str = "auto") -> str:
    """'auto' -> NVIDIA CUDA ('0'), else Apple Silicon GPU ('mps'), else 'cpu'.
    Ultralytics does NOT pick MPS by itself: without device='mps' a Mac trains on the CPU."""
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(dev: str) -> None:
    import torch
    if dev == "mps":
        torch.mps.synchronize()
    elif dev != "cpu":
        torch.cuda.synchronize()

def log(m: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no data)_\n"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
                     + ["| " + " | ".join(f(r.get(k, "")) for k in keys) + " |" for r in rows]) + "\n"


def save(name: str, rows: list[dict], title: str) -> None:
    write_csv(OUT / "tables" / f"{name}.csv", rows)
    (OUT / "tables" / f"{name}.md").write_text(f"### {title}\n\n{md_table(rows)}", encoding="utf-8")
    log(f"saved {name}")


def mean_sd(v: list[float]) -> str:
    v = [x for x in v if x is not None]
    if not v:
        return ""
    return f"{v[0]:.4f}" if len(v) == 1 else f"{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}"


def images_in(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.exists() else []


def label_rows(f: Path) -> list[list[str]]:
    return [l.split() for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def to_xywh(p: list[str]) -> tuple[int, float, float, float, float]:
    c, v = int(float(p[0])), [float(x) for x in p[1:]]
    if len(v) == 4:
        return (c, *v)
    xs, ys = v[0::2], v[1::2]
    x1, x2, y1, y2 = max(0.0, min(xs)), min(1.0, max(xs)), max(0.0, min(ys)), min(1.0, max(ys))
    return c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def read_names(root: Path) -> list[str]:
    import yaml
    for y in [root / "data.yaml", *root.glob("*.yaml")]:
        if y.exists():
            n = yaml.safe_load(y.read_text()).get("names")
            if isinstance(n, dict):
                return [n[k] for k in sorted(n)]
            if isinstance(n, list):
                return n
    raise FileNotFoundError(f"no names in data.yaml under {root}")


def write_yaml(root: Path, names: list[str], path: Path | None = None,
               train: str | list[str] = "train/images") -> Path:
    import yaml
    root = root.resolve()
    d = {"path": str(root), "train": train, "val": "valid/images", "test": "test/images",
         "nc": len(names), "names": {i: n for i, n in enumerate(names)}}
    path = path or root / "data.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False))
    return path


def link_images(src: Path, dst: Path, files: list[Path] | None = None) -> None:
    """Per-file symlinks. A symlinked directory would be resolved by Ultralytics back to the source labels."""
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for im in files if files is not None else images_in(src):
        os.symlink(im.resolve(), dst / im.name)


def build_copy(root: Path, dst: Path, names_src: list[str], names_dst: list[str],
               mapping: dict[str, str] | None = None) -> Path:
    """Copy a YOLO dataset as bbox-only labels, optionally remapping/dropping classes."""
    mapping = mapping or {}
    for s in SPLITS:
        if not (root / s / "images").exists():
            continue
        (dst / s / "labels").mkdir(parents=True, exist_ok=True)
        link_images(root / s / "images", dst / s / "images")
        for f in (root / s / "labels").glob("*.txt"):
            out = []
            for r in label_rows(f):
                c, cx, cy, w, h = to_xywh(r)
                name = mapping.get(names_src[c], names_src[c])
                if name == "drop" or name not in names_dst:
                    continue
                out.append(f"{names_dst.index(name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (dst / s / "labels" / f.name).write_text("\n".join(out) + ("\n" if out else ""))
    return write_yaml(dst, names_dst)


def evaluate(weights: str, yaml: Path, split: str, tag: str) -> dict:
    from ultralytics import YOLO
    r = YOLO(weights).val(data=str(yaml), split=split, imgsz=640, batch=16, plots=True, device=DEVICE,
                          project=str(OUT / "runs" / "val"), name=tag, exist_ok=True, verbose=False)
    p, rc = float(r.box.mp), float(r.box.mr)
    res = {"precision": p, "recall": rc, "F1": 2 * p * rc / (p + rc) if p + rc else 0.0,
           "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map)}
    for i, c in enumerate(r.box.ap_class_index):
        res[f"AP50_95_{r.names[int(c)]}"] = float(r.box.class_result(i)[3])
    return res


def train(model: str, yaml: Path, tag: str, seed: int, epochs: int, batch: int) -> str:
    from ultralytics import YOLO
    run = OUT / "runs" / "train" / tag
    best = run / "weights" / "best.pt"
    if (run / "DONE").exists() and best.exists():
        log(f"skip {tag} (done)")
        return str(best)
    log(f"train {tag}: {model}, epochs={epochs}, seed={seed}")
    m = YOLO(model)
    m.train(data=str(yaml), project=str(run.parent), name=run.name, exist_ok=True, epochs=epochs,
            batch=batch, seed=seed, device=DEVICE, **TRAIN, **PHOTOMETRIC)
    (run / "DONE").write_text(datetime.now().isoformat())
    return str(Path(m.trainer.best))


# ───────────────────────────── PATH B ─────────────────────────────
def audit_dataset(root: Path) -> dict:
    files = mixed = poly_only = box_rows = poly_rows = corrupted = 0
    for s in SPLITS:
        for f in (root / s / "labels").glob("*.txt"):
            files += 1
            lens = [len(r) for r in label_rows(f)]
            b, p = sum(n == 5 for n in lens), sum(n > 6 for n in lens)
            box_rows += b
            poly_rows += p
            if p and b:
                mixed += 1
                corrupted += b  # bbox rows Ultralytics reads as 2-point polygons
            elif p:
                poly_only += 1
    return {"dataset": root.name, "label_files": files, "bbox_rows": box_rows, "polygon_rows": poly_rows,
            "polygon_only_files": poly_only, "mixed_files": mixed,
            "pct_mixed_files": round(100 * mixed / files, 2) if files else 0.0,
            "corrupted_bbox_rows": corrupted,
            "pct_corrupted_rows": round(100 * corrupted / max(box_rows + poly_rows, 1), 2)}


def cmd_scan(a) -> None:
    rows = [audit_dataset(Path(d)) for d in a.datasets if (Path(d) / "train" / "labels").exists()]
    affected = sum(r["mixed_files"] > 0 for r in rows)
    rows.append({"dataset": f"TOTAL ({affected}/{len(rows)} affected)",
                 **{k: sum(r[k] for r in rows) for k in ["label_files", "bbox_rows", "polygon_rows",
                                                         "polygon_only_files", "mixed_files",
                                                         "corrupted_bbox_rows"]}})
    save("B1_label_integrity_scan", rows, "B1. Mixed bbox/polygon label files across datasets")


def cmd_download(a) -> None:
    """CSV columns: workspace,project,version. Needs ROBOFLOW_API_KEY in the environment (never hard-code it)."""
    from roboflow import Roboflow
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("set ROBOFLOW_API_KEY as an environment variable first")
    rf = Roboflow(api_key=key)
    dst = Path(a.dst)
    for row in csv.DictReader(open(a.list, encoding="utf-8")):
        loc = dst / f"{row['workspace']}__{row['project']}__v{row['version']}"
        if loc.exists():
            continue
        try:
            rf.workspace(row["workspace"]).project(row["project"]).version(int(row["version"])).download(
                "yolov8", location=str(loc))
        except Exception as e:  # noqa: BLE001
            log(f"download failed {row}: {e}")


def cmd_impact(a) -> None:
    """Same model, same seed, same settings: original labels vs cleaned labels, both scored on CLEAN test labels."""
    rows = []
    for d in map(Path, a.datasets):
        names = read_names(d)
        info = audit_dataset(d)
        if info["mixed_files"] == 0 and not a.force:
            log(f"{d.name}: no mixed files, skipped (use --force to run anyway)")
            continue
        orig_yaml = write_yaml(d, names, OUT / "yaml" / f"{d.name}_original.yaml")
        clean_yaml = build_copy(d, OUT / "clean" / d.name, names, names)
        for seed in a.seeds:
            rec = {"dataset": d.name, "seed": seed, "mixed_files": info["mixed_files"]}
            for variant, y in [("original", orig_yaml), ("cleaned", clean_yaml)]:
                best = train(a.model, y, f"impact_{d.name}_{variant}_s{seed}", seed, a.epochs, a.batch)
                m = evaluate(best, clean_yaml, "test", f"impact_{d.name}_{variant}_s{seed}")
                rec.update({f"{variant}_{k}": v for k, v in m.items()})
            rec["delta_mAP50_95"] = rec["cleaned_mAP50_95"] - rec["original_mAP50_95"]
            rec["delta_mAP50"] = rec["cleaned_mAP50"] - rec["original_mAP50"]
            rows.append(rec)
            write_csv(OUT / "tables" / "B2_impact_raw.csv", rows)
    summary = []
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        rs = [r for r in rows if r["dataset"] == ds]
        summary.append({"dataset": ds, "n_seeds": len(rs), "mixed_files": rs[0]["mixed_files"],
                        "mAP50_95_original": mean_sd([r["original_mAP50_95"] for r in rs]),
                        "mAP50_95_cleaned": mean_sd([r["cleaned_mAP50_95"] for r in rs]),
                        "delta_mAP50_95": mean_sd([r["delta_mAP50_95"] for r in rs]),
                        "delta_mAP50": mean_sd([r["delta_mAP50"] for r in rs])})
        if len(rs) >= 2:
            try:
                from scipy.stats import ttest_rel
                t = ttest_rel([r["cleaned_mAP50_95"] for r in rs], [r["original_mAP50_95"] for r in rs])
                summary[-1]["paired_t_p"] = round(float(t.pvalue), 4)
            except Exception:  # noqa: BLE001
                pass
    save("B2_label_fix_impact", summary,
         "B2. Effect of fixing mixed-format labels (same model/seed/settings; evaluated on cleaned test labels)")


# ───────────────────────────── PATH A ─────────────────────────────
def dhash(img, size: int = 16) -> int:
    import cv2
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size + 1, size))
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def cmd_frames(a) -> None:
    import cv2
    dst = Path(a.dst) / "images"
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    vids = [p for p in sorted(Path(a.videos).rglob("*")) if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step, i, kept, last = max(1, int(round(fps * a.every))), 0, 0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                h = dhash(frame)
                # compare with the last KEPT frame only: a static CCTV background must not
                # cause frames with different vehicles to be discarded
                if last is None or bin(h ^ last).count("1") > a.hamming:
                    last = h
                    name = f"{v.stem}_{int(i / fps):06d}.jpg"
                    cv2.imwrite(str(dst / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    kept += 1
            i += 1
        cap.release()
        rows.append({"video": v.name, "group": v.stem, "fps": round(fps, 2), "frames_read": i,
                     "frames_kept": kept, "duration_s": round(i / fps, 1)})
    save("A1_frame_extraction", rows, f"A1. Frame extraction (every {a.every}s, dHash Hamming > {a.hamming})")


def cmd_prelabel(a) -> None:
    """Model-assisted pre-annotation. Every box MUST be checked and corrected by a human afterwards."""
    from ultralytics import YOLO
    m = YOLO(a.weights)
    imgs = images_in(Path(a.images))
    lbl = Path(a.images).parent / "labels_prelabel"
    lbl.mkdir(parents=True, exist_ok=True)
    target = a.names
    n_box = 0
    for k in range(0, len(imgs), 32):
        for r in m.predict([str(p) for p in imgs[k:k + 32]], conf=a.conf, imgsz=640, device=DEVICE, verbose=False):
            out = []
            for (cx, cy, w, h), c in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist()):
                n = r.names[int(c)]
                if n in target:
                    out.append(f"{target.index(n)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (lbl / f"{Path(r.path).stem}.txt").write_text("\n".join(out) + ("\n" if out else ""))
            n_box += len(out)
    (lbl / "classes.txt").write_text("\n".join(target) + "\n")
    log(f"pre-labels: {len(imgs)} images, {n_box} boxes -> {lbl}. Import into CVAT/Label Studio/Roboflow and correct.")


def cmd_split(a) -> None:
    """Grouped split: all frames of one video/location stay in the same split (prevents leakage)."""
    src, dst = Path(a.src), Path(a.dst)
    imgs = images_in(src / "images")
    groups: dict[str, list[Path]] = {}
    for im in imgs:
        g = im.stem.rsplit("_", 1)[0] if a.group_by == "prefix" else im.stem
        groups.setdefault(g, []).append(im)
    keys = sorted(groups)
    random.Random(a.seed).shuffle(keys)
    total, acc, assign = len(imgs), 0, {}
    for g in keys:
        frac = acc / max(total, 1)
        assign[g] = "train" if frac < a.ratios[0] else "valid" if frac < a.ratios[0] + a.ratios[1] else "test"
        acc += len(groups[g])
    rows = []
    for s in SPLITS:
        files = [im for g in keys if assign[g] == s for im in groups[g]]
        (dst / s / "labels").mkdir(parents=True, exist_ok=True)
        link_images(src / "images", dst / s / "images", files)
        cnt = {n: 0 for n in a.names}
        for im in files:
            f = src / "labels" / f"{im.stem}.txt"
            if f.exists():
                shutil.copy2(f, dst / s / "labels" / f.name)
                for r in label_rows(f):
                    cnt[a.names[int(r[0])]] += 1
        rows.append({"split": s, "groups": sum(assign[g] == s for g in keys), "images": len(files),
                     "instances": sum(cnt.values()), **cnt})
    for r in rows:
        if r["images"] == 0:
            log(f"WARNING: split '{r['split']}' is empty — record more videos/locations (need >= 3 groups)")
    write_yaml(dst, a.names)
    write_csv(dst / "split_manifest.csv", [{"group": g, "split": assign[g], "images": len(groups[g])} for g in keys])
    save("A2_sby_dataset_composition", rows, "A2. Surabaya dataset composition (grouped by video/location)")


def cmd_crossdomain(a) -> None:
    ph, sby = Path(a.ph), Path(a.sby)
    mapping = dict(m.split(":") for m in a.map)
    ph_y = build_copy(ph, OUT / "shared" / "ph", read_names(ph), SHARED)
    sby_y = build_copy(sby, OUT / "shared" / "sby", a.sby_names, SHARED, mapping)
    joint_y = write_yaml(OUT / "shared" / "ph", SHARED, OUT / "yaml" / "joint.yaml",
                         train=[str((OUT / "shared" / "ph" / "train" / "images").resolve()),
                                str((OUT / "shared" / "sby" / "train" / "images").resolve())])
    rows = []
    for seed in a.seeds:
        models = {"PH-trained": train(a.model, ph_y, f"cd_ph_s{seed}", seed, a.epochs, a.batch),
                  "SBY-trained": train(a.model, sby_y, f"cd_sby_s{seed}", seed, a.epochs, a.batch),
                  "Joint (PH+SBY)": train(a.model, joint_y, f"cd_joint_s{seed}", seed, a.epochs, a.batch)}
        if a.include_deployed and seed == a.seeds[0]:
            models["Deployed TRACE model"] = a.weights
        for mname, w in models.items():
            for tname, y in [("PH test", ph_y), ("SBY test", sby_y)]:
                rows.append({"model": mname, "test_set": tname, "seed": seed,
                             **evaluate(w, y, "test", f"cd_{mname[:3]}_{tname[:3]}_s{seed}".replace(" ", ""))})
        write_csv(OUT / "tables" / "A3_crossdomain_raw.csv", rows)
    summary = []
    for mname in dict.fromkeys(r["model"] for r in rows):
        for tname in ["PH test", "SBY test"]:
            rs = [r for r in rows if r["model"] == mname and r["test_set"] == tname]
            summary.append({"model": mname, "test_set": tname, "n_seeds": len(rs),
                            **{k: mean_sd([r.get(k) for r in rs]) for k in
                               ["precision", "recall", "F1", "mAP50", "mAP50_95"] +
                               [f"AP50_95_{c}" for c in SHARED]}})
    save("A3_crossdomain", summary, f"A3. Cross-domain generalisation on the shared schema {SHARED} "
                                    f"(class mapping: {mapping or 'none'})")


def cmd_report(a) -> None:
    parts = [f"# TRACE novelty experiments\n\nGenerated {datetime.now():%Y-%m-%d %H:%M}\n"]
    parts += [p.read_text(encoding="utf-8") for p in sorted((OUT / "tables").glob("*.md"))]
    (OUT / "novelty_summary.md").write_text("\n".join(parts), encoding="utf-8")
    import zipfile
    zp = OUT.parent / f"{OUT.name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            if p.is_file() and not p.is_symlink() and p.suffix not in {".pt", ".cache"} \
                    and "clean" not in p.parts and "shared" not in p.parts:
                z.write(p, p.relative_to(OUT.parent))
    log(f"report: {OUT / 'novelty_summary.md'} | {zp}")


# ───────────────────────────── main ─────────────────────────────
def main() -> None:
    global OUT, DEVICE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results_novelty")
    ap.add_argument("--device", default="auto", help="auto | 0 (NVIDIA) | mps (Apple Silicon) | cpu")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_train(p):
        p.add_argument("--model", default="yolo26m.pt")
        p.add_argument("--epochs", type=int, default=50)
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--seeds", nargs="+", type=int, default=[0])

    p = sub.add_parser("scan"); p.add_argument("--datasets", nargs="+", required=True)
    p = sub.add_parser("download"); p.add_argument("--list", required=True); p.add_argument("--dst", default="rf_datasets")
    p = sub.add_parser("impact"); p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--force", action="store_true"); common_train(p)
    p = sub.add_parser("frames"); p.add_argument("--videos", required=True); p.add_argument("--dst", default="sby_raw")
    p.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
    p.add_argument("--hamming", type=int, default=10, help="min 256-bit dHash distance to the last kept frame")
    p = sub.add_parser("prelabel"); p.add_argument("--images", required=True)
    p.add_argument("--weights", default="app/models/best.pt"); p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--names", nargs="+", default=SHARED)
    p = sub.add_parser("split"); p.add_argument("--src", required=True); p.add_argument("--dst", default="sby_dataset")
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15])
    p.add_argument("--group-by", choices=["prefix", "none"], default="prefix")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("crossdomain"); p.add_argument("--ph", required=True); p.add_argument("--sby", required=True)
    p.add_argument("--sby-names", nargs="+", required=True)
    p.add_argument("--map", nargs="*", default=[], help="e.g. angkot:van truck:drop")
    p.add_argument("--weights", default="app/models/best.pt")
    p.add_argument("--include-deployed", action="store_true"); common_train(p)
    sub.add_parser("report")

    a = ap.parse_args()
    if a.cmd in {"impact", "crossdomain", "prelabel"}:
        DEVICE = auto_device(a.device)
        log(f"device: {DEVICE}")
    OUT = Path(a.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    globals()[f"cmd_{a.cmd}"](a)
    log(f"{a.cmd} finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
