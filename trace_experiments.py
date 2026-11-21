#!/usr/bin/env python3
"""
trace_experiments.py — complete experiment runner for the TRACE / VehicleScope AI manuscript.

Produces every missing table and figure in one place:
  audit     : label-format audit (bbox vs polygon, mixed files) + cleaned dataset copy
  env       : T5  hardware & software environment
  eval      : T6  overall val/test metrics, T7 per-class metrics, F8 confusion matrix, F9 PR curves
  train     : T8  baselines (YOLO26m, YOLOv8m, YOLO11m, RT-DETR-l) and
              T9  augmentation ablation (photometric / default / photometric+light-geometric), multi-seed
  latency   : T10 inference latency & throughput (GPU, CPU, optional n8n end-to-end)
  errors    : T11 TIDE-style error taxonomy + F10 qualitative success/failure figures
  report    : collects everything into results/summary.md + results.zip

Usage (Google Colab, GPU runtime):
  !pip install -U ultralytics
  !python trace_experiments.py --data-root /content/data --weights app/models/best.pt --stages audit env eval latency errors report
  # long runs (hours) — start with one seed:
  !python trace_experiments.py --data-root /content/data --stages train report --seeds 0
  # later add seeds 1 2 (finished runs are skipped automatically):
  !python trace_experiments.py --data-root /content/data --stages train report --seeds 0 1 2

--data-root must contain train/ valid/ test/ each with images/ and labels/.
Nothing is invented: every number in the output comes from a run of this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

CLASS_NAMES = ["bus", "car", "van"]
SPLITS = ["train", "valid", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Training configuration copied from the deployed checkpoint (best.pt train_args)
BASE_TRAIN = dict(epochs=50, imgsz=640, batch=16, optimizer="auto", lr0=0.01, lrf=0.01,
                  momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, patience=100,
                  amp=True, deterministic=True, workers=8, close_mosaic=10, cache=False)

AUG_PHOTOMETRIC = dict(hsv_h=0.015, hsv_s=0.3, hsv_v=0.3, degrees=0.0, translate=0.0, scale=0.0,
                       shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0,
                       cutmix=0.0, copy_paste=0.0, erasing=0.0)
AUG_DEFAULT = {}  # Ultralytics defaults
AUG_PHOTO_GEO = {**AUG_PHOTOMETRIC, "fliplr": 0.5, "translate": 0.1, "scale": 0.25}

BASELINES = {  # name -> (weights, loader)
    "YOLO26m": ("yolo26m.pt", "YOLO"),
    "YOLOv8m": ("yolov8m.pt", "YOLO"),
    "YOLO11m": ("yolo11m.pt", "YOLO"),
    "RT-DETR-l": ("rtdetr-l.pt", "RTDETR"),
}
ABLATIONS = {
    "photometric_only": AUG_PHOTOMETRIC,
    "ultralytics_default": AUG_DEFAULT,
    "photometric_plus_light_geometric": AUG_PHOTO_GEO,
}


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

def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def md_table(rows: list[dict], floatfmt: str = "{:.4f}") -> str:
    if not rows:
        return "_(no data)_\n"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    fmt = lambda v: floatfmt.format(v) if isinstance(v, float) else str(v)
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    out += ["| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |" for r in rows]
    return "\n".join(out) + "\n"


def save_table(out: Path, name: str, rows: list[dict], title: str) -> None:
    write_csv(out / "tables" / f"{name}.csv", rows)
    (out / "tables" / f"{name}.md").write_text(f"### {title}\n\n{md_table(rows)}", encoding="utf-8")
    log(f"saved table {name}")


def mean_sd(vals: list[float]) -> str:
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"


def images_in(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.exists() else []


def write_yaml(root: Path, path: Path) -> Path:
    root = root.resolve()
    lines = [f"path: {root}", "train: train/images", "val: valid/images", "test: test/images",
             f"nc: {len(CLASS_NAMES)}", "names:"] + [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    path.write_text("\n".join(lines) + "\n")
    return path


def read_label_rows(f: Path) -> list[list[str]]:
    return [l.split() for l in f.read_text().splitlines() if l.strip()]


def row_to_xywh(p: list[str]) -> tuple[int, float, float, float, float]:
    """Convert a YOLO bbox row (5 values) or polygon row (>5 values) to (cls, cx, cy, w, h)."""
    c = int(float(p[0]))
    v = [float(x) for x in p[1:]]
    if len(v) == 4:
        return (c, *v)
    xs, ys = v[0::2], v[1::2]
    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
    return c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


# ───────────────────────────── stage: audit ─────────────────────────────
def stage_audit(args, out: Path) -> Path:
    """Count label formats, detect mixed bbox/polygon files, write a cleaned bbox-only copy."""
    root = Path(args.data_root)
    clean = out / "data_clean"
    comp, fmt, mixed_rows = [], [], []
    for s in SPLITS:
        imgs = images_in(root / s / "images")
        lbl_dir = root / s / "labels"
        cnt = {n: 0 for n in CLASS_NAMES}
        nbox = npoly = empty = nmixed = mixed_bbox_rows = 0
        (clean / s / "labels").mkdir(parents=True, exist_ok=True)
        for f in sorted(lbl_dir.glob("*.txt")):
            rows = read_label_rows(f)
            if not rows:
                empty += 1
            lens = [len(r) for r in rows]
            is_mixed = any(n > 6 for n in lens) and any(n == 5 for n in lens)
            if is_mixed:
                nmixed += 1
                mixed_bbox_rows += sum(n == 5 for n in lens)
                mixed_rows.append({"split": s, "file": f.name,
                                   "bbox_rows": sum(n == 5 for n in lens),
                                   "polygon_rows": sum(n > 6 for n in lens)})
            conv = []
            for r in rows:
                if len(r) == 5:
                    nbox += 1
                else:
                    npoly += 1
                c, cx, cy, w, h = row_to_xywh(r)
                cnt[CLASS_NAMES[c]] += 1
                conv.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (clean / s / "labels" / f.name).write_text("\n".join(conv) + ("\n" if conv else ""))
        # Per-file symlinks (no copy). A symlinked *directory* would not work: Ultralytics calls
        # Path.resolve() on the split folder (data/utils.py) and would read the ORIGINAL labels again.
        link = clean / s / "images"
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            shutil.rmtree(link)
        link.mkdir(parents=True)
        for im in imgs:
            os.symlink(im.resolve(), link / im.name)
        total = sum(cnt.values())
        comp.append({"split": s, "images": len(imgs), "instances": total, **cnt,
                     "instances_per_image": round(total / max(len(imgs), 1), 3),
                     "empty_label_files": empty})
        fmt.append({"split": s, "bbox_rows": nbox, "polygon_rows": npoly,
                    "mixed_files": nmixed, "bbox_rows_in_mixed_files": mixed_bbox_rows})
    tot_img = sum(r["images"] for r in comp)
    for r in comp:
        r["pct_images"] = round(100 * r["images"] / max(tot_img, 1), 2)
    save_table(out, "T2_dataset_composition", comp, "T2. Dataset composition per split")
    save_table(out, "audit_label_formats", fmt, "Label-format audit (bbox vs polygon)")
    write_csv(out / "tables" / "audit_mixed_files.csv", mixed_rows)
    write_yaml(clean, clean / "data.yaml")
    write_yaml(root, out / "data_original.yaml")
    n_mixed = sum(r["mixed_files"] for r in fmt)
    log(f"audit done — mixed files: {n_mixed}. "
        + ("Ultralytics would mis-read bbox rows in these files; use the cleaned dataset."
           if n_mixed else "No mixed files; cleaned copy is equivalent to the original."))
    return clean / "data.yaml"


def data_yaml(args, out: Path, which: str = "clean") -> Path:
    p = out / "data_clean" / "data.yaml" if which == "clean" else out / "data_original.yaml"
    if not p.exists():
        stage_audit(args, out)
    return p


# ───────────────────────────── stage: env ─────────────────────────────
def stage_env(args, out: Path) -> None:
    import torch
    import ultralytics
    info = {"python": platform.python_version(), "os": platform.platform(),
            "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    info["device_used"] = DEVICE
    info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if platform.system() == "Darwin":
        try:
            import subprocess
            info["mac_chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                              capture_output=True, text=True).stdout.strip()
            info["mac_ram_gb"] = round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                                          text=True).stdout.strip()) / 1024 ** 3, 1)
        except Exception:  # noqa: BLE001
            pass
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info.update(gpu=prop.name, gpu_memory_gb=round(prop.total_memory / 1024 ** 3, 1),
                    gpu_count=torch.cuda.device_count())
    try:
        cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")]
        info.update(cpu=cpu[0] if cpu else platform.processor(), cpu_threads=len(cpu) or os.cpu_count())
    except OSError:
        info.update(cpu=platform.processor(), cpu_threads=os.cpu_count())
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        pass
    save_table(out, "T5_environment", [{"item": k, "value": v} for k, v in info.items()],
               "T5. Hardware and software environment")


# ───────────────────────────── stage: eval ─────────────────────────────
def evaluate(weights: str, yaml: Path, split: str, out: Path, tag: str, loader: str = "YOLO") -> dict:
    import ultralytics
    Model = getattr(ultralytics, loader)
    m = Model(weights)
    r = m.val(data=str(yaml), split=split, imgsz=640, batch=16, plots=True, device=DEVICE,
              project=str(out / "runs" / "val"), name=tag, exist_ok=True, verbose=False)
    names = r.names
    overall = {"precision": float(r.box.mp), "recall": float(r.box.mr),
               "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map)}
    p, rc = overall["precision"], overall["recall"]
    overall["F1"] = 2 * p * rc / (p + rc) if p + rc else 0.0
    per_class = []
    for i, c in enumerate(r.box.ap_class_index):
        cp, cr, a50, a = r.box.class_result(i)
        per_class.append({"class": names[int(c)], "precision": float(cp), "recall": float(cr),
                          "F1": float(2 * cp * cr / (cp + cr)) if cp + cr else 0.0,
                          "AP50": float(a50), "AP50_95": float(a)})
    cm = getattr(r, "confusion_matrix", None)
    return {"overall": overall, "per_class": per_class, "speed": dict(r.speed),
            "save_dir": str(r.save_dir), "cm": cm.matrix.tolist() if cm is not None else None,
            "names": [names[i] for i in sorted(names)]}


def stage_eval(args, out: Path) -> None:
    rows6, rows7 = [], []
    for which in ["clean", "original"]:
        yaml = data_yaml(args, out, which)
        for split in ["val", "test"]:
            tag = f"deployed_{which}_{split}"
            log(f"evaluating {args.weights} on {split} ({which} labels)")
            res = evaluate(args.weights, yaml, split, out, tag)
            rows6.append({"labels": which, "split": split, **res["overall"],
                          **{f"{k}_ms": round(v, 2) for k, v in res["speed"].items()}})
            if which == "clean":
                for pc in res["per_class"]:
                    rows7.append({"split": split, **pc})
                fig = out / "figures"
                fig.mkdir(parents=True, exist_ok=True)
                for png in Path(res["save_dir"]).glob("*.png"):
                    if split == "test" and ("confusion" in png.name or "curve" in png.name.lower()):
                        shutil.copy2(png, fig / f"{'F8' if 'confusion' in png.name else 'F9'}_{png.name}")
                if res["cm"] is not None:
                    labels = res["names"] + ["background"]
                    write_csv(out / "tables" / f"confusion_matrix_{split}_raw.csv",
                              [{"predicted\\true": labels[i], **{labels[j]: v for j, v in enumerate(row)}}
                               for i, row in enumerate(res["cm"])])
    save_table(out, "T6_overall_performance", rows6,
               "T6. Overall detection performance of the deployed model (clean = polygon labels converted to boxes)")
    save_table(out, "T7_per_class_performance", rows7, "T7. Per-class performance (clean labels)")


# ───────────────────────────── stage: train ─────────────────────────────
def train_one(name: str, weights: str, loader: str, aug: dict, seed: int, yaml: Path, out: Path,
              epochs: int, batch: int) -> dict | None:
    import ultralytics
    run_dir = out / "runs" / "train" / f"{name}_s{seed}"
    done = run_dir / "test_metrics.json"
    if done.exists():
        log(f"skip {name} seed {seed} (already done)")
        return json.loads(done.read_text())
    Model = getattr(ultralytics, loader)
    model = Model(weights)
    cfg = {**BASE_TRAIN, **aug, "epochs": epochs, "batch": batch, "seed": seed}
    log(f"training {name} seed {seed}: {weights}, epochs={epochs}, batch={batch}")
    t0 = time.time()
    model.train(data=str(yaml), project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
                plots=True, verbose=False, device=DEVICE, **cfg)
    train_min = (time.time() - t0) / 60
    best = Path(model.trainer.best)
    res = evaluate(str(best), yaml, "test", out, f"{name}_s{seed}_test", loader)
    params = sum(p.numel() for p in Model(str(best)).model.parameters())
    rec = {"name": name, "seed": seed, "weights": weights, **res["overall"],
           "per_class": res["per_class"], "inference_ms": res["speed"].get("inference"),
           "params_M": round(params / 1e6, 2), "train_minutes": round(train_min, 1),
           "best_epoch_weights": str(best)}
    done.write_text(json.dumps(rec, indent=2))
    return rec


def aggregate(recs: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in recs:
        groups.setdefault(r[key], []).append(r)
    rows = []
    for k, rs in groups.items():
        row = {key: k, "n_seeds": len(rs)}
        for m in ["precision", "recall", "F1", "mAP50", "mAP50_95", "inference_ms"]:
            row[m] = mean_sd([float(r[m]) for r in rs if r.get(m) is not None])
        row["params_M"] = rs[0].get("params_M")
        row["train_minutes"] = mean_sd([float(r["train_minutes"]) for r in rs])
        for cls in CLASS_NAMES:
            row[f"AP50_95_{cls}"] = mean_sd([pc["AP50_95"] for r in rs for pc in r["per_class"]
                                             if pc["class"] == cls])
        rows.append(row)
    return rows


def stage_train(args, out: Path) -> None:
    yaml = data_yaml(args, out, "clean")
    base, abl = [], []
    for seed in args.seeds:
        for name in args.models:
            w, loader = BASELINES[name]
            r = train_one(name, w, loader, AUG_PHOTOMETRIC, seed, yaml, out, args.epochs, args.batch)
            if r:
                base.append(r)
                if name == "YOLO26m":
                    abl.append({**r, "augmentation": "photometric_only"})
        if not args.skip_ablation:
            for aname, aug in ABLATIONS.items():
                if aname == "photometric_only":
                    continue  # identical to the YOLO26m baseline run above
                r = train_one(f"YOLO26m_{aname}", "yolo26m.pt", "YOLO", aug, seed, yaml, out,
                              args.epochs, args.batch)
                if r:
                    abl.append({**r, "augmentation": aname})
    save_table(out, "T8_baseline_comparison", aggregate(base, "name"),
               "T8. Comparison with baseline detectors on the test set (mean ± SD over seeds)")
    if abl:
        save_table(out, "T9_augmentation_ablation", aggregate(abl, "augmentation"),
                   "T9. Augmentation ablation for YOLO26m on the test set (mean ± SD over seeds)")


# ───────────────────────────── stage: latency ─────────────────────────────
def stage_latency(args, out: Path) -> None:
    import numpy as np
    import torch
    from ultralytics import YOLO
    imgs = images_in(Path(args.data_root) / "test" / "images")
    rows = []
    devices = ([DEVICE] if DEVICE != "cpu" else []) + ["cpu"]
    for dev in devices:
        n = args.latency_n if dev != "cpu" else min(args.latency_n, 50)
        sample = imgs[:n]
        m = YOLO(args.weights)
        for im in sample[:5]:  # warm-up
            m.predict(str(im), device=dev, imgsz=640, conf=0.40, verbose=False)
        wall, pre, inf, post = [], [], [], []
        for im in sample:
            sync(dev)
            t = time.perf_counter()
            r = m.predict(str(im), device=dev, imgsz=640, conf=0.40, verbose=False)[0]
            sync(dev)
            wall.append((time.perf_counter() - t) * 1000)
            pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
        w = np.array(wall)
        rows.append({"setting": {"cpu": "CPU", "mps": "Apple GPU (MPS)"}.get(dev, "NVIDIA GPU (CUDA)"), "images": len(sample),
                     "preprocess_ms": round(float(np.mean(pre)), 2),
                     "inference_ms": round(float(np.mean(inf)), 2),
                     "postprocess_ms": round(float(np.mean(post)), 2),
                     "end_to_end_mean_ms": round(float(w.mean()), 2),
                     "end_to_end_median_ms": round(float(np.median(w)), 2),
                     "end_to_end_p95_ms": round(float(np.percentile(w, 95)), 2),
                     "throughput_fps": round(1000 / float(w.mean()), 2)})
    if args.n8n_url:
        import requests
        m = YOLO(args.weights)
        wall, ok = [], 0
        for im in imgs[:min(30, len(imgs))]:
            t = time.perf_counter()
            r = m.predict(str(im), imgsz=640, conf=0.40, device=DEVICE, verbose=False)[0]
            counts = {n: 0 for n in CLASS_NAMES}
            for c in r.boxes.cls.tolist():
                counts[r.names[int(c)]] += 1
            confs = r.boxes.conf.tolist()
            payload = {"timestamp": datetime.now().astimezone().isoformat(), "image_name": im.name,
                       "detections": counts, "total_vehicles": sum(counts.values()),
                       "avg_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
                       "alert": sum(counts.values()) > 10, "source": "latency_benchmark"}
            try:
                resp = requests.post(args.n8n_url, json=payload, timeout=15)
                ok += resp.ok
            except requests.RequestException:
                pass
            wall.append((time.perf_counter() - t) * 1000)
        w = np.array(wall)
        rows.append({"setting": "Inference + n8n webhook round-trip", "images": len(wall),
                     "end_to_end_mean_ms": round(float(w.mean()), 2),
                     "end_to_end_median_ms": round(float(np.median(w)), 2),
                     "end_to_end_p95_ms": round(float(np.percentile(w, 95)), 2),
                     "throughput_fps": round(1000 / float(w.mean()), 2),
                     "webhook_success": f"{ok}/{len(wall)}"})
    m = YOLO(args.weights)
    try:
        layers, params, _, gflops = m.info(verbose=True)
        rows.append({"setting": "Model size", "params_M": round(params / 1e6, 2),
                     "GFLOPs": round(float(gflops), 1),
                     "weights_MB": round(Path(args.weights).stat().st_size / 1024 ** 2, 1)})
    except Exception as e:  # noqa: BLE001
        log(f"model.info failed: {e}")
    save_table(out, "T10_latency", rows,
               "T10. Inference latency and throughput (Streamlit Cloud latency must be measured in the browser)")


# ───────────────────────────── stage: errors ─────────────────────────────
def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def size_bucket(box) -> str:
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def match_image(gts: list[dict], preds: list[dict], t_fg: float = 0.5, t_bg: float = 0.1) -> list[dict]:
    """TIDE-style assignment (Bolya et al., 2020). gts/preds: dicts with cls, box (xyxy px), [conf]."""
    events, matched, explained = [], set(), set()
    for p in sorted(preds, key=lambda d: -d["conf"]):
        ious = [iou(p["box"], g["box"]) for g in gts]
        same = [(v, j) for j, v in enumerate(ious) if gts[j]["cls"] == p["cls"]]
        other = [(v, j) for j, v in enumerate(ious) if gts[j]["cls"] != p["cls"]]
        best_same_free = max([(v, j) for v, j in same if j not in matched], default=(0.0, -1))
        best_same = max(same, default=(0.0, -1))
        best_other = max(other, default=(0.0, -1))
        best_any = max(ious, default=0.0)
        ev = {"pred_cls": p["cls"], "conf": round(p["conf"], 3)}
        if best_same_free[0] >= t_fg:
            j = best_same_free[1]; matched.add(j)
            events.append({**ev, "type": "TP", "gt_cls": gts[j]["cls"], "iou": round(best_same_free[0], 3),
                           "size": size_bucket(gts[j]["box"])})
        elif best_same[0] >= t_fg:
            j = best_same[1]
            events.append({**ev, "type": "Duplicate", "gt_cls": gts[j]["cls"], "iou": round(best_same[0], 3),
                           "size": size_bucket(gts[j]["box"])})
        elif best_other[0] >= t_fg:
            j = best_other[1]; explained.add(j)
            events.append({**ev, "type": "Classification", "gt_cls": gts[j]["cls"], "iou": round(best_other[0], 3),
                           "size": size_bucket(gts[j]["box"])})
        elif best_same[0] >= t_bg:
            j = best_same[1]; explained.add(j)
            events.append({**ev, "type": "Localization", "gt_cls": gts[j]["cls"], "iou": round(best_same[0], 3),
                           "size": size_bucket(gts[j]["box"])})
        elif best_any >= t_bg:
            j = best_other[1]; explained.add(j)
            events.append({**ev, "type": "Both (cls+loc)", "gt_cls": gts[j]["cls"], "iou": round(best_other[0], 3),
                           "size": size_bucket(gts[j]["box"])})
        else:
            events.append({**ev, "type": "Background", "gt_cls": "", "iou": round(best_any, 3),
                           "size": size_bucket(p["box"])})
    for j, g in enumerate(gts):
        if j not in matched and j not in explained:
            events.append({"type": "Missed", "gt_cls": g["cls"], "pred_cls": "", "conf": "", "iou": "",
                           "size": size_bucket(g["box"])})
    return events


def load_gt(lbl: Path, w: int, h: int) -> list[dict]:
    gts = []
    if lbl.exists():
        for r in read_label_rows(lbl):
            c, cx, cy, bw, bh = row_to_xywh(r)
            gts.append({"cls": CLASS_NAMES[c], "box": [(cx - bw / 2) * w, (cy - bh / 2) * h,
                                                       (cx + bw / 2) * w, (cy + bh / 2) * h]})
    return gts


def draw(img, gts, preds, events):
    import cv2
    im = img.copy()
    for g in gts:
        x1, y1, x2, y2 = map(int, g["box"])
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(im, f"GT {g['cls']}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    bad = sum(e["type"] not in ("TP",) for e in events)
    for p in preds:
        x1, y1, x2, y2 = map(int, p["box"])
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 120, 0), 2)
        cv2.putText(im, f"{p['cls']} {p['conf']:.2f}", (x1, min(im.shape[0] - 4, y2 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)
    cv2.putText(im, f"errors: {bad}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return im


def stage_errors(args, out: Path) -> None:
    import cv2
    from ultralytics import YOLO
    root = out / "data_clean" / "test"
    if not root.exists():
        stage_audit(args, out)
    m = YOLO(args.weights)
    all_events, per_image = [], []
    for im in images_in(root / "images"):
        r = m.predict(str(im), imgsz=640, conf=args.conf, device=DEVICE, verbose=False)[0]
        h, w = r.orig_shape
        gts = load_gt(root / "labels" / f"{im.stem}.txt", w, h)
        preds = [{"cls": r.names[int(c)], "conf": float(s), "box": b}
                 for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        ev = match_image(gts, preds)
        for e in ev:
            all_events.append({"image": im.name, **e})
        n_err = sum(e["type"] != "TP" for e in ev)
        per_image.append({"image": str(im), "n_gt": len(gts), "n_err": n_err,
                          "n_tp": sum(e["type"] == "TP" for e in ev), "gts": gts, "preds": preds, "events": ev})
    write_csv(out / "tables" / "T11_error_events.csv", all_events)
    types = ["TP", "Classification", "Localization", "Both (cls+loc)", "Duplicate", "Background", "Missed"]
    rows = []
    for cls in CLASS_NAMES + ["all"]:
        sel = [e for e in all_events if cls == "all" or (e["gt_cls"] or e["pred_cls"]) == cls]
        rows.append({"class": cls, **{t: sum(e["type"] == t for e in sel) for t in types}})
    for sz in ["small", "medium", "large"]:
        sel = [e for e in all_events if e["size"] == sz]
        rows.append({"class": f"size:{sz}", **{t: sum(e["type"] == t for e in sel) for t in types}})
    save_table(out, "T11_error_taxonomy", rows,
               f"T11. TIDE-style error taxonomy on the test set (conf ≥ {args.conf}, IoU_fg=0.5, IoU_bg=0.1)")
    pairs: dict[str, int] = {}
    for e in all_events:
        if e["type"] == "Classification":
            k = f"{e['gt_cls']} → {e['pred_cls']}"
            pairs[k] = pairs.get(k, 0) + 1
    save_table(out, "T11b_confusion_pairs", [{"true → predicted": k, "count": v}
                                             for k, v in sorted(pairs.items(), key=lambda x: -x[1])],
               "T11b. Classification-error pairs")
    # F10: qualitative figures
    fig = out / "figures" / "F10_qualitative"
    fig.mkdir(parents=True, exist_ok=True)
    good = sorted([p for p in per_image if p["n_err"] == 0 and p["n_gt"] >= 3], key=lambda p: -p["n_gt"])[:4]
    bad = sorted(per_image, key=lambda p: -p["n_err"])[:8]
    for tag, group in [("success", good), ("failure", bad)]:
        for i, p in enumerate(group):
            img = cv2.imread(p["image"])
            cv2.imwrite(str(fig / f"{tag}_{i + 1}_{Path(p['image']).stem[:40]}.jpg"),
                        draw(img, p["gts"], p["preds"], p["events"]))
    log(f"F10 saved: {len(good)} success, {len(bad)} failure examples (GT green, prediction blue)")


# ───────────────────────────── stage: report ─────────────────────────────
def stage_report(args, out: Path) -> None:
    parts = [f"# TRACE experiment results\n\nGenerated {datetime.now():%Y-%m-%d %H:%M}, weights `{args.weights}`.\n"]
    for md in sorted((out / "tables").glob("*.md")):
        parts.append(md.read_text(encoding="utf-8"))
    figs = sorted(p.relative_to(out) for p in (out / "figures").rglob("*") if p.is_file()) \
        if (out / "figures").exists() else []
    parts.append("### Figures\n\n" + "\n".join(f"- `{f}`" for f in figs) + "\n")
    parts.append("### Still manual\n\n- F11: screenshots of the Streamlit app and the Google Sheets log\n"
                 "- Streamlit Cloud latency: time from upload to result in the browser (5–10 trials)\n")
    (out / "summary.md").write_text("\n".join(parts), encoding="utf-8")
    zp = out.parent / f"{out.name}.zip"
    import zipfile
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file() and p.suffix not in {".pt", ".cache"} and "data_clean" not in p.parts:
                z.write(p, p.relative_to(out.parent))
            elif p.is_file() and "data_clean" in p.parts and p.suffix == ".txt":
                pass  # cleaned labels are reproducible from the audit stage
    log(f"summary: {out / 'summary.md'}  |  zip (without weights): {zp}")


# ───────────────────────────── main ─────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder with train/ valid/ test/")
    ap.add_argument("--weights", default="app/models/best.pt", help="deployed model checkpoint")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--stages", nargs="+", default=["audit", "env", "eval", "latency", "errors", "report"],
                    choices=["audit", "env", "eval", "train", "latency", "errors", "report"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--models", nargs="+", default=list(BASELINES), choices=list(BASELINES))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16, help="lower to 8 if the GPU runs out of memory")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--device", default="auto", help="auto | 0 (NVIDIA) | mps (Apple Silicon) | cpu")
    ap.add_argument("--conf", type=float, default=0.40, help="confidence threshold used by the app")
    ap.add_argument("--latency-n", type=int, default=200)
    ap.add_argument("--n8n-url", default=os.getenv("N8N_WEBHOOK_URL", ""),
                    help="optional webhook URL for end-to-end timing (sends test payloads)")
    args = ap.parse_args()

    global DEVICE
    DEVICE = auto_device(args.device)
    log(f"device: {DEVICE}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_args.json").write_text(json.dumps(vars(args), indent=2))
    stages = {"audit": stage_audit, "env": stage_env, "eval": stage_eval, "train": stage_train,
              "latency": stage_latency, "errors": stage_errors, "report": stage_report}
    for s in ["audit", "env", "eval", "train", "latency", "errors", "report"]:
        if s in args.stages:
            log(f"═══ stage: {s} ═══")
            try:
                stages[s](args, out)
            except Exception as e:  # keep going so one failure doesn't lose the other results
                log(f"stage {s} FAILED: {type(e).__name__}: {e}")
                if s == "audit":
                    sys.exit(1)


if __name__ == "__main__":
    main()
