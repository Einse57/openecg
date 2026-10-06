#!/usr/bin/env python3
"""One-command Martinez macro-F1 on LUDB + ISP + QTDB for TFLite vs OpenVINO.

Matches the README headline path (lead II for LUDB/ISP; first lead for QTDB
T-subset) and reports the mean over the three corpora vs README 0.9274.

Requires::

    export OPENECG_LUDB_ZIP=... OPENECG_LUDB_CACHE=...
    export OPENECG_ISP_ZIP=...  OPENECG_ISP_CACHE=...   # optional but needed for full mean
    export OPENECG_QTDB_CACHE=...                       # or OPENECG_QTDB_ZIP

Run::

    python -m scripts.eval_macro_f1_openvino --device CPU --leads ii
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from openecg import ludb
from openecg.deploy import Inference, WINDOW_SAMPLES, preprocess_window
from openecg.stage2.evaluate import (
    BOUNDARY_KEYS,
    MARTINEZ_TOLERANCE_MS,
    signed_boundary_metrics,
)
from openecg.stage2.infer import (
    apply_reg_to_boundaries,
    extract_boundaries,
    post_process_frames,
)
from openecg.stage2.multi_dataset import _decimate_to_250

ROOT = Path(__file__).resolve().parents[1]
FS = 250
FRAME_MS = 20


def _boundaries_from_logits(cls_logits: np.ndarray, reg_offsets: np.ndarray | None):
    frames = cls_logits.argmax(-1).astype(np.uint8)
    pp = post_process_frames(frames, frame_ms=FRAME_MS)
    bds = extract_boundaries(pp, fs=FS, frame_ms=FRAME_MS)
    if reg_offsets is not None:
        bds = apply_reg_to_boundaries(
            bds, reg_offsets, samples_per_frame=5, max_window=WINDOW_SAMPLES,
        )
    return bds


def _tflite_predict(det: Inference, sig_250: np.ndarray):
    x = preprocess_window(sig_250).astype(np.float32)
    det._interp.set_tensor(det._input_detail["index"], x[None, ...])
    det._interp.invoke()
    cls = det._interp.get_tensor(det._cls_detail["index"])[0]
    reg = det._interp.get_tensor(det._reg_detail["index"])[0] if det._reg_detail else None
    return _boundaries_from_logits(cls, reg)


def _ov_predict(ov_bound, sig_250: np.ndarray):
    cls, reg = ov_bound.forward_window_cls_reg(sig_250)
    return _boundaries_from_logits(cls, reg)


def _macro_f1(bp: dict, bt: dict) -> dict:
    per = {}
    f1s = []
    for k in BOUNDARY_KEYS:
        m = signed_boundary_metrics(bp.get(k, []), bt.get(k, []), MARTINEZ_TOLERANCE_MS[k], fs=FS)
        per[k] = m
        if not np.isnan(m["f1"]):
            f1s.append(m["f1"])
    return {
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "per_boundary": {
            k: {
                kk: float(vv) if isinstance(vv, (float, np.floating, int, np.integer)) else vv
                for kk, vv in v.items()
                if kk in ("f1", "sens", "ppv", "n_true", "n_pred", "median_abs_ms")
            }
            for k, v in per.items()
        },
    }


def eval_ludb(predict_fn, leads_subset, edge_margin_ms=100):
    from openecg.stage2.dataset import LUDBFrameDataset

    rec_ids = ludb.load_split()["val"]
    ds = LUDBFrameDataset(rec_ids)
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    n = 0
    margin_250 = int(round(edge_margin_ms * FS / 1000.0))
    t0 = time.time()
    for idx in range(len(ds)):
        rid, lead = ds.items[idx]
        if leads_subset and lead not in leads_subset:
            continue
        sig_250, _lead_idx, _ = ds.cache[(rid, lead)]
        sig_250 = sig_250[:WINDOW_SAMPLES]
        if len(sig_250) < WINDOW_SAMPLES:
            continue
        sig_raw = sig_250.astype(np.float32)
        rng = ludb.labeled_range(rid, lead)
        if rng is None:
            continue
        lo_250 = max(0, rng[0] // 2 - margin_250)
        hi_250 = min(WINDOW_SAMPLES, rng[1] // 2 + margin_250 + 1)
        preds = predict_fn(sig_raw)
        for k, vs in preds.items():
            for s in vs:
                if lo_250 <= s < hi_250:
                    bp[k].append(int(s) + cum)
        try:
            gt = ludb.load_annotations(rid, lead)
            for k, v in gt.items():
                if k.endswith("_on") or k.endswith("_off"):
                    for s in v:
                        s250 = int(s // 2)
                        if lo_250 <= s250 < hi_250:
                            bt[k].append(s250 + cum)
        except Exception:
            pass
        cum += WINDOW_SAMPLES
        n += 1
    return _macro_f1(bp, bt) | {"n_windows": n, "elapsed_s": time.time() - t0}


def eval_isp(predict_fn, leads_subset):
    """ISP test, native 1000 Hz → 250 Hz (same as scripts/benchmark_v56c.py)."""
    try:
        from openecg import isp
        isp.ensure_extracted()
    except Exception as e:
        return {"skipped": True, "reason": str(e)}

    from openecg import isp

    # README / benchmark_v56c use load_split()['test'] (72), not eval_test_records()
    rec_ids = isp.load_split()["test"]
    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    n = 0
    t0 = time.time()
    for rid in rec_ids:
        try:
            record = isp.load_record(rid, split="test")
            ann = isp.load_annotations_as_super(rid, split="test")
        except Exception:
            continue
        for lead_idx, lead in enumerate(isp.LEADS_12):
            if leads_subset and lead not in leads_subset:
                continue
            sig_1000 = record[lead]
            sig_250 = _decimate_to_250(sig_1000, 1000)
            if len(sig_250) < WINDOW_SAMPLES:
                pad = np.zeros(WINDOW_SAMPLES - len(sig_250), dtype=sig_250.dtype)
                sig_250 = np.concatenate([sig_250, pad])
            sig_250 = sig_250[:WINDOW_SAMPLES].astype(np.float32)
            preds = predict_fn(sig_250)
            for k, vs in preds.items():
                for s in vs:
                    bp[k].append(int(s) + cum)
            for k, v in ann.items():
                if k.endswith("_on") or k.endswith("_off"):
                    for s in v:
                        s250 = int(s // 4)
                        if 0 <= s250 < WINDOW_SAMPLES:
                            bt[k].append(s250 + cum)
            cum += WINDOW_SAMPLES
            n += 1
    if n == 0:
        return {"skipped": True, "reason": "no ISP windows scored"}
    return _macro_f1(bp, bt) | {"n_windows": n, "elapsed_s": time.time() - t0}


def eval_qtdb(predict_fn):
    try:
        from openecg import qtdb
        qtdb.ensure_extracted()
    except Exception as e:
        return {"skipped": True, "reason": str(e)}

    from openecg import qtdb

    rids = []
    for rid in qtdb.records_with_q1c():
        ann = qtdb.load_q1c(rid)
        win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES, fs=FS)
        if win is None:
            continue
        start, end = win
        n_q = sum(1 for s in ann["qrs_on"] if start <= s < end)
        n_t = sum(1 for s in ann["t_on"] if start <= s < end)
        if n_q > 0 and n_t / n_q >= 0.8:
            rids.append(rid)

    bp, bt = defaultdict(list), defaultdict(list)
    cum = 0
    n = 0
    t0 = time.time()
    for rid in rids:
        try:
            rec = qtdb.load_record(rid)
            lead = next(iter(rec))
            sig = rec[lead]
            if len(sig) < WINDOW_SAMPLES:
                continue
            ann = qtdb.load_q1c(rid)
            win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES, fs=FS)
            if win is None:
                continue
            start, end = win
            if end > len(sig):
                end = len(sig)
                start = max(0, end - WINDOW_SAMPLES)
            sig_250 = sig[start:start + WINDOW_SAMPLES].astype(np.float32)
            if len(sig_250) < WINDOW_SAMPLES:
                continue
            preds = predict_fn(sig_250)
            for k, vs in preds.items():
                for s in vs:
                    bp[k].append(int(s) + cum)
            for k, v in ann.items():
                if k.endswith("_on") or k.endswith("_off"):
                    for s in v:
                        if start <= s < start + WINDOW_SAMPLES:
                            bt[k].append(int(s - start) + cum)
            cum += WINDOW_SAMPLES
            n += 1
        except Exception:
            continue
    if n == 0:
        return {"skipped": True, "reason": "no QTDB windows scored"}
    return _macro_f1(bp, bt) | {"n_windows": n, "elapsed_s": time.time() - t0, "n_t_subset": len(rids)}


def _mean_over(report: dict, backend: str) -> dict:
    """Mean of per-corpus macro-F1 for corpora that actually ran."""
    keys = [f"ludb_{backend}", f"isp_{backend}", f"qtdb_{backend}"]
    vals = []
    present = []
    for k in keys:
        block = report.get(k)
        if not block or block.get("skipped"):
            continue
        f1 = block.get("macro_f1")
        if f1 is None or (isinstance(f1, float) and np.isnan(f1)):
            continue
        vals.append(float(f1))
        present.append(k)
    return {
        "macro_f1_mean": float(np.mean(vals)) if vals else float("nan"),
        "n_corpora": len(vals),
        "corpora": present,
        "readme_baseline": 0.9274,
        "delta_vs_readme": (float(np.mean(vals)) - 0.9274) if vals else float("nan"),
    }


def _ensure_env() -> None:
    if "OPENECG_LUDB_ZIP" not in os.environ:
        cand = ROOT / "data/physionet/ludb-1.0.1.zip"
        if cand.exists():
            os.environ["OPENECG_LUDB_ZIP"] = str(cand)
            os.environ.setdefault("OPENECG_LUDB_CACHE", str(ROOT / "data/physionet/ludb_cache"))
    if "OPENECG_ISP_ZIP" not in os.environ:
        cand = ROOT / "data/physionet/isp_delineation_dataset.zip"
        if cand.exists():
            os.environ["OPENECG_ISP_ZIP"] = str(cand)
            os.environ.setdefault("OPENECG_ISP_CACHE", str(ROOT / "data/physionet/isp_cache"))
    if "OPENECG_QTDB_CACHE" not in os.environ:
        cand = ROOT / "data/physionet/qtdb_cache"
        if cand.exists():
            os.environ["OPENECG_QTDB_CACHE"] = str(cand)
    if "OPENECG_QTDB_ZIP" not in os.environ:
        cand = ROOT / "data/physionet/qt-database-1.0.0.zip"
        if cand.exists() and cand.stat().st_size > 80_000_000:
            os.environ["OPENECG_QTDB_ZIP"] = str(cand)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--leads", default="ii", help="'ii' or 'all' (LUDB/ISP)")
    ap.add_argument("--out", type=Path, default=ROOT / "out/macro_f1_openvino.json")
    ap.add_argument("--skip-isp", action="store_true")
    ap.add_argument("--skip-qtdb", action="store_true")
    args = ap.parse_args()

    leads = None if args.leads == "all" else {args.leads.lower()}
    _ensure_env()

    from openecg.openvino_backend import OpenVINOBoundary

    det = Inference()
    ovb = OpenVINOBoundary(device=args.device)

    report = {
        "readme_baseline_macro_f1_int8": 0.9274,
        "readme_note": (
            "0.9274 is mean over LUDB val + ISP test + QTDB T-subset "
            "(TFLite int8, lead II / first lead)."
        ),
        "leads": args.leads,
        "device": args.device.upper(),
    }

    print("=== LUDB val / TFLite ===", flush=True)
    report["ludb_tflite"] = eval_ludb(lambda s: _tflite_predict(det, s), leads)
    print("macro_f1", report["ludb_tflite"]["macro_f1"], "n", report["ludb_tflite"]["n_windows"], flush=True)

    print("=== LUDB val / OpenVINO ===", flush=True)
    report["ludb_openvino"] = eval_ludb(lambda s: _ov_predict(ovb, s), leads)
    print("macro_f1", report["ludb_openvino"]["macro_f1"], "n", report["ludb_openvino"]["n_windows"], flush=True)

    if not args.skip_isp:
        print("=== ISP test / TFLite ===", flush=True)
        report["isp_tflite"] = eval_isp(lambda s: _tflite_predict(det, s), leads)
        print(report["isp_tflite"].get("macro_f1", report["isp_tflite"]), flush=True)
        print("=== ISP test / OpenVINO ===", flush=True)
        report["isp_openvino"] = eval_isp(lambda s: _ov_predict(ovb, s), leads)
        print(report["isp_openvino"].get("macro_f1", report["isp_openvino"]), flush=True)

    if not args.skip_qtdb:
        print("=== QTDB T-subset / TFLite ===", flush=True)
        report["qtdb_tflite"] = eval_qtdb(lambda s: _tflite_predict(det, s))
        print(report["qtdb_tflite"].get("macro_f1", report["qtdb_tflite"]), flush=True)
        print("=== QTDB T-subset / OpenVINO ===", flush=True)
        report["qtdb_openvino"] = eval_qtdb(lambda s: _ov_predict(ovb, s))
        print(report["qtdb_openvino"].get("macro_f1", report["qtdb_openvino"]), flush=True)

    report["mean_tflite"] = _mean_over(report, "tflite")
    report["mean_openvino"] = _mean_over(report, "openvino")
    print("=== MEAN TFLite ===", report["mean_tflite"], flush=True)
    print("=== MEAN OpenVINO ===", report["mean_openvino"], flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
