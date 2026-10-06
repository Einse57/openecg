#!/usr/bin/env python3
"""One-command CPU/GPU/NPU bench: reference runtimes vs OpenVINO.

Examples
--------
    python bench_openvino.py --device CPU --runs 50
    python bench_openvino.py --device CPU --runs 20 --sample mitdb100

Prints latency per 10 s window (mean / p50 / p90), throughput (windows/s),
and peak RSS for:
  * boundary delineator: TFLite/LiteRT (reference); OpenVINO if loadable
  * layered codec: ONNX Runtime (reference) vs OpenVINO

Numbers from this script on a shared cloud box are **not** Intel Core/Xeon
target numbers — label them as box numbers when reporting.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent
SAMPLES = {
    "synth_sinus": {
        "250": ROOT / "data/samples/synth_sinus_10s_250hz.npy",
        "500": ROOT / "data/samples/synth_sinus_10s_500hz.npy",
    },
    "mitdb100": {
        "250": ROOT / "data/samples/mitdb100_10s_250hz.npy",
        "500": ROOT / "data/samples/mitdb100_10s_500hz.npy",
    },
}


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _bench(fn, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    rss0 = _rss_mb()
    times = []
    peak = rss0
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
        peak = max(peak, _rss_mb())
    mean = statistics.fmean(times)
    return {
        "runs": runs,
        "warmup": warmup,
        "latency_ms_mean": mean,
        "latency_ms_p50": _pct(times, 50),
        "latency_ms_p90": _pct(times, 90),
        "throughput_windows_per_s": 1000.0 / mean if mean > 0 else float("inf"),
        "peak_rss_mb": peak,
        "rss_delta_mb": peak - rss0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="CPU", help="OpenVINO device: CPU|GPU|NPU")
    ap.add_argument("--runs", type=int, default=30, help="timed iterations")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--sample", default="mitdb100", choices=sorted(SAMPLES))
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    samp = SAMPLES[args.sample]
    for p in samp.values():
        if not p.exists():
            raise SystemExit(
                f"missing sample {p}; generate via scripts/make_samples.py "
                f"or re-run the smoke setup"
            )
    sig250 = np.load(samp["250"]).astype(np.float32)
    sig500 = np.load(samp["500"]).astype(np.float32)
    assert sig250.size == 2500 and sig500.size == 5000

    from openecg.deploy import Inference, OnnxCodec, preprocess_window
    from openecg.dsp import rank_normalize
    from openecg.openvino_backend import (
        BOUNDARY_TFLITE_ERROR,
        OpenVINOBoundary,
        OpenVINOCodec,
        available_devices,
    )

    print(f"host_label: BOX (not Intel-target)")
    print(f"sample: {args.sample}")
    print(f"OpenVINO devices available: {available_devices()}")
    print(f"requested --device {args.device.upper()}")
    print(f"runs={args.runs} warmup={args.warmup}")
    print(f"baseline README latency: 44 ms / 10 s window (TFLite int8, published)")
    print()

    results: dict = {
        "host_label": "BOX",
        "sample": args.sample,
        "device_requested": args.device.upper(),
        "devices_available": available_devices(),
        "readme_baseline_ms": 44,
    }

    # --- Boundary reference (TFLite / LiteRT) ---
    det = Inference()
    window250 = sig250.copy()

    def run_tflite():
        det.forward_window(window250)

    results["boundary_tflite"] = _bench(run_tflite, args.runs, args.warmup)
    print("[boundary] TFLite/LiteRT (reference)")
    for k, v in results["boundary_tflite"].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Boundary OpenVINO (may fail) ---
    try:
        ov_bound = OpenVINOBoundary(device=args.device)

        def run_ov_bound():
            ov_bound.forward_window(window250)

        results["boundary_openvino"] = _bench(run_ov_bound, args.runs, args.warmup)
        print("[boundary] OpenVINO")
        for k, v in results["boundary_openvino"].items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    except Exception as e:  # noqa: BLE001
        results["boundary_openvino"] = {
            "status": "unavailable",
            "error": str(e).split("\n")[0],
            "note": BOUNDARY_TFLITE_ERROR[:200] + "...",
        }
        print("[boundary] OpenVINO: UNAVAILABLE")
        print(f"  {results['boundary_openvino']['error']}")

    # --- Codec ORT ---
    ort = OnnxCodec()
    x500 = np.asarray(rank_normalize(sig500), dtype=np.float32)[:5000]

    def run_ort():
        ort._sess.run(None, {ort._inp: x500[None, :]})

    results["codec_onnxruntime"] = _bench(run_ort, args.runs, args.warmup)
    print("[codec] ONNX Runtime (reference)")
    for k, v in results["codec_onnxruntime"].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Codec OpenVINO ---
    ov_codec = OpenVINOCodec(device=args.device)

    def run_ov_codec():
        ov_codec.forward_logits(x500)

    results["codec_openvino"] = _bench(run_ov_codec, args.runs, args.warmup)
    print("[codec] OpenVINO")
    for k, v in results["codec_openvino"].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    out = args.json_out or (ROOT / "out" / f"bench_{args.device.upper()}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
