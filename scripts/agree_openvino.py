#!/usr/bin/env python3
"""Compare OpenVINO codec outputs to ONNX Runtime on bundled samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openecg.deploy import OnnxCodec
from openecg.dsp import rank_normalize
from openecg.eval import ALL_SUPER_QRS_CLASSES
from openecg.openvino_backend import OpenVINOCodec

ROOT = Path(__file__).resolve().parents[1]


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s)), [float(x) for x in f1s]


def pack(a, b, ncls):
    arg_a, arg_b = a.argmax(-1), b.argmax(-1)
    f1m, f1c = macro_f1(arg_b, arg_a, ncls)
    return {
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
        "argmax_agreement": float(np.mean(arg_a == arg_b)),
        "per_sample_macro_f1_vs_ort": f1m,
        "per_class_f1": f1c,
        "n_disagree": int(np.sum(arg_a != arg_b)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--out", type=Path, default=ROOT / "out/agreement/agreement.json")
    args = ap.parse_args()

    ort = OnnxCodec()
    ov = OpenVINOCodec(device=args.device)
    report = {"codec": {}, "device": args.device.upper()}
    for name in ("synth_sinus", "mitdb100"):
        p = ROOT / f"data/samples/{name}_10s_500hz.npy"
        if not p.exists():
            print("skip missing", p)
            continue
        sig = np.load(p)
        x = np.asarray(rank_normalize(sig.astype(np.float32)), dtype=np.float32)
        x = x[:5000] if x.size >= 5000 else np.pad(x, (0, 5000 - x.size))
        ort_outs = ort._sess.run(None, {ort._inp: x[None, :]})
        frame_o = ort_outs[ort._idx["frame"]][0]
        beat_o = ort_outs[ort._idx["beat"]][0]
        rhythm_o = ort_outs[ort._idx["rhythm"]][0]
        ov_outs = ov.forward_logits(x)
        codec_ort = ort.encode(sig, fs=500)
        frame_pred = ov_outs["frame"].argmax(-1).astype(np.uint8)
        beat_pred = ov_outs["beat"].argmax(-1).astype(np.uint8)
        rhythm_pred = ov_outs["rhythm"].argmax(-1).astype(np.uint8)
        beat_pred = np.where(np.isin(frame_pred, ALL_SUPER_QRS_CLASSES), beat_pred, 0).astype(np.uint8)
        ch_agree = float(np.mean(np.stack([frame_pred, beat_pred, rhythm_pred]) == codec_ort.channels))
        report["codec"][name] = {
            "frame": pack(ov_outs["frame"], frame_o, 4),
            "beat": pack(ov_outs["beat"], beat_o, 6),
            "rhythm": pack(ov_outs["rhythm"], rhythm_o, 6),
            "gated_channels_agreement": ch_agree,
            "note": "codec emits labels — reconstruction error N/A",
        }
        print(name, "frame agree", report["codec"][name]["frame"]["argmax_agreement"],
              "gated", ch_agree)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
