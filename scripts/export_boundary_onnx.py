#!/usr/bin/env python3
"""Export v56c boundary detector to ONNX for OpenVINO.

Upstream does **not** ship ``stage2_v45k_noaux_L8_d128_1ch_v56c.pt`` — only
``boundary_int8.tflite``. This script:

1. Instantiates ``vit_transformer_noaux_1ch`` (0.99 M params)
2. Dequantizes weight tensors from the bundled TFLite via LiteRT
3. Fits the few missing params (upper LayerNorm + patch_embed.bias) to match
   TFLite logits on calibration windows
4. Writes ``openecg/models/boundary_v56c_from_tflite.pt`` and
   ``openecg/models/boundary_v56c.onnx`` (opset 17, static 1×2500)
5. Optionally NNCF-compresses to ``boundary_v56c_wc8.xml``

Re-run::

    python -m scripts.export_boundary_onnx --nncf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Implementation lives inline so a one-file re-run works without a private ckpt.
# (Heavy lifting duplicated from the feasibility notebook path.)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nncf", action="store_true", help="NNCF INT8 weight compression")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from ai_edge_litert.interpreter import Interpreter

    from openecg.deploy import Inference, WINDOW_SAMPLES, preprocess_window
    from openecg.stage2.model_variants import FrameClassifierTransformerNoAux1Ch

    out_dir = ROOT / "openecg" / "models"
    tflite = out_dir / "boundary_int8.tflite"
    ckpt_path = out_dir / "boundary_v56c_from_tflite.pt"
    onnx_path = out_dir / "boundary_v56c.onnx"

    model = FrameClassifierTransformerNoAux1Ch(
        patch_size=5, n_leads=12, d_model=128, n_heads=4, n_layers=8, ff=256,
        n_classes=4, dropout=0.0, use_lead_emb=False, pos_type="sinusoidal",
        conv_stem=False, n_reg=6, mid_split=4, lower_kernel=7,
    ).eval()

    interp = Interpreter(model_path=str(tflite))
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    interp.set_tensor(inp_d["index"], np.zeros(inp_d["shape"], np.float32))
    interp.invoke()
    details = {d["index"]: d for d in interp.get_tensor_details()}

    def get(idx: int) -> np.ndarray:
        d = details[idx]
        arr = interp.get_tensor(idx)
        q = d.get("quantization_parameters") or {}
        scales = np.asarray(q.get("scales", []), np.float32).reshape(-1)
        zps = np.asarray(q.get("zero_points", []), np.int32).reshape(-1)
        if arr.dtype == np.int8 and scales.size > 0:
            qdim = int(q.get("quantized_dimension", 0))
            if scales.size == 1:
                return (arr.astype(np.float32) - float(zps[0])) * float(scales[0])
            deq = np.moveaxis(arr.astype(np.float32), qdim, 0)
            for i in range(deq.shape[0]):
                deq[i] = (deq[i] - float(zps[i])) * float(scales[i])
            return np.moveaxis(deq, 0, qdim)
        return arr.astype(np.float32)

    sd = model.state_dict()

    def set_(key, arr):
        t = torch.from_numpy(np.ascontiguousarray(arr))
        assert sd[key].shape == t.shape, (key, sd[key].shape, t.shape)
        sd[key].copy_(t)

    set_("head.bias", get(2)); set_("head.weight", get(32))
    set_("reg_head.bias", get(1)); set_("reg_head.weight", get(31))
    set_("patch_embed.weight", get(41))
    sd["pos_enc"][:, :500, :].copy_(torch.from_numpy(get(42)))
    for i, (w_idx, b_idx) in enumerate([(51, 30), (49, 29), (48, 28), (47, 27)]):
        w = np.transpose(np.squeeze(get(w_idx), axis=2), (0, 2, 1))
        set_(f"lower_convs.{i}.weight", w)
        set_(f"lower_convs.{i}.bias", get(b_idx).reshape(128))
    ln = [get(i) for i in (70, 71, 72, 73, 74, 75, 76, 77)]
    for i in range(4):
        set_(f"lower_norms.{i}.bias", ln[2 * i])
        set_(f"lower_norms.{i}.weight", ln[2 * i + 1])
    attn_in_w = {0: 25, 1: 19, 2: 13, 3: 7}
    attn_in_b = {0: 26, 1: 20, 2: 14, 3: 8}
    attn_out_w = {0: 40, 1: 38, 2: 36, 3: 34}
    attn_out_b = {0: 24, 1: 18, 2: 12, 3: 6}
    lin1_w = {0: 22, 1: 16, 2: 10, 3: 4}
    lin1_b = {0: 23, 1: 17, 2: 11, 3: 5}
    lin2_w = {0: 39, 1: 37, 2: 35, 3: 33}
    lin2_b = {0: 21, 1: 15, 2: 9, 3: 3}
    for L in range(4):
        set_(f"upper_transformer.layers.{L}.self_attn.in_proj_weight", get(attn_in_w[L]))
        set_(f"upper_transformer.layers.{L}.self_attn.in_proj_bias", get(attn_in_b[L]))
        set_(f"upper_transformer.layers.{L}.self_attn.out_proj.weight", get(attn_out_w[L]))
        set_(f"upper_transformer.layers.{L}.self_attn.out_proj.bias", get(attn_out_b[L]))
        set_(f"upper_transformer.layers.{L}.linear1.weight", get(lin1_w[L]))
        set_(f"upper_transformer.layers.{L}.linear1.bias", get(lin1_b[L]))
        set_(f"upper_transformer.layers.{L}.linear2.weight", get(lin2_w[L]))
        set_(f"upper_transformer.layers.{L}.linear2.bias", get(lin2_b[L]))
    model.load_state_dict(sd)

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = [model.patch_embed.bias]
    model.patch_embed.bias.requires_grad_(True)
    for layer in model.upper_transformer.layers:
        for n in (layer.norm1, layer.norm2):
            n.weight.requires_grad_(True)
            n.bias.requires_grad_(True)
            trainable.extend([n.weight, n.bias])

    det = Inference()
    cals = []
    sample_dir = ROOT / "data" / "samples"
    for path in [sample_dir / "mitdb100_10s_250hz.npy", sample_dir / "synth_sinus_10s_250hz.npy"]:
        if not path.exists():
            continue
        sig = np.load(path)
        x = preprocess_window(sig).astype(np.float32)
        ref_cls = det.forward_window(sig)
        det._interp.set_tensor(det._input_detail["index"], x[None, ...])
        det._interp.invoke()
        ref_reg = det._interp.get_tensor(det._reg_detail["index"])[0]
        cals.append((x, ref_cls, ref_reg))
    rng = np.random.default_rng(0)
    base = np.load(sample_dir / "mitdb100_10s_250hz.npy") if (sample_dir / "mitdb100_10s_250hz.npy").exists() else rng.standard_normal(WINDOW_SAMPLES).astype(np.float32)
    for _ in range(16):
        sig = (base + rng.standard_normal(WINDOW_SAMPLES).astype(np.float32) * 0.05).astype(np.float32)
        x = preprocess_window(sig).astype(np.float32)
        ref_cls = det.forward_window(sig)
        det._interp.set_tensor(det._input_detail["index"], x[None, ...])
        det._interp.invoke()
        ref_reg = det._interp.get_tensor(det._reg_detail["index"])[0]
        cals.append((x, ref_cls, ref_reg))

    opt = torch.optim.Adam(trainable, lr=5e-2)
    model.train()
    for step in range(args.steps):
        opt.zero_grad()
        loss = torch.zeros(())
        for x, ref_cls, ref_reg in cals:
            cls, reg, _, _ = model(torch.from_numpy(x[None, ...]), torch.zeros(1, dtype=torch.long))
            loss = loss + torch.nn.functional.mse_loss(cls[0], torch.from_numpy(ref_cls))
            loss = loss + 0.25 * torch.nn.functional.mse_loss(reg[0], torch.from_numpy(ref_reg))
        (loss / len(cals)).backward()
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step} loss={float(loss)/len(cals):.4f}")

    model.eval()
    torch.save({
        "model_state": model.state_dict(),
        "model_config": dict(model.model_config),
        "note": "Recovered from boundary_int8.tflite; upper LN + patch bias fitted.",
        "source_tflite": str(tflite),
    }, ckpt_path)
    print("wrote", ckpt_path)

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, sig):
            lead = torch.zeros(sig.shape[0], dtype=torch.long, device=sig.device)
            cls, reg, _, _ = self.m(sig, lead)
            return cls, reg

    torch.onnx.export(
        Wrap(model).eval(),
        torch.zeros(1, WINDOW_SAMPLES),
        str(onnx_path),
        input_names=["signal"],
        output_names=["cls_logits", "reg_offsets"],
        opset_version=17,
        dynamo=False,
    )
    print("wrote", onnx_path, onnx_path.stat().st_size)

    if args.nncf:
        import openvino as ov
        import nncf
        from nncf import CompressWeightsMode
        core = ov.Core()
        ov_model = core.read_model(str(onnx_path))
        compressed = nncf.compress_weights(ov_model, mode=CompressWeightsMode.INT8_ASYM)
        wc8 = out_dir / "boundary_v56c_wc8.xml"
        ov.save_model(compressed, str(wc8))
        print("wrote", wc8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
