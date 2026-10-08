# openecg-boundary v56c — ONNX / OpenVINO IR export of `boundary_int8.tflite`

The **same v56c boundary detector** as the bundled `boundary_int8.tflite`, in two
formats that OpenVINO can load: an fp32 ONNX graph and an OpenVINO IR with 8-bit
weight compression. No retraining and no fitting — the weights come from the
TFLite file.

```
OpenVINOBoundary().forward_window(sig_250hz_2500)  ->  cls_logits[500, 4]   # none, P, QRS, T
```

> **Why a second format.** The OpenVINO TFLite frontend cannot load
> `boundary_int8.tflite` (`GeneralFailure` on the `ADD` after `patch_embed`), and the
> v56c torch checkpoint is not bundled. `boundary_int8.tflite` is weight-only int8, so
> its dequantized weights are the trained weights up to int8 rounding; rebuilding the
> torch graph from them gives an fp32 model that reproduces the TFLite logits.

## Architecture & build
`vit_transformer_noaux_1ch` (`FrameClassifierTransformerNoAux1Ch`), L8 / d=128, 4 heads,
ff=256, patch 5, 4 lower conv blocks + 4 upper transformer layers, **0.99 M params**.
Single-lead, 10 s @ 250 Hz, rank-normalized (`openecg.deploy.preprocess_window`).

Built by `scripts/export_boundary_onnx.py --nncf` from `boundary_int8.tflite`:

1. **Dequantize** every weight tensor of the TFLite file (per-channel scale / zero point).
2. **Map exactly** onto the torch module. Three converter folds are undone, nothing is fitted:
   - `patch_embed.bias` was folded into the positional-encoding constant → set to 0;
   - the upper pre-LN LayerNorm affine params were folded into the following
     `in_proj` / `linear1` weights → upper `norm1` / `norm2` are identity;
   - the lower LayerNorm constants are stored in reverse block order (block 0 uses
     MUL 77 / ADD 76 … block 3 uses MUL 71 / ADD 70).
3. **Check**: the script asserts max |torch − TFLite| on the cls logits < 1e-3 on the
   sample windows (observed ~1e-5).
4. **Export** `torch.onnx.export`, opset 17, static input `(1, 2500)` →
   `boundary_v56c.onnx` (torch 2.14.1).
5. **Compress** `nncf.compress_weights(mode=INT8_ASYM)`, data-free (no calibration set),
   all other settings default → `boundary_v56c_wc8.xml` / `.bin` (NNCF 3.4.0,
   OpenVINO 2026.4.1). Weights are stored as int8 and dequantized at run time;
   activations stay floating point.

## Evaluation (held-out, same protocol as the README table)
Martínez boundary macro-F1, **lead II** (first lead for the QTDB T-subset), repo splits,
OpenVINO 2026.4.1 `CPU`. Reproduce with `python -m scripts.eval_macro_f1_openvino --device CPU --leads ii`.

| corpus (n records) | `boundary_int8.tflite` | `boundary_v56c.onnx` (fp32) | `boundary_v56c_wc8.xml` (8-bit) |
|---|---|---|---|
| LUDB val (41) | 0.9633 | 0.9633 | 0.9633 |
| ISP test (72) | 0.9711 | 0.9711 | 0.9715 |
| QTDB T-subset (44) | 0.9076 | 0.9076 | 0.9075 |
| **mean** | **0.9473** | **0.9473** | **0.9474** |

Frame-argmax agreement with the TFLite model on the sample windows (`CPU`, f32):
fp32 ONNX 1.000, 8-bit IR 0.994–0.996. Results on `GPU` / `NPU` and the tested hosts
are in [`docs/benchmarks/openvino.md`](../../docs/benchmarks/openvino.md).

## Limitations (read before use)
- **Inherits everything about v56c** — training data, accuracy and limits are those of
  `boundary_int8.tflite` (see the README *Performance* section). This export adds no
  training and no new data.
- The 8-bit IR is a **re-quantization** of already int8-rounded weights; its macro-F1
  differs from the TFLite model in the 4th decimal.
- Static batch 1 × 2500 samples (10 s @ 250 Hz); longer signals are windowed by the caller.
- `OpenVINOBoundary` runs `CPU` in f32 by default (`INFERENCE_PRECISION_HINT=f32`) for
  reproducibility; OpenVINO would otherwise pick bf16 on CPUs with native bf16 support.
  To opt into bf16, pass `config={"INFERENCE_PRECISION_HINT": "bf16"}`; results above
  are for f32.
- Single-lead. Research and educational use only. **Not a medical device; not for diagnosis.**

## Artifacts
| file | size | notes |
|------|------|-------|
| `boundary_v56c.onnx` | 4.3 MB | fp32, opset 17; `OpenVINOBoundary(prefer_wc8=False)` |
| `boundary_v56c_wc8.xml` + `.bin` | 0.4 MB + 1.3 MB | NNCF INT8_ASYM weight compression; `OpenVINOBoundary()` default |

I/O (both): input `signal (1, 2500)` float32 (rank-normalized); outputs
`cls_logits (1, 500, 4)` and `reg_offsets (1, 500, 6)` float32, one frame per 5 samples.
