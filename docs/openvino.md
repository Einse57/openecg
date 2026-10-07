# OpenVINO path

This tree adds an OpenVINO path for both bundled deploy artifacts, alongside
the reference runtimes.

## Artifacts

| Model | Reference runtime | OpenVINO input |
|---|---|---|
| Boundary delineator (v56c) | `boundary_int8.tflite` via TFLite / `ai-edge-litert` | `boundary_v56c.onnx` (fp32) or `boundary_v56c_wc8.xml` (NNCF INT8 weight compression), recovered from the TFLite file |
| Layered codec (v6) | `codec_v6_int8.onnx` via ONNX Runtime | same ONNX file, read directly |

`OpenVINOBoundary` prefers `boundary_v56c_wc8.xml` and falls back to
`boundary_v56c.onnx` (`prefer_wc8=False` selects the fp32 ONNX).

### Static input shape

The codec ONNX declares a dynamic batch dimension. `openvino_backend._compile`
fixes every dynamic input dim to 1 (codec input `[1, 5000]`) before
`compile_model`, so devices that need static shapes can compile it. All
callers run single-window batch-1 inference.

## Boundary: TFLite → fp32 ONNX

The native OpenVINO TFLite frontend cannot load `boundary_int8.tflite`:

```
FrontEnd API failed with GeneralFailure:
This tensor should be either input, constant or should be already produced
by previous operators: ...Linear_patch_embed...
Error is encountered while working with operation of type ADD and name 3.
```

(`ov.convert_model`, `tflite2onnx` and `tf2onnx.convert.from_tflite` also
fail.) Upstream does not ship the v56c torch checkpoint, so
`scripts/export_boundary_onnx.py` rebuilds it from the TFLite file. The TFLite
model is weight-only int8, so the dequantized weights are the trained weights
up to int8 rounding. The mapping is exact; nothing is fitted:

- `patch_embed.bias` is folded into the positional-encoding constant, so it is set to 0.
- The upper (pre-LN) LayerNorm affine params are folded into the following FC
  weights, so upper `norm1` / `norm2` are identity.
- The lower LayerNorm constants are consumed in reverse tensor order.

The rebuilt fp32 graph reproduces the TFLite logits to ~1e-5, and the export
script asserts this on the bundled samples. Regenerate with:

```bash
python -m scripts.export_boundary_onnx --nncf
```

| Artifact | Notes |
|---|---|
| `boundary_v56c.onnx` | opset 17, static `(1, 2500)`, fp32 |
| `boundary_v56c_wc8.xml` / `.bin` | NNCF `INT8_ASYM` weight compression of the above |
| `boundary_v56c_from_tflite.pt` | torch state dict (gitignored; written by the export) |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-openvino.lock.txt
pip install -e .
python scripts/make_samples.py --with-mitdb   # optional PhysioNet pull
python scripts/agree_openvino.py --device CPU
python bench_openvino.py --device CPU --runs 50
```

`--device` takes an OpenVINO device name available on the host (default
`CPU`). The scripts raise a clear error if the device is absent.

## Results

### Boundary macro-F1 (Martínez tolerances)

Lead II for LUDB/ISP, first lead for the QTDB T-subset; repo splits.

| Corpus | n | TFLite (ref) | OpenVINO fp32 ONNX | OpenVINO wc8 |
|---|---:|---:|---:|---:|
| LUDB val | 41 | 0.9633 | 0.9633 | 0.9633 |
| ISP test | 72 | 0.9711 | 0.9711 | 0.9715 |
| QTDB T-subset | 44 | 0.9076 | 0.9076 | 0.9075 |
| **Unweighted mean** | — | **0.9473** | **0.9473** | **0.9474** |

The TFLite per-corpus cells match the README table (~0.963 / 0.971 / 0.908).

### Agreement

- Boundary vs TFLite (frame argmax on bundled samples): fp32 ONNX 1.000;
  wc8 0.994–0.996.
- Codec vs ONNX Runtime (gated channel agreement): ≥ 0.998
  (synth_sinus 0.9995, mitdb100 0.9987).

### Reproduce macro-F1

```bash
export OPENECG_LUDB_ZIP=data/physionet/ludb-1.0.1.zip
export OPENECG_LUDB_CACHE=data/physionet/ludb_cache
export OPENECG_ISP_ZIP=data/physionet/isp_delineation_dataset.zip   # Zenodo 14679837
export OPENECG_ISP_CACHE=data/physionet/isp_cache
export OPENECG_QTDB_CACHE=data/physionet/qtdb_cache
python -m scripts.eval_macro_f1_openvino --device CPU --leads ii
```

## License

Code and bundled weights: **Apache-2.0** (`LICENSE`, Copyright 2026 Hyung-Chul Lee
and OpenECG contributors). There is no separate weights license file, and the model
cards do not state a different license. The training corpora cited in the README are
PhysioNet / Zenodo (CC-BY / ODC-BY) and are **not** redistributed here.
