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

On `GPU` devices, `_compile` also rewrites 3-D `Interpolate` nodes into an
equivalent 4-D form, and `OpenVINOCodec` defaults to
`INFERENCE_PRECISION_HINT=f32`. Pass `config={}` to use the device default.
See *Tested on*.

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

`--device` takes any OpenVINO device name available on the host, e.g. `CPU`
(default), `GPU`, `NPU`, or `AUTO`. The scripts raise a clear error if the device is absent.

## Tested on

| | Host 1 | Host 2 |
|---|---|---|
| CPU | Intel Core Ultra 7 265F | Intel Core Ultra 9 285H |
| OpenVINO devices tested | `CPU`, `NPU` (Intel AI Boost) | `CPU`, `GPU` (Intel Arc 140T integrated GPU), `NPU` (Intel AI Boost) |
| OS / runtime | Windows 11 (10.0.26200), Python 3.12.10, OpenVINO 2026.4.1 | Windows 11 Pro (10.0.26200), Python 3.12.10, OpenVINO 2026.4.1 |
| Drivers | — | GPU 32.0.101.8860, NPU 32.0.100.4841 |
| Not tested | Integrated GPU (this CPU has none). The discrete NVIDIA GeForce RTX 4070 exposed as `GPU` was only spot-checked: codec agreement 0.9996 / 0.9992, no latency or macro-F1 runs. | — |

Reference runtimes (TFLite / LiteRT for the boundary model, ONNX Runtime for the codec) always run on the host CPU. Bench commit: `b15fbae`/`1c39ea8` on Host 1 and `4bc2406` on Host 2. The commits differ only in the GPU-only codec path below, so CPU/NPU code paths are unchanged.

### Device-specific compile steps (applied automatically by `openvino_backend`)

- All devices: dynamic input dims are fixed to batch 1 (codec input `[1, 5000]`). `NPU` needs this to compile the codec.
- `GPU`:
  - 3-D `Interpolate` nodes are run as Unsqueeze → 4-D Interpolate → Squeeze, which is numerically identical. Without this, the codec's 1-D nearest upsample followed by a 1-D Conv fails GPU program build ("Data batch and filters rank do not match"), on OpenVINO 2025.4.1 through 2026.4.1.
  - The codec compiles with `INFERENCE_PRECISION_HINT=f32` by default. With the default f16, gated agreement vs the int8 ONNX Runtime reference is 0.981 / 0.993 (synth_sinus / mitdb100), below the 0.998 gate.

### Boundary macro-F1 (Martínez tolerances)

Lead II for LUDB/ISP, first lead for the QTDB T-subset; repo splits (LUDB val n=41, ISP test n=72, QTDB T-subset n=44). "fp32" = `boundary_v56c.onnx`, "8-bit" = `boundary_v56c_wc8.xml` (NNCF INT8 weight compression). TFLite results are identical on both hosts.

| Corpus | TFLite (ref) | `CPU` fp32 | `CPU` 8-bit | `GPU` fp32 | `GPU` 8-bit | `NPU` fp32 | `NPU` 8-bit |
|---|---:|---:|---:|---:|---:|---:|---:|
| LUDB val | 0.9633 | 0.9633 | 0.9633 | 0.9638 | 0.9634 | 0.9638 | 0.9633 |
| ISP test | 0.9711 | 0.9711 | 0.9715 | 0.9711 | 0.9715 | 0.9711 | 0.9713 |
| QTDB T-subset | 0.9076 | 0.9076 | 0.9075 | 0.9076 | 0.9079 | 0.9083 | 0.9090 |
| **Unweighted mean** | **0.9473** | **0.9473** | **0.9474** | **0.9475** | **0.9476** | **0.9477** | **0.9479** |

`CPU` and `NPU` values are identical on both hosts; `GPU` is Host 2 (Arc 140T). The TFLite per-corpus cells match the README table (~0.963 / 0.971 / 0.908).

### Codec agreement vs ONNX Runtime (gated channels, `agree_openvino`)

| Device | synth_sinus | mitdb100 |
|---|---:|---:|
| `CPU` (both hosts) | 0.9995 | 0.9987 |
| `GPU` (Host 2, f32 hint) | 0.9989 | 0.9989 |
| `NPU` (both hosts) | 0.9985 | 0.9988 |

Boundary frame-argmax agreement vs TFLite on the bundled samples (`CPU`): fp32 1.000; 8-bit 0.994–0.996.

### Latency per 10 s window (`bench_openvino.py --runs 50 --warmup 5`, sample mitdb100)

mean / p50 / p90 in ms. Reference rows run on the host CPU in the same process.

**Host 1: Intel Core Ultra 7 265F**

| Path | `--device CPU` | `--device NPU` |
|---|---:|---:|
| Boundary, TFLite / LiteRT (ref, CPU) | 23.80 / 23.71 / 24.12 | 23.77 / 23.70 / 24.07 |
| Boundary, OpenVINO 8-bit | 2.32 / 2.21 / 2.66 | 4.41 / 3.78 / 6.45 |
| Codec, ONNX Runtime int8 (ref, CPU) | 5.71 / 5.83 / 5.97 | 5.63 / 5.41 / 6.05 |
| Codec, OpenVINO (bundled int8 ONNX) | 6.37 / 6.22 / 6.87 | 19.82 / 19.93 / 22.00 |

**Host 2: Intel Core Ultra 9 285H (Arc 140T iGPU)**

| Path | `--device CPU` | `--device GPU` | `--device NPU` |
|---|---:|---:|---:|
| Boundary, TFLite / LiteRT (ref, CPU) | 25.45 / 25.27 / 26.44 | 25.25 / 25.08 / 25.86 | 25.30 / 25.17 / 26.11 |
| Boundary, OpenVINO 8-bit | 3.75 / 3.73 / 4.09 | 1.45 / 1.42 / 1.57 | 4.78 / 4.11 / 7.43 |
| Codec, ONNX Runtime int8 (ref, CPU) | 9.36 / 7.47 / 12.21 | 10.15 / 7.84 / 15.50 | 9.48 / 6.93 / 11.82 |
| Codec, OpenVINO (bundled int8 ONNX) | 9.23 / 9.20 / 9.93 | 7.29 / 7.22 / 7.68 (f32 hint) | 18.65 / 18.62 / 19.14 |

On Host 2, the `GPU` codec with the default f16 precision runs at 2.81 / 2.76 / 3.01 ms, but at the lower agreement noted above.

Host 2 latencies were measured from the interactive desktop session. When the same commands were launched from a non-interactive SSH session, Windows scheduled them onto the low-power E-cores, and CPU-side latencies were 3–10× higher and bimodal. Accuracy results did not change.

Notes:

- On `NPU`, the shipped codec (dynamic-quantized int8 ONNX) takes ~19 ms per window on both hosts.
- In a separate experiment (not included in this tree), a static-shape fp32 ONNX re-export of `codec_v6.pt` ran at 5.67 / 5.51 / 6.22 ms on Host 1 `NPU` and 4.26 / 4.17 / 4.58 ms on Host 1 `CPU`. It agrees with the torch fp32 model at 0.999 (frame argmax), but its gated agreement vs the int8 ONNX Runtime reference is 0.9946. That is below the 0.998 gate used here; the torch fp32 model itself scores 0.9948 vs that reference.

## Reproduce macro-F1

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
