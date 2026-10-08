# OpenVINO path

`openecg.openvino_backend` runs both bundled deploy models on OpenVINO, alongside the
reference runtimes. It is optional; nothing else in openecg imports it.

```bash
pip install "openecg[openvino]"
```

```python
from openecg.openvino_backend import OpenVINOBoundary, OpenVINOCodec, available_devices
import openecg

print(available_devices())                     # e.g. ['CPU', 'GPU', 'NPU']
det = OpenVINOBoundary(device="CPU")           # v56c boundary, 8-bit IR
cls_logits = det.forward_window(sig_250hz)     # (500, 4) for one 2500-sample window
codec = openecg.encode(sig_500hz, fs=500, model=OpenVINOCodec(device="NPU"))
```

`device` takes any OpenVINO device name available on the host, e.g. `CPU` (default),
`GPU`, `NPU`, or `AUTO`; an absent device raises a clear error.

## Artifacts

| Model | Reference runtime | OpenVINO input |
|---|---|---|
| Boundary detector (v56c) | `boundary_int8.tflite` via TFLite / LiteRT | `boundary_v56c_wc8.xml` (8-bit weights, default) or `boundary_v56c.onnx` (fp32, `prefer_wc8=False`) |
| Layered codec (v6) | `codec_v6_int8.onnx` via ONNX Runtime | the same ONNX file, read directly |

The OpenVINO TFLite frontend cannot load `boundary_int8.tflite`, so the boundary model
ships as an exact fp32 rebuild of the TFLite weights plus an NNCF `INT8_ASYM`
weight-compressed IR. How they were built, and their macro-F1, is in
[`boundary_v56c_MODEL_CARD.md`](../openecg/models/boundary_v56c_MODEL_CARD.md).
Regenerate with:

```bash
pip install "openecg[openvino,openvino-export,openvino-bench]"
python -m scripts.export_boundary_onnx --nncf
```

## Compile steps (applied automatically)

- **All devices:** dynamic input dims are fixed to batch 1 before `compile_model`
  (codec input `[1, 5000]`). `NPU` needs static shapes to compile the codec.
- **`GPU`:** 3-D `Interpolate` nodes are run as Unsqueeze → 4-D Interpolate → Squeeze,
  which is numerically identical. Without it, the codec's 1-D nearest upsample followed
  by a 1-D Conv fails GPU program build ("Data batch and filters rank do not match").

## Precision

`OpenVINOCodec` and `OpenVINOBoundary` take an OpenVINO `config=` dict. When it is not
given, they set `INFERENCE_PRECISION_HINT=f32` on:

- **`CPU`** (both classes), so results are reproducible across hosts. Without the hint,
  OpenVINO picks bf16 on CPUs with native bf16 support.
- **`GPU`** (codec), because with the device default (f16) the gated agreement vs ONNX
  Runtime falls below 0.998.

Other devices use their own defaults. To opt into bf16 on a CPU that supports it, or
into the device default:

```python
OpenVINOBoundary(device="CPU", config={"INFERENCE_PRECISION_HINT": "bf16"})
OpenVINOCodec(device="GPU", config={})   # device default (f16 on GPU)
```

The macro-F1 and agreement results in `docs/benchmarks/openvino.md` are for the
defaults above.

## Tested on

- Intel Core Ultra 7 265F: `CPU`, `NPU`
- Intel Core Ultra 9 285H: `CPU`, `GPU` (Intel Arc 140T integrated GPU), `NPU`
- Windows 11, Python 3.12, OpenVINO 2026.4.1

On every tested device, boundary macro-F1 is within 0.001 of the TFLite reference
(0.9473, mean of LUDB / ISP / QTDB) and codec gated agreement vs ONNX Runtime is ≥ 0.998. Per-device macro-F1, agreement and latency are in
[`docs/benchmarks/openvino.md`](benchmarks/openvino.md).

## Reproduce

Sample windows are generated locally under `data/samples/` (not committed):

```bash
pip install "openecg[openvino,openvino-bench,loaders]"
python scripts/make_samples.py                # synthetic sinus @ 250 / 500 Hz
python scripts/make_samples.py --with-mitdb   # + PhysioNet MIT-BIH record 100 (needs network)

python scripts/agree_openvino.py --device CPU                       # codec vs ONNX Runtime
python scripts/bench_openvino.py --device CPU --runs 50 --sample mitdb100
```

Boundary macro-F1 on LUDB / ISP / QTDB (PhysioNet / Zenodo data, not redistributed):

```bash
export OPENECG_LUDB_ZIP=data/physionet/ludb-1.0.1.zip
export OPENECG_LUDB_CACHE=data/physionet/ludb_cache
export OPENECG_ISP_ZIP=data/physionet/isp_delineation_dataset.zip   # Zenodo 14679837
export OPENECG_ISP_CACHE=data/physionet/isp_cache
export OPENECG_QTDB_CACHE=data/physionet/qtdb_cache
python -m scripts.eval_macro_f1_openvino --device CPU --leads ii
```

Tests: `pytest tests/test_openvino_backend.py` (skipped when `openvino` is not installed).
