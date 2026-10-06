# OpenVINO feasibility (CPU smoke)

This tree adds an OpenVINO path for the bundled deploy artifacts so Intel
Core / Xeon / NPU hosts can run the same smoke + bench that was validated
on a shared Linux box.

## Artifacts

| Model | File | Size | Reference runtime | OpenVINO |
|---|---|---:|---|---|
| Boundary delineator (v56c) | `openecg/models/boundary_int8.tflite` | ~1.5 MB | TFLite / `ai-edge-litert` | **Blocked** — TFLite frontend fails |
| Layered codec (v6) | `openecg/models/codec_v6_int8.onnx` | ~3.6 MB | ONNX Runtime | **Works** — direct ONNX read |

## Boundary TFLite failure (exact)

```
FrontEnd API failed with GeneralFailure:
This tensor should be either input, constant or should be already produced
by previous operators: ...Linear_patch_embed...
Error is encountered while working with operation of type ADD and name 3.
```

Tried and failed: `ov.Core().read_model(.tflite)`, `ov.convert_model(.tflite)`,
`tflite2onnx`, `tf2onnx.convert.from_tflite` (unsupported `TFL_GELU`, broken
dequant `Add`).

**Next step for full parity:** export v56c boundary to ONNX from the torch
checkpoint (not shipped in this repo) and load that via OpenVINO.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-openvino.lock.txt
pip install -e .
python scripts/make_samples.py --with-mitdb   # optional PhysioNet pull
python scripts/agree_openvino.py --device CPU
python bench_openvino.py --device CPU --runs 50
```

`--device` accepts `CPU|GPU|NPU`. Only CPU is required for this smoke; GPU/NPU
raise a clear error if absent.

## License

Code + bundled weights: **Apache-2.0** (`LICENSE`, Copyright 2026 Hyung-Chul Lee
and OpenECG contributors). No separate weights license file; model cards do not
state a different license. Training corpora cited in the README are PhysioNet /
Zenodo (CC-BY / ODC-BY) and are **not** redistributed here.
