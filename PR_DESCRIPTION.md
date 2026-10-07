# OpenVINO path for openecg deploy artifacts

## Summary

Adds an OpenVINO path for both bundled deploy artifacts:

| Artifact | Reference | OpenVINO |
|---|---|---|
| Boundary v56c | TFLite / LiteRT | fp32 ONNX (exact rebuild from the TFLite file) or NNCF INT8 weight-compressed IR |
| Codec v6 | ONNX Runtime | bundled `codec_v6_int8.onnx`, read directly |

On LUDB / ISP / QTDB (lead II), boundary macro-F1 through OpenVINO matches the TFLite reference. Codec gated channel agreement vs ONNX Runtime is ≥ 0.998.

## What was added

- `openecg/openvino_backend.py`: `OpenVINOCodec`, `OpenVINOBoundary`. Dynamic input dims are fixed to batch 1 before compile (codec input `[1, 5000]`) so devices that need static shapes can compile the codec.
- `scripts/export_boundary_onnx.py`: exact TFLite → fp32 torch mapping → ONNX (opset 17) → optional NNCF INT8 weight compression.
- `openecg/models/boundary_v56c.onnx`, `boundary_v56c_wc8.xml` (+ `.bin`)
- `bench_openvino.py`: one-command latency / throughput / RSS
- `scripts/agree_openvino.py`, `scripts/eval_macro_f1_openvino.py` (LUDB + ISP + QTDB)
- `scripts/make_samples.py`, pinned `requirements-openvino*.txt`, `docs/openvino.md`

## Boundary export

Upstream ships only `boundary_int8.tflite`, which is weight-only int8. The OpenVINO TFLite frontend can't load it, so the export rebuilds the `vit_transformer_noaux_1ch` graph (0.99 M params) from the dequantized TFLite tensors. Nothing is fitted:

- `patch_embed.bias` is folded into the positional-encoding constant, so it is set to 0.
- The upper pre-LN LayerNorm affine params are folded into the following FC weights, so upper `norm1` / `norm2` are identity.
- The lower LayerNorm constants are consumed in reverse tensor order.

The fp32 graph reproduces the TFLite logits to ~1e-5, and the export asserts this on the bundled samples.

## Results

### Macro-F1 (Martínez), lead II / first lead, repo splits

| Corpus | n | TFLite ref | OpenVINO fp32 ONNX | OpenVINO wc8 | README |
|---|---:|---:|---:|---:|---:|
| LUDB val | 41 | 0.9633 | 0.9633 | 0.9633 | 0.963 |
| ISP test | 72 | 0.9711 | 0.9711 | 0.9715 | 0.971 |
| QTDB T-subset | 44 | 0.9076 | 0.9076 | 0.9075 | 0.908 |
| **Unweighted mean** | — | **0.9473** | **0.9473** | **0.9474** | deploy row 0.9274 |

The deploy-table **0.9274** is cited as-is; its exact aggregation isn't specified beyond LUDB + ISP + QTDB.

### Agreement

| Check | mitdb100 | synth_sinus |
|---|---:|---:|
| Boundary fp32 ONNX vs TFLite, frame argmax | 1.000 | 1.000 |
| Boundary wc8 vs TFLite, frame argmax | 0.994 | 0.996 |
| Codec vs ONNX Runtime, gated channels | 0.9987 | 0.9995 |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-openvino.lock.txt
pip install -e .
python scripts/make_samples.py --with-mitdb
# Boundary ONNX/IR are committed; to regenerate:
#   python -m scripts.export_boundary_onnx --nncf
python scripts/agree_openvino.py --device CPU
python -m scripts.eval_macro_f1_openvino --device CPU --leads ii
python bench_openvino.py --device CPU --runs 50
```

PhysioNet / Zenodo eval data (gitignored under `data/physionet/`):

```bash
export OPENECG_LUDB_ZIP=data/physionet/ludb-1.0.1.zip
export OPENECG_LUDB_CACHE=data/physionet/ludb_cache
export OPENECG_ISP_ZIP=data/physionet/isp_delineation_dataset.zip   # Zenodo 14679837
export OPENECG_ISP_CACHE=data/physionet/isp_cache
export OPENECG_QTDB_CACHE=data/physionet/qtdb_cache                 # full 105 records
```

## License

- **Code and bundled weights:** Apache-2.0 (`LICENSE`, Copyright 2026 Hyung-Chul Lee and OpenECG contributors).
- There is no separate weights license file.
