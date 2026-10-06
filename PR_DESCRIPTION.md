# OpenVINO CPU smoke + bench for openecg

**Target fork:** `Einse57/openecg` (draft; local branch only — no push yet)

**Branch:** `openvino-cpu-npu`

## Summary

OpenVINO path for both bundled deploy artifacts on CPU (`--device CPU|GPU|NPU`):

| Artifact | Reference | OpenVINO |
|---|---|---|
| Boundary v56c | TFLite / LiteRT ✅ | ✅ via recovered ONNX + NNCF wc8 (TFLite frontend still broken) |
| Codec v6 | ONNX Runtime ✅ | ✅ direct ONNX read |

## What was added

- `openecg/openvino_backend.py` — `OpenVINOCodec`, `OpenVINOBoundary` (ONNX/wc8)
- `scripts/export_boundary_onnx.py` — TFLite→torch recover → ONNX (opset 17) → NNCF INT8
- `openecg/models/boundary_v56c.onnx`, `boundary_v56c_wc8.xml` (+ `.bin`)
- `bench_openvino.py` — one-command latency / throughput / RSS
- `scripts/agree_openvino.py`, `scripts/eval_macro_f1_openvino.py`
- `scripts/make_samples.py`, pinned `requirements-openvino*.txt`, `docs/openvino.md`

## License

- **Code + bundled weights:** Apache-2.0 (`LICENSE`, Copyright 2026 Hyung-Chul Lee and OpenECG contributors).
- No separate weights license file.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-openvino.lock.txt
pip install -e .
python scripts/make_samples.py --with-mitdb
# Boundary ONNX already committed; to regenerate:
#   python -m scripts.export_boundary_onnx --nncf
python scripts/agree_openvino.py --device CPU
python -m scripts.eval_macro_f1_openvino --device CPU --leads ii
python bench_openvino.py --device CPU --runs 50
```

PhysioNet LUDB (required for macro-F1):

```bash
# zip already used in this smoke lives under data/physionet/ (gitignored)
export OPENECG_LUDB_ZIP=data/physionet/ludb-1.0.1.zip
export OPENECG_LUDB_CACHE=data/physionet/ludb_cache
```

## Export result (boundary)

- **No upstream boundary `.pt` in the public tree** (only `boundary_int8.tflite` + codec `.pt` files).
- Recovered dequantized TFLite weights into `vit_transformer_noaux_1ch` (0.99 M params).
- Fitted upper LayerNorm + `patch_embed.bias` to match TFLite logits.
- Exported `boundary_v56c.onnx` (opset 17, static 10 s / 2500 samples) and NNCF `INT8_ASYM` → `boundary_v56c_wc8.xml`.

## Agreement (OpenVINO boundary vs TFLite reference)

Smoke windows (mitdb100 / synth_sinus), event F1 within ±20 samples:

| Sample | argmax agree | max abs diff (logits) | boundary event F1 |
|---|---:|---:|---:|
| mitdb100 | 0.978 | 3.01 | **1.00** |
| synth_sinus | 0.968 | 3.11 | **1.00** |

Codec OV vs ORT (unchanged from prior): gated channel agreement ≥ 0.997.

## Macro-F1 (Martinez, lead II) vs README 0.9274

README **0.9274** = mean over LUDB+ISP+QTDB TFLite int8. This smoke:

| Backend | LUDB val (n=41) | QTDB T-subset (partial download, n=13) |
|---|---:|---:|
| **TFLite reference** | **0.9633** (matches README LUDB **0.963**) | 0.940 |
| **OpenVINO (wc8 ONNX)** | 0.872 | 0.831 |

- ISP not downloaded here (loader needs separate zip).
- OV gap vs TFLite is from imperfect TFLite→torch weight recovery (no official `.pt`); event-level smoke F1 is still 1.0.
- Mean LUDB+QTDB TFLite on this partial set ≠ published 0.9274 (missing ISP + full QTDB T-subset).

## Box bench (**BOX only**, not Intel-target)

Host: shared Linux box, OpenVINO `CPU` = Xeon. Sample `mitdb100`, runs=40, warmup=5.

| Path | mean ms | p50 | p90 | win/s | peak RSS MB |
|---|---:|---:|---:|---:|---:|
| Boundary TFLite | 15.2 | 15.1 | 15.7 | 65.6 | 115 |
| Boundary OpenVINO (wc8) | **2.9** | 2.9 | 3.1 | 342 | 174 |
| Codec ORT | 9.0 | 8.2 | 10.9 | 112 | 226 |
| Codec OpenVINO | 10.3 | 10.3 | 11.3 | 97 | 287 |

README published baseline: **44 ms** / 10 s window (different machine).

## Checklist for Andrew — Core / Xeon / NPU

- [ ] `pip install -r requirements-openvino.lock.txt && pip install -e .`
- [ ] `python scripts/make_samples.py --with-mitdb`
- [ ] `python -m scripts.eval_macro_f1_openvino --device CPU --leads ii` (with LUDB zip)
- [ ] Optional: full ISP + complete QTDB for README-mean 0.9274 reproduction
- [ ] `python bench_openvino.py --device CPU --runs 100` — label SKU
- [ ] Repeat with `--device GPU` / `--device NPU` if present
- [ ] Optional: obtain official v56c `.pt` and re-export ONNX for bit-exact OV parity
- [ ] Paste SKU-labeled numbers before marking PR ready

## Blockers / honesty

1. Official boundary `.pt` not in public repo — ONNX recovered from TFLite (OV LUDB F1 0.87 vs TFLite 0.96).
2. ISP not fetched; QTDB partial at time of writing.
3. Still **no push / PR** from this agent (`gh` auth not set up).
