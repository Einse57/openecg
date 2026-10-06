# OpenVINO CPU smoke + bench for openecg

**Target fork:** `Einse57/openecg` (draft; local branch only — no push yet)

**Branch:** `openvino-cpu-npu`

## Summary

OpenVINO path for both bundled deploy artifacts on CPU (`--device CPU|GPU|NPU`):

| Artifact | Reference | OpenVINO |
|---|---|---|
| Boundary v56c | TFLite / LiteRT ✅ | ✅ via recovered ONNX + NNCF wc8 (TFLite frontend still broken) |
| Codec v6 | ONNX Runtime ✅ | ✅ direct ONNX read — **clean path, no weight recovery** |

## What was added

- `openecg/openvino_backend.py` — `OpenVINOCodec`, `OpenVINOBoundary` (ONNX/wc8)
- `scripts/export_boundary_onnx.py` — TFLite→torch recover → ONNX (opset 17) → NNCF INT8
- `openecg/models/boundary_v56c.onnx`, `boundary_v56c_wc8.xml` (+ `.bin`)
- `bench_openvino.py` — one-command latency / throughput / RSS
- `scripts/agree_openvino.py`, `scripts/eval_macro_f1_openvino.py` (LUDB+ISP+QTDB)
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

PhysioNet / Zenodo eval data (gitignored under `data/physionet/`):

```bash
export OPENECG_LUDB_ZIP=data/physionet/ludb-1.0.1.zip
export OPENECG_LUDB_CACHE=data/physionet/ludb_cache
export OPENECG_ISP_ZIP=data/physionet/isp_delineation_dataset.zip   # Zenodo 14679837
export OPENECG_ISP_CACHE=data/physionet/isp_cache
export OPENECG_QTDB_CACHE=data/physionet/qtdb_cache                 # full 105 records
```

## Export result (boundary)

- **No upstream boundary `.pt` in the public tree** (only `boundary_int8.tflite` + codec `.pt` files).
- Recovered dequantized TFLite weights into `vit_transformer_noaux_1ch` (0.99 M params).
- Fitted upper LayerNorm + `patch_embed.bias` to match TFLite logits.
- Exported `boundary_v56c.onnx` (opset 17, static 10 s / 2500 samples) and NNCF `INT8_ASYM` → `boundary_v56c_wc8.xml`.

### Follow-up (weight recovery gap)

The OV boundary Martinez gap vs TFLite (e.g. LUDB **0.872 vs 0.963**) is from **imperfect TFLite→torch weight recovery** — there was no upstream v56c `.pt` to export from. Event-level smoke agreement is still **F1 = 1.00** (±20 samples) despite that Martinez gap.

**Ask upstream for** the official `stage2_…_v56c.pt` checkpoint, **or** an official ONNX export of the boundary model, then re-run `scripts/export_boundary_onnx.py` / replace the recovered IR for bit-exact parity.

## Codec OpenVINO path (clean)

Codec is a **clean OpenVINO path**: bundled `codec_v6_int8.onnx` loads and compiles directly (no TFLite recovery, no re-export). Gated channel agreement vs ONNX Runtime is **≥ 0.997**. Treat codec OV as production-ready for CPU smoke; boundary OV is provisional until an official checkpoint lands.

## Agreement (OpenVINO boundary vs TFLite reference)

Smoke windows (mitdb100 / synth_sinus), event F1 within ±20 samples:

| Sample | argmax agree | max abs diff (logits) | boundary event F1 |
|---|---:|---:|---:|
| mitdb100 | 0.978 | 3.01 | **1.00** |
| synth_sinus | 0.968 | 3.11 | **1.00** |

Codec OV vs ORT: gated channel agreement ≥ 0.997 (no weight recovery).

## Macro-F1 (Martinez) vs README 0.9274

README per-corpus table (lead II / first lead) and deploy-table headline **0.9274** (TFLite int8 mean over LUDB+ISP+QTDB). This smoke used the repo splits: LUDB val (41), ISP test (72), QTDB T-subset (44).

| Corpus | n | TFLite ref | OpenVINO wc8 | README |
|---|---:|---:|---:|---:|
| LUDB val (lead II) | 41 | **0.9633** | 0.8715 | 0.963 |
| ISP test (lead II) | 72 | **0.9711** | 0.9166 | 0.971 |
| QTDB T-subset (first lead) | 44 | **0.9076** | 0.7207 | 0.908 |
| **Unweighted mean (3 corpora)** | — | **0.9473** | **0.8363** | deploy row **0.9274** |

- TFLite **per-corpus** numbers match the README table to ~0.001.
- Unweighted mean of the three README per-corpus cells is ~0.947; the deploy-table **0.9274** is cited as-is (exact aggregation not specified beyond LUDB+ISP+QTDB).
- OV gap is the TFLite→torch recovery limitation above (event smoke F1 still 1.00).
- Data status: **LUDB** zip ✅, **ISP** Zenodo `isp_delineation_dataset.zip` ✅, **QTDB** full 105 records ✅ (via PhysioNet files; T-subset n=44).

## Box bench (**BOX only**, not Intel-target)

Host: shared Linux box, OpenVINO `CPU` = Xeon. Sample `mitdb100`, runs=40, warmup=5. Re-run after full PhysioNet land (code/models unchanged; latency noise only).

| Path | mean ms | p50 | p90 | win/s | peak RSS MB |
|---|---:|---:|---:|---:|---:|
| Boundary TFLite | 15.8 | 15.5 | 18.3 | 63.2 | 115 |
| Boundary OpenVINO (wc8) | **3.8** | 3.5 | 5.1 | 263 | 175 |
| Codec ORT | 12.6 | 11.8 | 15.8 | 79.4 | 227 |
| Codec OpenVINO | 18.5 | 18.4 | 21.5 | 53.9 | 287 |

README published baseline: **44 ms** / 10 s window (different machine).

## Checklist for Andrew — Core / Xeon / NPU

- [ ] `pip install -r requirements-openvino.lock.txt && pip install -e .`
- [ ] `python scripts/make_samples.py --with-mitdb`
- [ ] Fetch LUDB + ISP + QTDB; set `OPENECG_*` env vars (see above)
- [ ] `python -m scripts.eval_macro_f1_openvino --device CPU --leads ii`
- [ ] `python bench_openvino.py --device CPU --runs 100` — label SKU
- [ ] Repeat with `--device GPU` / `--device NPU` if present
- [ ] **Follow-up:** obtain official v56c `.pt` (or official ONNX) from upstream; re-export for bit-exact OV boundary parity
- [ ] Paste SKU-labeled numbers before marking PR ready

## Blockers / honesty

1. Official boundary `.pt` not in public repo — ONNX recovered from TFLite (OV LUDB F1 ~0.87 vs TFLite 0.96). Ask upstream for checkpoint / official ONNX.
2. Codec OV is clean (direct ONNX, agree ≥0.997).
3. Still **no push / PR** from this agent (`gh` auth not set up). Working tree is ready to push once auth is green.
