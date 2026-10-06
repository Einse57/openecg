# OpenVINO CPU smoke + bench for openecg

**Target fork:** `Einse57/openecg` (draft PR; this branch is local-only until `gh` auth / push is configured)

**Branch:** `openvino-cpu-npu`

## Summary

Feasibility pass for running the public OpenECG deploy artifacts through **Intel OpenVINO** on CPU (with `--device CPU|GPU|NPU` plumbing for Andrew's Core / Xeon / NPU machines).

| Artifact | Reference path | OpenVINO |
|---|---|---|
| `boundary_int8.tflite` (~1.5 MB, v56c delineator) | TFLite / `ai-edge-litert` ✅ | ❌ TFLite frontend fails (see below) |
| `codec_v6_int8.onnx` (~3.6 MB, layered codec) | ONNX Runtime ✅ | ✅ Direct ONNX read + CPU infer |

## What was added

- `openecg/openvino_backend.py` — `OpenVINOCodec` (drop-in for `OnnxCodec`) + `OpenVINOBoundary` (raises a clear error today) + `available_devices()`
- `bench_openvino.py` — one-command bench: `--device CPU|GPU|NPU --runs N`
- `scripts/agree_openvino.py` — ORT vs OpenVINO logit / argmax agreement
- `scripts/make_samples.py` — synth sinus + optional PhysioNet MIT-BIH 100
- `docs/openvino.md` — failure notes + quickstart
- `requirements-openvino.txt` + `requirements-openvino.lock.txt` — pinned smoke env
- `data/samples/README.md` — how to regenerate `.npy` windows (binaries gitignored)

## License findings

- **Code license:** Apache License 2.0 (`LICENSE`; `pyproject.toml` `license = "Apache-2.0"`).
- **Copyright notice:** `Copyright 2026 Hyung-Chul Lee and OpenECG contributors`.
- **Weights license:** No separate weights LICENSE. Bundled `.tflite` / `.onnx` / `.pt` ship inside the Apache-2.0 tree; model cards do not declare a different license. Treat weights as **Apache-2.0** with the package unless upstream clarifies otherwise.
- Training corpora cited in the README are PhysioNet/Zenodo (CC-BY / ODC-BY) and are **not** redistributed in-repo.

## How to run

```bash
git clone <this-fork> openecg && cd openecg
git checkout openvino-cpu-npu
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-openvino.lock.txt
pip install -e .
python scripts/make_samples.py --with-mitdb   # needs network + wfdb
python scripts/agree_openvino.py --device CPU
python bench_openvino.py --device CPU --runs 50
```

Code supports `--device CPU|GPU|NPU`. Missing devices raise a clear `RuntimeError` listing `available_devices`.

## Reference reproduction

- Bundled TFLite boundary + ONNX codec both run on CPU with the pinned env (`ai-edge-litert` used because `tflite-runtime` has no cp313 wheel).
- Samples: synthetic sinus + PhysioNet MIT-BIH record 100 lead MLII resampled to 250 / 500 Hz.
- Example TFLite output on MIT-BIH 100: P/QRS boundaries decoded; codec report: *"Sinus rhythm, HR 75 bpm, regular, no ectopy. (AFib rule: negative.)"*
- **README macro-F1 0.9274 / 44 ms baseline not re-measured on LUDB/ISP/QTDB** — the repo ships only `data/splits/*.json` IDs, not the waveforms. Full eval needs PhysioNet downloads + `scripts/benchmark_v56c.py`.

## OpenVINO conversion

### Codec ONNX — success

```text
ov.Core().read_model("openecg/models/codec_v6_int8.onnx")  # OK
compile_model(..., "CPU")  # OK
outputs: frame_logits [?,5000,4], beat_logits [?,5000,6], rhythm_logits [?,5000,6]
```

### Boundary TFLite — failed (exact error)

```text
FrontEnd API failed with GeneralFailure:
This tensor should be either input, constant or should be already produced by
previous operators: ...Linear_patch_embed...
Error is encountered while working with operation of type ADD and name 3.
```

Also failed: `ov.convert_model(.tflite)`, `tflite2onnx` (AssertionError on Add inputs), `tf2onnx.from_tflite` (unsupported `TFL_GELU`, broken dequant Add / topological_sort).

**Unblock path:** export v56c boundary to ONNX from the torch checkpoint (not in this public tree) and load via OpenVINO ONNX frontend.

## Agreement (OpenVINO codec vs ONNX Runtime, box)

| Sample | Head | max abs diff | argmax agreement | gated channels agree |
|---|---|---:|---:|---:|
| synth_sinus | frame | 0.504 | 0.9982 | |
| synth_sinus | beat | 0.765 | 0.9978 | **0.999** |
| synth_sinus | rhythm | 0.531 | 1.000 | |
| mitdb100 | frame | 0.444 | 0.9924 | |
| mitdb100 | beat | 0.876 | 0.9998 | **0.997** |
| mitdb100 | rhythm | 0.587 | 1.000 | |

Notes:

- Per-sample delineation agreement is high; logit MAD ~0.4–0.9 is expected across int8 ORT vs OpenVINO numerics.
- Macro-F1 over unused beat/rhythm classes is misleadingly low (many zero-support classes on sinus-only clips) — prefer argmax agreement / gated channel agreement.
- Codec emits **labels**, not a reconstructed waveform → reconstruction error N/A.
- Boundary OpenVINO agreement: **N/A** (load failed).

## Box bench numbers (NOT Intel-target)

Host: shared Linux box, OpenVINO device `CPU` = `Intel(R) Xeon(R) Processor`. Sample: `mitdb100` 10 s window. `runs=40`, `warmup=5`.

| Path | mean ms | p50 ms | p90 ms | windows/s | peak RSS MB |
|---|---:|---:|---:|---:|---:|
| Boundary TFLite/LiteRT (ref) | **16.6** | 16.3 | 18.2 | 60.1 | 114 |
| Boundary OpenVINO | — | — | — | — | unavailable |
| Codec ONNX Runtime (ref) | **13.7** | 13.1 | 18.6 | 73.0 | 176 |
| Codec OpenVINO | **17.5** | 14.3 | 21.1 | 57.1 | 265 |

**README published baseline:** 44 ms / 10 s window (TFLite int8), int8 macro-F1 0.9274 — different machine; do not compare directly to the box table above.

## Checklist for Andrew — Core / Xeon / NPU runs

- [ ] Clone fork branch `openvino-cpu-npu`, create venv, `pip install -r requirements-openvino.lock.txt && pip install -e .`
- [ ] `python scripts/make_samples.py --with-mitdb` (or point `--sample` / paths at local ECG)
- [ ] `python scripts/agree_openvino.py --device CPU` — confirm gated channel agreement ≥ ~0.99
- [ ] `python bench_openvino.py --device CPU --runs 100` — record mean/p50/p90 / RSS; label as Core or Xeon SKU + OS
- [ ] If iGPU / discrete GPU present: `bench_openvino.py --device GPU --runs 100`
- [ ] If Intel NPU present: `bench_openvino.py --device NPU --runs 100`
- [ ] Compare OpenVINO codec latency to ORT on the same SKU; note any driver / OpenVINO version deltas
- [ ] (Optional unblock) Export `boundary_int8` to ONNX from v56c torch ckpt; add to `openecg/models/` and wire `OpenVINOBoundary` to the ONNX path
- [ ] (Optional) Full LUDB/ISP/QTDB macro-F1 via `scripts/benchmark_v56c.py` after PhysioNet download — reproduce README 0.9274
- [ ] Paste SKU-labeled numbers back into this PR before marking ready-for-review

## Blockers

1. **Boundary TFLite → OpenVINO blocked** (frontend ADD / GELU). Codec path is the working OpenVINO deliverable today.
2. **No push / PR from this agent** — `gh` not logged in; branch is local under `/workspace/openecg`.
3. **README macro-F1 not reproduced here** — eval waveforms not shipped.
