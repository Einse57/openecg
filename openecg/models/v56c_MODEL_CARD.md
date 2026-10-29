# openecg boundary v56c — fp32 reference checkpoint

The **boundary delineation** model behind `openecg/models/boundary_int8.tflite`,
published in full precision so third-party runtimes can anchor to the original
weights instead of reconstructing them from the quantized artifact.

```
window (2500 samples, 10 s @ 250 Hz, rank-normalized)
  ->  logits[500 frames, 4 classes]        # other / P / QRS / T
```

| asset | what it is | checksums |
|---|---|---|
| `v56c.pt` | torch fp32 checkpoint (4,259,434 B) | md5 `4354112e1ff326314b839a558b013169`<br>sha256 `bde39ab38fcfe461f52b4b97d632c8e8c04b777727964f225de04523fe718156` |
| `v56c_fp32.onnx` | fp32 ONNX, opset 17, self-contained (4,272,229 B) | md5 `d3ef04d94d6b841f8dba8f7f8a55bf7d`<br>sha256 `6ae046ee058cbe05b28176d77760e19c2587efcab027ef13ea1990b1283a6c8e` |
| `boundary_int8.tflite` | int8 deploy artifact, ships in-repo (1.4 MB) | — |

## Architecture & build
1-channel Conv + Transformer, **992,266 params** — `n_layers=8`, `d_model=128`,
`mid_split=4`, `patch_size=5`, no lead embedding. Trained with **soft-T α=0.9**
(the `v56` soft-boundary sweep's optimum; α ∈ {0.3, 0.5, 0.9, 1.0} were run,
α=0.9 won), best epoch **10** of 30, selection score **0.9299**.

## Training data — public corpora only
**LUDB + QTDB + ISP** wave annotation, plus a **synthetic AV-block mix**. All
real corpora are PhysioNet-/Zenodo-distributed under CC-BY / ODC-BY. No hospital
data and no patient-identifiable material went into this model — the lydus
hospital rhythm corpus enters only at the `codec_v5`/`codec_v6` rhythm head, not
here.

## Evaluation

Two protocols appear in this repo and they are **not interchangeable** — quote
whichever matches the comparison you are making.

**(a) Baseline comparison** — `scripts/benchmark_v56c.py`, lead II, Martínez 2004
per-boundary tolerances (P 50 ms, QRS 40 ms, T_on 50 ms, T_off 100 ms). This is
the table in [`docs/benchmarks/v56c_vs_baselines.md`](../../docs/benchmarks/v56c_vs_baselines.md)
and in the README:

| dataset (n records) | **v56c** | NeuroKit2 DWT | WTdelineator |
|---|---:|---:|---:|
| LUDB val (41) | **0.963** | 0.788 | 0.596 |
| ISP test (72) | **0.971** | 0.703 | 0.604 |
| QTDB T-subset (44) | **0.908** | 0.605 | 0.535 |

**(b) Training-time selection metric** — edge-filtered / random-subset splits used
to pick the checkpoint, *not* comparable to (a):

| split | boundary macro-F1 |
|---|---:|
| `ludb_edge_filtered` | 0.9456 |
| `isp_test` | 0.9737 |
| `qtdb_pu0_random` | 0.8704 |

**(c) fp32 ONNX parity.** The ONNX asset is the same graph the TFLite artifact is
built from (`_wrap_cls_reg_only` → `cls_logits[B,500,4]`, `reg_offsets[B,500,6]`),
exported by [`scripts/export_boundary_onnx.py`](../../scripts/export_boundary_onnx.py)
and scored against the torch fp32 model on 64 real LUDB windows:

| check | result |
|---|---|
| max \|Δ\| `cls_logits` | 1.14e−05 |
| max \|Δ\| `reg_offsets` | 6.49e−05 |
| frame argmax agreement | **100.0000%** (32,000 / 32,000) |
| batch 1 / 2 / 8 | OK |

The export is deterministic for a given toolchain — re-running reproduces the md5 above with torch 2.15.0,
onnx 1.23.2, onnxscript 0.7.2 (other versions change only file metadata; parity numbers are unchanged).
Reproduce with:

```bash
python -m scripts.export_boundary_onnx --ckpt data/checkpoints/v56c.pt \
    --out data/deploy/v56c_fp32.onnx --verify data/cache/v57_500hz/ludb_val.npz
```

**Quantization.** int8 is post-training, no fine-tune, and matches torch fp32
within sub-0.001 macro-F1 on this model (Δ macro-F1 = −0.0025 as measured in the
README's deploy table). An fp32 export is therefore a convenience for other
runtimes, not an accuracy upgrade.

## ⚠ Do not confuse v56c with codec_v6
`v56c` is the **boundary** model (4 frame classes, 250 Hz, 2500-sample window).
`codec_v6` is the **layered codec** (frame + beat + rhythm, 500 Hz) and reports a
frame boundary macro-F1 of **0.855** on LUDB val. That number and the **0.963**
above are different metrics on different benchmarks at different sample rates —
they are not a regression or an improvement over one another. Pin any comparison
to one protocol and say which.

## Intended use & limits
Research use. Single-lead (lead II or first available lead), 10 s windows. **Not a
medical device and not for clinical decision-making.** Performance outside the
training corpora's demographics, lead placements and acquisition hardware is
unvalidated.

## License
Apache-2.0, same as the repository.

## Citation
See the repository README.
