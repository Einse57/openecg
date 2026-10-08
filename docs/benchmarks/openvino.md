# OpenVINO path — tested hosts and results

Accuracy and latency of `openecg.openvino_backend` (see [`docs/openvino.md`](../openvino.md))
next to the reference runtimes, on the hosts below.

## Hosts

| | Host 1 | Host 2 |
|---|---|---|
| CPU | Intel Core Ultra 7 265F | Intel Core Ultra 9 285H |
| OpenVINO devices tested | `CPU`, `NPU` (Intel AI Boost) | `CPU`, `GPU` (Intel Arc 140T integrated GPU), `NPU` (Intel AI Boost) |
| OS / runtime | Windows 11 (10.0.26200), Python 3.12.10, OpenVINO 2026.4.1 | Windows 11 Pro (10.0.26200), Python 3.12.10, OpenVINO 2026.4.1 |
| Drivers | — | GPU 32.0.101.8860, NPU 32.0.100.4841 |

Host 1 has no integrated GPU, so `GPU` results are from Host 2 only.

Reference runtimes (TFLite / LiteRT for the boundary model, ONNX Runtime for the codec)
always run on the host CPU.

## Boundary macro-F1 (Martínez tolerances)

Lead II for LUDB / ISP, first lead for the QTDB T-subset; repo splits (LUDB val n=41,
ISP test n=72, QTDB T-subset n=44). "fp32" = `boundary_v56c.onnx`, "8-bit" =
`boundary_v56c_wc8.xml`. `scripts/eval_macro_f1_openvino.py --leads ii`.

| Corpus | TFLite (ref) | `CPU` fp32 | `CPU` 8-bit | `GPU` fp32 | `GPU` 8-bit | `NPU` fp32 | `NPU` 8-bit |
|---|---:|---:|---:|---:|---:|---:|---:|
| LUDB val | 0.9633 | 0.9633 | 0.9633 | 0.9638 | 0.9634 | 0.9638 | 0.9633 |
| ISP test | 0.9711 | 0.9711 | 0.9715 | 0.9711 | 0.9715 | 0.9711 | 0.9713 |
| QTDB T-subset | 0.9076 | 0.9076 | 0.9075 | 0.9076 | 0.9079 | 0.9083 | 0.9090 |
| **Unweighted mean** | **0.9473** | **0.9473** | **0.9474** | **0.9475** | **0.9476** | **0.9477** | **0.9479** |

TFLite, `CPU` and `NPU` values are identical on both hosts; `GPU` is Host 2 (Arc 140T).
The TFLite cells match the README table (0.963 / 0.971 / 0.908).

## Codec agreement vs ONNX Runtime

Gated channel agreement (frame / QRS-gated beat / rhythm) with the int8 ONNX Runtime
reference, `scripts/agree_openvino.py`. Gate: ≥ 0.998.

| Device | synth_sinus | mitdb100 |
|---|---:|---:|
| `CPU` (both hosts) | 0.9995 | 0.9987 |
| `GPU` (Host 2, f32 hint) | 0.9989 | 0.9989 |
| `GPU` (Host 2, device default f16) | 0.981 | 0.993 |
| `NPU` (both hosts) | 0.9985 | 0.9988 |

Boundary frame-argmax agreement vs TFLite on the same samples (`CPU`): fp32 1.000;
8-bit 0.994–0.996.

## Latency per 10 s window

`scripts/bench_openvino.py --runs 50 --warmup 5 --sample mitdb100`; mean / p50 / p90 in ms.
Reference rows run on the host CPU in the same process.

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

With the device-default f16 precision, the Host 2 `GPU` codec runs at 2.81 / 2.76 / 3.01 ms,
at the lower agreement shown above.

Notes:

- Host 2 latencies were measured from the interactive desktop session. Launched from a
  non-interactive SSH session, Windows scheduled the process onto the E-cores and
  CPU-side latencies were 3–10× higher and bimodal; accuracy did not change.
- On `NPU`, the bundled codec (dynamic-quantized int8 ONNX) takes ~19 ms per window on
  both hosts.
