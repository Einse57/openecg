"""Export a 1-channel boundary checkpoint → fp32 ONNX, and verify parity.

Companion to :mod:`scripts.export_boundary_tflite`. Exports the *same* graph
the TFLite artifact is built from — ``_wrap_cls_reg_only`` → ``(cls_logits,
reg_offsets)`` — so the ONNX is a drop-in fp32 reference for third-party
runtimes rather than a differently-shaped model.

Two things this script does deliberately:

* **Exports with a batch>1 example and an explicit dynamic ``Dim``.** Tracing
  at batch=1 bakes the batch size into the attention reshapes; the resulting
  graph loads fine and then fails at run time for any batch > 1
  (``gemm_input_reshape``: cannot reshape {500,64,128} to {500,128}).
* **Writes one self-contained file.** The exporter defaults to a sidecar
  ``.onnx.data``; a two-file release asset is an easy footgun downstream.

``--verify`` scores the export against the torch fp32 model on real LUDB
windows and fails loudly if they disagree.

Run::

    python -m scripts.export_boundary_onnx \\
        --ckpt data/checkpoints/v56c.pt \\
        --out data/deploy/v56c_fp32.onnx \\
        --verify data/cache/v57_500hz/ludb_val.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from openecg.deploy import DEFAULT_CKPT, WINDOW_SAMPLES, _wrap_cls_reg_only
from openecg.stage2.model import load_model_from_ckpt


def _load(ckpt: str | Path, tmp_dir: Path):
    """Load a ckpt, stripping the ``m.`` training-wrapper prefix if present."""
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    st = blob.get("model_state", blob)
    if any(k.startswith("m.") for k in st):
        blob["model_state"] = {(k[2:] if k.startswith("m.") else k): v
                               for k, v in st.items()}
        tmp = tmp_dir / "_stripped.pt"
        torch.save(blob, tmp)
        ckpt = tmp
    return load_model_from_ckpt(str(ckpt), device="cpu")


def _windows_250hz(npz_path: str, n: int = 64) -> np.ndarray:
    """Real LUDB windows from the 500 Hz cache, decimated to the 250 Hz the
    boundary model was trained at."""
    from openecg.dsp import rank_normalize

    sigs = np.load(npz_path)["signals"]
    out = []
    for i in range(min(n, len(sigs))):
        s = np.asarray(sigs[i], dtype=np.float32)[::2][:WINDOW_SAMPLES]
        if len(s) < WINDOW_SAMPLES:
            s = np.pad(s, (0, WINDOW_SAMPLES - len(s)))
        out.append(rank_normalize(s))
    return np.stack(out).astype(np.float32)


def export(ckpt: str | Path, out_path: str | Path, opset: int = 17) -> Path:
    import onnx

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, cfg = _load(ckpt, out_path.parent)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    arch = cfg.get("model_config", cfg).get("arch", "?")
    print(f"arch={arch}  params={n_params:,}")

    wrap = _wrap_cls_reg_only(model).eval()
    raw = out_path.with_name(out_path.stem + "_raw.onnx")
    batch = torch.export.Dim("batch", min=1, max=4096)
    torch.onnx.export(
        wrap, (torch.zeros(2, WINDOW_SAMPLES, dtype=torch.float32),), str(raw),
        input_names=["signal"], output_names=["cls_logits", "reg_offsets"],
        dynamic_shapes={"sig": {0: batch}},
        opset_version=opset, do_constant_folding=True, dynamo=True,
    )

    # Collapse the sidecar into a single file.
    m = onnx.load(str(raw))
    onnx.checker.check_model(m)
    onnx.save_model(m, str(out_path), save_as_external_data=False)
    for leftover in (raw, raw.with_suffix(".onnx.data"),
                     out_path.parent / "_stripped.pt"):
        leftover.unlink(missing_ok=True)
    print(f"wrote {out_path}  ({out_path.stat().st_size:,} B)")
    return out_path


def verify(ckpt: str | Path, onnx_path: str | Path, ludb_npz: str,
           tol: float = 1e-4) -> bool:
    import onnxruntime as ort

    X = _windows_250hz(ludb_npz)
    print(f"LUDB windows: {X.shape}")

    model, _ = _load(ckpt, Path(onnx_path).parent)
    wrap = _wrap_cls_reg_only(model).eval()
    with torch.no_grad():
        t_cls, t_reg = wrap(torch.from_numpy(X))
    t_cls, t_reg = t_cls.numpy(), t_reg.numpy()
    (Path(onnx_path).parent / "_stripped.pt").unlink(missing_ok=True)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o_cls, o_reg = sess.run(["cls_logits", "reg_offsets"], {"signal": X})

    d_cls = float(np.abs(t_cls - o_cls).max())
    d_reg = float(np.abs(t_reg - o_reg).max())
    agree = float((t_cls.argmax(-1) == o_cls.argmax(-1)).mean())
    n_frames = t_cls.shape[0] * t_cls.shape[1]
    print(f"max |Δ| cls_logits : {d_cls:.3e}")
    print(f"max |Δ| reg_offsets: {d_reg:.3e}")
    print(f"argmax agreement   : {agree * 100:.4f}%  ({int(agree * n_frames)}/{n_frames} frames)")

    # A batch-baked graph loads fine and only fails at run time — check shapes.
    for b in (1, 2, 8):
        c, r = sess.run(None, {"signal": np.zeros((b, WINDOW_SAMPLES), np.float32)})
        assert c.shape == (b, 500, 4) and r.shape == (b, 500, 6), (b, c.shape, r.shape)
    print("batch 1/2/8        : OK")

    ok = d_cls < tol and d_reg < tol and agree == 1.0
    print("PARITY:", "OK" if ok else "FAIL")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default="data/deploy/v56c_fp32.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", metavar="LUDB_NPZ", default=None,
                    help="LUDB 500 Hz cache .npz to score parity against")
    args = ap.parse_args()

    out = export(args.ckpt, args.out, args.opset)
    if args.verify:
        return 0 if verify(args.ckpt, out, args.verify) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
