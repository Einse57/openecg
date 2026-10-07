"""OpenVINO backends for openecg deploy artifacts.

Supports:
  * ``codec_v6_int8.onnx`` via direct ONNX read
  * ``boundary_v56c.onnx`` / ``boundary_v56c_wc8.xml`` (ONNX export recovered
    from the bundled TFLite; native TFLite frontend still fails — see
    :data:`BOUNDARY_TFLITE_ERROR`)

Device selection: any OpenVINO device name (default ``CPU``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from openecg.deploy import (
    CODEC_FS,
    CODEC_SEQ,
    N_CLASSES,
    WINDOW_SAMPLES,
    bundled_codec_onnx_path,
    bundled_model_path,
    preprocess_window,
)
from openecg.dsp import rank_normalize

BOUNDARY_TFLITE_ERROR = (
    "OpenVINO TFLite frontend cannot load boundary_int8.tflite: "
    "GeneralFailure on ADD near patch_embed. "
    "Use the ONNX export (boundary_v56c.onnx / boundary_v56c_wc8.xml) instead."
)


def available_devices() -> list[str]:
    import openvino as ov
    return list(ov.Core().available_devices)


def bundled_boundary_onnx_path(*, prefer_wc8: bool = True) -> Path:
    """Path to the OpenVINO-friendly boundary ONNX / IR artifact."""
    root = Path(__file__).resolve().parent / "models"
    wc8 = root / "boundary_v56c_wc8.xml"
    onnx = root / "boundary_v56c.onnx"
    if prefer_wc8 and wc8.exists():
        return wc8
    if onnx.exists():
        return onnx
    raise FileNotFoundError(
        f"No OpenVINO boundary artifact at {wc8} or {onnx}. "
        f"Run scripts/export_boundary_onnx.py first."
    )


def _compile(model_path: str | Path, device: str):
    import openvino as ov
    core = ov.Core()
    device = device.upper()
    virtual = device.split(":", 1)[0] in ("AUTO", "HETERO", "MULTI", "BATCH")
    if not virtual and device not in core.available_devices:
        raise RuntimeError(
            f"OpenVINO device {device!r} not available; have {core.available_devices}. "
            f"Pass any OpenVINO device name, e.g. CPU."
        )
    model = core.read_model(str(model_path))
    _make_static(model)
    return core.compile_model(model, device)


def _make_static(model) -> None:
    """Pin dynamic input dims (e.g. the codec's ``batch``) to 1.

    Every caller here runs single-window batch-1 inference. Some device
    compilers cannot derive output bounds from a dynamic batch dim
    (codec ``node_unsqueeze``), so fix the shapes before compile.
    """
    import openvino as ov
    new_shapes = {}
    for inp in model.inputs:
        ps = inp.get_partial_shape()
        if ps.is_dynamic:
            dims = [1 if d.is_dynamic else d.get_length() for d in ps]
            new_shapes[inp.get_any_name()] = ov.PartialShape(dims)
    if new_shapes:
        model.reshape(new_shapes)


class OpenVINOCodec:
    """OpenVINO-backed layered codec — drop-in for ``OnnxCodec`` / ``model=``."""

    SEQ = CODEC_SEQ

    def __init__(
        self,
        onnx_path: str | Path | None = None,
        *,
        version: str = "v6",
        device: str = "CPU",
    ):
        if onnx_path is None:
            onnx_path = bundled_codec_onnx_path(version)
        self.path = str(onnx_path)
        self.device = device.upper()
        self._compiled = _compile(self.path, self.device)
        self._input = self._compiled.inputs[0]
        self._out = {}
        for o in self._compiled.outputs:
            name = o.get_any_name()
            for head in ("frame", "beat", "rhythm"):
                if head in name:
                    self._out[head] = o
                    break
        missing = {"frame", "beat", "rhythm"} - set(self._out)
        if missing:
            raise RuntimeError(f"codec ONNX missing heads: {missing}")

    def encode(self, signal: np.ndarray, fs: int = CODEC_FS):
        from openecg.eval import ALL_SUPER_QRS_CLASSES, SUPER_OTHER
        from openecg.layered import LayeredCodec

        sig = np.asarray(signal, dtype=np.float32).ravel()
        n = int(sig.size)
        sig = np.asarray(rank_normalize(sig), dtype=np.float32)
        if sig.size >= self.SEQ:
            x = sig[: self.SEQ]
        else:
            x = np.zeros(self.SEQ, dtype=np.float32)
            x[: sig.size] = sig
        rq = self._compiled.create_infer_request()
        rq.infer({self._input: x[None, :].astype(np.float32)})
        outs = {h: rq.get_tensor(o).data.copy()[0] for h, o in self._out.items()}

        def _arg(head, n_keep):
            return outs[head].argmax(-1).astype(np.uint8)[:n_keep]

        frame = _arg("frame", min(n, self.SEQ))
        beat = _arg("beat", min(n, self.SEQ))
        rhythm = _arg("rhythm", min(n, self.SEQ))
        if n > self.SEQ:
            pad = n - self.SEQ
            frame = np.concatenate([frame, np.full(pad, SUPER_OTHER, np.uint8)])
            beat = np.concatenate([beat, np.zeros(pad, np.uint8)])
            rhythm = np.concatenate([rhythm, np.zeros(pad, np.uint8)])
        qrs_mask = np.isin(frame, ALL_SUPER_QRS_CLASSES)
        beat = np.where(qrs_mask, beat, 0).astype(np.uint8)
        channels = np.stack([frame, beat, rhythm], axis=0)
        return LayeredCodec(fs=int(fs), channels=channels)

    def forward_logits(self, window_5000: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(window_5000, dtype=np.float32).ravel()
        if x.size != self.SEQ:
            raise ValueError(f"expected {self.SEQ} samples, got {x.size}")
        rq = self._compiled.create_infer_request()
        rq.infer({self._input: x[None, :]})
        return {h: rq.get_tensor(o).data.copy()[0] for h, o in self._out.items()}


class OpenVINOBoundary:
    """OpenVINO-backed v56c boundary detector (ONNX / NNCF-compressed IR).

    Prefers ``boundary_v56c_wc8.xml`` (NNCF INT8 weight compression), then
    ``boundary_v56c.onnx``. Direct TFLite load is still unsupported
    (:data:`BOUNDARY_TFLITE_ERROR`).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str = "CPU",
        prefer_wc8: bool = True,
    ):
        if model_path is None:
            model_path = bundled_boundary_onnx_path(prefer_wc8=prefer_wc8)
        self.path = str(model_path)
        self.device = device.upper()
        if self.path.endswith(".tflite"):
            # Explicit legacy path — document the failure clearly.
            try:
                self._compiled = _compile(self.path, self.device)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"{BOUNDARY_TFLITE_ERROR}\nUnderlying: {e}"
                ) from e
        else:
            self._compiled = _compile(self.path, self.device)
        self._input = self._compiled.inputs[0]
        self._cls = None
        self._reg = None
        for o in self._compiled.outputs:
            name = o.get_any_name() or ""
            # Prefer named outputs from our ONNX export.
            if "cls" in name:
                self._cls = o
            elif "reg" in name:
                self._reg = o
        if self._cls is None:
            for o in self._compiled.outputs:
                shape = list(o.partial_shape)
                # last dim == 4
                try:
                    last = int(str(shape[-1]))
                except Exception:
                    last = -1
                if last == N_CLASSES or "4" in str(shape[-1]):
                    self._cls = o
                    break
        if self._cls is None:
            raise RuntimeError(f"could not locate cls_logits output in {self.path}")

    def forward_window(self, window: np.ndarray) -> np.ndarray:
        """Return (N_FRAMES, 4) cls logits for a single 2500-sample window."""
        x = preprocess_window(window)[None, ...].astype(np.float32)
        rq = self._compiled.create_infer_request()
        rq.infer({self._input: x})
        return rq.get_tensor(self._cls).data.copy()[0]

    def forward_window_cls_reg(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        x = preprocess_window(window)[None, ...].astype(np.float32)
        rq = self._compiled.create_infer_request()
        rq.infer({self._input: x})
        cls = rq.get_tensor(self._cls).data.copy()[0]
        reg = None
        if self._reg is not None:
            reg = rq.get_tensor(self._reg).data.copy()[0]
        return cls, reg


__all__ = [
    "OpenVINOCodec",
    "OpenVINOBoundary",
    "BOUNDARY_TFLITE_ERROR",
    "available_devices",
    "bundled_boundary_onnx_path",
]
