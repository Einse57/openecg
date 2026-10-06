"""OpenVINO backends for openecg deploy artifacts.

Supports:
  * ``codec_v6_int8.onnx`` (and other ``codec_*_int8.onnx``) via direct ONNX read
  * ``boundary_int8.tflite`` via OpenVINO's TFLite frontend (currently fails on
    the ai-edge-torch residual ADD graph — see ``BOUNDARY_TFLITE_ERROR``)

Device selection: ``CPU`` (default), ``GPU``, or ``NPU``. Code accepts the flag
even when only CPU is present on the host; compile will raise if the device is
missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    "GeneralFailure on ADD near patch_embed "
    "('This tensor should be either input, constant or should be already "
    "produced by previous operators'). "
    "Workarounds tried: ov.convert_model, tflite2onnx, tf2onnx — all failed "
    "(TFL_GELU / dequant Add). Use TFLite/LiteRT for the boundary detector "
    "until an ONNX export of v56c ships or the TFLite frontend is fixed."
)


def available_devices() -> list[str]:
    import openvino as ov
    return list(ov.Core().available_devices)


def _compile(model_path: str | Path, device: str):
    import openvino as ov
    core = ov.Core()
    device = device.upper()
    if device not in core.available_devices:
        raise RuntimeError(
            f"OpenVINO device {device!r} not available; have {core.available_devices}. "
            f"Code supports --device CPU|GPU|NPU."
        )
    model = core.read_model(str(model_path))
    return core.compile_model(model, device)


class OpenVINOCodec:
    """OpenVINO-backed layered codec — drop-in for ``OnnxCodec`` / ``model=``.

    Loads ``codec_{version}_int8.onnx`` through OpenVINO's ONNX frontend and
    exposes :meth:`encode` returning :class:`~openecg.layered.LayeredCodec`.
    """

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
        # Map outputs by name (frame/beat/rhythm).
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
        """Raw head logits for a single rank-normalized 5000-sample window."""
        x = np.asarray(window_5000, dtype=np.float32).ravel()
        if x.size != self.SEQ:
            raise ValueError(f"expected {self.SEQ} samples, got {x.size}")
        rq = self._compiled.create_infer_request()
        rq.infer({self._input: x[None, :]})
        return {h: rq.get_tensor(o).data.copy()[0] for h, o in self._out.items()}


class OpenVINOBoundary:
    """Attempt to load ``boundary_int8.tflite`` into OpenVINO.

    Raises :class:`RuntimeError` with :data:`BOUNDARY_TFLITE_ERROR` on failure
    (current OpenVINO TFLite frontend cannot parse this graph).
    """

    def __init__(
        self,
        tflite_path: str | Path | None = None,
        *,
        device: str = "CPU",
    ):
        if tflite_path is None:
            tflite_path = bundled_model_path()
        self.path = str(tflite_path)
        self.device = device.upper()
        try:
            self._compiled = _compile(self.path, self.device)
        except Exception as e:  # noqa: BLE001 — surface OV frontend error
            raise RuntimeError(f"{BOUNDARY_TFLITE_ERROR}\nUnderlying: {e}") from e
        self._input = self._compiled.inputs[0]
        self._cls = next(
            o for o in self._compiled.outputs
            if list(o.get_partial_shape())[-1] == N_CLASSES
            or (o.shape is not None and len(o.shape) and o.shape[-1] == N_CLASSES)
        )

    def forward_window(self, window: np.ndarray) -> np.ndarray:
        x = preprocess_window(window)[None, ...].astype(np.float32)
        rq = self._compiled.create_infer_request()
        rq.infer({self._input: x})
        return rq.get_tensor(self._cls).data.copy()[0]


__all__ = [
    "OpenVINOCodec",
    "OpenVINOBoundary",
    "BOUNDARY_TFLITE_ERROR",
    "available_devices",
]
