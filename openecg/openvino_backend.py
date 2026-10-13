"""OpenVINO backends for openecg deploy artifacts.

Supports:
  * ``codec_v6_int8.onnx`` via direct ONNX read
  * ``boundary_v56c.onnx`` / ``boundary_v56c_wc8.xml`` (ONNX export recovered
    from the bundled TFLite; native TFLite frontend still fails — see
    :data:`BOUNDARY_TFLITE_ERROR`)

Device selection: any OpenVINO device name, e.g. ``CPU`` (default), ``GPU``,
``NPU``, or ``AUTO``.
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


def device_full_names() -> dict[str, str]:
    """``{device: FULL_DEVICE_NAME}`` for every available OpenVINO device."""
    import openvino as ov
    core = ov.Core()
    out = {}
    for d in core.available_devices:
        try:
            out[d] = str(core.get_property(d, "FULL_DEVICE_NAME"))
        except Exception:  # noqa: BLE001
            out[d] = "unknown"
    return out


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


def _compile(model_path: str | Path, device: str, config: dict | None = None):
    import openvino as ov
    core = ov.Core()
    device = device.upper()
    virtual = device.split(":", 1)[0] in ("AUTO", "HETERO", "MULTI", "BATCH")
    if not virtual and device not in core.available_devices:
        raise RuntimeError(
            f"OpenVINO device {device!r} not available; have {core.available_devices}. "
            f"Pass any OpenVINO device name, e.g. CPU, GPU, NPU, or AUTO."
        )
    model = core.read_model(str(model_path))
    _make_static(model)
    if device.startswith("GPU"):
        _wrap_3d_interpolate(model)
    return core.compile_model(model, device, config or {})


def _wrap_3d_interpolate(model) -> int:
    """Run 3-D ``Interpolate`` nodes as Unsqueeze -> 4-D Interpolate -> Squeeze.

    Math is unchanged (the extra trailing dim has size 1 and scale 1). The
    codec's 1-D nearest upsample (``node_upsample_nearest1d``) followed by a
    1-D Conv fails GPU program build ("Data batch and filters rank do not
    match"); the 4-D form compiles. Applied only for ``GPU`` devices.
    """
    import numpy as np
    import openvino.opset11 as o11
    import openvino.opset13 as ops

    n = 0
    for node in model.get_ordered_ops():
        if node.get_type_name() != "Interpolate":
            continue
        ps = node.get_input_partial_shape(0)
        if ps.rank.is_dynamic or ps.rank.get_length() != 3:
            continue
        attrs = node.get_attributes()
        axis = ops.constant(np.array([3], np.int64))
        x4 = ops.unsqueeze(node.input_value(0), axis)
        axes = None
        if node.get_input_size() >= 3:
            scales = node.input_value(1)
            axes = node.input_value(2)
        else:
            src = node.input_value(1).get_node()
            if not hasattr(src, "get_data"):
                continue
            arr = src.get_data()
            if arr.size != 3:
                continue
            scales = ops.constant(np.append(arr, 1).astype(arr.dtype))
        new = o11.interpolate(
            x4, scales,
            mode=attrs["mode"],
            shape_calculation_mode=attrs["shape_calculation_mode"],
            pads_begin=list(attrs["pads_begin"]) + [0],
            pads_end=list(attrs["pads_end"]) + [0],
            coordinate_transformation_mode=attrs["coordinate_transformation_mode"],
            nearest_mode=attrs["nearest_mode"],
            antialias=attrs["antialias"],
            cube_coeff=attrs["cube_coeff"],
            axes=axes,
        )
        node.output(0).replace(ops.squeeze(new.output(0), axis).output(0))
        n += 1
    if n:
        model.validate_nodes_and_infer_types()
    return n


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
        config: dict | None = None,
    ):
        if onnx_path is None:
            onnx_path = bundled_codec_onnx_path(version)
        self.path = str(onnx_path)
        self.device = device.upper()
        if config is None and self.device.startswith("GPU"):
            # GPU defaults to f16; f32 keeps codec agreement with the int8
            # ONNX Runtime reference above the 0.998 gate.
            config = {"INFERENCE_PRECISION_HINT": "f32"}
        self.config = dict(config or {})
        self._compiled = _compile(self.path, self.device, self.config)
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
    "device_full_names",
    "bundled_boundary_onnx_path",
]
