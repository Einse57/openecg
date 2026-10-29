"""OpenVINO backends for the bundled deploy models.

Supports:
  * ``codec_v*_int8.onnx`` (layered codec, bundled), read directly
  * ``v56c_fp32.onnx`` — the fp32 v56c boundary detector published as a
    release asset (``boundary-v56c``; see ``openecg/models/v56c_MODEL_CARD.md``),
    read directly. It is not bundled: pass its path, set
    ``OPENECG_V56C_ONNX``, or call :func:`fetch_v56c_fp32_onnx` once.

Device selection: any OpenVINO device name, e.g. ``CPU`` (default), ``GPU``,
``NPU``, or ``AUTO``.

Precision: on ``CPU`` and ``GPU`` both classes compile with
``INFERENCE_PRECISION_HINT=f32`` by default (OpenVINO would otherwise pick bf16
on CPUs with native bf16 support and f16 on GPUs), so results match the
reference runtimes. ``NPU`` uses its own default. Pass ``config=`` to override, e.g.
``config={"INFERENCE_PRECISION_HINT": "bf16"}`` or ``config={}`` for the device
default.

Requires the ``openvino`` extra::

    pip install "openecg[openvino]"
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from openecg.deploy import (
    CODEC_FS,
    CODEC_SEQ,
    bundled_codec_onnx_path,
    preprocess_window,
)
from openecg.dsp import rank_normalize

F32_HINT = {"INFERENCE_PRECISION_HINT": "f32"}


def _device_base(device: str) -> str:
    """``"GPU.0"`` -> ``"GPU"``; virtual devices (``AUTO:...``) keep their prefix."""
    return device.upper().split(":", 1)[0].split(".", 1)[0]


BOUNDARY_TFLITE_ERROR = (
    "OpenVINO TFLite frontend cannot load boundary_int8.tflite: "
    "GeneralFailure on ADD near patch_embed. "
    "Use the fp32 ONNX release asset (v56c_fp32.onnx) instead."
)

#: fp32 ONNX release asset of the v56c boundary detector (Apache-2.0).
V56C_ONNX_URL = (
    "https://github.com/vitaldb/openecg/releases/download/boundary-v56c/v56c_fp32.onnx"
)
#: sha256 from ``openecg/models/v56c_MODEL_CARD.md``.
V56C_ONNX_SHA256 = "6ae046ee058cbe05b28176d77760e19c2587efcab027ef13ea1990b1283a6c8e"
#: Environment variable that points at a local copy of ``v56c_fp32.onnx``.
V56C_ONNX_ENV = "OPENECG_V56C_ONNX"


def _import_openvino():
    try:
        import openvino as ov
    except ImportError as e:
        raise ImportError(
            "openecg.openvino_backend requires OpenVINO: "
            'pip install "openecg[openvino]"'
        ) from e
    return ov


def available_devices() -> list[str]:
    ov = _import_openvino()
    return list(ov.Core().available_devices)


def device_full_names() -> dict[str, str]:
    """``{device: FULL_DEVICE_NAME}`` for every available OpenVINO device."""
    ov = _import_openvino()
    core = ov.Core()
    out = {}
    for d in core.available_devices:
        try:
            out[d] = str(core.get_property(d, "FULL_DEVICE_NAME"))
        except Exception:  # noqa: BLE001
            out[d] = "unknown"
    return out


def _default_cache_dir() -> Path:
    return Path(os.environ.get("OPENECG_CACHE", Path.home() / ".cache" / "openecg"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def boundary_onnx_path() -> Path:
    """Locate ``v56c_fp32.onnx``.

    Looks at ``$OPENECG_V56C_ONNX``, then ``data/deploy/v56c_fp32.onnx`` under the
    current directory (the default output of ``scripts/export_boundary_onnx.py``),
    then the cache used by :func:`fetch_v56c_fp32_onnx`.
    """
    cands = []
    env = os.environ.get(V56C_ONNX_ENV)
    if env:
        cands.append(Path(env).expanduser())
    cands.append(Path("data") / "deploy" / "v56c_fp32.onnx")
    cands.append(_default_cache_dir() / "v56c_fp32.onnx")
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "v56c_fp32.onnx not found (looked at: "
        + ", ".join(str(c) for c in cands)
        + "). Download it once with "
        "`python -c \"from openecg.openvino_backend import fetch_v56c_fp32_onnx; "
        "fetch_v56c_fp32_onnx()\"`, or set "
        f"{V56C_ONNX_ENV} to a local copy of {V56C_ONNX_URL}."
    )


def fetch_v56c_fp32_onnx(dest: str | Path | None = None, *, url: str = V56C_ONNX_URL) -> Path:
    """Download the v56c fp32 ONNX release asset and verify its sha256.

    Writes to ``dest`` (default ``~/.cache/openecg/v56c_fp32.onnx``, or
    ``$OPENECG_CACHE``). An existing file with the right sha256 is reused.
    """
    dest = Path(dest) if dest is not None else _default_cache_dir() / "v56c_fp32.onnx"
    if dest.is_file() and _sha256(dest) == V56C_ONNX_SHA256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    tmp = Path(tmp)
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        got = _sha256(tmp)
        if got != V56C_ONNX_SHA256:
            raise RuntimeError(
                f"sha256 mismatch for {url}: got {got}, expected {V56C_ONNX_SHA256}"
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _compile(model_path: str | Path, device: str, config: dict | None = None):
    ov = _import_openvino()
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
    _import_openvino()
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
    ov = _import_openvino()
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
        if config is None and _device_base(self.device) in ("CPU", "GPU"):
            # CPU: reproducible f32 instead of the bf16 default on CPUs with
            # native bf16. GPU: the f16 default drops codec agreement with the
            # int8 ONNX Runtime reference below the 0.998 gate.
            config = F32_HINT
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
    """OpenVINO-backed v56c boundary detector, read from the fp32 ONNX release asset.

    ``model_path`` defaults to :func:`boundary_onnx_path`. Direct TFLite load is
    unsupported (:data:`BOUNDARY_TFLITE_ERROR`). Inference is batch 1, one
    2500-sample window at a time.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str = "CPU",
        config: dict | None = None,
    ):
        if model_path is None:
            model_path = boundary_onnx_path()
        self.path = str(model_path)
        self.device = device.upper()
        if config is None and _device_base(self.device) in ("CPU", "GPU"):
            # f32 instead of the bf16 (CPU) / f16 (GPU) device defaults, so the
            # outputs track the fp32 ONNX reference.
            config = F32_HINT
        self.config = dict(config or {})
        if self.path.endswith(".tflite"):
            # Explicit legacy path — document the failure clearly.
            try:
                self._compiled = _compile(self.path, self.device, self.config)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"{BOUNDARY_TFLITE_ERROR}\nUnderlying: {e}"
                ) from e
        else:
            self._compiled = _compile(self.path, self.device, self.config)
        self._input = self._compiled.inputs[0]
        self._cls = None
        self._reg = None
        for o in self._compiled.outputs:
            names = o.get_names()
            if "cls_logits" in names:
                self._cls = o
            elif "reg_offsets" in names:
                self._reg = o
        if self._cls is None:
            raise RuntimeError(f"no 'cls_logits' output in {self.path}")

    def forward_preprocessed(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """``(cls_logits[500, 4], reg_offsets[500, 6])`` for one window that is
        already rank-normalized (the ONNX ``signal`` input, shape ``(2500,)``)."""
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        rq = self._compiled.create_infer_request()
        rq.infer({self._input: x})
        cls = rq.get_tensor(self._cls).data.copy()[0]
        reg = rq.get_tensor(self._reg).data.copy()[0] if self._reg is not None else None
        return cls, reg

    def forward_window(self, window: np.ndarray) -> np.ndarray:
        """Return (N_FRAMES, 4) cls logits for a single 2500-sample window."""
        return self.forward_preprocessed(preprocess_window(window))[0]

    def forward_window_cls_reg(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """``(cls_logits, reg_offsets)`` for a single raw 2500-sample 250 Hz window."""
        return self.forward_preprocessed(preprocess_window(window))

__all__ = [
    "OpenVINOCodec",
    "OpenVINOBoundary",
    "BOUNDARY_TFLITE_ERROR",
    "F32_HINT",
    "available_devices",
    "device_full_names",
    "boundary_onnx_path",
    "fetch_v56c_fp32_onnx",
    "V56C_ONNX_URL",
    "V56C_ONNX_SHA256",
    "V56C_ONNX_ENV",
]
