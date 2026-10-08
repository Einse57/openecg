"""Optional OpenVINO backend tests.

Skipped unless ``openvino`` is installed (``pip install "openecg[openvino]"``).
The TFLite comparison additionally needs a TFLite interpreter
(``tflite-runtime`` or ``ai-edge-litert``); the codec comparison needs
``onnxruntime``. Everything runs on the ``CPU`` device.
"""
import sys

import numpy as np
import pytest

ov = pytest.importorskip("openvino")

from openecg import openvino_backend as ovb  # noqa: E402

F32 = {"INFERENCE_PRECISION_HINT": "f32"}


def _synth_sinus(fs: int, seconds: float = 10.0, rr_s: float = 0.75, seed: int = 0) -> np.ndarray:
    """Deterministic synthetic sinus ECG (P / QRS / T bumps + wander + noise)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    sig = 0.03 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.008, n)
    qrs = np.hanning(max(3, int(0.08 * fs)))
    pw = 0.25 * np.hanning(max(3, int(0.10 * fs)))
    tw = 0.35 * np.hanning(max(3, int(0.18 * fs)))
    c = int(rr_s * fs)
    while c + int(0.16 * fs) + len(tw) < n:
        lo = c - len(qrs) // 2
        sig[lo:lo + len(qrs)] += qrs
        p0 = c - int(0.18 * fs)
        if p0 >= 0:
            sig[p0:p0 + len(pw)] += pw
        t0 = c + int(0.16 * fs)
        sig[t0:t0 + len(tw)] += tw
        c += int(rr_s * fs)
    return sig.astype(np.float32)


def test_import_error_names_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "openvino", None)
    with pytest.raises(ImportError, match=r"openecg\[openvino\]"):
        ovb._import_openvino()


def test_cpu_defaults_to_f32_and_config_overrides():
    # The OpenVINO CPU plugin would pick bf16 on CPUs with native bf16 support.
    for cls in (ovb.OpenVINOCodec, ovb.OpenVINOBoundary):
        default = cls(device="CPU")
        assert default.config == F32
        assert default._compiled.get_property("INFERENCE_PRECISION_HINT") == ov.Type.f32
        assert cls(device="CPU", config={}).config == {}


def test_make_static_pins_codec_batch():
    from openecg.deploy import CODEC_SEQ, bundled_codec_onnx_path

    model = ov.Core().read_model(str(bundled_codec_onnx_path("v6")))
    assert any(i.get_partial_shape().is_dynamic for i in model.inputs)
    ovb._make_static(model)
    assert all(i.get_partial_shape().is_static for i in model.inputs)
    assert list(model.inputs[0].get_shape()) == [1, CODEC_SEQ]
    assert all(o.get_partial_shape().is_static for o in model.outputs)


def _tiny_upsample_conv(with_axes: bool):
    """x[1,4,8] -> nearest x2 upsample (3-D Interpolate) -> Conv1d -> y[1,3,16]."""
    import openvino.opset11 as o11
    import openvino.opset13 as ops

    rng = np.random.default_rng(0)
    x = ops.parameter([1, 4, 8], np.float32, name="x")
    if with_axes:
        scales = ops.constant(np.array([2.0], np.float32))
        axes = ops.constant(np.array([2], np.int64))
    else:
        scales = ops.constant(np.array([1.0, 1.0, 2.0], np.float32))
        axes = None
    up = o11.interpolate(
        x, scales, mode="nearest", shape_calculation_mode="scales",
        coordinate_transformation_mode="asymmetric", nearest_mode="floor", axes=axes,
    )
    w = ops.constant(rng.standard_normal((3, 4, 3)).astype(np.float32))
    y = ops.convolution(up, w, strides=[1], pads_begin=[1], pads_end=[1], dilations=[1])
    return ov.Model([y], [x], "tiny_upsample_conv")


def _interp_ranks(model) -> list[int]:
    return [
        n.get_input_partial_shape(0).rank.get_length()
        for n in model.get_ordered_ops()
        if n.get_type_name() == "Interpolate"
    ]


@pytest.mark.parametrize("with_axes", [True, False], ids=["axes", "no_axes"])
def test_wrap_3d_interpolate_is_equivalent(with_axes):
    core = ov.Core()
    x = np.random.default_rng(1).standard_normal((1, 4, 8)).astype(np.float32)

    ref_model = _tiny_upsample_conv(with_axes)
    assert _interp_ranks(ref_model) == [3]
    ref = core.compile_model(ref_model, "CPU")(x)[0]

    model = _tiny_upsample_conv(with_axes)
    assert ovb._wrap_3d_interpolate(model) == 1
    assert _interp_ranks(model) == [4]
    got = core.compile_model(model, "CPU")(x)[0]

    assert got.shape == ref.shape == (1, 3, 16)
    np.testing.assert_array_equal(got, ref)


def test_codec_cpu_agrees_with_onnxruntime():
    ort = pytest.importorskip("onnxruntime")
    from openecg.deploy import OnnxCodec

    ref_codec = OnnxCodec()
    # Single-threaded reference session for a deterministic int8 baseline.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    ref_codec._sess = ort.InferenceSession(
        ref_codec.path, opts, providers=["CPUExecutionProvider"])

    sig = _synth_sinus(500)
    ref = ref_codec.encode(sig, fs=500).channels
    got = ovb.OpenVINOCodec(device="CPU").encode(sig, fs=500).channels
    assert got.shape == ref.shape == (3, 5000)
    assert float(np.mean(got == ref)) >= 0.998


@pytest.mark.parametrize(
    "prefer_wc8, artifact, min_agree",
    [(True, "boundary_v56c_wc8.xml", 0.98), (False, "boundary_v56c.onnx", 0.998)],
    ids=["8bit_ir", "fp32_onnx"],
)
def test_boundary_frame_argmax_vs_tflite(prefer_wc8, artifact, min_agree):
    from openecg.deploy import N_FRAMES, Inference

    try:
        det = Inference()
    except ImportError:
        pytest.skip("no TFLite interpreter (tflite-runtime / ai-edge-litert)")
    ob = ovb.OpenVINOBoundary(device="CPU", prefer_wc8=prefer_wc8)
    assert ob.path.endswith(artifact)

    sig = _synth_sinus(250)
    ref = det.forward_window(sig).argmax(-1)
    got = ob.forward_window(sig).argmax(-1)
    assert got.shape == ref.shape == (N_FRAMES,)
    assert float(np.mean(got == ref)) >= min_agree
