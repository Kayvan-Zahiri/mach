"""Tests for the inverted-loop I/Q kernel and FP16 channel-data storage.

The inverted kernel is auto-dispatched for I/Q data with nearest/linear
interpolation when n_frames % 8 == 0. The ``use_inverted_kernel=False``
keyword of the nanobind functions forces the original per-frame kernel, which
these tests use as the reference.
"""

import numpy as np
import pytest

import mach
from mach._cuda_impl import beamform_fp16
from mach.kernel import InterpolationType, nb_beamform

cp = pytest.importorskip("cupy")

N_RX, N_SAMPLES, N_SCAN = 256, 512, 40_000
BEAMFORM_KWARGS = {
    "f_number": 1.0,
    "rx_start_s": 0.0,
    "sampling_freq_hz": 7.8e6,
    "sound_speed_m_s": 1540.0,
    "modulation_freq_hz": 7.8e6,
}


@pytest.fixture(scope="module")
def iq_data():
    """Random I/Q channel data and a random voxel cloud that straddles the aperture and trace bounds."""
    rng = np.random.default_rng(1)
    chan = (rng.standard_normal((N_RX, N_SAMPLES, 64)) + 1j * rng.standard_normal((N_RX, N_SAMPLES, 64))).astype(
        np.complex64
    )
    rx = rng.uniform(-5e-3, 5e-3, (N_RX, 3)).astype(np.float32)
    rx[:, 2] = 0.0
    scan = (rng.uniform(0.0, 1.0, (N_SCAN, 3)) * np.array([1e-2, 1e-2, 1e-2]) + np.array([-5e-3, -5e-3, 8e-3])).astype(
        np.float32
    )
    tx_arrivals = (scan[:, 2] / BEAMFORM_KWARGS["sound_speed_m_s"]).astype(np.float32)
    return {
        "chan": cp.asarray(chan),
        "rx": cp.asarray(rx),
        "scan": cp.asarray(scan),
        "tx_arrivals": cp.asarray(tx_arrivals),
    }


def _beamform(data, chan, **kwargs):
    """Public API path (auto-dispatch)."""
    return mach.beamform(
        channel_data=chan,
        rx_coords_m=data["rx"],
        scan_coords_m=data["scan"],
        tx_wave_arrivals_s=data["tx_arrivals"],
        **{**BEAMFORM_KWARGS, **kwargs},
    )


def _beamform_original(data, chan, **kwargs):
    """Reference: the original per-frame kernel via the nanobind function."""
    out = cp.zeros((N_SCAN, chan.shape[2]), dtype=cp.complex64)
    nb_beamform(
        channel_data=chan,
        rx_coords_m=data["rx"],
        scan_coords_m=data["scan"],
        tx_wave_arrivals_s=data["tx_arrivals"],
        out=out,
        use_inverted_kernel=False,
        **{**BEAMFORM_KWARGS, **kwargs},
    )
    return out


def _max_rel_err(a, b):
    return float(cp.abs(a - b).max() / cp.abs(b).max())


@pytest.mark.parametrize("interp_type", [InterpolationType.NearestNeighbor, InterpolationType.Linear])
@pytest.mark.parametrize("tukey_alpha", [0.0, 0.5])
@pytest.mark.parametrize("modulation_freq_hz", [7.8e6, 0.0])
@pytest.mark.parametrize("n_frames", [8, 16, 24, 64])  # small counts change the block shape
def test_inverted_kernel_matches_original(iq_data, interp_type, tukey_alpha, modulation_freq_hz, n_frames):
    """n_frames % 8 == 0 dispatches the inverted kernel; it must agree with the original to FP32 rounding."""
    chan = cp.ascontiguousarray(iq_data["chan"][:, :, :n_frames])
    kwargs = {"tukey_alpha": tukey_alpha, "interp_type": interp_type, "modulation_freq_hz": modulation_freq_hz}
    reference = _beamform_original(iq_data, chan, **kwargs)
    inverted = _beamform(iq_data, chan, **kwargs)
    assert inverted.shape == reference.shape
    if modulation_freq_hz != 0.0:
        # The folded phase rotation reorders the arithmetic, so a dispatched inverted kernel is
        # detectably (but only by rounding) different from the original.
        assert not cp.array_equal(inverted, reference), "inverted kernel was not dispatched (bitwise-equal output)"
    assert _max_rel_err(inverted, reference) < 1e-5


def test_use_inverted_kernel_flag_selects_original(iq_data):
    """use_inverted_kernel=False on the nanobind function reproduces the original kernel bitwise."""
    chan = cp.ascontiguousarray(iq_data["chan"][:, :, :64])
    reference = _beamform_original(iq_data, chan)
    out = cp.zeros((N_SCAN, 64), dtype=cp.complex64)
    nb_beamform(chan, iq_data["rx"], iq_data["scan"], iq_data["tx_arrivals"], out, **BEAMFORM_KWARGS)
    assert not cp.array_equal(out, reference)  # default dispatches the inverted kernel
    out.fill(0)
    nb_beamform(
        chan, iq_data["rx"], iq_data["scan"], iq_data["tx_arrivals"], out, use_inverted_kernel=False, **BEAMFORM_KWARGS
    )
    assert cp.array_equal(out, reference)


@pytest.mark.parametrize("n_frames", [50, 63])
def test_non_multiple_of_8_frames_falls_back(iq_data, n_frames):
    """Frame counts that are not a multiple of 8 must use the original kernel (bitwise-identical output)."""
    chan = cp.ascontiguousarray(iq_data["chan"][:, :, :n_frames])
    assert cp.array_equal(_beamform(iq_data, chan), _beamform_original(iq_data, chan))


def test_quadratic_falls_back(iq_data):
    chan = cp.ascontiguousarray(iq_data["chan"][:, :, :64])
    kwargs = {"interp_type": InterpolationType.Quadratic}
    assert cp.array_equal(_beamform(iq_data, chan, **kwargs), _beamform_original(iq_data, chan, **kwargs))


def test_unaligned_view_falls_back(iq_data):
    """A C-contiguous view whose base is not 16-byte aligned must use the original kernel, not crash."""
    n_frames = 64
    buf = cp.zeros(N_RX * N_SAMPLES * n_frames + 1, dtype=cp.complex64)
    chan = buf[1:].reshape(N_RX, N_SAMPLES, n_frames)  # 8-byte offset, still C-contiguous
    assert chan.data.ptr % 16 == 8
    chan[...] = iq_data["chan"][:, :, :n_frames]
    assert cp.array_equal(_beamform(iq_data, chan), _beamform_original(iq_data, chan))


@pytest.mark.parametrize(
    "interp_type", [InterpolationType.NearestNeighbor, InterpolationType.Linear, InterpolationType.Quadratic]
)
@pytest.mark.parametrize("tukey_alpha", [0.0, 0.5])
@pytest.mark.parametrize("n_frames", [64, 50])
def test_fp16_storage_matches_fp32(iq_data, interp_type, tukey_alpha, n_frames):
    """beamform_fp16 (half2 storage, FP32 compute) must match beamform() to FP16 input rounding."""
    chan = cp.ascontiguousarray(iq_data["chan"][:, :, :n_frames])
    chan16 = cp.ascontiguousarray(chan.view(cp.float32).astype(cp.float16)).view(cp.uint16)
    assert chan16.shape == (N_RX, N_SAMPLES, 2 * n_frames)
    reference = _beamform(iq_data, chan, tukey_alpha=tukey_alpha, interp_type=interp_type)
    out = cp.zeros((N_SCAN, n_frames), dtype=cp.complex64)
    beamform_fp16(
        chan16,
        iq_data["rx"],
        iq_data["scan"],
        iq_data["tx_arrivals"],
        out,
        tukey_alpha=tukey_alpha,
        interp_type=interp_type,
        **BEAMFORM_KWARGS,
    )
    assert _max_rel_err(out, reference) < 2e-3
    corr = cp.abs(cp.vdot(out, reference)) / (cp.linalg.norm(out) * cp.linalg.norm(reference))
    assert float(corr) > 0.99999


def test_fp16_rejects_wrong_shape(iq_data):
    """channel_data must be (n_rx, n_samples, 2 * n_frames) uint16."""
    chan = cp.ascontiguousarray(iq_data["chan"][:, :, :64])
    chan16 = cp.ascontiguousarray(chan.view(cp.float32).astype(cp.float16)).view(cp.uint16)
    out = cp.zeros((N_SCAN, 63), dtype=cp.complex64)  # 2 * 63 != chan16.shape[2]
    with pytest.raises(RuntimeError, match="2\\*n_frames"):
        beamform_fp16(chan16, iq_data["rx"], iq_data["scan"], iq_data["tx_arrivals"], out, **BEAMFORM_KWARGS)


def test_fp16_rejects_cpu_arrays(iq_data):
    chan = cp.ascontiguousarray(iq_data["chan"][:, :, :64])
    chan16 = cp.asnumpy(cp.ascontiguousarray(chan.view(cp.float32).astype(cp.float16)).view(cp.uint16))
    out = cp.zeros((N_SCAN, 64), dtype=cp.complex64)
    with pytest.raises(RuntimeError, match="GPU"):
        beamform_fp16(chan16, iq_data["rx"], iq_data["scan"], iq_data["tx_arrivals"], out, **BEAMFORM_KWARGS)


def test_fp16_rejects_odd_offset_view(iq_data):
    """half2 loads need 4-byte alignment; a uint16 view at an odd element offset must be rejected up front."""
    n_frames = 64
    buf = cp.zeros(N_RX * N_SAMPLES * 2 * n_frames + 1, dtype=cp.uint16)
    chan16 = buf[1:].reshape(N_RX, N_SAMPLES, 2 * n_frames)  # 2-byte offset
    out = cp.zeros((N_SCAN, n_frames), dtype=cp.complex64)
    with pytest.raises(RuntimeError, match="aligned"):
        beamform_fp16(chan16, iq_data["rx"], iq_data["scan"], iq_data["tx_arrivals"], out, **BEAMFORM_KWARGS)
