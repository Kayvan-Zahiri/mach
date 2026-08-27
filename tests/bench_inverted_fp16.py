#!/usr/bin/env python3
"""Benchmark mach's inverted-loop I/Q kernel and FP16 channel-data storage.

Three kernel paths are timed on the same synthetic plane-wave dataset
(matrix array, 128^3-voxel volume, `batch` frames):

  original   the baseline per-frame beamformKernel (use_inverted_kernel=False)
  inverted   beamformKernelInvIQ with FP32 storage (the auto-dispatched default)
  inv+fp16   beamformKernelInvIQ with half2 storage (_cuda_impl.beamform_fp16)

Each path is checked against `original` (complex correlation r and max
relative error), and the GPU's name, L2 size, and SM count are reported so
runs on different GPUs can be compared. Only cupy and a GPU build of this
branch are required.

Examples::

    python tests/bench_inverted_fp16.py                    # 4 default configs
    python tests/bench_inverted_fp16.py --sweep-batch      # + L2-footprint sweep
    python tests/bench_inverted_fp16.py --check            # correctness matrix only
    python tests/bench_inverted_fp16.py --json results.json

Memory: the largest default config (4096 ch x 2048 samples x 128 frames)
needs about 18 GB of free GPU memory (a 24 GB card); configs that do not fit
are skipped and reported. Timings drift ~10% run to run with GPU clocks and
thermals, so compare ratios within one run rather than absolute numbers
across runs.
"""

import argparse
import json
import math
import platform
import statistics
import sys
import time

import cupy as cp
import numpy as np

import mach
from mach._cuda_impl import InterpolationType, beamform, beamform_fp16

C0, F0, FS_RF, PITCH = 1540.0, 7.81e6, 40e6, 300e-6

DEFAULT_CONFIGS = [
    # name,            elements, n_samples, iq,    batch
    ("1024ch IQ-256", "32x32", 256, True, 128),
    ("1024ch RF-2048", "32x32", 2048, False, 128),
    ("4096ch IQ-256", "128x32", 256, True, 128),
    ("4096ch RF-2048", "128x32", 2048, False, 128),
]
SWEEP_BATCHES = (32, 64, 128, 256)

# Peak GPU memory per channel-data element and per output element, used to skip configs that do not fit
BYTES_PER_CHANNEL_ELEMENT = 8 + 4  # complex64 data + its float16 copy
BYTES_PER_OUTPUT_ELEMENT = 8 * 2  # complex64 output + the reference copy


def gpu_info(dev: int = 0) -> dict:
    p = cp.cuda.runtime.getDeviceProperties(dev)
    name = p["name"].decode() if isinstance(p["name"], bytes) else str(p["name"])
    _, total = cp.cuda.Device(dev).mem_info
    return {
        "gpu": name,
        "compute_capability": f"{p['major']}.{p['minor']}",
        "sm_count": int(p["multiProcessorCount"]),
        "l2_cache_mb": p["l2CacheSize"] / 2**20,
        "vram_gb": total / 2**30,
        "driver": cp.cuda.runtime.driverGetVersion(),
        "runtime": cp.cuda.runtime.runtimeGetVersion(),
        "nvcc": getattr(mach._cuda_impl, "__nvcc_version__", "?"),
        "mach": mach.__version__,
        "cupy": cp.__version__,
        "python": platform.python_version(),
    }


def make_dataset(el: str, nt: int, iq: bool, batch: int, n: int) -> dict:
    """Synthetic matrix-array plane-wave acquisition of five point scatterers."""
    fs = F0 if iq else FS_RF
    enx, eny = map(int, el.split("x"))
    elx = (cp.arange(enx, dtype=cp.float32) - (enx - 1) / 2.0) * PITCH
    ely = (cp.arange(eny, dtype=cp.float32) - (eny - 1) / 2.0) * PITCH
    ex, ey = cp.meshgrid(elx, ely, indexing="ij")
    n_ch = enx * eny
    rx_pos = cp.column_stack([ex.ravel(), ey.ravel(), cp.zeros(n_ch, cp.float32)])

    x = cp.linspace(-6.4e-3, 6.4e-3, n, dtype=cp.float32)
    z = cp.linspace(7.5e-3, 19.5e-3, n, dtype=cp.float32)
    zz, yy, xx = cp.meshgrid(z, x, x, indexing="ij")
    vox = cp.ascontiguousarray(cp.stack([xx, yy, zz], axis=-1).reshape(-1, 3))
    tx_arrivals = cp.ascontiguousarray((vox[:, 2] / C0).astype(cp.float32))

    scat = cp.asarray(
        np.array(
            [[0, 0, 10e-3], [3e-3, 0, 12e-3], [-3e-3, 0, 12e-3], [0, 3e-3, 15e-3], [2e-3, -2e-3, 18e-3]], np.float32
        )
    )
    t = cp.arange(nt, dtype=cp.float32) / fs
    sigma = 2.0 / F0
    rf1 = cp.zeros((n_ch, nt), cp.complex64)
    for s in scat:
        tau = (s[2] + cp.linalg.norm(rx_pos - s[None, :], axis=1)) / C0
        dtau = t[None, :] - tau[:, None]
        env = cp.exp(-0.5 * (dtau / sigma) ** 2)
        if iq:  # baseband I/Q sampled at fs = f0 (short traces)
            rf1 += (env * cp.exp(-2j * np.pi * F0 * tau[:, None])).astype(cp.complex64)
        else:  # analytic RF at 40 MHz (long traces)
            rf1 += (env * cp.exp(2j * np.pi * F0 * dtau)).astype(cp.complex64)
    chan = cp.ascontiguousarray(cp.broadcast_to(rf1[:, :, None], (n_ch, nt, batch)))
    del rf1
    return {
        "chan": chan,
        "rx_pos": rx_pos,
        "vox": vox,
        "tx_arrivals": tx_arrivals,
        "fs": fs,
        "mod_hz": F0 if iq else 0.0,
        "n_ch": n_ch,
    }


def to_fp16_view(chan: cp.ndarray) -> cp.ndarray:
    """complex64 (n_rx, n_samples, n_frames) -> uint16 view of interleaved float16 (re, im)."""
    return cp.ascontiguousarray(chan.view(cp.float32).astype(cp.float16)).view(cp.uint16)


def compare(a: cp.ndarray, b: cp.ndarray, chunk: int = 1 << 18) -> tuple[float, float]:
    """(|<a,b>| / (|a||b|), max|a-b| / max|b|) accumulated in float64, chunked to bound memory."""
    num = 0j
    na = nb = mx = mb = 0.0
    for i in range(0, a.shape[0], chunk):
        x, y = a[i : i + chunk], b[i : i + chunk]
        num += complex(cp.sum(cp.conj(x) * y, dtype=cp.complex128))
        na += float(cp.sum(cp.abs(x) ** 2, dtype=cp.float64))
        nb += float(cp.sum(cp.abs(y) ** 2, dtype=cp.float64))
        mx = max(mx, float(cp.abs(x - y).max()))
        mb = max(mb, float(cp.abs(y).max()))
    return abs(num) / math.sqrt(na * nb), mx / mb


def timed(fn, rep: int) -> float:
    fn()
    cp.cuda.Device().synchronize()
    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def run_config(
    name: str, el: str, nt: int, iq: bool, batch: int, n: int, rep: int, tukey: float, interp: InterpolationType
) -> dict:
    n_vox = n**3
    enx, eny = map(int, el.split("x"))
    n_ch = enx * eny
    need = n_ch * nt * batch * BYTES_PER_CHANNEL_ELEMENT + n_vox * batch * BYTES_PER_OUTPUT_ELEMENT + 64 * 2**20
    free, _ = cp.cuda.Device().mem_info
    if need > 0.9 * free:
        return {"name": name, "skipped": f"needs ~{need / 2**30:.1f} GB, {free / 2**30:.1f} GB free"}

    d = make_dataset(el, nt, iq, batch, n)
    chan, rx_pos, vox, tx_arrivals = d["chan"], d["rx_pos"], d["vox"], d["tx_arrivals"]
    kw = {
        "f_number": 1.0,
        "rx_start_s": 0.0,
        "sampling_freq_hz": d["fs"],
        "sound_speed_m_s": C0,
        "modulation_freq_hz": d["mod_hz"],
        "tukey_alpha": tukey,
        "interp_type": interp,
    }
    out = cp.zeros((n_vox, batch), cp.complex64)

    t0 = time.perf_counter()
    chan16 = to_fp16_view(chan)
    cp.cuda.Device().synchronize()
    convert_ms = (time.perf_counter() - t0) * 1e3

    def call_original():
        out.fill(0)
        beamform(chan, rx_pos, vox, tx_arrivals, out, use_inverted_kernel=False, **kw)

    def call_inverted():
        out.fill(0)
        beamform(chan, rx_pos, vox, tx_arrivals, out, **kw)

    def call_fp16():
        out.fill(0)
        beamform_fp16(chan16, rx_pos, vox, tx_arrivals, out, **kw)

    res = {
        "name": name,
        "elements": el,
        "n_ch": n_ch,
        "n_samples": nt,
        "iq": iq,
        "batch": batch,
        "n_vox": n_vox,
        "tukey_alpha": tukey,
        "interp": str(interp).split(".")[-1],
        "fp16_convert_ms": convert_ms,
    }
    pts = n_vox * n_ch * batch

    res["original_ms"] = timed(call_original, rep)
    ref = out.copy()
    for label, fn in (("inverted", call_inverted), ("inv_fp16", call_fp16)):
        res[f"{label}_ms"] = timed(fn, rep)
        res[f"{label}_r"], res[f"{label}_max_rel_err"] = compare(out, ref)
    res["tpts_per_s"] = {k: pts / (res[f"{k}_ms"] * 1e-3) / 1e12 for k in ("original", "inverted", "inv_fp16")}
    return res  # GPU arrays are released on return; the caller drains the cupy pool


def correctness_matrix(seed: int = 1) -> list[dict]:
    """Small randomized check: every interpolator x tukey x (dispatched | fallback) path."""
    cp.random.seed(seed)
    n_ch, nt, n_vox = 256, 512, 40000
    chan_all = (cp.random.standard_normal((n_ch, nt, 64)) + 1j * cp.random.standard_normal((n_ch, nt, 64))).astype(
        cp.complex64
    )
    rx = cp.random.uniform(-5e-3, 5e-3, (n_ch, 3)).astype(cp.float32)
    rx[:, 2] = 0
    vox = cp.ascontiguousarray(
        cp.random.uniform(0, 1, (n_vox, 3)).astype(cp.float32) * cp.asarray([1e-2, 1e-2, 1e-2], cp.float32)
        + cp.asarray([-5e-3, -5e-3, 8e-3], cp.float32)
    )
    arr = cp.ascontiguousarray((vox[:, 2] / C0).astype(cp.float32))
    kw = {
        "f_number": 1.0,
        "rx_start_s": 0.0,
        "sampling_freq_hz": 7.8e6,
        "sound_speed_m_s": C0,
        "modulation_freq_hz": 7.8e6,
    }
    rows = []
    for interp in (InterpolationType.NearestNeighbor, InterpolationType.Linear, InterpolationType.Quadratic):
        for tukey in (0.0, 0.5):
            for batch in (64, 50):  # 64 -> inverted path; 50 -> not %8, falls back to original
                chan = cp.ascontiguousarray(chan_all[:, :, :batch])
                chan16 = to_fp16_view(chan)
                a = cp.zeros((n_vox, batch), cp.complex64)
                b = cp.zeros((n_vox, batch), cp.complex64)
                c = cp.zeros((n_vox, batch), cp.complex64)
                beamform(chan, rx, vox, arr, a, tukey_alpha=tukey, interp_type=interp, use_inverted_kernel=False, **kw)
                beamform(chan, rx, vox, arr, b, tukey_alpha=tukey, interp_type=interp, **kw)
                beamform_fp16(chan16, rx, vox, arr, c, tukey_alpha=tukey, interp_type=interp, **kw)
                r_inv, e_inv = compare(b, a)
                r_16, e_16 = compare(c, a)
                rows.append({
                    "interp": str(interp).split(".")[-1],
                    "tukey": tukey,
                    "batch": batch,
                    "inverted_r": r_inv,
                    "inverted_max_rel_err": e_inv,
                    "inv_fp16_r": r_16,
                    "inv_fp16_max_rel_err": e_16,
                })
                print(
                    f"  {rows[-1]['interp']:16s} tukey={tukey} batch={batch:3d}: "
                    f"inverted r={r_inv:.7f} err={e_inv:.1e} | inv+fp16 r={r_16:.7f} err={e_16:.1e}"
                )
    return rows


def print_table(results: list[dict]) -> None:
    print()
    print(
        "| config | batch | original fp32 | inverted fp32 | inv+fp16 | speedup | r / max rel err (inv+fp16 vs original) |"
    )
    print("|---|---|---|---|---|---|---|")
    for r in results:
        if "skipped" in r:
            print(f"| {r['name']} | | skipped: {r['skipped']} | | | | |")
            continue
        print(
            f"| {r['name']} | {r['batch']} | {r['original_ms']:.0f} ms | {r['inverted_ms']:.0f} ms "
            f"({r['original_ms'] / r['inverted_ms']:.2f}x) | {r['inv_fp16_ms']:.0f} ms | "
            f"**{r['original_ms'] / r['inv_fp16_ms']:.2f}x** | {r['inv_fp16_r']:.5f} / {r['inv_fp16_max_rel_err']:.1e} |"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=128, help="voxels per axis (default 128 -> 128^3 volume)")
    ap.add_argument("--rep", type=int, default=3, help="timed repetitions per kernel (median reported)")
    ap.add_argument("--tukey", type=float, default=0.0, help="Tukey apodization alpha (0 = none)")
    ap.add_argument(
        "--interp",
        choices=["nearest", "linear"],
        default="linear",
        help="interpolation (quadratic is excluded because it always uses the original kernel)",
    )
    ap.add_argument("--configs", nargs="*", help="subset of default config names to run")
    ap.add_argument("--el", help="extra array geometry AZxEL, e.g. 140x40 (run in both regimes)")
    ap.add_argument(
        "--sweep-batch", action="store_true", help="also sweep batch=32..256 at 1024 ch (L2 footprint sensitivity)"
    )
    ap.add_argument("--check", action="store_true", help="run only the small randomized correctness matrix")
    ap.add_argument("--json", help="write all results + GPU info to this JSON file")
    a = ap.parse_args(argv)

    interp = {"nearest": InterpolationType.NearestNeighbor, "linear": InterpolationType.Linear}[a.interp]
    info = gpu_info()
    print(
        f"GPU: {info['gpu']} (sm_{info['compute_capability'].replace('.', '')}, {info['sm_count']} SMs, "
        f"L2 {info['l2_cache_mb']:.0f} MB, {info['vram_gb']:.0f} GB) | driver {info['driver']} runtime {info['runtime']} "
        f"nvcc {info['nvcc']} | mach {info['mach']} cupy {info['cupy']}"
    )

    payload: dict = {"gpu_info": info, "args": vars(a)}
    if a.check:
        print("Correctness matrix (vs original kernel, random data):")
        payload["check"] = correctness_matrix()
    else:
        configs = [c for c in DEFAULT_CONFIGS if not a.configs or c[0] in a.configs]
        if a.el:
            n_ch = math.prod(map(int, a.el.split("x")))
            configs += [(f"{n_ch}ch IQ-256", a.el, 256, True, 128), (f"{n_ch}ch RF-2048", a.el, 2048, False, 128)]
        if a.sweep_batch:
            configs += [("1024ch IQ-256", "32x32", 256, True, b) for b in SWEEP_BATCHES if b != 128]
            configs += [("1024ch RF-2048", "32x32", 2048, False, b) for b in SWEEP_BATCHES if b != 128]
        results = []
        for name, el, nt, iq, batch in configs:
            print(f"[{name} batch={batch}] ...", end="", flush=True)
            r = run_config(name, el, nt, iq, batch, a.n, a.rep, a.tukey, interp)
            cp.get_default_memory_pool().free_all_blocks()
            results.append(r)
            if "skipped" in r:
                print(f" skipped ({r['skipped']})")
            else:
                print(
                    f" orig {r['original_ms']:.0f} ms | inv {r['inverted_ms']:.0f} ms | inv+fp16 {r['inv_fp16_ms']:.0f} ms "
                    f"| r={r['inv_fp16_r']:.5f}"
                )
        print_table(results)
        payload["results"] = results
    if a.json:
        with open(a.json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
