# Performance Benchmarks

## Methodology

### Test Environment

- **Python**: 3.11
- **GPU**: NVIDIA GeForce RTX 5090
- **CPU**: Intel Core Ultra 9 285K
- **OS**: Linux 6.11.0-29-generic

### What We Measure

**Kernel Performance Only**: All benchmarks measure only the core delay-and-sum beamforming kernel execution time. We explicitly **exclude**:
- CPU↔GPU memory transfers (depends on PCIe bandwidth, motherboard, etc., and pre-/post-processing steps may keep data on the GPU)
- JIT compilation (amortized over multiple function-calls)
- Data format reshaping (varies by scanner or file format)
- Pre-processing and post-processing steps (bit-unpacking, demodulation, clutter filtering, etc.; varies by ultrasound sequence)

The benchmark focuses on the most compute-/memory-bandwidth-intensive part of beamforming rather than system-specific overheads.

### Benchmark Dataset

We use PyMUST's [rotating-disk Doppler dataset](https://github.com/creatis-ULTIM/PyMUST/blob/170ba68/examples/rotatingDisk_real.ipynb) as our primary benchmark:

- **128 receive elements** (L7-4 linear array)
- **63,001 voxels** (25mm × 25mm grid, 0.1mm spacing)
- **32 frames** (temporal ensemble)

This represents a realistic ultrafast imaging workload, although it is a small microbenchmark compared to the 3D, high-channel count datasets we're actually interested in.

The benchmark scripts use the following settings:
* linear interpolation
* f-number: `1.0`

## Performance Results

![Benchmark Results](assets/benchmark-doppler_disk.svg)

| Implementation | Median Runtime | Points/Second | "Mach factor" |
|---------------|----------------|---------------|----------------|
| **mach (GPU)** | **0.23 ms** | **1.13 × 10¹²** | **6.5×** |
| Speed-of-Sound (35mm) | 1.5 ms |  | 1× |
| vbeam (JAX/GPU) | 3.6 ms | 7.2 × 10¹⁰ | 0.42× |
| PyMUST (CPU) | 67 ms | 3.8 × 10⁹ | 0.022× |

### What does "beamforming at the speed of sound" even mean?

The **speed-of-sound** ("Mach 1"), represents the theoretical minimum time required for ultrasound waves to travel to the deepest imaging point and back, multiplied by the number of frames in the dataset.
This is specific to each imaging scenario. For our benchmark with PyMUST's [rotating-disk Doppler dataset](https://github.com/creatis-ULTIM/PyMUST/blob/170ba68/examples/rotatingDisk_real.ipynb):

- **Maximum imaging depth**: 35 mm
- **Speed of sound in rotating disk**: 1,480 m/s
- **Round-trip time**: 2 × 0.035 m ÷ 1,480 m/s = 47 μs per frame
- **Total for 32 frames**: 47.3 μs × 32 = **1.5 ms**

mach's processing time depends on various factors (see [Computational Complexity](#computational-complexity)),
so the "Mach factor" description is specific to this benchmark.

## Benchmark Reproduction

All benchmarks can be reproduced using the included test suite:

```bash
# Run full benchmark suite
make benchmark

# Generate performance plots
uv run --group compare tests/plot_benchmark.py --output assets/benchmark-doppler_disk.svg
uv run --group compare tests/plot_benchmark.py --points-per-second --output assets/benchmark-doppler_disk_pps.svg
```

The benchmark job in our CI pipeline ([`test_gpu.yml`](https://github.com/Forest-Neurotech/mach/blob/main/.github/workflows/test_gpu.yml)) automatically runs these benchmarks across different commits, providing continuous performance monitoring.

## CUDA Optimizations

mach optimizes GPU memory access patterns to improve performance. For those interested in learning more about CUDA optimization, excellent resources include:

- [CUDA Crash Course](https://github.com/CoffeeBeforeArch/cuda_programming/) by CoffeeBeforeArch
- [How CUDA Programming Works](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41487/) - CUDA Architect presentation on CUDA best-practices
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) - NVIDIA example of optimizing a different algorithm

### Key Optimizations in mach

#### 1. **Coalesced Memory Access**
- Channel data organized as `[n_receive_elements, n_samples, n_frames]`
- Frames dimension is contiguous for [coalesced access](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/) to reduce global memory reads

#### 2. **Pre-computed Transmit Wavefront Arrivals**
- Pre-compute transmit arrival times to amortize delay calculation across repeated kernel calls
- However, cannot pre-compute the full transmit+receive delay matrix (like PyMUST) for large datasets: (would require transmits × voxels × channels x `size(float)` memory)

#### 3. **Shared Memory for Delay Tables**
- Transmit-to-receive delays computed only once per voxel (not 32× for 32 frames)
- Delay and apodization tables cached in shared memory
- Reused across all frames for each voxel

### Memory Bandwidth and Compute Utilization

The current thread-block and L1-cache parameters were selected for an [RTX4090 (Blackwell)](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf). These should work well across [Blackwell Architecture GPUs](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

If you're wondering if different thread-block or L1-cache parameters might be better for your specific GPU, [Nsight Compute](https://developer.nvidia.com/nsight-compute) can be helpful to identify bottlenecks.

![Nsight Compute Profile Summary](assets/profile_nsight_compute.png)

*Figure: Nsight Compute profiling results shows 94% memory throughput and 78% compute throughput for the mach kernel on an RTX4090. (Different dataset)*

> **Note**: These utilization percentages are kernel-specific metrics from Nsight Compute, not the overall GPU utilization shown by `nvidia-smi` or `nvtop`.

![Nsight Compute memory workload analysis](assets/memory_throughput_nsight_compute.png)

*Figure: Example Nsight Compute memory workload analysis. (Different dataset)*

Anecdotally, kernel-duration seems to hit a pareto-optimum at >70% memory+compute-efficiency. Changing parameters at that point tends to trade-off memory/compute in a way that doesn't change the overall kernel-time.

## Scaling Performance

(computational-complexity)=
### Computational Complexity

The beamforming algorithm scales as:
```
O(n_voxels × n_elements × n_frames)
```

For the PyMUST dataset: `63,001 voxels × 128 elements × 32 frames ≈ 2.6 × 10⁸` points

### GPU Memory (VRAM) Usage

mach allocates GPU memory only for input arguments and the output result; no intermediate arrays are created during computation, simplifying memory management.

Total GPU memory usage scales as:
```
O(n_voxels × n_frames + n_elements × n_samples × n_frames)
```

Where:
- **First term** (`n_voxels × n_frames`): Output array size—dominates for large imaging grids
- **Second term** (`n_elements × n_samples × n_frames`): Input data size—dominates for high-channel-count systems

#### Memory Usage Example

Here is an example functional ultrasound imaging (fUSI) workload:
- **Imaging grid**: 100×100×100 voxels (1M points)
- **Temporal frames**: 200 frames
- **Matrix probe**: 1024 elements
- **Samples per channel**: 100
- **Data type**: `complex64` (8 bytes per sample)

**Memory breakdown:**
```
channel_data:       (1024, 100, 200) → 164 MB
rx_coords_m:        (1024, 3)        → 12 KB
scan_coords_m:      (1M, 3)          → 12 MB
tx_wave_arrivals_s: (1M,)            → 4 MB
out:                (1M, 200)        → 1.6 GB
                                    ─────────
Total GPU memory:                    ~1.78 GB
```

In this example, the output array (`out`) represents 90% of memory usage, demonstrating how large imaging grids dominate memory requirements for volumetric datasets.

### Performance Scaling with Dataset Size

Typical functional ultrasound imaging (fUSI) datasets we're targeting:
- **1024+ receive elements** (high-density arrays)
- **1M+ voxels** (volumetric or high-resolution imaging)
- **100+ frames** (longer temporal windows)

The performance scaling tests measure how mach's beamforming performance scales with different dataset dimensions:

- **Voxel scaling**: Testing grid resolution from 1e-4 (default) to 1e-5 meters (63k to 6.3M voxels)
- **Element scaling**: Testing 1x to 64x receive elements (128 to 8,192 elements)
- **Frame scaling**: Testing 1/32x to 16x ensemble size (1 to 512 frames)

To run the scaling benchmarks and then generate plots:

```bash
# Run pytest-benchmark
make benchmark

# Generate plots
python tests/plot_scaling.py --output assets/benchmark-scaling.svg
```

![Benchmark scaling workload size](assets/benchmark-scaling.svg)

*Figure: throughput is largely consistent across dataset sizes, except for a small decrease for <16 frames.*

## Performance Suggestions

For Maximum Throughput:

1. **Keep data on GPU**: Use CuPy/JAX arrays to avoid CPU↔GPU transfers
2. **Use sufficient ensemble size**: Use ≥16 frames for complex64 or ≥32 frames for float32 to fully coalesce reads to global memory
3. **Ensure contiguous frame dimension**: The kernel requires frame-contiguous memory layouts.

## Experimental: Inverted-Loop I/Q Kernel and FP16 Storage

Two stacked changes, developed together because the second is what makes the first pay off.

**FP16 channel-data storage** (`_cuda_impl.beamform_fp16`). Only the storage format of `channel_data` changes, to interleaved float16 (re, im) pairs passed as a `uint16` view of shape `(n_rx, n_samples, 2 * n_frames)`. Interpolation, apodization, phase rotation, accumulation, and `out` all stay float32 / complex64, so no complex32 dtype is ever needed on the Python side. Results match `beamform()` to float16 input rounding (max relative error ~2e-4 on random data, ~2e-5 on band-limited data). On its own this gave no speedup on an RTX A6000, because the original kernel is bound by load latency rather than by memory bandwidth.

```python
import cupy as cp
from mach._cuda_impl import beamform_fp16

chan16 = iq.view(cp.float32).astype(cp.float16).view(cp.uint16)  # once, 4-40 ms
out = cp.zeros((n_voxels, n_frames), cp.complex64)
beamform_fp16(chan16, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, out,
              f_number=1.0, rx_start_s=0.0, sampling_freq_hz=fs,
              sound_speed_m_s=1540.0, modulation_freq_hz=f0, tukey_alpha=0.5)
```

**Inverted-loop I/Q kernel** (`beamformKernelInvIQ`). The receive-element loop moves outermost and each thread owns four consecutive frames in registers, fetched with 128-bit vector loads. Sample index, bounds check, apodization weight, and the phase-rotation `sincos` are computed once per element instead of once per (element, frame), and the serial load-then-FMA dependency chain becomes four independent accumulator streams. It is dispatched automatically for I/Q data with nearest or linear interpolation when `n_frames % 8 == 0`. RF data, quadratic interpolation, and other frame counts use the original kernel unchanged. Set `MACH_NO_INVERT=1` in the environment to force the original kernel for A/B comparisons. Results agree with the original kernel to float32 rounding (max relative error ~6e-7, the summation order differs).

### Results on an RTX A6000 (Ampere, 84 SMs, 6 MB L2)

128³ voxels, matrix array with 300 µm pitch, linear interpolation, no apodization, single GPU, median of 5 runs. "IQ-256" is baseband I/Q sampled at f0 with 256 samples per trace. "RF-2048" is analytic RF sampled at 40 MHz with 2048 samples per trace.

| config | batch | original fp32 | inverted fp32 | inverted + fp16 | speedup |
|---|---|---|---|---|---|
| 1024 ch, IQ-256 | 128 | 647 ms | 417 ms (1.55x) | 331 ms | **1.96x** |
| 1024 ch, RF-2048 | 128 | 622 ms | 553 ms (1.13x) | 345 ms | **1.80x** |
| 4096 ch, IQ-256 | 128 | 1464 ms | 896 ms (1.63x) | 776 ms | **1.89x** |
| 4096 ch, RF-2048 | 128 | 1447 ms | 1248 ms (1.16x) | 820 ms | **1.76x** |

Batch sweep at 1024 channels (the working-set footprint grows with batch, so this is the axis most sensitive to L2 size):

| config | batch 32 | batch 64 | batch 128 | batch 256 |
|---|---|---|---|---|
| IQ-256, inverted + fp16 vs original | 1.64x | 1.92x | 1.96x | 2.01x |
| RF-2048, inverted + fp16 vs original | 1.48x | 1.70x | 1.80x | 1.96x |

Inversion alone helps most in the short-trace I/Q regime (1.5 to 1.6x) and little on long RF traces at large batch (about 1.0x), where the restructured kernel becomes bandwidth bound. FP16 storage then halves the bytes and restores the gain, which is why the two changes are worth roughly 2x together but far less individually. Timings drift about 10% run to run with clocks and thermals, so compare ratios within one run.

**Open question, GPUs with larger L2.** The A6000 has 6 MB of L2. Cards like the RTX 4090 (72 MB) and RTX 5090 (96 MB) keep far more of a tile's sample window resident, which should shift where the inverted kernel stops being latency bound and may change the fp32 / fp16 split of the gain. Numbers from those GPUs are wanted.

### Reproducing on another GPU

```bash
git clone https://github.com/Forest-Neurotech/mach.git && cd mach && git checkout feat/fp16-storage
pip install scikit-build-core nanobind cmake ninja cupy-cuda12x
pip install --no-build-isolation .          # needs nvcc; CMakeLists targets sm_75/89/90 (+PTX)
python tests/bench_inverted_fp16.py --check              # correctness matrix, ~10 s
python tests/bench_inverted_fp16.py --sweep-batch --json results.json   # ~5 min, ~16 GB VRAM
```

The script prints the GPU name, L2 size, and SM count with the table. For a Blackwell card (RTX 5090, sm_120) the compute_90 PTX is JIT-compiled by the driver, or add `120` to `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` with CUDA 12.8 or newer. `--el 140x40` adds a 5600-channel configuration in both regimes.
