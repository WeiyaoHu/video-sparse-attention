# Hardware-Efficient Sparse Attention for Video Generation

> **Status:** Research project in progress  
> **Focus:** Video Generation · Sparse Attention · GPU Kernel Optimization · Triton/CUDA · AI Systems

[中文版 / Chinese Version](README.md)

---

## 1. Overview

This project studies **hardware-efficient sparse attention for video generation models**, with an emphasis on understanding, reproducing, profiling, and optimizing GPU kernels for sparse attention workloads.

The project is motivated by a practical systems question:

> **How can algorithmic sparsity in video attention be translated into real GPU speedup?**

Reducing FLOPs alone does not guarantee lower latency. Sparse attention introduces additional challenges such as irregular memory access, metadata overhead, workload imbalance, poor Tensor Core utilization, and kernel launch overhead. This project therefore focuses not only on sparse attention algorithms, but also on their **GPU execution behavior**.

The long-term goal is to build an end-to-end research pipeline:

```text
Attention Fundamentals
        ↓
FlashAttention-style Tiling
        ↓
Generic Block-Sparse Attention
        ↓
Video Sparse Attention
        ↓
Reproduction of State-of-the-Art Methods
        ↓
GPU Profiling
        ↓
Hardware-Aware Kernel Optimization
        ↓
Integration into Real Video Generation Models
        ↓
End-to-End Evaluation
```

---

## 2. Project Objectives

This project aims to:

1. Understand the GPU execution characteristics of dense and sparse attention.
2. Implement attention kernels from scratch using **Triton and/or CUDA**.
3. Reproduce representative video sparse attention methods, with **Sparse VideoGen (SVG)** as the initial main reference.
4. Analyze GPU bottlenecks with **Nsight Systems** and **Nsight Compute**.
5. Develop at least one **hardware-aware optimization** for sparse attention.
6. Integrate optimized kernels into real video generation workloads such as **Wan** or **HunyuanVideo**.
7. Evaluate both **operator-level** and **end-to-end** performance.

---

## 3. Research Questions

This project is organized around several core questions:

- Why does attention become expensive for long video token sequences?
- Why does reducing theoretical FLOPs not lead to proportional GPU speedup?
- How do different sparse attention patterns affect GPU utilization?
- How should sparse blocks be scheduled to improve memory locality and load balance?
- How do tile size, sparsity ratio, head dimension, and sequence length interact?
- When is a sparse kernel compute-bound, memory-bound, or launch-overhead-bound?
- How can sparse attention kernels be designed to better utilize modern GPU hardware?
- Can a hardware-aware implementation improve both operator latency and end-to-end video generation latency?

---

## 4. Scope

### In Scope

- Dense attention baselines
- PyTorch SDPA baseline
- FlashAttention-style tiled attention
- Block-sparse attention
- Spatial / temporal sparse patterns
- Triton kernel implementation
- CUDA kernel implementation where useful
- Kernel benchmarking
- GPU profiling
- Hardware-aware block scheduling
- Layout transformation
- Sparse metadata optimization
- Kernel fusion experiments
- Video generation model integration
- End-to-end latency and memory evaluation

### Out of Scope for the Initial Stage

- Training a large video generation model from scratch
- Designing a completely new video foundation model
- Large-scale distributed training
- Claiming new algorithmic contributions before a strong reproducible baseline is established

---

## 5. Hardware Strategy

The project uses a two-level hardware workflow.

### Local Development GPU

Primary local development platform:

- **NVIDIA GeForce RTX 4060**
- **8 GB dedicated VRAM**

Main use cases:

- Triton/CUDA development
- Unit tests
- Correctness checks
- Synthetic Q/K/V benchmarks
- Small- and medium-scale profiling
- Nsight Compute / Nsight Systems analysis

### Cloud GPU

Large-scale experiments will be performed on rented GPUs when needed.

Potential platforms:

- RTX 4090 24 GB
- A100 40/80 GB
- H100 80 GB

Main use cases:

- Large sequence lengths
- Full video generation models
- End-to-end inference
- Formal benchmark runs
- Cross-GPU evaluation

The project does **not** treat shared system memory as equivalent to GPU VRAM.

---

## 6. Repository Structure

Planned repository layout:

```text
video-sparse-attention/
│
├── kernels/
│   ├── dense/
│   │   ├── torch_reference.py
│   │   ├── triton_dense.py
│   │   └── cuda_dense/
│   │
│   ├── flash/
│   │   ├── triton_flash.py
│   │   └── online_softmax.py
│   │
│   ├── sparse/
│   │   ├── block_sparse.py
│   │   ├── sparse_patterns.py
│   │   └── sparse_metadata.py
│   │
│   └── video_sparse/
│       ├── svg_reference.py
│       ├── svg_triton.py
│       └── optimizations/
│
├── benchmarks/
│   ├── benchmark_attention.py
│   ├── sweep_seq_len.py
│   ├── sweep_sparsity.py
│   ├── sweep_block_size.py
│   └── benchmark_utils.py
│
├── profiling/
│   ├── nsight_compute/
│   ├── nsight_systems/
│   └── analysis/
│
├── integration/
│   ├── wan/
│   └── hunyuanvideo/
│
├── scripts/
│   ├── setup.sh
│   ├── run_baselines.sh
│   ├── run_benchmarks.sh
│   └── run_profile.sh
│
├── results/
│   ├── csv/
│   ├── figures/
│   ├── logs/
│   └── profiler/
│
├── docs/
│   ├── notes/
│   └── technical_report/
│
├── tests/
│   ├── test_correctness.py
│   └── test_numerics.py
│
├── README.md
└── README_EN.md
```

---

## 7. Benchmark Methodology

All kernels should be evaluated under a consistent benchmark framework.

### Core Input Parameters

```text
Batch size:        B
Number of heads:   H
Sequence length:   N
Head dimension:    D
Data type:         FP16 / BF16
Sparsity ratio:    0%–93.75%
Block size:        32 / 64 / 128
```

Example sweep:

```text
B = 1

H = 8, 16

N =
512
1024
2048
4096
8192
16384
32768

D =
64
128

Sparsity =
0%
25%
50%
75%
87.5%
93.75%

Block size =
32
64
128
```

### Metrics

Each benchmark should record:

- Mean latency
- Median / P50 latency
- P95 latency
- Peak GPU memory
- Effective TFLOPS
- Effective memory bandwidth
- Speedup vs baseline
- Numerical error
- GPU utilization
- Occupancy
- DRAM throughput
- L2 cache behavior
- Tensor Core utilization where applicable

### Baselines

At minimum:

1. Naive PyTorch attention
2. PyTorch SDPA
3. Optimized dense attention / FlashAttention equivalent
4. Official sparse implementation
5. Reimplemented sparse kernel
6. Optimized sparse kernel

---

## 8. Correctness Criteria

Performance results are only meaningful after correctness is established.

Each custom implementation must be compared with a trusted reference.

Example checks:

```python
torch.testing.assert_close(
    output_custom,
    output_reference,
    rtol=...,
    atol=...
)
```

Correctness evaluation should include:

- Forward output error
- Multiple sequence lengths
- Multiple head dimensions
- Multiple sparsity patterns
- FP16/BF16 numerical behavior
- Edge cases for irregular sparse layouts

No speedup result should be reported if the kernel fails correctness checks.

---

## 9. Research Roadmap

### Phase 0 — Infrastructure

**Goal:** Build a reproducible research environment.

Tasks:

- Create project repository
- Freeze software versions
- Build benchmark harness
- Build correctness test framework
- Add CSV logging
- Record GPU / CUDA / PyTorch / Triton versions

Deliverables:

- `benchmark_utils.py`
- `test_correctness.py`
- environment documentation
- first benchmark CSV

---

### Phase 1 — Dense Attention

**Goal:** Understand the baseline computation.

Implement:

- Naive PyTorch attention
- PyTorch SDPA baseline
- Simple Triton attention

Study:

```text
QK^T
Softmax
PV
```

Questions:

- Where does the O(N²) cost come from?
- Where does memory usage explode?
- How much of the runtime is GEMM vs softmax vs memory traffic?

Deliverables:

- Latency vs sequence length
- Peak memory vs sequence length
- Correctness tests

---

### Phase 2 — FlashAttention-Style Tiling

**Goal:** Understand IO-aware attention.

Implement:

- Tiled Q/K/V processing
- Online softmax
- On-chip accumulation

Study:

- Tile size
- Register pressure
- Shared memory
- Occupancy
- Memory traffic

Example sweep:

```text
BLOCK_M = 32 / 64 / 128
BLOCK_N = 32 / 64 / 128
```

Deliverables:

- Tile size vs latency
- Profiling report
- Short note explaining online softmax

---

### Phase 3 — Generic Block-Sparse Attention

**Goal:** Compute only active blocks.

Sparse patterns:

- Local window
- Strided
- Random block
- Spatial-like
- Temporal-like

Important requirement:

> Inactive blocks must be skipped during computation, not masked after dense attention.

Study:

- Sparsity ratio vs actual speedup
- Metadata overhead
- Load balance
- Memory locality
- Block granularity

Deliverables:

- Sparse kernel
- Sparsity vs speedup plot
- Block size vs speedup plot

---

### Phase 4 — GPU Profiling

**Goal:** Explain performance instead of only measuring it.

Use:

- Nsight Systems
- Nsight Compute

Analyze:

- SM utilization
- DRAM throughput
- L2 cache behavior
- Occupancy
- Warp efficiency
- Tensor Core utilization
- Register pressure
- Kernel launch overhead
- Synchronization overhead

Deliverable:

> **Performance Characterization of Block-Sparse Attention on GPUs**

Expected key question:

> Why does 90% sparsity not necessarily produce 10× speedup?

---

### Phase 5 — Video Sparse Attention Study

**Goal:** Move from generic sparse attention to real video workloads.

Initial main reference:

- **Sparse VideoGen (SVG)**

Study:

- Video token organization
- Spatial sparse patterns
- Temporal sparse patterns
- Head-wise behavior
- Online profiling
- Layout transformation
- GPU-friendly sparse execution

Deliverables:

- Algorithm flow diagram
- Kernel execution diagram
- Notes mapping algorithm design to GPU behavior

---

### Phase 6 — Reimplement the Core Sparse Kernel

**Goal:** Independently reproduce the core computation.

Implement:

1. PyTorch reference
2. Custom Triton implementation
3. Optional CUDA implementation

Compare:

```text
Reference
vs
Official implementation
vs
Custom implementation
```

Metrics:

- Correctness
- Latency
- Peak memory
- GPU utilization

Important rule:

> Read the official implementation for understanding, but independently implement the core kernel.

---

### Phase 7 — Real Model Integration

**Goal:** Replace synthetic Q/K/V with real video generation workloads.

Candidate models:

- Wan
- HunyuanVideo

Keep inference settings fixed:

- Prompt
- Random seed
- Resolution
- Number of frames
- Sampling steps
- Precision

Compare:

```text
Dense/default attention
Official sparse implementation
Custom sparse implementation
```

Record:

- Attention latency
- End-to-end generation time
- Peak VRAM
- Output quality / similarity

---

### Phase 8 — Bottleneck Analysis

**Goal:** Identify the real limiting factor.

Break down:

```text
mask generation
block metadata
layout transformation
QK
softmax
PV
scatter/gather
kernel launch
```

Possible bottlenecks:

- Irregular block layout
- Poor memory coalescing
- Sparse metadata overhead
- Low Tensor Core utilization
- Register pressure
- Load imbalance
- Excessive kernel launch overhead

Deliverable:

- Bottleneck breakdown chart
- Profiling-based optimization hypothesis

---

### Phase 9 — Hardware-Aware Optimization

At least one original optimization must be attempted.

Candidate directions:

#### A. Hardware-Aware Block Reordering

Reorder sparse blocks to improve:

- Memory locality
- Coalescing
- Warp utilization
- Load balance

#### B. Adaptive Tile Selection

Select tile size based on:

- Sequence length
- Head dimension
- Sparsity
- Sparse pattern
- GPU architecture

#### C. Sparse Metadata Optimization

Reduce:

- Indexing overhead
- Metadata bandwidth
- Scheduling cost

Possible methods:

- Compressed block metadata
- Precomputed schedules
- Compact block index structures

#### D. Kernel Fusion

Explore fusing stages such as:

```text
layout transform
+
sparse attention
```

or

```text
mask processing
+
attention dispatch
```

Optimization must be justified by profiling evidence.

---

### Phase 10 — Formal Benchmark

Formal experiments should be run on a consistent hardware platform.

Recommended matrix:

```text
Sequence length:
2K / 4K / 8K / 16K / 32K

Head dimension:
64 / 128

Sparsity:
50% / 75% / 87.5% / 93.75%

Block size:
32 / 64 / 128

dtype:
FP16 / BF16
```

Core figures:

1. Sequence Length vs Latency
2. Sparsity vs Speedup
3. Block Size vs Speedup
4. Sequence Length vs Peak Memory
5. GPU Utilization
6. Effective Bandwidth
7. End-to-End Video Generation Latency
8. Original vs Optimized Kernel

---

## 10. Expected Milestones

### Milestone A — Attention Kernel Foundation

Expected capabilities:

- Dense attention
- FlashAttention-style tiling
- Block-sparse attention
- Triton programming
- Nsight profiling

At this point, the project becomes a serious GPU kernel engineering project.

---

### Milestone B — Research Reproduction

Expected capabilities:

- Understand video sparse attention
- Reimplement the main kernel
- Compare against official implementation
- Integrate into a real video model

At this point, the project is strong enough for a resume entry.

---

### Milestone C — Research Contribution

Expected capabilities:

- Identify a real performance bottleneck
- Propose a hardware-aware optimization
- Implement the optimization
- Show reproducible benchmark improvements
- Validate on real video generation workloads

At this point, the project becomes a research-oriented AI Systems project.

---

## 11. Experimental Principles

This repository follows several rules:

### Rule 1

**Correctness before performance.**

### Rule 2

**Do not compare only against naive PyTorch.**

Always include a strong optimized baseline.

### Rule 3

**FLOPs reduction is not equivalent to latency reduction.**

Always report real GPU latency.

### Rule 4

**Every optimization requires profiling evidence.**

### Rule 5

**Do not mix results from different GPU architectures without clearly labeling them.**

### Rule 6

**Formal benchmark settings must be reproducible.**

### Rule 7

**No fabricated or estimated speedup numbers.**

All reported results must come from actual experiments.

---

## 12. Planned Results Directory

Example:

```text
results/
├── csv/
│   ├── dense_attention.csv
│   ├── block_sparse.csv
│   ├── svg_reproduction.csv
│   └── optimized_kernel.csv
│
├── figures/
│   ├── latency_vs_seq_len.png
│   ├── memory_vs_seq_len.png
│   ├── sparsity_vs_speedup.png
│   ├── block_size_vs_latency.png
│   └── end_to_end_latency.png
│
└── profiler/
    ├── ncu/
    └── nsys/
```

---

## 13. Technical Report Plan

A final technical report will be written with the following structure:

```text
1. Introduction
2. Background
3. Video Sparse Attention
4. GPU Kernel Design
5. Reproduction Methodology
6. Performance Characterization
7. Hardware-Aware Optimization
8. Experiments
9. Limitations
10. Conclusion
```

Target length:

- 6–10 pages
- Reproducible figures and tables
- Clear distinction between reproduced work and original optimization

---

## 14. Resume Goal

The final resume description should only include experimentally verified results.

Example structure:

> **Hardware-Efficient Sparse Attention for Video Generation**  
> Implemented and optimized custom Triton/CUDA kernels for block-sparse attention in video diffusion transformer workloads, reproducing representative video sparse attention mechanisms and integrating them into end-to-end generation pipelines.

> Profiled GPU execution using Nsight Compute/Systems, identified bottlenecks in memory access, sparse scheduling, and block utilization, and developed hardware-aware optimizations achieving **X× kernel-level acceleration**.

> Evaluated kernels across sequence lengths, sparsity ratios, and GPU platforms, achieving **Y× end-to-end acceleration** and **Z% peak-memory reduction** while preserving generation quality.

`X`, `Y`, and `Z` must be replaced only after formal experiments.

---

## 15. Current Status

- [ ] Research environment
- [ ] Benchmark framework
- [ ] Naive attention baseline
- [ ] PyTorch SDPA baseline
- [ ] Triton dense attention
- [ ] FlashAttention-style kernel
- [ ] Generic block-sparse attention
- [ ] Nsight profiling
- [ ] Sparse VideoGen study
- [ ] SVG reference implementation
- [ ] SVG Triton kernel
- [ ] Real video model integration
- [ ] Bottleneck analysis
- [ ] Hardware-aware optimization
- [ ] Formal benchmark
- [ ] Technical report
- [ ] Final README results section

---

## 16. References

The initial research direction is centered on modern work in:

- FlashAttention and IO-aware exact attention
- Block-sparse attention
- Sparse VideoGen
- Sparse VideoGen2
- Efficient attention for video diffusion / video DiT models
- Triton and CUDA kernel optimization

Detailed paper citations will be added as the reproduction progresses.

---

## 17. Disclaimer

This repository is a research and learning project. Reproduced methods will be clearly attributed to their original authors. Any original optimization will be explicitly separated from reproduced work.

Performance numbers will only be published after reproducible experiments on clearly specified hardware and software environments.
