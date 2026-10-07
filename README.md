# 面向视频生成的硬件高效稀疏注意力

> **项目状态：** 研究进行中  
> **研究方向：** 视频生成 · 稀疏注意力 · GPU Kernel Optimization · Triton/CUDA · AI Systems

[English Version](README_EN.md)

---

## 1. 项目简介

本项目研究 **Video Generation 场景下的 Hardware-Efficient Sparse Attention**，重点不是单纯复现某个模型，而是系统地理解、实现、分析并优化稀疏注意力在 GPU 上的实际执行过程。

项目围绕一个核心问题展开：

> **算法层面的稀疏性，如何真正转化为 GPU 上的实际加速？**

理论 FLOPs 减少并不意味着 GPU latency 会同比下降。Sparse Attention 往往会引入新的系统问题，例如：

- 不规则内存访问
- Sparse metadata 开销
- GPU workload 不均衡
- Tensor Core 利用率降低
- Kernel launch overhead
- Layout transformation 成本
- Sparse block 调度开销

因此，本项目不仅研究稀疏注意力算法本身，更关注其 **GPU execution behavior** 和 **hardware-aware optimization**。

完整技术路线为：

```text
Attention 基础
      ↓
FlashAttention-style Tiling
      ↓
Generic Block-Sparse Attention
      ↓
Video Sparse Attention
      ↓
复现前沿方法
      ↓
GPU Profiling
      ↓
Hardware-Aware Kernel Optimization
      ↓
接入真实 Video Generation Model
      ↓
End-to-End Evaluation
```

---

## 2. 项目目标

本项目最终希望完成以下目标：

1. 理解 Dense Attention 与 Sparse Attention 在 GPU 上的执行特征。
2. 使用 **Triton 和/或 CUDA** 独立实现 Attention Kernel。
3. 复现代表性的 Video Sparse Attention 工作，第一阶段以 **Sparse VideoGen（SVG）** 为主要研究对象。
4. 使用 **Nsight Systems** 和 **Nsight Compute** 分析真实 GPU 瓶颈。
5. 完成至少一个 **hardware-aware optimization**。
6. 将优化后的 Kernel 接入 **Wan / HunyuanVideo** 等真实视频生成 workload。
7. 同时完成 **operator-level** 与 **end-to-end** 性能评估。

---

## 3. 核心研究问题

本项目主要回答以下问题：

- 为什么视频生成中的 Attention 会随着 token 数增加迅速变得昂贵？
- 为什么减少理论 FLOPs 并不会带来同比例 GPU speedup？
- 不同 sparse pattern 会如何影响 GPU 利用率？
- Sparse block 应该如何调度，才能改善 memory locality 和 load balance？
- Tile size、sparsity、head dimension、sequence length 之间如何相互影响？
- 一个 sparse kernel 在什么情况下是 compute-bound、memory-bound 或 launch-overhead-bound？
- 如何设计更适合现代 GPU 执行的 sparse attention kernel？
- Hardware-aware optimization 能否同时改善 kernel latency 和 end-to-end generation latency？

---

## 4. 项目范围

### 当前重点研究内容

- Dense Attention baseline
- PyTorch SDPA baseline
- FlashAttention-style tiled attention
- Block-Sparse Attention
- Spatial / Temporal sparse pattern
- Triton Kernel
- CUDA Kernel
- Kernel Benchmark
- GPU Profiling
- Hardware-aware block scheduling
- Layout transformation
- Sparse metadata optimization
- Kernel fusion
- Video generation model integration
- End-to-end latency / memory evaluation

### 第一阶段暂不作为核心目标

- 从零训练大型视频生成模型
- 设计全新的 Video Foundation Model
- 大规模分布式训练
- 在没有充分复现和 profiling 的情况下直接追求全新算法创新

---

## 5. 硬件规划

本项目采用两级硬件环境。

### 本地开发 GPU

主要开发平台：

- **NVIDIA GeForce RTX 4060**
- **8 GB 独立显存**

主要负责：

- Triton/CUDA 开发
- Unit Test
- Correctness Test
- Synthetic Q/K/V Benchmark
- 中小规模 Attention 实验
- Nsight Compute / Nsight Systems Profiling

### 云端 GPU

需要大显存或完整模型时租用 GPU。

计划根据需要使用：

- RTX 4090 24 GB
- A100 40/80 GB
- H100 80 GB

主要负责：

- 大 sequence length
- 完整 Video Generation Model
- End-to-End Inference
- 正式 Benchmark
- Cross-GPU Evaluation

注意：

> Windows 显示的 shared GPU memory 不等价于独立 GPU VRAM，因此本项目按照真实独立显存容量规划实验。

---

## 6. 仓库结构

计划中的项目结构：

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

## 7. Benchmark 设计

所有 Kernel 使用统一 Benchmark Framework。

### 输入参数

```text
Batch Size:       B
Attention Heads:  H
Sequence Length:  N
Head Dimension:   D
Data Type:        FP16 / BF16
Sparsity Ratio:   0%–93.75%
Block Size:       32 / 64 / 128
```

建议 Sweep：

```text
B = 1

H =
8
16

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

Block Size =
32
64
128
```

### 记录指标

每次 Benchmark 至少记录：

- Mean Latency
- Median / P50 Latency
- P95 Latency
- Peak GPU Memory
- Effective TFLOPS
- Effective Memory Bandwidth
- Speedup
- Numerical Error
- GPU Utilization
- Occupancy
- DRAM Throughput
- L2 Cache Behavior
- Tensor Core Utilization

### Baseline

至少包含：

1. Naive PyTorch Attention
2. PyTorch SDPA
3. 优化 Dense Attention / FlashAttention Equivalent
4. Official Sparse Implementation
5. 自己复现的 Sparse Kernel
6. 自己优化后的 Sparse Kernel

---

## 8. Correctness 标准

所有性能测试必须先通过正确性验证。

自定义 Kernel 必须与可信 reference 进行比较。

例如：

```python
torch.testing.assert_close(
    output_custom,
    output_reference,
    rtol=...,
    atol=...
)
```

Correctness Test 至少覆盖：

- Forward Output Error
- 多种 Sequence Length
- 多种 Head Dimension
- 多种 Sparse Pattern
- FP16 / BF16 数值行为
- Irregular Sparse Layout 边界情况

原则：

> 如果 Kernel 没有通过正确性测试，就不能报告 Speedup。

---

# 9. 研究路线

## Phase 0 — Research Infrastructure

### 目标

建立可复现的研究环境。

### 任务

- 建立独立 GitHub Repository
- 固定软件版本
- 建立 Benchmark Harness
- 建立 Correctness Test
- 自动保存 CSV
- 记录 GPU / CUDA / PyTorch / Triton Version

### 产出

```text
benchmark_utils.py
test_correctness.py
environment.md
benchmark CSV
```

---

## Phase 1 — Dense Attention

### 目标

真正理解 Attention Baseline。

实现：

- Naive PyTorch Attention
- PyTorch SDPA
- Simple Triton Attention

重点分析：

```text
QK^T
Softmax
PV
```

必须回答：

- O(N²) 来自哪里？
- 显存在哪一步快速增长？
- GEMM、Softmax 和 Memory Traffic 分别占多大成本？

### 输出

- Sequence Length vs Latency
- Sequence Length vs Peak Memory
- Correctness Results

---

## Phase 2 — FlashAttention-style Tiling

### 目标

理解 IO-aware Attention。

实现：

- Tiled Q/K/V
- Online Softmax
- On-Chip Accumulator

重点分析：

- Tile Size
- Register Pressure
- Shared Memory
- Occupancy
- Memory Traffic

建议：

```text
BLOCK_M =
32
64
128

BLOCK_N =
32
64
128
```

### 输出

- Tile Size vs Latency
- Profiling Report
- Online Softmax 推导笔记

---

## Phase 3 — Generic Block-Sparse Attention

### 目标

真正做到：

> 只计算 Active Blocks。

Sparse Pattern：

- Local Window
- Strided
- Random Block
- Spatial-like
- Temporal-like

特别注意：

错误方式：

```text
Dense Attention
↓
Mask
```

正确方式：

```text
直接跳过 Inactive Blocks
```

研究：

- Sparsity vs Actual Speedup
- Sparse Metadata Overhead
- Load Balance
- Memory Locality
- Block Granularity

### 输出

- Sparse Attention Kernel
- Sparsity vs Speedup
- Block Size vs Speedup

---

## Phase 4 — GPU Profiling

### 目标

不仅知道“快不快”，还要知道“为什么”。

工具：

- Nsight Systems
- Nsight Compute

分析指标：

```text
SM Utilization
DRAM Throughput
L2 Hit Rate
Occupancy
Warp Efficiency
Tensor Core Utilization
Register Pressure
Kernel Launch Overhead
Synchronization Overhead
```

需要回答：

> 为什么 90% sparsity 不一定带来 10× speedup？

### 输出

一份内部技术报告：

> **Performance Characterization of Block-Sparse Attention on GPUs**

---

## Phase 5 — Video Sparse Attention

### 目标

从 Generic Sparse Attention 进入真实 Video Workload。

第一阶段主要研究：

> **Sparse VideoGen（SVG）**

需要理解：

- Video Token Organization
- Spatial Sparse Pattern
- Temporal Sparse Pattern
- Head-wise Sparse Behavior
- Online Profiling
- Layout Transformation
- GPU-Friendly Sparse Execution

### 输出

一张完整数据流：

```text
Video
 ↓
Tokenization
 ↓
Q / K / V
 ↓
Sparsity Detection
 ↓
Sparse Pattern
 ↓
Layout
 ↓
GPU Kernel
```

以及：

> Algorithm Design → Tensor Layout → GPU Execution

的对应关系。

---

## Phase 6 — 独立复现核心 Sparse Kernel

### 目标

不依赖官方实现，自己重写核心计算。

实现：

```text
1. PyTorch Reference
2. Triton Kernel
3. Optional CUDA Kernel
```

比较：

```text
Reference
vs
Official Implementation
vs
Custom Implementation
```

指标：

- Correctness
- Latency
- Peak Memory
- GPU Utilization

原则：

> 可以阅读官方实现理解思路，但核心 Kernel 必须自己重新实现。

---

## Phase 7 — 真实 Video Model Integration

### 目标

从 Synthetic Q/K/V 进入真实视频生成模型。

候选：

- Wan
- HunyuanVideo

固定：

- Prompt
- Random Seed
- Resolution
- Number of Frames
- Sampling Steps
- Precision

比较：

```text
Dense / Default Attention

Official Sparse Attention

Custom Sparse Attention
```

记录：

- Attention Latency
- End-to-End Generation Time
- Peak VRAM
- Output Quality / Similarity

到这一步，项目已经具备较强简历价值。

---

## Phase 8 — Bottleneck Analysis

### 目标

找到真实瓶颈。

拆分：

```text
Mask Generation
Block Metadata
Layout Transformation
QK
Softmax
PV
Scatter / Gather
Kernel Launch
```

可能出现的问题：

- Sparse Block 过于分散
- Memory Coalescing 差
- Metadata Overhead
- Tensor Core 利用率低
- Register Pressure
- Load Imbalance
- Kernel Launch 过多

### 输出

- Bottleneck Breakdown
- Profiling Evidence
- Optimization Hypothesis

---

## Phase 9 — Hardware-Aware Optimization

项目必须至少尝试一个自己的优化。

### A. Hardware-Aware Block Reordering

通过重排 sparse blocks 改善：

- Memory Locality
- Coalescing
- Warp Utilization
- Load Balance

---

### B. Adaptive Tile Selection

根据以下 workload 特征选择 Tile Size：

```text
Sequence Length
Head Dimension
Sparsity
Sparse Pattern
GPU Architecture
```

可能使用：

```text
32
64
128
```

或自动调优。

---

### C. Sparse Metadata Optimization

降低：

- Indexing Overhead
- Metadata Bandwidth
- Scheduling Cost

可能方向：

- Compressed Sparse Block Metadata
- Precomputed Schedule
- Compact Block Index

---

### D. Kernel Fusion

探索融合：

```text
Layout Transformation
+
Sparse Attention
```

或者：

```text
Mask Processing
+
Attention Dispatch
```

所有 Optimization 都必须来源于 Profiling Evidence，而不是拍脑袋。

---

## Phase 10 — 正式 Benchmark

正式实验应尽量在同一 GPU 平台完成。

建议实验矩阵：

```text
Sequence Length:
2K / 4K / 8K / 16K / 32K

Head Dimension:
64 / 128

Sparsity:
50% / 75% / 87.5% / 93.75%

Block Size:
32 / 64 / 128

dtype:
FP16 / BF16
```

核心结果图：

1. Sequence Length vs Latency
2. Sparsity vs Speedup
3. Block Size vs Speedup
4. Sequence Length vs Peak Memory
5. GPU Utilization
6. Effective Bandwidth
7. End-to-End Generation Latency
8. Original vs Optimized Kernel

---

# 10. 项目里程碑

## Milestone A — Attention Kernel Foundation

能力：

```text
Dense Attention
FlashAttention-style Tiling
Block-Sparse Attention
Triton
Nsight
```

做到这里，已经真正进入 GPU Attention Kernel。

---

## Milestone B — Research Reproduction

能力：

```text
理解 Video Sparse Attention
重写核心 Kernel
与 Official Implementation 对比
接入真实 Video Model
```

做到这里，项目已经可以正式写入简历。

---

## Milestone C — Research Contribution

能力：

```text
发现真实性能瓶颈
↓
提出 Hardware-Aware Optimization
↓
实现
↓
Benchmark
↓
真实 Video Workload 验证
```

做到这里，项目才真正具备成为简历第一项目的资格。

---

# 11. 实验原则

### Rule 1

**Correctness before performance.**

先正确，再优化。

### Rule 2

**不能只和 Naive PyTorch 比。**

必须加入强 Baseline。

### Rule 3

**FLOPs Reduction ≠ Latency Reduction。**

必须报告真实 GPU Latency。

### Rule 4

**每一个 Optimization 都必须有 Profiling Evidence。**

### Rule 5

**不同 GPU 架构的数据不能不加说明地直接混合比较。**

### Rule 6

**正式实验必须可复现。**

### Rule 7

**禁止提前编造 Speedup。**

所有最终结果必须来自真实实验。

---

# 12. Results 目录规划

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

# 13. Technical Report 计划

最终整理一份完整 Technical Report。

建议结构：

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

建议长度：

- 6–10 页
- 图表可复现
- 清楚区分 Reproduction 和 Original Optimization

---

# 14. 最终简历目标

最终简历中的描述必须只使用真实实验结果。

建议结构：

> **Hardware-Efficient Sparse Attention for Video Generation**  
> Implemented and optimized custom Triton/CUDA kernels for block-sparse attention in video diffusion transformer workloads, reproducing representative video sparse attention mechanisms and integrating them into end-to-end generation pipelines.

> Profiled GPU execution using Nsight Compute/Systems, identified bottlenecks in memory access, sparse scheduling, and block utilization, and developed hardware-aware optimizations achieving **X× kernel-level acceleration**.

> Evaluated kernels across sequence lengths, sparsity ratios, and GPU platforms, achieving **Y× end-to-end acceleration** and **Z% peak-memory reduction** while preserving generation quality.

其中：

```text
X
Y
Z
```

必须等正式实验完成后填写。

---

# 15. 当前进度

- [ ] Research Environment
- [ ] Benchmark Framework
- [ ] Naive Attention Baseline
- [ ] PyTorch SDPA Baseline
- [ ] Triton Dense Attention
- [ ] FlashAttention-style Kernel
- [ ] Generic Block-Sparse Attention
- [ ] Nsight Profiling
- [ ] Sparse VideoGen Study
- [ ] SVG Reference Implementation
- [ ] SVG Triton Kernel
- [ ] Real Video Model Integration
- [ ] Bottleneck Analysis
- [ ] Hardware-Aware Optimization
- [ ] Formal Benchmark
- [ ] Technical Report
- [ ] Final README Results

---

# 16. References

第一阶段重点关注：

- FlashAttention / IO-Aware Attention
- Block-Sparse Attention
- Sparse VideoGen
- Sparse VideoGen2
- Video Diffusion / Video DiT Efficient Attention
- Triton Kernel Optimization
- CUDA Performance Optimization

随着项目推进，将逐步补充详细论文、代码仓库和引用信息。

---

# 17. 项目原则

本仓库是一个研究型复现与优化项目。

对于已有工作：

- 明确注明原论文与原作者
- 清楚区分 reproduced component 和 original contribution
- 不把已有方法包装成自己的创新
- 不报告未经验证的性能数字

本项目真正希望回答的不是：

> “我能不能把某个 repo 跑起来？”

而是：

> **“我是否能够理解一个前沿 Sparse Attention 方法，把它映射到真实 GPU execution，自己实现核心 Kernel，找到性能瓶颈，并进一步做 Hardware-Aware Optimization？”**
