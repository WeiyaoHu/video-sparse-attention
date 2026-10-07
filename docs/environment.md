# 实验环境

在发布任何基准测试结果前，请记录以下信息，避免混合不同软硬件环境的数据。

## 硬件

- GPU：NVIDIA GeForce RTX 4060（桌面版，功耗上限 115 W，WDDM 模式，同时驱动显示器）
- 显存：8188 MiB（桌面空闲时已占用约 0.9 GB，实际可用约 7.1 GB）
- CPU：Intel Core i9-13900K
- 系统内存：待记录

## 软件

- 操作系统：待记录
- NVIDIA Driver：560.94（2026-10-07 记录）
- CUDA：驱动最高支持 12.6；CUDA Toolkit 版本待记录
- Python：待记录
- PyTorch：待记录
- Triton：待记录
- Nsight Systems：待记录
- Nsight Compute：待记录

## 可复现性

- Git commit：待记录
- 随机种子：待记录
- 精度：待记录
- 关键环境变量：待记录

<!-- AUTO-ENV-START -->

## 自动采集（由 scripts/collect_env.py 生成，请勿手改）

| 项目 | 值 |
|---|---|
| 记录时间 | 2026-10-07T18:53:56 |
| GPU | NVIDIA GeForce RTX 4060 |
| 显存 (MiB) | 8188 |
| NVIDIA Driver | 560.94 |
| CPU | 13th Gen Intel(R) Core(TM) i9-13900K |
| 系统内存（当前系统可见） | 31.2 GiB |
| 操作系统 | Linux-5.10.16.3-microsoft-standard-WSL2-x86_64-with-glibc2.31 |
| 内核 | 5.10.16.3-microsoft-standard-WSL2 |
| Python | 3.11.17 |
| PyTorch | 2.6.0+cu124 |
| PyTorch CUDA runtime | 12.4 |
| Triton | 3.2.0 |
| Nsight Compute | 未安装 |
| Nsight Systems | 未安装 |
| Git commit | 18296b8（有未提交修改） |

<!-- AUTO-ENV-END -->
