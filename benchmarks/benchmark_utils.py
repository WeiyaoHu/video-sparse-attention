"""统一的 benchmark 工具：造数据、计时、显存统计、CSV 记录、环境信息。"""

from __future__ import annotations

import csv
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def make_qkv(B, H, N, D, dtype=torch.float16, device="cuda", seed=0):
    """生成合成 Q/K/V。固定 seed，先在 CPU 上用 fp32 生成再转换，保证不同设备/精度下输入一致。"""
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, H, N, D, generator=g) for _ in range(3))
    return tuple(t.to(device=device, dtype=dtype) for t in (q, k, v))


def attention_flops(B, H, N, D, density: float = 1.0) -> float:
    """两次 GEMM（QK^T 与 PV）的浮点运算数，各 2*N*N*D。density 为 active 比例（1 - sparsity）。"""
    return 4.0 * B * H * N * N * D * density


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(sorted_vals, p: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


@torch.inference_mode()
def benchmark(fn, *args, device, warmup: int = 5, iters: int = 20, slow_threshold_s: float = 2.0, **kwargs):
    """对 fn(*args, **kwargs) 计时。

    - 每次迭代前后都做 cuda.synchronize，测的是真实执行时间而不是提交时间。
    - 先试跑一次：如果单次超过 slow_threshold_s，自动减少迭代次数，避免大 N 时跑太久。
    - 显存 OOM 时返回 status="oom"，不抛异常，方便 sweep 继续。
    返回 dict：status / mean_ms / median_ms / p95_ms / min_ms / std_ms / iters / peak_mem_mb / extra_mem_mb
    """
    is_cuda = device.type == "cuda"
    result = {"status": "ok"}
    try:
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            base_mem = torch.cuda.memory_allocated(device)

        # 试跑（同时触发 lazy init / Triton JIT 编译）
        _sync(device)
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        _sync(device)
        first = time.perf_counter() - t0

        if first > slow_threshold_s:
            warmup, iters = 0, min(iters, 3)

        for _ in range(warmup):
            fn(*args, **kwargs)
        _sync(device)

        times = []
        for _ in range(iters):
            _sync(device)
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1e3)

        times.sort()
        result.update(
            mean_ms=statistics.fmean(times),
            median_ms=_percentile(times, 0.5),
            p95_ms=_percentile(times, 0.95),
            min_ms=times[0],
            std_ms=statistics.pstdev(times),
            iters=iters,
        )
        if is_cuda:
            peak = torch.cuda.max_memory_allocated(device)
            result["peak_mem_mb"] = peak / 2**20
            result["extra_mem_mb"] = (peak - base_mem) / 2**20  # 扣除输入张量后的额外显存
    except torch.cuda.OutOfMemoryError:
        result = {"status": "oom"}
        if is_cuda:
            torch.cuda.empty_cache()
    return result


def free_vram_bytes(device: torch.device):
    """当前空闲显存（字节）；非 CUDA 返回 None。"""
    if device.type != "cuda":
        return None
    free, _total = torch.cuda.mem_get_info(device)
    return free


def append_csv(path, row: dict, fieldnames) -> None:
    """追加一行到 CSV；文件不存在时写表头。fieldnames 固定列顺序，缺失的列留空。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=20).stdout.strip()
    except Exception:
        return ""


def env_info() -> dict:
    """采集软硬件环境，写进每条 benchmark 记录，避免混用不同环境的数据。"""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "",
        "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = ""
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**20)
        info["driver"] = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    else:
        info["gpu"] = ""
    return info
