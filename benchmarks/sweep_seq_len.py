"""Phase 1：sequence length sweep（naive vs SDPA）。

用法（在仓库根目录）：
    python -m benchmarks.sweep_seq_len
    python -m benchmarks.sweep_seq_len --seq-lens 512 1024 2048 --heads 8 --head-dims 64

结果追加写入 results/csv/dense_attention.csv。
"""

from __future__ import annotations

import argparse
import datetime as dt

import torch

from benchmarks.benchmark_utils import (
    DTYPES,
    REPO_ROOT,
    append_csv,
    attention_flops,
    benchmark,
    env_info,
    free_vram_bytes,
    get_device,
    make_qkv,
)
from kernels.dense.torch_reference import naive_attention, reference_attention, sdpa_attention

KERNELS = {"naive": naive_attention, "sdpa": sdpa_attention}

FIELDS = [
    "timestamp", "kernel", "B", "H", "N", "D", "dtype", "status",
    "mean_ms", "median_ms", "p95_ms", "min_ms", "std_ms", "iters",
    "peak_mem_mb", "extra_mem_mb", "tflops", "max_abs_err",
    "gpu", "gpu_mem_mb", "driver", "torch", "torch_cuda", "triton", "python", "platform",
    "git_commit", "git_dirty",
]

# 只在 N 不超过该值时计算与 float64 参考的误差（参考实现本身是 O(N^2) 显存）
ERR_CHECK_MAX_N = 2048


def naive_mem_estimate(B, H, N, dtype) -> int:
    """naive attention 的峰值额外显存估计：scores 和 probs 两个 [B,H,N,N] 同时存在。"""
    return 2 * B * H * N * N * torch.empty((), dtype=dtype).element_size()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--heads", type=int, nargs="+", default=[8, 16])
    p.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", choices=list(DTYPES), default="fp16")
    p.add_argument("--kernels", nargs="+", choices=list(KERNELS), default=list(KERNELS))
    p.add_argument("--device", default="auto")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "csv" / "dense_attention.csv"))
    args = p.parse_args()

    device = get_device(args.device)
    dtype = DTYPES[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        print("CPU 上改用 fp32（仅用于功能验证，数据不要用于报告）")
        args.dtype, dtype = "fp32", torch.float32

    env = env_info()
    print(f"device={device}  gpu={env.get('gpu')}  torch={env['torch']}  dtype={args.dtype}")
    B = args.batch

    for H in args.heads:
        for D in args.head_dims:
            for N in args.seq_lens:
                q, k, v = make_qkv(B, H, N, D, dtype=dtype, device=device)
                ref = None
                if N <= ERR_CHECK_MAX_N:
                    with torch.inference_mode():
                        ref = reference_attention(q, k, v)

                for name in args.kernels:
                    fn = KERNELS[name]
                    row = {
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                        "kernel": name, "B": B, "H": H, "N": N, "D": D, "dtype": args.dtype,
                        **env,
                    }

                    # Windows/WSL 下显存不足时驱动可能退到共享内存而不是报 OOM，
                    # 结果会慢几个数量级且没有意义，所以 naive 先做显存预估，放不下就直接跳过。
                    free = free_vram_bytes(device)
                    if name == "naive" and free is not None and naive_mem_estimate(B, H, N, dtype) > 0.9 * free:
                        res = {"status": "skipped_mem"}
                    else:
                        res = benchmark(fn, q, k, v, device=device, warmup=args.warmup, iters=args.iters)

                    row.update(res)
                    if res["status"] == "ok":
                        row["tflops"] = attention_flops(B, H, N, D) / (res["median_ms"] * 1e-3) / 1e12
                        if ref is not None:
                            with torch.inference_mode():
                                row["max_abs_err"] = (fn(q, k, v).double() - ref).abs().max().item()
                        print(
                            f"{name:>6} H={H:<3} D={D:<4} N={N:<6} "
                            f"median={res['median_ms']:9.3f} ms  p95={res['p95_ms']:9.3f} ms  "
                            f"extra_mem={res.get('extra_mem_mb', float('nan')):9.1f} MB"
                        )
                    else:
                        print(f"{name:>6} H={H:<3} D={D:<4} N={N:<6} {res['status']}")
                    append_csv(args.out, row, FIELDS)

                del q, k, v, ref
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
