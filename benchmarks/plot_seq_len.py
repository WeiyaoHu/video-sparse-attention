"""把 dense_attention.csv 画成两张图：N vs latency、N vs 额外显存。

用法：python -m benchmarks.plot_seq_len [--H 8] [--D 64]
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.benchmark_utils import REPO_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(REPO_ROOT / "results" / "csv" / "dense_attention.csv"))
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--D", type=int, default=64)
    args = p.parse_args()

    # 同一配置多次运行时取最后一次
    latest = {}
    with open(args.csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok" and int(r["H"]) == args.H and int(r["D"]) == args.D:
                latest[(r["kernel"], int(r["N"]))] = r
    if not latest:
        raise SystemExit("CSV 里没有匹配的数据，先运行 python -m benchmarks.sweep_seq_len")

    kernels = sorted({k for k, _ in latest})
    gpu = next(iter(latest.values())).get("gpu", "")
    dtype = next(iter(latest.values())).get("dtype", "")
    out_dir = REPO_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for col, ylabel, fname in [
        ("median_ms", "Median latency (ms)", "latency_vs_seq_len.png"),
        ("extra_mem_mb", "Peak extra memory (MB)", "memory_vs_seq_len.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for kern in kernels:
            pts = sorted((n, float(r[col])) for (kk, n), r in latest.items() if kk == kern and r.get(col))
            if pts:
                ax.plot(*zip(*pts), marker="o", label=kern)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Sequence length N")
        ax.set_ylabel(ylabel)
        ax.set_title(f"H={args.H}, D={args.D}, {dtype}, {gpu}", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        print("saved", out_dir / fname)


if __name__ == "__main__":
    main()
