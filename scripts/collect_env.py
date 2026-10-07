"""采集当前环境并写入 docs/environment.md 的自动区块。

用法（仓库根目录）：python scripts/collect_env.py
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_utils import env_info  # noqa: E402

START, END = "<!-- AUTO-ENV-START -->", "<!-- AUTO-ENV-END -->"


def _read(path, key):
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def main():
    info = env_info()
    mem_kb = _read("/proc/meminfo", "MemTotal").split()[0] if _read("/proc/meminfo", "MemTotal") else ""
    rows = [
        ("记录时间", dt.datetime.now().isoformat(timespec="seconds")),
        ("GPU", info.get("gpu", "")),
        ("显存 (MiB)", info.get("gpu_mem_mb", "")),
        ("NVIDIA Driver", info.get("driver", "")),
        ("CPU", _read("/proc/cpuinfo", "model name")),
        ("系统内存（当前系统可见）", f"{int(mem_kb) / 2**20:.1f} GiB" if mem_kb else ""),
        ("操作系统", info["platform"]),
        ("内核", _run(["uname", "-r"])),
        ("Python", info["python"]),
        ("PyTorch", info["torch"]),
        ("PyTorch CUDA runtime", info["torch_cuda"]),
        ("Triton", info["triton"]),
        ("Nsight Compute", _run(["ncu", "--version"]).splitlines()[0] if _run(["ncu", "--version"]) else "未安装"),
        ("Nsight Systems", _run(["nsys", "--version"]).splitlines()[0] if _run(["nsys", "--version"]) else "未安装"),
        ("Git commit", info["git_commit"] + ("（有未提交修改）" if info["git_dirty"] else "")),
    ]
    block = [START, "", "## 自动采集（由 scripts/collect_env.py 生成，请勿手改）", "", "| 项目 | 值 |", "|---|---|"]
    block += [f"| {k} | {v} |" for k, v in rows]
    block += ["", END]
    block = "\n".join(block)

    doc = ROOT / "docs" / "environment.md"
    text = doc.read_text(encoding="utf-8")
    if START in text and END in text:
        text = text[: text.index(START)] + block + text[text.index(END) + len(END):]
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    doc.write_text(text, encoding="utf-8")

    for k, v in rows:
        print(f"{k:<28} {v}")
    print(f"\n已更新 {doc}")


if __name__ == "__main__":
    main()
