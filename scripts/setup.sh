#!/usr/bin/env bash
# 一键搭建开发环境（在 WSL2 / Linux 里运行）：
#   bash scripts/setup.sh
#
# 可用环境变量：
#   VENV_DIR      虚拟环境位置，默认 ~/.venvs/vsa（放在 WSL 内部，比放 /mnt/d 快）
#   TORCH_INDEX   PyTorch wheel 源，默认 cu124
#   UV_DEFAULT_INDEX 其余 Python 包的源（下载慢时可设为国内镜像）
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/.venvs/vsa}"
PY_VER="3.11"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

cd "$(dirname "$0")/.."

echo "==> [1/4] 检查 GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "错误：找不到 nvidia-smi。"
  echo "如果这是 WSL：请在 PowerShell 里运行 'wsl -l -v' 确认 VERSION 为 2，"
  echo "若为 1 则执行 'wsl --set-version Ubuntu 2' 后重试。"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "==> [2/4] 准备 Python 环境（uv + venv，不依赖 conda）"
# 说明：本机 Anaconda 的 conda 在解析包列表时会 segfault，所以改用 uv。
# uv 是单个可执行文件，自己管理 Python 版本，和现有 conda 互不影响。
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "安装 uv"
  python3 -m pip install --user --quiet uv 2>/dev/null \
    || curl -LsSf https://astral.sh/uv/install.sh | sh
  hash -r
fi
uv --version

if [ ! -x "$VENV_DIR/bin/python" ]; then
  # 优先用 3.11（uv 会自动下载）；下载失败时退回本机已有的 >=3.9 的 Python
  uv venv --seed --python "$PY_VER" "$VENV_DIR" \
    || uv venv --seed --python ">=3.9" "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python --version

echo "==> [3/4] 安装 Python 包"
# Linux 版 PyTorch 自带与之匹配的 Triton，不要单独升级 triton，否则容易版本不兼容
uv pip install torch --index-url "$TORCH_INDEX"
uv pip install pytest numpy pandas matplotlib

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "PyTorch 看不到 GPU"
print("gpu  ", torch.cuda.get_device_name(0))
import triton
print("triton", triton.__version__)
PY

echo "==> [4/4] 记录环境并运行正确性测试"
python scripts/collect_env.py
python -m pytest -q

cat <<MSG

完成。之后每次开新终端先执行：
    source ${VENV_DIR}/bin/activate
然后跑第一个 benchmark：
    python -m benchmarks.sweep_seq_len
    python -m benchmarks.plot_seq_len
MSG
