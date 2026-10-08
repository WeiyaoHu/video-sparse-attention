"""正确性测试：所有 kernel 都必须先通过这里，才能报告性能数字。

新增 kernel 时，把它注册到 KERNELS（签名 fn(q, k, v, mask=None) -> out）即可自动获得全部测试。
运行：pytest -q
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.benchmark_utils import make_qkv
from kernels.dense.torch_reference import naive_attention, reference_attention, sdpa_attention

KERNELS = {
    "naive": naive_attention,
    "sdpa": sdpa_attention,
}

# 只能在 GPU 上运行的 kernel
CUDA_ONLY = set()

try:
    from kernels.dense.triton_dense import triton_dense_attention

    KERNELS["triton_dense"] = triton_dense_attention
    CUDA_ONLY.add("triton_dense")
except ImportError:  # 没装 triton 的环境（例如纯 CPU）跳过
    pass

# 相对 float64 参考的容差，按精度区分
TOL = {
    torch.float32: dict(rtol=1e-4, atol=1e-5),
    torch.float16: dict(rtol=5e-2, atol=5e-3),
    torch.bfloat16: dict(rtol=1e-1, atol=3e-2),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_matches_reference(fn, q, k, v, mask=None):
    """把 fn 的输出与 float64 参考实现比较。参考使用的是同一份（已转换精度的）输入。"""
    with torch.inference_mode():
        out = fn(q, k, v, mask) if mask is not None else fn(q, k, v)
        ref = reference_attention(q, k, v, mask)
    assert out.shape == ref.shape
    assert out.dtype == q.dtype
    assert torch.isfinite(out).all(), "输出包含 NaN/Inf"
    torch.testing.assert_close(out.double(), ref, **TOL[q.dtype])


def _skip_if_unsupported(name, dtype):
    if DEVICE.type == "cpu" and (dtype != torch.float32 or name in CUDA_ONLY):
        pytest.skip("需要 GPU")


def make_block_mask(N, block, density, device, seed=0):
    """随机 block 稀疏 mask，[N, N] bool。对角 block 恒为 True，保证每一行至少有一个可见位置。"""
    nb = (N + block - 1) // block
    g = torch.Generator().manual_seed(seed)
    m = torch.rand(nb, nb, generator=g) < density
    m |= torch.eye(nb, dtype=torch.bool)
    m = m.repeat_interleave(block, 0).repeat_interleave(block, 1)[:N, :N]
    return m.to(device)


@pytest.mark.parametrize("name", list(KERNELS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("N", [64, 256, 1000, 2048])  # 1000：故意不是 block size 的整数倍
def test_dense_matches_reference(name, dtype, D, N):
    _skip_if_unsupported(name, dtype)
    q, k, v = make_qkv(1, 4, N, D, dtype=dtype, device=DEVICE, seed=N + D)
    assert_matches_reference(KERNELS[name], q, k, v)


@pytest.mark.parametrize("name", list(KERNELS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("density", [0.5, 0.125])
@pytest.mark.parametrize("N,block", [(512, 64), (1000, 64), (1024, 128)])
def test_block_mask_matches_reference(name, dtype, density, N, block):
    _skip_if_unsupported(name, dtype)
    q, k, v = make_qkv(1, 4, N, 64, dtype=dtype, device=DEVICE, seed=N)
    mask = make_block_mask(N, block, density, DEVICE)
    assert_matches_reference(KERNELS[name], q, k, v, mask)


@pytest.mark.parametrize("name", list(KERNELS))
def test_batch_and_head_independence(name):
    """每个 (batch, head) 的结果应当只依赖自己的 Q/K/V。"""
    _skip_if_unsupported(name, torch.float32)
    q, k, v = make_qkv(2, 3, 128, 64, dtype=torch.float32, device=DEVICE)
    fn = KERNELS[name]
    with torch.inference_mode():
        full = fn(q, k, v)
        single = fn(q[1:2, 2:3], k[1:2, 2:3], v[1:2, 2:3])
    torch.testing.assert_close(full[1:2, 2:3], single, rtol=1e-4, atol=1e-5)
