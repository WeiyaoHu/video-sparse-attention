# ===========================================================================
# 存档：triton_dense 的第一版（v0），保留供对照学习，不被任何代码导入。
#
# 与现行版本 kernels/dense/triton_dense.py 的唯一区别在 kernel 2（softmax）：
#   v0 让每个 program 一次读入整行，BLOCK = next_power_of_2(N)，num_warps 随 BLOCK 增大。
#   结果：N <= 4096 时测试全部通过、性能正常；
#         N = 8192 时 NVIDIA 的 ptxas 编译器在这个超大块上卡死（2026-10-08，RTX 4060，Triton 3.2.0），
#         traceback 停在 triton/backends/nvidia/compiler.py 的 make_cubin -> ptxas 子进程。
#   现行版本改为固定 BLOCK=1024 沿行循环，两遍扫描 + online softmax。
#
# 如需运行：from kernels.dense.legacy.triton_dense_v0 import triton_dense_attention
# 注意不要在 N >= 8192 时调用。
# ===========================================================================

"""Phase 1：最简单的 Triton dense attention —— 三个独立 kernel，不做融合。

与 naive_attention 一一对应：
    kernel 1  _qk_kernel       S = Q K^T * scale   -> 写回显存 [B*H, N, N]
    kernel 2  _softmax_kernel  P = softmax(S)      -> 写回显存 [B*H, N, N]
    kernel 3  _pv_kernel       O = P V             -> [B*H, N, D]

它和 naive 一样需要 O(N^2) 显存，这是故意的：
先熟悉 Triton 写法，Phase 2 再把三步融合成一个 kernel、用 online softmax 去掉 S 和 P，
前后对比最清楚。

接口与 kernels/dense/torch_reference.py 一致：
    q, k, v : [B, H, N, D]，CUDA 张量，fp32 / fp16 / bf16
    mask    : bool，可广播到 [B, H, N, N]，True 表示参与 attention
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)


# ---------------------------------------------------------------------------
# kernel 1：S = Q K^T * scale
# grid = (N 方向的 query 块数, N 方向的 key 块数, B*H)
# 每个 program 计算 S 的一个 [BLOCK_M, BLOCK_N] 小块。
# ---------------------------------------------------------------------------
@triton.jit
def _qk_kernel(
    Q, K, S, scale, N,
    stride_qbh, stride_qn,
    stride_kbh, stride_kn,
    stride_sbh, stride_sm,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, IEEE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    bh = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # 这个 program 负责的 query 行
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # 这个 program 负责的 key 行
    offs_d = tl.arange(0, HEAD_DIM)

    # 读入 Q 块 [BLOCK_M, D] 和 K 块 [BLOCK_N, D]；越界的行补 0（N 不是块大小整数倍时）
    q = tl.load(Q + bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :],
                mask=offs_m[:, None] < N, other=0.0)
    k = tl.load(K + bh * stride_kbh + offs_n[:, None] * stride_kn + offs_d[None, :],
                mask=offs_n[:, None] < N, other=0.0)

    # fp32 输入时强制 IEEE 精度，否则 Tensor Core 默认走 TF32，误差约 1e-3，过不了 fp32 测试
    if IEEE:
        s = tl.dot(q, tl.trans(k), input_precision="ieee")
    else:
        s = tl.dot(q, tl.trans(k))
    s = s * scale

    tl.store(S + bh * stride_sbh + offs_m[:, None] * stride_sm + offs_n[None, :],
             s.to(S.dtype.element_ty),
             mask=(offs_m[:, None] < N) & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------
# kernel 2：P = softmax(S)（按行）
# grid = (N, B*H)，每个 program 处理一整行；BLOCK 取不小于 N 的 2 的幂。
# 一整行一次放进寄存器，所以这种写法只适合 N 不太大的情况——这也是 Phase 2 要解决的问题。
# ---------------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    S, P, M, N, H,
    stride_sbh, stride_sm,
    stride_pbh, stride_pm,
    stride_mb, stride_mh, stride_mm, stride_mn,
    HAS_MASK: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    bh = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    valid = cols < N

    # 在 fp32 里做 softmax，越界位置填 -inf，exp 之后自然为 0
    x = tl.load(S + bh * stride_sbh + row * stride_sm + cols, mask=valid, other=-float("inf")).to(tl.float32)

    if HAS_MASK:
        b = bh // H
        h = bh % H
        keep = tl.load(M + b * stride_mb + h * stride_mh + row * stride_mm + cols * stride_mn,
                       mask=valid, other=0)
        x = tl.where(keep != 0, x, -float("inf"))

    x = x - tl.max(x, axis=0)  # 减去最大值，防止 exp 溢出
    e = tl.exp(x)
    p = e / tl.sum(e, axis=0)

    tl.store(P + bh * stride_pbh + row * stride_pm + cols, p.to(P.dtype.element_ty), mask=valid)


# ---------------------------------------------------------------------------
# kernel 3：O = P V
# grid = (N 方向的 query 块数, B*H)
# 每个 program 计算 O 的一个 [BLOCK_M, D] 块，沿 key 方向按 BLOCK_K 循环累加。
# ---------------------------------------------------------------------------
@triton.jit
def _pv_kernel(
    P, V, O, N,
    stride_pbh, stride_pm,
    stride_vbh, stride_vn,
    stride_obh, stride_om,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, IEEE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, HEAD_DIM)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)  # 累加器始终用 fp32
    for start in range(0, N, BLOCK_K):
        k_idx = start + offs_k
        p = tl.load(P + bh * stride_pbh + offs_m[:, None] * stride_pm + k_idx[None, :],
                    mask=(offs_m[:, None] < N) & (k_idx[None, :] < N), other=0.0)
        v = tl.load(V + bh * stride_vbh + k_idx[:, None] * stride_vn + offs_d[None, :],
                    mask=k_idx[:, None] < N, other=0.0)
        if IEEE:
            acc += tl.dot(p, v, input_precision="ieee")
        else:
            acc += tl.dot(p, v)

    tl.store(O + bh * stride_obh + offs_m[:, None] * stride_om + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N)


# ---------------------------------------------------------------------------
# Python 入口
# ---------------------------------------------------------------------------
def _softmax_num_warps(block: int) -> int:
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    return 4


def triton_dense_attention(q, k, v, mask=None):
    assert q.is_cuda, "Triton kernel 需要 CUDA 张量"
    assert q.shape == k.shape == v.shape, "目前只支持 q/k/v 形状相同（自注意力）"
    B, H, N, D = q.shape
    assert D in SUPPORTED_HEAD_DIMS, f"head_dim 需为 {SUPPORTED_HEAD_DIMS} 之一，当前 {D}"

    BH = B * H
    q_, k_, v_ = (t.contiguous().reshape(BH, N, D) for t in (q, k, v))
    # S、P 用与输入相同的精度存储，和 naive 的显存占用保持一致，方便公平对比
    S = torch.empty((BH, N, N), device=q.device, dtype=q.dtype)
    P = torch.empty_like(S)
    O = torch.empty((BH, N, D), device=q.device, dtype=q.dtype)

    ieee = q.dtype == torch.float32
    # fp32 每个元素 4 字节，块取小一点，避免超出 4060 的 shared memory
    blk = 32 if ieee else 64

    # kernel 1
    grid = (triton.cdiv(N, blk), triton.cdiv(N, blk), BH)
    _qk_kernel[grid](
        q_, k_, S, 1.0 / math.sqrt(D), N,
        q_.stride(0), q_.stride(1),
        k_.stride(0), k_.stride(1),
        S.stride(0), S.stride(1),
        HEAD_DIM=D, BLOCK_M=blk, BLOCK_N=blk, IEEE=ieee,
        num_warps=4,
    )

    # kernel 2
    if mask is not None:
        assert mask.dtype == torch.bool, "mask 需为 bool"
        m = mask.to(q.device).expand(B, H, N, N).view(torch.uint8)  # 广播维度的 stride 为 0，不额外占显存
        has_mask, m_strides = True, m.stride()
    else:
        m, has_mask, m_strides = S, False, (0, 0, 0, 0)  # 占位，不会被读取
    BLOCK = triton.next_power_of_2(N)
    _softmax_kernel[(N, BH)](
        S, P, m, N, H,
        S.stride(0), S.stride(1),
        P.stride(0), P.stride(1),
        *m_strides,
        HAS_MASK=has_mask, BLOCK=BLOCK,
        num_warps=_softmax_num_warps(BLOCK),
    )

    # kernel 3
    _pv_kernel[(triton.cdiv(N, 64), BH)](
        P, v_, O, N,
        P.stride(0), P.stride(1),
        v_.stride(0), v_.stride(1),
        O.stride(0), O.stride(1),
        HEAD_DIM=D, BLOCK_M=64, BLOCK_K=32, IEEE=ieee,
        num_warps=4, num_stages=2,
    )

    return O.reshape(B, H, N, D)
