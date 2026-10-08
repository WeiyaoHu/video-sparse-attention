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
# grid = (N, B*H)，每个 program 处理一整行，但每次只读 BLOCK 个元素、沿行循环。
#
# 为什么不一次读完整行：最初的版本让 BLOCK = next_pow2(N)，N=8192 时一个 program
# 要把 8192 个元素放进寄存器，NVIDIA 的 ptxas 编译器在这种超大块上卡死。
# 现在 BLOCK 固定，与 N 无关，所有 N 共用一份编译结果。
#
# 两遍扫描：
#   第 1 遍：边读边维护"当前最大值 m"和"以 m 为基准的指数和 l"（online softmax），
#            遇到更大的值时把旧的 l 乘以 exp(m_old - m_new) 重新缩放。
#   第 2 遍：再读一次，写出 exp(x - m) / l。
# 这里每个 lane（块内位置）各自维护一份 m、l，最后再合并成整行的结果。
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
    b = bh // H
    h = bh % H
    s_row = S + bh * stride_sbh + row * stride_sm
    p_row = P + bh * stride_pbh + row * stride_pm
    m_row = M + b * stride_mb + h * stride_mh + row * stride_mm
    offs = tl.arange(0, BLOCK)

    # ---- 第 1 遍：求每个 lane 的最大值 m_i 和指数和 l_i ----
    m_i = tl.full([BLOCK], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK], tl.float32)
    for start in range(0, N, BLOCK):
        cols = start + offs
        valid = cols < N
        x = tl.load(s_row + cols, mask=valid, other=-float("inf")).to(tl.float32)
        if HAS_MASK:
            keep = tl.load(m_row + cols * stride_mn, mask=valid, other=0)
            x = tl.where(keep != 0, x, -float("inf"))
        m_new = tl.maximum(m_i, x)
        seen = m_new != -float("inf")  # 这个 lane 至少见过一个有效值；否则保持 0，避免 (-inf) - (-inf) = nan
        alpha = tl.where(seen, tl.exp(m_i - m_new), 0.0)
        l_i = l_i * alpha + tl.where(seen, tl.exp(x - m_new), 0.0)
        m_i = m_new

    # 合并所有 lane：整行最大值 m，以及以 m 为基准的指数和 l
    m = tl.max(m_i, axis=0)
    l = tl.sum(tl.where(m_i != -float("inf"), l_i * tl.exp(m_i - m), 0.0), axis=0)

    # ---- 第 2 遍：写出归一化结果 ----
    for start in range(0, N, BLOCK):
        cols = start + offs
        valid = cols < N
        x = tl.load(s_row + cols, mask=valid, other=-float("inf")).to(tl.float32)
        if HAS_MASK:
            keep = tl.load(m_row + cols * stride_mn, mask=valid, other=0)
            x = tl.where(keep != 0, x, -float("inf"))
        p = tl.exp(x - m) / l
        tl.store(p_row + cols, p.to(P.dtype.element_ty), mask=valid)


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
    _softmax_kernel[(N, BH)](
        S, P, m, N, H,
        S.stride(0), S.stride(1),
        P.stride(0), P.stride(1),
        *m_strides,
        HAS_MASK=has_mask, BLOCK=1024,
        num_warps=4,
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
