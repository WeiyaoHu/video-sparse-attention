"""Dense attention baselines（Phase 1）。

约定（全项目统一）：
    q, k, v : [B, H, N, D]
    mask    : bool，可广播到 [B, H, N, N]，True 表示该位置参与 attention
              （与 torch.nn.functional.scaled_dot_product_attention 的 bool mask 语义一致）
    返回值  : [B, H, N, D]
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def naive_attention(q, k, v, mask=None):
    """教科书写法：显式构造 N×N 的 score 矩阵。

    显存随 N^2 增长的来源就在这里：scores 和 probs 都是 [B, H, N, N]。
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # QK^T
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)  # Softmax
    return torch.matmul(probs, v)  # PV


def sdpa_attention(q, k, v, mask=None):
    """PyTorch 内置 SDPA（强 baseline，CUDA 上通常走 FlashAttention 路径）。"""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def reference_attention(q, k, v, mask=None):
    """可信参考实现：在 float64 下计算，仅用于正确性比较，不用于计时。"""
    return naive_attention(q.double(), k.double(), v.double(), mask)
