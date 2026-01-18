import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转角度的复数表示（cos + i*sin）
    Args:
        dim: 每个 head 的维度（必须是偶数）
        end: 最大序列长度
        theta: 基础频率缩放因子（默认 1e4）
    Returns:
        freqs_cis: [end, dim // 2] 的复数张量（实际用两个实数张量 cos/sin 表示）
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # [end, dim//2]
    freqs_cos = freqs.cos()
    freqs_sin = freqs.sin()
    return freqs_cos, freqs_sin  # 分开存储更便于后续操作，不使用torch.complex类型，因为部分系统不支持


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 query 和 key 应用 RoPE。
    Args:
        xq: [B, L, H, D]
        xk: [B, L, H, D]
        freqs_cos: [L, D//2]
        freqs_sin: [L, D//2]
    Returns:
        xq_rot, xk_rot: 旋转后的 q 和 k
    """
    # 将 xq, xk 从 [..., D] 拆分为 [..., D//2, 2]，视为复数的实部和虚部
    xq_r, xq_i = xq[..., 0::2], xq[..., 1::2]  # 偶数位为实部，奇数位为虚部
    xk_r, xk_i = xk[..., 0::2], xk[..., 1::2]

    # 自动扩展 freqs 到 batch 和 head 维度
    freqs_cos = freqs_cos[None, :, None, :]  # [1, L, 1, D//2]
    freqs_sin = freqs_sin[None, :, None, :]

    # 复数乘法: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    xq_rot_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_rot_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_rot_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_rot_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 合并回 [..., D]
    xq_out = torch.stack([xq_rot_r, xq_rot_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_rot_r, xk_rot_i], dim=-1).flatten(-2)

    return xq_out, xk_out