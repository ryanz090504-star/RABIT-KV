from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log10, sqrt
from typing import Any

import numpy as np

from kvquant.packing import QuantizedKVBlock, quantization_error


def perplexity_from_losses(losses: list[float]) -> float:
    """从交叉熵损失列表计算困惑度（PPL）。

    PPL = exp(mean(losses))，越低越好。
    """
    if not losses:
        raise ValueError("无法从空损失列表计算困惑度")
    return float(exp(sum(losses) / len(losses)))


def kv_error_summary(block: QuantizedKVBlock, keys: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    """计算量化块相对于原始K/V张量的误差摘要。

    返回: 包含key误差、value误差、压缩比、原始字节数和估算payload字节数的字典。
    """
    key_deq, value_deq = block.dequantize()
    key_error = quantization_error(keys, key_deq)
    value_error = quantization_error(values, value_deq)
    return {
        "key": asdict(key_error),
        "value": asdict(value_error),
        "compression_ratio": block.compression_ratio(),
        "original_nbytes": block.original_nbytes(),
        "estimated_payload_nbytes": block.estimated_payload_nbytes(),
    }


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的softmax实现。"""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


# ── 量化误差详细指标 ────────────────────────────────────────────────

@dataclass
class QuantizationErrorStats:
    """量化误差的完整统计。

    可以用于两个场景：
    1. 单张量: original vs reconstructed 的误差分布
    2. 聚合: 多个层的误差汇总 (取 mean/max)
    """

    mse: float
    """均方误差: mean((original - reconstructed)^2)"""

    rmse: float
    """均方根误差: sqrt(MSE)"""

    max_abs: float
    """最大绝对误差: max(|original - reconstructed|)"""

    mean_abs: float
    """平均绝对误差 (MAE): mean(|original - reconstructed|)"""

    cosine_similarity: float
    """余弦相似度: dot(original, reconstructed) / (||original|| * ||reconstructed||)
    1.0 = 方向完全一致, 0 = 正交, -1 = 方向相反"""

    relative_l2: float
    """相对L2误差: ||original - reconstructed|| / ||original||"""

    snr_db: float
    """信噪比 (dB): 10 * log10(||original||^2 / ||error||^2)
    > 30dB = 好, > 20dB = 可接受, < 10dB = 严重退化"""

    outlier_count: int
    """异常值数量: |error| > 3 * std(error) 的元素数"""

    total_elements: int
    """总元素数"""

    @property
    def outlier_ratio(self) -> float:
        """异常值比例"""
        return self.outlier_count / self.total_elements if self.total_elements > 0 else 0.0


def quantization_error_stats(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    outlier_threshold_sigma: float = 3.0,
) -> QuantizationErrorStats:
    """计算量化误差的完整统计。

    参数:
      original: 原始浮点张量
      reconstructed: 量化→反量化后的张量
      outlier_threshold_sigma: 异常值判定为 |error| > k * std(error) 的标准差倍数

    返回:
      QuantizationErrorStats 包含 MSE, RMSE, max_abs, mean_abs,
      cosine_similarity, relative_l2, snr_db, outlier_count, total_elements
    """
    orig = np.asarray(original, dtype=np.float32)
    recons = np.asarray(reconstructed, dtype=np.float32)

    if orig.shape != recons.shape:
        raise ValueError(
            f"shape mismatch: original {orig.shape} vs reconstructed {recons.shape}"
        )

    error = orig - recons
    total = int(error.size)

    # 基础指标
    mse = float(np.mean(error * error)) if total > 0 else 0.0
    rmse = sqrt(mse)
    max_abs = float(np.max(np.abs(error))) if total > 0 else 0.0
    mean_abs = float(np.mean(np.abs(error))) if total > 0 else 0.0

    # 余弦相似度
    orig_flat = orig.ravel()
    recons_flat = recons.ravel()
    norm_orig = float(np.linalg.norm(orig_flat))
    norm_recons = float(np.linalg.norm(recons_flat))
    if norm_orig > 0 and norm_recons > 0:
        cosine = float(np.dot(orig_flat, recons_flat) / (norm_orig * norm_recons))
    else:
        cosine = 1.0

    # 相对 L2 误差
    relative_l2 = float(np.linalg.norm(error.ravel())) / norm_orig if norm_orig > 0 else 0.0

    # 信噪比
    signal_power = float(np.sum(orig_flat * orig_flat))
    noise_power = float(np.sum(error.ravel() * error.ravel()))
    if noise_power > 0:
        snr = 10.0 * log10(signal_power / noise_power)
    else:
        snr = float("inf")

    # 异常值计数
    error_flat = error.ravel()
    error_std = float(np.std(error_flat))
    if error_std > 0:
        threshold = outlier_threshold_sigma * error_std
        outlier_count = int(np.sum(np.abs(error_flat) > threshold))
    else:
        outlier_count = 0

    return QuantizationErrorStats(
        mse=mse,
        rmse=rmse,
        max_abs=max_abs,
        mean_abs=mean_abs,
        cosine_similarity=cosine,
        relative_l2=relative_l2,
        snr_db=snr,
        outlier_count=outlier_count,
        total_elements=total,
    )


def aggregate_error_stats(stats_list: list[QuantizationErrorStats]) -> QuantizationErrorStats:
    """聚合多个层的误差统计（取均值）。

    用于: "整个模型所有层的平均量化误差"
    """
    if not stats_list:
        raise ValueError("stats_list is empty")

    n = len(stats_list)
    return QuantizationErrorStats(
        mse=sum(s.mse for s in stats_list) / n,
        rmse=sqrt(sum(s.rmse**2 for s in stats_list) / n),
        max_abs=max(s.max_abs for s in stats_list),
        mean_abs=sum(s.mean_abs for s in stats_list) / n,
        cosine_similarity=sum(s.cosine_similarity for s in stats_list) / n,
        relative_l2=sum(s.relative_l2 for s in stats_list) / n,
        snr_db=sum(s.snr_db for s in stats_list) / n,
        outlier_count=sum(s.outlier_count for s in stats_list),
        total_elements=sum(s.total_elements for s in stats_list),
    )


# ── KL divergence and attention score error ──────────────────────────


def kl_divergence(logits_fp: np.ndarray, logits_quant: np.ndarray, eps: float = 1e-10) -> float:
    """Compute KL divergence between full-precision and quantized logits.

    KL(P_fp || P_quant) = sum(p_fp * log(p_fp / p_quant))

    Lower values indicate the quantized model's output distribution is close
    to the full-precision reference.

    Args:
        logits_fp: Logits from the full-precision model [vocab_size] or [batch, vocab_size].
        logits_quant: Logits from the quantized model, same shape as logits_fp.
        eps: Small epsilon for numerical stability.

    Returns:
        KL divergence in nats.
    """
    fp = np.asarray(logits_fp, dtype=np.float64)
    quant = np.asarray(logits_quant, dtype=np.float64)

    # Softmax along the last axis
    def _softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / (np.sum(exp_vals, axis=-1, keepdims=True) + eps)

    p = _softmax(fp)
    q = _softmax(quant)
    q = np.clip(q, eps, 1.0)

    kl = np.sum(p * (np.log(p + eps) - np.log(q)), axis=-1)
    return float(np.mean(kl))


def attention_score_error(
    query: np.ndarray,
    key_fp: np.ndarray,
    key_quant: np.ndarray,
    scale: float | None = None,
) -> dict[str, float]:
    """Compute attention score error between full-precision and quantized keys.

    Measures the distortion in dot-product attention scores caused by quantization
    of the key tensor. This is the principal signal-fidelity metric used in
    TurboQuant and related work.

    Args:
        query: Query tensor [batch, heads, head_dim] or [heads, head_dim].
        key_fp: Full-precision key tensor [batch, heads, seq, head_dim].
        key_quant: Dequantized key tensor, same shape as key_fp.
        scale: Optional softmax scale (default: 1/sqrt(head_dim)).

    Returns:
        Dict with keys: ``mse``, ``cosine_similarity``, ``top1_recall``,
        ``top5_recall``, ``relative_error``.
    """
    q = np.asarray(query, dtype=np.float32)
    k_fp = np.asarray(key_fp, dtype=np.float32)
    k_quant = np.asarray(key_quant, dtype=np.float32)

    if q.ndim == k_fp.ndim - 1:
        q = q[..., None, :]  # [..., 1, head_dim]

    if scale is None:
        head_dim = k_fp.shape[-1]
        scale = 1.0 / sqrt(float(head_dim))

    scores_fp = (q * scale) @ _swap_last_two(k_fp)
    scores_quant = (q * scale) @ _swap_last_two(k_quant)

    delta = scores_fp - scores_quant
    mse = float(np.mean(delta * delta))

    fp_flat = scores_fp.reshape(-1)
    quant_flat = scores_quant.reshape(-1)
    denom = float(np.linalg.norm(fp_flat)) * float(np.linalg.norm(quant_flat))
    cosine = (float(np.dot(fp_flat, quant_flat)) / denom) if denom > 0 else 1.0

    rel_error = float(np.linalg.norm(delta)) / float(np.linalg.norm(fp_flat)) if float(np.linalg.norm(fp_flat)) > 0 else 0.0

    # Top-k recall per query row
    k_values = [1, 5]
    recalls: dict[str, float] = {}
    for k in k_values:
        topk_fp = _topk_indices(scores_fp, k)
        topk_quant = _topk_indices(scores_quant, k)
        overlap = sum(
            1
            for fp_set, q_set in zip(topk_fp, topk_quant)
            if len(set(fp_set) & set(q_set)) > 0
        )
        recalls[f"top{k}_recall"] = overlap / max(len(topk_fp), 1)

    return {
        "mse": mse,
        "cosine_similarity": cosine,
        "relative_error": rel_error,
        **recalls,
    }


def _swap_last_two(arr: np.ndarray) -> np.ndarray:
    """Swap the last two axes of an array (transpose)."""
    if arr.ndim < 2:
        return arr
    axes = list(range(arr.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return np.transpose(arr, axes)


def _topk_indices(scores: np.ndarray, k: int) -> list[list[int]]:
    """Return top-k indices along the last axis for each row."""
    flat_leading = scores.reshape(-1, scores.shape[-1])
    return [list(row) for row in np.argpartition(-flat_leading, k - 1, axis=-1)[:, :k]]
