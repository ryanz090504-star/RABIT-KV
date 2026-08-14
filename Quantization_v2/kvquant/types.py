from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TokenType(str, Enum):
    """VLA场景中的token类型。

    后续VLA实验可以区分文本、视觉、动作、系统提示等不同模态的token，
    以便对不同类型的缓存区域采用不同的量化策略。
    """
    TEXT = "text"
    VISUAL = "visual"
    ACTION = "action"
    SYSTEM = "system"
    PROMPT = "prompt"
    HISTORY = "history"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModalitySpan:
    """标记序列中的一段多模态token区间。

    start: 区间起始位置（包含）
    end: 区间结束位置（不包含）
    token_type: 该区间内token的类型
    name: 该区间的可读名称（如 "image_1"）
    """
    start: int
    end: int
    token_type: TokenType
    name: str = ""

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("span start 必须为非负数")
        if self.end < self.start:
            raise ValueError("span end 必须大于等于 start")

    def contains(self, position: int) -> bool:
        """检查给定位置是否在此区间内。"""
        return self.start <= position < self.end


@dataclass
class QuantizationContext:
    """量化操作的上下文信息。

    layer_idx: attention层索引（0-based，只计数attention层，不是model layer索引）
    token_positions: 各token在序列中的位置
    attention_scores: attention分数矩阵，用于attention感知的量化策略
    importance_state: 重要性状态的额外信息
    modality_spans: 多模态区间标记（VLA扩展预留）
    metadata: 附加元数据
    """
    layer_idx: int
    token_positions: np.ndarray | None = None
    attention_scores: np.ndarray | None = None
    importance_state: dict[str, Any] = field(default_factory=dict)
    modality_spans: tuple[ModalitySpan, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantizationError:
    """量化误差指标。

    mse: 原始张量和反量化后张量之间的均方误差
    cosine: 原始张量和反量化后张量之间的余弦相似度（1.0 = 完美）
    max_abs: 最大绝对误差
    """
    mse: float
    cosine: float
    max_abs: float


def as_float_array(value: Any) -> np.ndarray:
    """将任意值转换为 float32 numpy 数组。"""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    return array
