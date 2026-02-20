"""Dropout 层。"""

from __future__ import annotations

import numpy as np

from ..autograd import Tensor
from ..backend.ndarray import NDArray
from .module import Module


class Dropout(Module):
    """训练时以概率 p 随机置零并缩放（inverted dropout），推理时直通。

    参数
    ----
    p : 置零概率（默认 0.5）
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        # 生成 Bernoulli mask：以概率 (1-p) 保留
        mask_np = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
        mask = Tensor(
            NDArray.from_numpy(mask_np, device=x.device),
            device=x.device,
            requires_grad=False,
        )
        return x * mask / (1.0 - self.p)
