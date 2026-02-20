from __future__ import annotations

from typing import List

from ..autograd import Tensor


class Optimizer:
    """优化器基类。"""

    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        """将所有参数的 .grad 置为 None。"""
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        """执行一步参数更新（子类实现）。"""
        raise NotImplementedError
