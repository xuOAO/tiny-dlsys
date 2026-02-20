from __future__ import annotations

from typing import List

from ..autograd import Tensor
from .optimizer import Optimizer


class SGD(Optimizer):
    """随机梯度下降（支持动量和权重衰减）。

    更新规则
    --------
    v_t = momentum * v_{t-1} + (grad + weight_decay * param)
    param = param - lr * v_t
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        # 为每个参数维护动量缓冲，初始为全零
        self.velocities = [None] * len(self.params)

    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # 使用 detach 后的值，避免将更新纳入计算图
            grad = param.grad.data
            p = param.data

            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p

            if self.momentum != 0.0:
                if self.velocities[i] is None:
                    self.velocities[i] = grad
                else:
                    self.velocities[i] = self.momentum * self.velocities[i] + grad
                update = self.velocities[i]
            else:
                update = grad

            param.data = p - self.lr * update
