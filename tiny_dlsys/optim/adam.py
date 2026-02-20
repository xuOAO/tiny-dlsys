from __future__ import annotations

from typing import List

from ..autograd import Tensor
from .optimizer import Optimizer


class Adam(Optimizer):
    """Adam 优化器（带偏差修正）。

    更新规则
    --------
    m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    param = param - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        # 一阶矩（均值）和二阶矩（未中心化方差）缓冲
        self.m = [None] * len(self.params)
        self.v = [None] * len(self.params)
        self.t = 0  # 全局步数计数器

    def step(self) -> None:
        self.t += 1
        beta1_t = self.beta1 ** self.t
        beta2_t = self.beta2 ** self.t

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data
            p = param.data

            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p

            # 更新一阶矩
            if self.m[i] is None:
                self.m[i] = (1 - self.beta1) * grad
            else:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # 更新二阶矩
            if self.v[i] is None:
                self.v[i] = (1 - self.beta2) * grad ** 2
            else:
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            # 偏差修正
            m_hat = self.m[i] / (1 - beta1_t)
            v_hat = self.v[i] / (1 - beta2_t)

            param.data = p - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
