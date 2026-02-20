"""激活函数模块。"""

from __future__ import annotations

from ..autograd import Tensor
from .. import ops
from .module import Module


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        # sigmoid(x) = 1 / (1 + exp(-x))
        return (ops.exp(-x) + 1) ** (-1)
