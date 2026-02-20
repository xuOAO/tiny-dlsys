"""容器模块：Sequential。"""

from __future__ import annotations

from ..autograd import Tensor
from .module import Module


class Sequential(Module):
    """顺序容器，依次调用各子模块。"""

    def __init__(self, *modules: Module):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x
