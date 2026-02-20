"""线性层。"""

from __future__ import annotations

from ..autograd import Tensor
from .. import ops
from ..init import initializers as init
from .module import Module, Parameter


class Linear(Module):
    """全连接线性层：y = x @ W + b。

    参数
    ----
    weight : (in_features, out_features)，Kaiming uniform 初始化
    bias   : (out_features,)，零初始化（可选）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype: str = "float32",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype),
            device=device,
            dtype=dtype,
        )
        if bias:
            self.bias = Parameter(
                init.zeros(out_features, device=device, dtype=dtype),
                device=device,
                dtype=dtype,
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias.broadcast_to(out.shape)
        return out
