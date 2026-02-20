"""归一化层：BatchNorm 和 LayerNorm。"""

from __future__ import annotations

import numpy as np

from ..autograd import Tensor
from .. import ops
from ..init import initializers as init
from ..backend.ndarray import NDArray
from .module import Module, Parameter


class BatchNorm(Module):
    """批归一化。

    训练时用 batch 统计量（均值/方差），推理时用滑动统计量。

    参数
    ----
    dim       : 特征维度（输入最后一维）
    eps       : 数值稳定项
    momentum  : 滑动统计量更新系数
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device=None,
        dtype: str = "float32",
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype), device=device, dtype=dtype
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype), device=device, dtype=dtype
        )

        # 滑动统计量（不参与梯度）
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., dim)  假设最后一维是特征维
        # 计算沿除最后维之外的所有轴的均值/方差
        # 对形状 (N, dim) 或 (N, ..., dim) 都适用
        shape = x.shape
        N = 1
        for s in shape[:-1]:
            N *= s

        # 计算 batch 均值和方差（沿前面所有维度）
        reduce_axes = tuple(range(len(shape) - 1))

        if self.training:
            mean = ops.summation(x, axes=reduce_axes) / N  # (dim,)
            # 广播均值做差
            mean_bc = mean.reshape((1,) * (len(shape) - 1) + (self.dim,)).broadcast_to(shape)
            diff = x - mean_bc
            var = ops.summation(diff ** 2, axes=reduce_axes) / N  # (dim,)

            # 更新滑动统计量（使用 numpy，不参与计算图）
            mean_np = mean.numpy()
            var_np = var.numpy()
            rm_np = self.running_mean.numpy()
            rv_np = self.running_var.numpy()
            new_rm = (1 - self.momentum) * rm_np + self.momentum * mean_np
            new_rv = (1 - self.momentum) * rv_np + self.momentum * var_np
            dev = self.running_mean.device
            self.running_mean = Tensor(
                NDArray.from_numpy(new_rm.astype(rm_np.dtype), device=dev),
                device=dev,
                requires_grad=False,
            )
            self.running_var = Tensor(
                NDArray.from_numpy(new_rv.astype(rv_np.dtype), device=dev),
                device=dev,
                requires_grad=False,
            )

            std = (var + self.eps) ** 0.5
            std_bc = std.reshape((1,) * (len(shape) - 1) + (self.dim,)).broadcast_to(shape)
            x_norm = diff / std_bc
        else:
            rm = Tensor(
                NDArray.from_numpy(self.running_mean.numpy(), device=x.device),
                device=x.device,
                requires_grad=False,
            )
            rv = Tensor(
                NDArray.from_numpy(self.running_var.numpy(), device=x.device),
                device=x.device,
                requires_grad=False,
            )
            mean_bc = rm.reshape((1,) * (len(shape) - 1) + (self.dim,)).broadcast_to(shape)
            std_bc = (rv + self.eps) ** 0.5
            std_bc = std_bc.reshape((1,) * (len(shape) - 1) + (self.dim,)).broadcast_to(shape)
            x_norm = (x - mean_bc) / std_bc

        w_bc = self.weight.reshape((1,) * (len(shape) - 1) + (self.dim,)).broadcast_to(shape)
        b_bc = self.bias.reshape((1,) * (len(shape) - 1) + (self.dim,)).broadcast_to(shape)
        return w_bc * x_norm + b_bc


class LayerNorm(Module):
    """层归一化：对最后一维做归一化。

    参数
    ----
    dim : 最后一维的大小
    eps : 数值稳定项
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device=None,
        dtype: str = "float32",
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype), device=device, dtype=dtype
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype), device=device, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        # 沿最后一维求均值和方差
        mean = ops.summation(x, axes=(-1,)) / self.dim  # (...,)
        # 插入最后一维以广播
        mean_bc = mean.reshape(shape[:-1] + (1,)).broadcast_to(shape)
        diff = x - mean_bc
        var = ops.summation(diff ** 2, axes=(-1,)) / self.dim  # (...,)
        std = (var + self.eps) ** 0.5
        std_bc = std.reshape(shape[:-1] + (1,)).broadcast_to(shape)
        x_norm = diff / std_bc

        w_bc = self.weight.reshape((1,) * (len(shape) - 1) + (self.dim,)).broadcast_to(shape)
        b_bc = self.bias.reshape((1,) * (len(shape) - 1) + (self.dim,)).broadcast_to(shape)
        return w_bc * x_norm + b_bc
