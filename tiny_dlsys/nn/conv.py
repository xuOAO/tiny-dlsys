"""卷积层：Conv2d，使用 im2col 实现。

输入/输出采用 NHWC 布局：
    x:      (N, H,     W,     C_in)
    weight: (K, K, C_in, C_out)
    output: (N, H_out, W_out, C_out)
"""

from __future__ import annotations

import builtins

from ..autograd import Tensor
from .. import ops
from ..init import initializers as init
from .module import Module, Parameter


class Conv2d(Module):
    """2D 卷积层，NHWC 布局，im2col 实现。

    参数
    ----
    in_channels  : 输入通道数
    out_channels : 输出通道数
    kernel_size  : 卷积核尺寸（正方形）
    stride       : 步长（默认 1）
    padding      : 四边填充大小（默认 0）
    bias         : 是否使用偏置（默认 True）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device=None,
        dtype: str = "float32",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # weight: (K, K, C_in, C_out)，Kaiming uniform 初始化
        fan_in = kernel_size * kernel_size * in_channels
        fan_out = out_channels
        w_data = init.kaiming_uniform(fan_in, fan_out, device=device, dtype=dtype)
        self.weight = Parameter(
            w_data.reshape((kernel_size, kernel_size, in_channels, out_channels)),
            device=device,
            dtype=dtype,
        )

        if bias:
            self.bias = Parameter(
                init.zeros(out_channels, device=device, dtype=dtype),
                device=device,
                dtype=dtype,
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (N, H, W, C_in)  NHWC 布局
        返回: (N, H_out, W_out, C_out)
        """
        N, H, W, C_in = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1

        # 1. 填充
        if P > 0:
            x = ops.pad(x, ((0, 0), (P, P), (P, P), (0, 0)))

        # 2. im2col：对每个卷积核位置 (ki, kj) 提取对应切片
        #    切片形状: (N, H_out, W_out, C_in) → 展平为 (N*H_out*W_out, C_in)
        patches = []
        for ki in range(K):
            for kj in range(K):
                slices = (
                    builtins.slice(None),
                    builtins.slice(ki, ki + H_out * S, S),
                    builtins.slice(kj, kj + W_out * S, S),
                    builtins.slice(None),
                )
                patch = ops.slice_op(x, slices)  # (N, H_out, W_out, C_in)
                patches.append(patch.reshape((N * H_out * W_out, C_in)))

        # 3. 沿新轴 0 拼接: (K*K, N*H_out*W_out, C_in)
        col = ops.stack(patches, axis=0)
        # 转置到 (N*H_out*W_out, K*K, C_in) 再展平为 (N*H_out*W_out, K*K*C_in)
        col = col.transpose((1, 0, 2)).reshape((N * H_out * W_out, K * K * C_in))

        # 4. 权重展平: (K, K, C_in, C_out) → (K*K*C_in, C_out)
        W_mat = self.weight.reshape((K * K * C_in, self.out_channels))

        # 5. 矩阵乘法并还原空间维度
        out = (col @ W_mat).reshape((N, H_out, W_out, self.out_channels))

        # 6. 加偏置
        if self.bias is not None:
            out = out + self.bias.broadcast_to(out.shape)

        return out
