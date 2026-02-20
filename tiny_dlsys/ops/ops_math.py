"""数学算子——前向 compute（NDArray）+ 反向 gradient（Tensor）。

每个算子实现 compute 和 gradient，供 autograd 调用。
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from ..autograd import Op, Tensor, TensorOp
from ..backend.ndarray import NDArray


# ---------------------------------------------------------------------------
# 广播梯度辅助
# ---------------------------------------------------------------------------


def _broadcast_gradient(out_grad: Tensor, shape: tuple) -> Tensor:
    """将 out_grad 规约到 shape，用于处理广播后的梯度。"""
    if out_grad.shape == shape:
        return out_grad
    ndim_added = len(out_grad.shape) - len(shape)
    axes_to_sum = list(range(ndim_added))
    for i in range(len(shape)):
        if shape[i] == 1:
            axes_to_sum.append(ndim_added + i)
    if not axes_to_sum:
        return out_grad.reshape(shape)
    return summation(out_grad, axes=tuple(axes_to_sum)).reshape(shape)


# ---------------------------------------------------------------------------
# 逐元素算子
# ---------------------------------------------------------------------------


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = node.inputs
        return _broadcast_gradient(out_grad, a.shape), _broadcast_gradient(out_grad, b.shape)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = node.inputs
        return _broadcast_gradient(out_grad * b, a.shape), _broadcast_gradient(out_grad * a, b.shape)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad * self.scalar


class EWiseDiv(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = node.inputs
        grad_a = _broadcast_gradient(out_grad / b, a.shape)
        grad_b = _broadcast_gradient(out_grad * (-a) / (b * b), b.shape)
        return grad_a, grad_b


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad / self.scalar


class PowerScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        a, = node.inputs
        a_data = a.realize_cached_data()
        grad_coef = (a_data ** (self.scalar - 1)) * self.scalar
        return multiply(out_grad, Tensor.make_const(grad_coef))


class Negate(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return -a

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return -out_grad


class Exp(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return a.exp()

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return multiply(out_grad, Tensor.make_const(node.realize_cached_data()))


class Log(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return a.log()

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        a, = node.inputs
        return out_grad / a


class ReLU(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return a.maximum(0)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        a, = node.inputs
        a_data = a.realize_cached_data()
        mask = Tensor.from_numpy(
            (a_data.numpy() >= 0).astype(a_data.dtype),
            device=out_grad.device,
            requires_grad=False,
        )
        return multiply(out_grad, mask)


class Tanh(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return a.tanh()

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        y_np = node.realize_cached_data().numpy()
        one_minus_y2 = 1 - y_np * y_np
        coef = Tensor.from_numpy(
            one_minus_y2.astype(y_np.dtype),
            device=out_grad.device,
            requires_grad=False,
        )
        return multiply(out_grad, coef)


# ---------------------------------------------------------------------------
# 矩阵 / 形状算子
# ---------------------------------------------------------------------------


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a @ b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = node.inputs
        grad_a = matmul(out_grad, transpose(b))
        grad_b = matmul(transpose(a), out_grad)
        return grad_a, grad_b


class Reshape(TensorOp):
    def __init__(self, shape: tuple):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return a.reshape(self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        a, = node.inputs
        return reshape(out_grad, a.shape)


class Transpose(TensorOp):
    def __init__(self, axes: Tuple[int, ...] = None):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        return a.transpose(self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        a, = node.inputs
        if self.axes is None:
            return transpose(out_grad)
        inv = [0] * len(self.axes)
        for i, j in enumerate(self.axes):
            inv[j] = i
        return transpose(out_grad, tuple(inv))


class BroadcastTo(TensorOp):
    def __init__(self, shape: tuple):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return a.broadcast_to(self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return _broadcast_gradient(out_grad, node.inputs[0].shape)


class Summation(TensorOp):
    def __init__(self, axes: Tuple[int, ...] = None):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        return a.sum(axis=self.axes, keepdims=False)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        a, = node.inputs
        if self.axes is None:
            return broadcast_to(out_grad, a.shape)
        out_shape = list(out_grad.shape)
        for ax in sorted(self.axes):
            out_shape.insert(ax, 1)
        return broadcast_to(reshape(out_grad, tuple(out_shape)), a.shape)


class Slice(TensorOp):
    def __init__(self, slices: tuple):
        self.slices = slices

    def compute(self, a: NDArray) -> NDArray:
        return a[self.slices]

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        a, = node.inputs
        a_np = a.realize_cached_data().numpy()
        out_np = out_grad.realize_cached_data().numpy()
        full_grad = np.zeros_like(a_np)
        full_grad[self.slices] = out_np
        return Tensor.from_numpy(
            full_grad,
            device=out_grad.device,
            requires_grad=False,
        )


class Flip(TensorOp):
    def __init__(self, axes: Tuple[int, ...]):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        return a.flip(self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return flip(out_grad, self.axes)


class Dilate(TensorOp):
    def __init__(self, axes: Tuple[int, ...], dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray) -> NDArray:
        return a.dilate(self.axes, self.dilation)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        a, = node.inputs
        slices = [slice(None)] * len(out_grad.shape)
        for ax in self.axes:
            if ax < len(slices):
                slices[ax] = slice(None, None, self.dilation + 1)
        return slice(out_grad, tuple(slices))


# ---------------------------------------------------------------------------
# 函数式接口
# ---------------------------------------------------------------------------


def add(a: Tensor, b: Tensor) -> Tensor:
    return EWiseAdd()(a, b)


def add_scalar(a: Tensor, scalar) -> Tensor:
    return AddScalar(scalar)(a)


def multiply(a: Tensor, b: Tensor) -> Tensor:
    return EWiseMul()(a, b)


def mul_scalar(a: Tensor, scalar) -> Tensor:
    return MulScalar(scalar)(a)


def divide(a: Tensor, b: Tensor) -> Tensor:
    return EWiseDiv()(a, b)


def divide_scalar(a: Tensor, scalar) -> Tensor:
    return DivScalar(scalar)(a)


def power_scalar(a: Tensor, scalar) -> Tensor:
    return PowerScalar(scalar)(a)


def negate(a: Tensor) -> Tensor:
    return Negate()(a)


def exp(a: Tensor) -> Tensor:
    return Exp()(a)


def log(a: Tensor) -> Tensor:
    return Log()(a)


def relu(a: Tensor) -> Tensor:
    return ReLU()(a)


def tanh(a: Tensor) -> Tensor:
    return Tanh()(a)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatMul()(a, b)


def reshape(a: Tensor, shape: tuple) -> Tensor:
    return Reshape(shape)(a)


def transpose(a: Tensor, axes: tuple = None) -> Tensor:
    return Transpose(axes)(a)


def broadcast_to(a: Tensor, shape: tuple) -> Tensor:
    return BroadcastTo(shape)(a)


def summation(a: Tensor, axes: tuple = None) -> Tensor:
    return Summation(axes)(a)


def slice_op(a: Tensor, slices: tuple) -> Tensor:
    return Slice(slices)(a)


# 别名，与 API 命名一致（避免与内置 slice 冲突）
slice = slice_op


def flip(a: Tensor, axes: tuple) -> Tensor:
    return Flip(axes)(a)


def dilate(a: Tensor, axes: tuple, dilation: int) -> Tensor:
    return Dilate(axes, dilation)(a)
