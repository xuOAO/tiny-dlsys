"""CPU 后端——基于 NumPy 实现全部 NDArray 原子操作。

每个函数接收 / 返回的都是原始 ``numpy.ndarray``（而非 NDArray），
由 NDArray 的方法负责封装和拆箱。
"""

from __future__ import annotations

import numpy as np

# ====================== 逐元素运算 ======================


def add(a, b):
    return a + b


def mul(a, b):
    return a * b


def divide(a, b):
    return a / b


def neg(a):
    return -a


def exp(a):
    return np.exp(a)


def log(a):
    return np.log(a)


def power(a, scalar):
    return a ** scalar


def maximum(a, b):
    return np.maximum(a, b)


def tanh(a):
    return np.tanh(a)


def sqrt(a):
    return np.sqrt(a)


# ====================== 规约运算 ======================


def reduce_sum(a, axis=None, keepdims=False):
    return np.sum(a, axis=axis, keepdims=keepdims)


def reduce_max(a, axis=None, keepdims=False):
    return np.max(a, axis=axis, keepdims=keepdims)


# ====================== 矩阵运算 ======================


def matmul(a, b):
    return a @ b


# ====================== 形状操作 ======================


def reshape(a, shape):
    return np.array(a).reshape(shape)


def transpose(a, axes=None):
    return np.transpose(a, axes)


def broadcast_to(a, shape):
    return np.broadcast_to(a, shape).copy()


def getitem(a, slices):
    return a[slices].copy()


def setitem(a, slices, val):
    a[slices] = val


def flip(a, axes):
    return np.flip(a, axes).copy()


def pad(a, pad_width, mode="constant", constant_values=0):
    return np.pad(a, pad_width, mode=mode, constant_values=constant_values)


def dilate(a, axes, dilation):
    """在指定轴上插入零行/列以实现膨胀。"""
    new_shape = list(a.shape)
    for ax in axes:
        if ax < len(new_shape):
            new_shape[ax] = new_shape[ax] * (dilation + 1)
    out = np.zeros(new_shape, dtype=a.dtype)
    slices = [slice(None)] * len(new_shape)
    for ax in axes:
        if ax < len(new_shape):
            slices[ax] = slice(None, None, dilation + 1)
    out[tuple(slices)] = a
    return out


# ====================== 比较运算 ======================


def eq(a, b):
    return (a == b).astype(a.dtype)


def ge(a, b):
    return (a >= b).astype(a.dtype)


# ====================== 工具函数 ======================


def full(shape, val, dtype="float32"):
    return np.full(shape, val, dtype=dtype)


def one_hot(n, indices, dtype="float32"):
    return np.eye(n, dtype=dtype)[indices]
