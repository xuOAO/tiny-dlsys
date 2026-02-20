"""与设备绑定的多维数组——Tensor 底层的数据载体。

NDArray 持有原始后端数据（CPU 上为 ``numpy.ndarray``，CUDA 上为 GPU buffer）
以及所属 ``Device`` 引用。所有运算方法内部通过 ``device.backend`` 分派到
对应后端实现，从而让上层 Op 代码保持设备无关。
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy

from .device import Device, default_device


class NDArray:
    """与设备绑定的多维数组。"""

    def __init__(self, data, device: Device):
        self._data = data
        self._device = device

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return len(self._data.shape)

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def dtype(self) -> str:
        return str(self._data.dtype)

    @property
    def device(self) -> Device:
        return self._device

    @property
    def data(self):
        """返回原始后端数据（仅供 Device / backend 层使用）。"""
        return self._data

    # ------------------------------------------------------------------
    # 静态工厂方法
    # ------------------------------------------------------------------

    @staticmethod
    def from_numpy(np_array: numpy.ndarray, device: Device = None) -> "NDArray":
        if device is None:
            device = default_device()
        raw = device.from_numpy(np_array)
        return NDArray(raw, device)

    @staticmethod
    def zeros(shape, device: Device = None, dtype="float32") -> "NDArray":
        if device is None:
            device = default_device()
        return NDArray(device.zeros(shape, dtype), device)

    @staticmethod
    def ones(shape, device: Device = None, dtype="float32") -> "NDArray":
        if device is None:
            device = default_device()
        return NDArray(device.ones(shape, dtype), device)

    @staticmethod
    def randn(shape, device: Device = None, dtype="float32") -> "NDArray":
        if device is None:
            device = default_device()
        return NDArray(device.randn(shape, dtype), device)

    @staticmethod
    def empty(shape, device: Device = None, dtype="float32") -> "NDArray":
        if device is None:
            device = default_device()
        return NDArray(device.empty(shape, dtype), device)

    @staticmethod
    def full(shape, val, device: Device = None, dtype="float32") -> "NDArray":
        if device is None:
            device = default_device()
        raw = device.backend.full(shape, val, dtype)
        return NDArray(raw, device)

    # ------------------------------------------------------------------
    # 设备迁移与 NumPy 互转
    # ------------------------------------------------------------------

    def to(self, device: Device) -> "NDArray":
        if self._device == device:
            return self
        np_data = self.numpy()
        return NDArray.from_numpy(np_data, device)

    def numpy(self) -> numpy.ndarray:
        return self._device.to_numpy(self._data)

    # ------------------------------------------------------------------
    # 后端分派辅助
    # ------------------------------------------------------------------

    def _b(self):
        return self._device.backend

    @staticmethod
    def _raw(other):
        """提取原始数据：NDArray → _data，标量保持原样。"""
        return other._data if isinstance(other, NDArray) else other

    def _wrap(self, raw_data) -> "NDArray":
        return NDArray(raw_data, self._device)

    # ------------------------------------------------------------------
    # 逐元素运算
    # ------------------------------------------------------------------

    def __add__(self, other) -> "NDArray":
        return self._wrap(self._b().add(self._data, self._raw(other)))

    def __radd__(self, other) -> "NDArray":
        return self.__add__(other)

    def __mul__(self, other) -> "NDArray":
        return self._wrap(self._b().mul(self._data, self._raw(other)))

    def __rmul__(self, other) -> "NDArray":
        return self.__mul__(other)

    def __truediv__(self, other) -> "NDArray":
        return self._wrap(self._b().divide(self._data, self._raw(other)))

    def __rtruediv__(self, other) -> "NDArray":
        return self._wrap(self._b().divide(self._raw(other), self._data))

    def __neg__(self) -> "NDArray":
        return self._wrap(self._b().neg(self._data))

    def __sub__(self, other) -> "NDArray":
        return self.__add__(-other if isinstance(other, NDArray) else -other)

    def __rsub__(self, other) -> "NDArray":
        return (-self).__add__(other)

    def __pow__(self, scalar) -> "NDArray":
        return self._wrap(self._b().power(self._data, scalar))

    def exp(self) -> "NDArray":
        return self._wrap(self._b().exp(self._data))

    def log(self) -> "NDArray":
        return self._wrap(self._b().log(self._data))

    def tanh(self) -> "NDArray":
        return self._wrap(self._b().tanh(self._data))

    def sqrt(self) -> "NDArray":
        return self._wrap(self._b().sqrt(self._data))

    def maximum(self, other) -> "NDArray":
        return self._wrap(self._b().maximum(self._data, self._raw(other)))

    # ------------------------------------------------------------------
    # 规约运算
    # ------------------------------------------------------------------

    def sum(self, axis=None, keepdims=False) -> "NDArray":
        return self._wrap(self._b().reduce_sum(self._data, axis=axis, keepdims=keepdims))

    def max(self, axis=None, keepdims=False) -> "NDArray":
        return self._wrap(self._b().reduce_max(self._data, axis=axis, keepdims=keepdims))

    # ------------------------------------------------------------------
    # 矩阵运算
    # ------------------------------------------------------------------

    def matmul(self, other: "NDArray") -> "NDArray":
        return self._wrap(self._b().matmul(self._data, other._data))

    def __matmul__(self, other: "NDArray") -> "NDArray":
        return self.matmul(other)

    # ------------------------------------------------------------------
    # 形状操作
    # ------------------------------------------------------------------

    def reshape(self, shape) -> "NDArray":
        return self._wrap(self._b().reshape(self._data, shape))

    def transpose(self, axes=None) -> "NDArray":
        return self._wrap(self._b().transpose(self._data, axes))

    def broadcast_to(self, shape) -> "NDArray":
        return self._wrap(self._b().broadcast_to(self._data, shape))

    def __getitem__(self, slices) -> "NDArray":
        return self._wrap(self._b().getitem(self._data, slices))

    def __setitem__(self, slices, value):
        self._b().setitem(self._data, slices, self._raw(value))

    def flip(self, axes) -> "NDArray":
        return self._wrap(self._b().flip(self._data, axes))

    def pad(self, pad_width, mode="constant", constant_values=0) -> "NDArray":
        return self._wrap(self._b().pad(self._data, pad_width, mode=mode, constant_values=constant_values))

    def dilate(self, axes, dilation) -> "NDArray":
        return self._wrap(self._b().dilate(self._data, axes, dilation))

    # ------------------------------------------------------------------
    # 比较运算
    # ------------------------------------------------------------------

    def eq(self, other) -> "NDArray":
        return self._wrap(self._b().eq(self._data, self._raw(other)))

    def ge(self, other) -> "NDArray":
        return self._wrap(self._b().ge(self._data, self._raw(other)))

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"NDArray({self._data}, device={self._device})"

    def __str__(self) -> str:
        return str(self._data)
