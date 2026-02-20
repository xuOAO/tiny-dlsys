"""设备抽象与具体实现。

所有 NDArray 都绑定到某个 Device 实例上。Device 定义了创建 / 转换
原始后端数据的统一接口，由具体子类（CPUDevice、CUDADevice）实现。

Device 方法操作的是**原始后端数据**（CPU 上为 ``numpy.ndarray``，
CUDA 上为 GPU buffer），而非上层的 ``NDArray``。``NDArray`` 会持有
Device 引用并调用这些方法来完成数据分配与格式转换。
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy


class Device:
    """设备基类。

    子类需实现全部以 ``NotImplementedError`` 标记的方法，
    以便 NDArray 可以在该设备上完成数据分配与格式转换。

    各方法返回 / 接受的 *data* 均为该设备对应的原始数组类型
    （CPU → ``numpy.ndarray``，CUDA → GPU buffer）。
    """

    # ------------------------------------------------------------------
    # 可用性
    # ------------------------------------------------------------------

    def enabled(self) -> bool:
        """当前设备是否可用。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 数据创建 —— 返回原始后端数组
    # ------------------------------------------------------------------

    def zeros(self, shape: Tuple[int, ...], dtype: str = "float32") -> Any:
        """创建全零数组。"""
        raise NotImplementedError

    def ones(self, shape: Tuple[int, ...], dtype: str = "float32") -> Any:
        """创建全一数组。"""
        raise NotImplementedError

    def randn(self, shape: Tuple[int, ...], dtype: str = "float32") -> Any:
        """创建标准正态分布随机数组。"""
        raise NotImplementedError

    def empty(self, shape: Tuple[int, ...], dtype: str = "float32") -> Any:
        """创建未初始化数组。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # NumPy 互转
    # ------------------------------------------------------------------

    def from_numpy(self, np_array: numpy.ndarray) -> Any:
        """将 *numpy.ndarray* 转换为该设备上的原始数据。"""
        raise NotImplementedError

    def to_numpy(self, data: Any) -> numpy.ndarray:
        """将该设备上的原始数据转换为 *numpy.ndarray*。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 比较与哈希 —— 同类型的 Device 视为相等
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Device) and self.__class__ is other.__class__

    def __hash__(self) -> int:
        return hash(self.__class__)

    def __repr__(self) -> str:
        return self.__class__.__name__


# ======================================================================
# CPUDevice
# ======================================================================


class CPUDevice(Device):
    """基于 NumPy 的 CPU 设备。原始数据类型为 ``numpy.ndarray``。"""
    backend : None

    def __init__(self):
        from . import backend_numpy as backend
        self.backend = backend

    def enabled(self) -> bool:
        return True

    def zeros(
        self, shape: Tuple[int, ...], dtype: str = "float32"
    ) -> numpy.ndarray:
        return numpy.zeros(shape, dtype=dtype)

    def ones(
        self, shape: Tuple[int, ...], dtype: str = "float32"
    ) -> numpy.ndarray:
        return numpy.ones(shape, dtype=dtype)

    def randn(
        self, shape: Tuple[int, ...], dtype: str = "float32"
    ) -> numpy.ndarray:
        return numpy.random.randn(*shape).astype(dtype)

    def empty(
        self, shape: Tuple[int, ...], dtype: str = "float32"
    ) -> numpy.ndarray:
        return numpy.empty(shape, dtype=dtype)

    def from_numpy(self, np_array: numpy.ndarray) -> numpy.ndarray:
        return numpy.array(np_array, copy=True, dtype=np_array.dtype)

    def to_numpy(self, data: numpy.ndarray) -> numpy.ndarray:
        return numpy.asarray(data)


# ======================================================================
# CUDADevice（桩 —— 待后续实现）
# ======================================================================


class CUDADevice(Device):
    """CUDA 设备（尚未实现）。"""
    backend : None

    def __init__(self):
        from . import backend_cuda as backend
        self.backend = backend

    def enabled(self) -> bool:
        return False


# ======================================================================
# 工厂函数 —— 单例模式
# ======================================================================

_cpu_device = CPUDevice()
_cuda_device = CUDADevice()


def cpu() -> CPUDevice:
    """返回全局唯一的 CPU 设备实例。"""
    return _cpu_device


def cuda() -> CUDADevice:
    """返回全局唯一的 CUDA 设备实例。"""
    return _cuda_device


def default_device() -> CPUDevice:
    """返回默认设备（CPU）。"""
    return _cpu_device
