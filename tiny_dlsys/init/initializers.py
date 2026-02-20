from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ..autograd import Tensor
from ..backend.ndarray import NDArray
from ..backend.device import Device, default_device


def _normalize_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return tuple(shape)


def _resolve_device(device: Optional[Device], ref: Optional[Tensor] = None) -> Device:
    if device is not None:
        return device
    if ref is not None:
        return ref.device
    return default_device()


def _calculate_gain(nonlinearity: str) -> float:
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity in ("linear", "conv1d", "conv2d", "conv3d", "sigmoid"):
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    return 1.0


def zeros(*shape, device=None, dtype="float32", requires_grad=False) -> Tensor:
    shape = _normalize_shape(shape)
    arr = NDArray.zeros(shape, device=_resolve_device(device), dtype=dtype)
    return Tensor(arr, device=arr.device, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False) -> Tensor:
    shape = _normalize_shape(shape)
    arr = NDArray.ones(shape, device=_resolve_device(device), dtype=dtype)
    return Tensor(arr, device=arr.device, requires_grad=requires_grad)


def randn(
    *shape,
    mean=0.0,
    std=1.0,
    device=None,
    dtype="float32",
    requires_grad=False,
) -> Tensor:
    shape = _normalize_shape(shape)
    dev = _resolve_device(device)
    np_arr = np.random.normal(loc=mean, scale=std, size=shape).astype(dtype)
    arr = NDArray.from_numpy(np_arr, device=dev)
    return Tensor(arr, device=arr.device, requires_grad=requires_grad)


def rand(
    *shape,
    low=0.0,
    high=1.0,
    device=None,
    dtype="float32",
    requires_grad=False,
) -> Tensor:
    shape = _normalize_shape(shape)
    dev = _resolve_device(device)
    np_arr = np.random.uniform(low=low, high=high, size=shape).astype(dtype)
    arr = NDArray.from_numpy(np_arr, device=dev)
    return Tensor(arr, device=arr.device, requires_grad=requires_grad)


def one_hot(n: int, i: Tensor, device=None, dtype="float32") -> Tensor:
    dev = _resolve_device(device, ref=i)
    indices = i.numpy().astype("int64").reshape(-1)
    np_arr = np.eye(n, dtype=dtype)[indices]
    arr = NDArray.from_numpy(np_arr, device=dev)
    return Tensor(arr, device=arr.device, requires_grad=False)


def xavier_uniform(fan_in: int, fan_out: int, gain=1.0, **kwargs) -> Tensor:
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def xavier_normal(fan_in: int, fan_out: int, gain=1.0, **kwargs) -> Tensor:
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)


def kaiming_uniform(
    fan_in: int, fan_out: int, nonlinearity="relu", **kwargs
) -> Tensor:
    gain = _calculate_gain(nonlinearity)
    bound = gain * math.sqrt(3.0 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)


def kaiming_normal(
    fan_in: int, fan_out: int, nonlinearity="relu", **kwargs
) -> Tensor:
    gain = _calculate_gain(nonlinearity)
    std = gain / math.sqrt(fan_in)
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
