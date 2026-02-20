from __future__ import annotations
from typing import List, Optional, Tuple, Union

import numpy as np

from .backend.ndarray import NDArray
from .backend.device import Device, cpu, default_device

LAZY_MODE = True


# ---------------------------------------------------------------------------
# Op 基类
# ---------------------------------------------------------------------------

class Op:
    def compute(self, *args: NDArray) -> NDArray:
        raise NotImplementedError

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value", ...]]:
        raise NotImplementedError


class TensorOp(Op):
    def __call__(self, *args) -> "Tensor":
        return Tensor.make_from_op(self, list(args))


class TensorTupleOp(Op):
    def __call__(self, *args) -> "TensorTuple":
        return TensorTuple.make_from_op(self, list(args))


# ---------------------------------------------------------------------------
# Value（计算图节点基类）
# ---------------------------------------------------------------------------

class Value:
    op: Optional[Op]
    inputs: List["Value"]
    cached_data: Optional[NDArray]
    requires_grad: bool

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Value"],
        *,
        cached_data: Optional[NDArray] = None,
        requires_grad: Optional[bool] = None,
    ) -> None:
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data: NDArray, *, requires_grad: bool = False) -> "Value":
        val = cls.__new__(cls)
        val._init(None, [], cached_data=data, requires_grad=requires_grad)
        return val

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]) -> "Value":
        val = cls.__new__(cls)
        val._init(op, inputs)
        if not LAZY_MODE:
            if not val.requires_grad:
                return val.detach()
            val.realize_cached_data()
        return val

    def realize_cached_data(self) -> NDArray:
        if self.cached_data is None:
            self.cached_data = self.op.compute(
                *[x.realize_cached_data() for x in self.inputs]
            )
        return self.cached_data

    def is_leaf(self) -> bool:
        return self.op is None

    def detach(self) -> "Value":
        return self.make_const(self.realize_cached_data(), requires_grad=False)

    def __del__(self):
        pass


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------

class Tensor(Value):
    grad: Optional["Tensor"]

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype: Optional[str] = None,
        requires_grad: bool = True,
    ):
        if isinstance(array, Tensor):
            device = device or array.device
            dtype = dtype or array.dtype
            if device == array.device and (dtype is None or dtype == array.dtype):
                cached_data = array.realize_cached_data()
            else:
                cached_data = NDArray(array.numpy(), device=device)
        elif isinstance(array, NDArray):
            device = device or array.device
            cached_data = array
        else:
            device = device or default_device()
            np_array = np.array(array, dtype=dtype or "float32")
            cached_data = NDArray(np_array, device=device)

        self._init(None, [], cached_data=cached_data, requires_grad=requires_grad)
        self.grad = None

    # ---------- 属性 ----------

    @property
    def shape(self) -> tuple:
        return self.realize_cached_data().shape

    @property
    def dtype(self) -> str:
        return self.realize_cached_data().dtype

    @property
    def device(self) -> Device:
        return self.realize_cached_data().device

    @property
    def data(self) -> "Tensor":
        return self.detach()

    @data.setter
    def data(self, value: "Tensor") -> None:
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype
        self.cached_data = value.realize_cached_data()

    # ---------- 设备迁移 ----------

    def to(self, device: Device) -> "Tensor":
        if self.device == device:
            return self
        return Tensor(self.numpy(), device=device, dtype=self.dtype,
                      requires_grad=self.requires_grad)

    def numpy(self) -> np.ndarray:
        return self.realize_cached_data().numpy()

    # ---------- 反向传播 ----------

    def backward(self, out_grad: Optional["Tensor"] = None) -> None:
        if out_grad is None:
            out_grad = Tensor(
                NDArray.ones(self.shape, device=self.device),
                device=self.device,
                requires_grad=False,
            )
        compute_gradient_of_variables(self, out_grad)

    # ---------- 运算符重载 ----------

    def __add__(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return ops_fn.add(self, other)
        return ops_fn.add_scalar(self, other)

    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)

    def __mul__(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return ops_fn.multiply(self, other)
        return ops_fn.mul_scalar(self, other)

    def __rmul__(self, other) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return ops_fn.divide(self, other)
        return ops_fn.divide_scalar(self, other)

    def __rtruediv__(self, other) -> "Tensor":
        return ops_fn.divide_scalar(ops_fn.negate(self), -other)

    def __neg__(self) -> "Tensor":
        return ops_fn.negate(self)

    def __sub__(self, other) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other) -> "Tensor":
        return (-self) + other

    def __pow__(self, scalar) -> "Tensor":
        return ops_fn.power_scalar(self, scalar)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return ops_fn.matmul(self, other)

    def reshape(self, shape) -> "Tensor":
        return ops_fn.reshape(self, shape)

    def transpose(self, axes=None) -> "Tensor":
        return ops_fn.transpose(self, axes)

    def sum(self, axes=None) -> "Tensor":
        return ops_fn.summation(self, axes)

    def broadcast_to(self, shape) -> "Tensor":
        return ops_fn.broadcast_to(self, shape)

    # ---------- 静态工厂 ----------

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]) -> "Tensor":
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        tensor.grad = None
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data: NDArray, *, requires_grad: bool = False) -> "Tensor":
        tensor = Tensor.__new__(Tensor)
        tensor._init(None, [], cached_data=data, requires_grad=requires_grad)
        tensor.grad = None
        return tensor

    @staticmethod
    def from_numpy(np_array: np.ndarray, device=None, requires_grad: bool = False) -> "Tensor":
        return Tensor(np_array, device=device, requires_grad=requires_grad)

    def __repr__(self) -> str:
        return f"tiny_dlsys.Tensor({self.realize_cached_data().numpy()})"


# ---------------------------------------------------------------------------
# TensorTuple
# ---------------------------------------------------------------------------

class TensorTuple(Value):
    def __len__(self) -> int:
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int) -> Tensor:
        return ops_fn.tuple_get_item(self, index)

    def tuple(self) -> tuple:
        return tuple(self[i] for i in range(len(self)))

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]) -> "TensorTuple":
        tt = TensorTuple.__new__(TensorTuple)
        tt._init(op, inputs)
        if not LAZY_MODE:
            tt.realize_cached_data()
        return tt

    @staticmethod
    def make_const(data, *, requires_grad: bool = False) -> "TensorTuple":
        tt = TensorTuple.__new__(TensorTuple)
        tt._init(None, [], cached_data=data, requires_grad=requires_grad)
        return tt

    def __repr__(self) -> str:
        return f"tiny_dlsys.TensorTuple({self.realize_cached_data()})"


# ---------------------------------------------------------------------------
# 反向传播算法
# ---------------------------------------------------------------------------

def find_topo_sort(node_list: List[Value]) -> List[Value]:
    visited = set()
    topo = []

    def dfs(node: Value):
        if id(node) in visited:
            return
        visited.add(id(node))
        for inp in node.inputs:
            dfs(inp)
        topo.append(node)

    for node in node_list:
        dfs(node)
    return topo


def compute_gradient_of_variables(output_tensor: Tensor, out_grad: Tensor) -> None:
    node_to_grad: dict = {id(output_tensor): [out_grad]}

    topo = find_topo_sort([output_tensor])

    for node in reversed(topo):
        grad = sum_node_list(node_to_grad[id(node)])
        node.grad = grad

        if node.is_leaf():
            continue

        grads = node.op.gradient(grad, node)
        if not isinstance(grads, (list, tuple)):
            grads = [grads]

        for inp, g in zip(node.inputs, grads):
            if inp.requires_grad:
                node_to_grad.setdefault(id(inp), []).append(g)


def sum_node_list(nodes: List[Tensor]) -> Tensor:
    result = nodes[0]
    for node in nodes[1:]:
        result = result + node
    return result


# ---------------------------------------------------------------------------
# 延迟导入 ops，避免循环依赖
# ---------------------------------------------------------------------------

class _OpsFn:
    """代理对象，第一次使用时才 import ops，打破 autograd ↔ ops 循环依赖。"""
    _mod = None

    def _get(self):
        if self._mod is None:
            from . import ops as _ops
            self._mod = _ops
        return self._mod

    # 访问不存在的属性时，会调__getattr__
    def __getattr__(self, name):
        return getattr(self._get(), name)


ops_fn = _OpsFn()
