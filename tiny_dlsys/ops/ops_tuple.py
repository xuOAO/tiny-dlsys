"""TensorTuple 算子——Stack、Split、TupleGetItem。

供 Conv2d im2col 及多输出场景使用。
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from ..autograd import Op, Tensor, TensorOp, TensorTuple, TensorTupleOp
from ..backend.ndarray import NDArray


# ---------------------------------------------------------------------------
# Stack —— 沿新轴拼接多个 Tensor
# ---------------------------------------------------------------------------


class Stack(TensorOp):
    """沿新轴 axis 拼接多个 Tensor。"""

    def __init__(self, axis: int = 0):
        self.axis = axis

    def compute(self, *args: NDArray) -> NDArray:
        arrays = [a.numpy() for a in args]
        stacked = np.stack(arrays, axis=self.axis)
        return NDArray.from_numpy(stacked, device=args[0].device)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, ...]:
        # Stack 的梯度：沿 axis 拆分后，去掉新增维度，恢复到各输入形状
        split_grads = split(out_grad, self.axis, len(node.inputs)).tuple()
        return tuple(g.reshape(inp.shape) for g, inp in zip(split_grads, node.inputs))


# ---------------------------------------------------------------------------
# Split —— 沿轴拆分，返回 TensorTuple
# ---------------------------------------------------------------------------


class Split(TensorTupleOp):
    """沿 axis 拆分，返回 TensorTuple。"""

    def __init__(self, axis: int, indices_or_sections: Union[int, Tuple[int, ...]]):
        self.axis = axis
        self.indices_or_sections = indices_or_sections

    def compute(self, a: NDArray) -> Tuple[NDArray, ...]:
        parts = np.split(a.numpy(), self.indices_or_sections, axis=self.axis)
        return tuple(NDArray.from_numpy(p, device=a.device) for p in parts)

    def gradient(self, out_grad: "TensorTuple", node: "TensorTuple") -> Tensor:
        # Split 的梯度是 concatenate（不是 stack），需沿原 axis 拼回输入形状
        in_tensor = node.inputs[0]
        parts = [t.realize_cached_data().numpy() for t in out_grad.tuple()]
        grad_np = np.concatenate(parts, axis=self.axis)
        return Tensor.from_numpy(
            grad_np,
            device=in_tensor.device,
            requires_grad=False,
        )


# ---------------------------------------------------------------------------
# TupleGetItem —— 从 TensorTuple 中取出第 index 个 Tensor
# ---------------------------------------------------------------------------


class TupleGetItem(TensorOp):
    """从 TensorTuple 中取出指定索引的 Tensor。"""

    def __init__(self, index: int):
        self.index = index

    def compute(self, data) -> NDArray:
        if isinstance(data, (tuple, list)):
            return data[self.index]
        raise TypeError(f"TupleGetItem expects tuple/list, got {type(data)}")

    def gradient(self, out_grad: Tensor, node: Tensor) -> "TensorTuple":
        # 梯度：构造新的 TensorTuple，index 处为 out_grad，其余为 zeros
        a = node.inputs[0]
        tuple_data = a.realize_cached_data()
        grad_list = []
        for i in range(len(tuple_data)):
            if i == self.index:
                grad_list.append(out_grad)
            else:
                nd = tuple_data[i]
                zeros = Tensor.from_numpy(
                    np.zeros(nd.shape, dtype=nd.dtype),
                    device=out_grad.device,
                    requires_grad=False,
                )
                grad_list.append(zeros)
        return TensorTuple.make_const(tuple(grad_list), requires_grad=False)


# ---------------------------------------------------------------------------
# 函数式接口
# ---------------------------------------------------------------------------


def stack(tensors: Tuple[Tensor, ...], axis: int = 0) -> Tensor:
    """沿新轴 axis 拼接多个 Tensor。"""
    return Stack(axis)(*tensors)


def split(a: Tensor, axis: int, indices_or_sections: Union[int, Tuple[int, ...]] = 2) -> TensorTuple:
    """沿 axis 拆分，返回 TensorTuple。indices_or_sections 默认 2 表示等分两份。"""
    return Split(axis, indices_or_sections)(a)


def tuple_get_item(tensor_tuple: TensorTuple, index: int) -> Tensor:
    """从 TensorTuple 中取出第 index 个 Tensor。"""
    return TupleGetItem(index)(tensor_tuple)
