from __future__ import annotations

from typing import Any, List

import numpy as np

from ..autograd import Tensor
from ..backend.ndarray import NDArray


class Parameter(Tensor):
    """需要梯度的可训练参数。"""

    def __init__(
        self,
        data: Any,
        *,
        device=None,
        dtype=None,
        requires_grad: bool = True,
    ):
        super().__init__(
            data, device=device, dtype=dtype, requires_grad=requires_grad
        )


def _unpack_params(value: Any) -> List[Parameter]:
    if isinstance(value, Parameter):
        return [value]
    if isinstance(value, Module):
        return value.parameters()
    if isinstance(value, dict):
        params: List[Parameter] = []
        for v in value.values():
            params.extend(_unpack_params(v))
        return params
    if isinstance(value, (list, tuple)):
        params: List[Parameter] = []
        for v in value:
            params.extend(_unpack_params(v))
        return params
    return []


def _child_modules(value: Any) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules: List[Module] = []
        for v in value.values():
            modules.extend(_child_modules(v))
        return modules
    if isinstance(value, (list, tuple)):
        modules: List[Module] = []
        for v in value:
            modules.extend(_child_modules(v))
        return modules
    return []


class Module:
    """神经网络模块基类。"""

    def __init__(self):
        self.training = True

    def parameters(self) -> List[Parameter]:
        params = _unpack_params(self.__dict__)
        seen = set()
        result: List[Parameter] = []
        for p in params:
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                result.append(p)
        return result

    def _children(self) -> List["Module"]:
        modules = _child_modules(self.__dict__)
        seen = set()
        result: List[Module] = []
        for m in modules:
            mid = id(m)
            if mid not in seen:
                seen.add(mid)
                result.append(m)
        return result

    def train(self) -> "Module":
        self.training = True
        for module in self._children():
            module.train()
        return self

    def eval(self) -> "Module":
        self.training = False
        for module in self._children():
            module.eval()
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Embedding(Module):
    """词嵌入查找表。

    参数
    ----
    num_embeddings : 词汇表大小
    embedding_dim  : 嵌入维度
    weight         : (num_embeddings, embedding_dim)，标准正态初始化
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype: str = "float32",
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        from ..init import initializers as init
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype),
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: 整数索引，形状 (seq_len,) 或 (batch, seq_len)
        返回: (..., embedding_dim)
        """
        from ..init import initializers as init
        indices = x.numpy().astype("int64").reshape(-1)
        # one_hot: (len, num_embeddings) @ weight: (num_embeddings, embedding_dim)
        oh = init.one_hot(self.num_embeddings, x.reshape((indices.shape[0],)),
                          device=self.weight.device)
        out = oh @ self.weight  # (flat_len, embedding_dim)
        # 还原原始批次维度
        original_shape = x.numpy().shape
        return out.reshape(original_shape + (self.embedding_dim,))
