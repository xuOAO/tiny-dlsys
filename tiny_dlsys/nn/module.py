from __future__ import annotations

from typing import Any, List

from ..autograd import Tensor


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
