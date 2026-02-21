"""损失函数：SoftmaxLoss（交叉熵）和 MSELoss（均方误差）。"""

from __future__ import annotations

import numpy as np

from ..autograd import Tensor
from .. import ops
from ..init import initializers as init
from ..backend.ndarray import NDArray
from .module import Module


class SoftmaxLoss(Module):
    """数值稳定的 Softmax 交叉熵损失（log-sum-exp 技巧）。

    参数
    ----
    logits : (batch, num_classes) 未归一化的对数概率
    y      : (batch,) 整数类别标签
    返回   : 标量，batch 平均交叉熵
    """

    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        batch, num_classes = logits.shape

        # 用 numpy 取最大值（作为常量，不参与梯度），保证数值稳定
        max_np = logits.numpy().max(axis=1, keepdims=True)  # (batch, 1)
        max_t = Tensor(
            NDArray.from_numpy(max_np.astype(logits.dtype), device=logits.device),
            device=logits.device,
            requires_grad=False,
        )
        # shifted = logits - max  (广播)
        shifted = logits - max_t.broadcast_to(logits.shape)

        # log_sum_exp(x) = log(sum(exp(x - max))) + max
        log_sum_exp = ops.log(
            ops.summation(ops.exp(shifted), axes=(1,))
        ) + max_t.reshape((batch,))   # (batch,)

        # 取出各样本正确类别的 logit
        y_onehot = init.one_hot(num_classes, y, device=logits.device)  # (batch, num_classes)
        correct_logit = ops.summation(logits * y_onehot, axes=(1,))    # (batch,)

        sum_loss = ops.summation(log_sum_exp - correct_logit)
        batch_t = Tensor(NDArray.from_numpy(np.array(batch, dtype=logits.dtype), device=logits.device), 
        device=logits.device, 
        requires_grad=False)

        loss = sum_loss / batch_t

        return loss


class MSELoss(Module):
    """均方误差损失：mean((pred - target)^2)。"""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return ops.summation(diff * diff) / pred.shape[0]
