from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

from .dataset import Dataset


class DataLoader:
    """按 batch_size 批量迭代数据集。

    Parameters
    ----------
    dataset   : Dataset 实例
    batch_size: 每个 batch 的样本数，默认 1
    shuffle   : 每个 epoch 开始前是否随机打乱样本顺序
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        """返回 batch 总数（向上取整）。"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[np.ndarray, ...]]:
        n = len(self.dataset)
        indices = (
            np.random.permutation(n) if self.shuffle else np.arange(n)
        )

        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            samples = [self.dataset[int(i)] for i in batch_idx]

            # 将样本列表转置为 (field_0_batch, field_1_batch, ...) 的元组
            # 每个 field 堆叠成 numpy 数组
            num_fields = len(samples[0]) if isinstance(samples[0], (tuple, list)) else 1
            if num_fields == 1:
                yield (np.array(samples),)
            else:
                yield tuple(
                    np.array([s[i] for s in samples]) for i in range(num_fields)
                )
