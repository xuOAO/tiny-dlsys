from __future__ import annotations

import gzip
import struct
from typing import List, Optional, Tuple

import numpy as np


class Dataset:
    """数据集基类。"""

    transforms: Optional[List]

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def apply_transforms(self, x: np.ndarray) -> np.ndarray:
        """依次应用 self.transforms 中的变换。"""
        if getattr(self, "transforms", None):
            for t in self.transforms:
                x = t(x)
        return x


class MNISTDataset(Dataset):
    """读取 gzip 压缩的 MNIST IDX 格式文件。

    图像以 (H, W) = (28, 28) 的 float32 数组返回，像素值归一化至 [0, 1]。
    标签以 uint8 整数返回。
    """

    def __init__(
        self,
        images_path: str,
        labels_path: str,
        transforms: Optional[List] = None,
    ):
        self.transforms = transforms
        self.images = _load_mnist_images(images_path)
        self.labels = _load_mnist_labels(labels_path)

    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        img = self.images[index].copy()   # (28, 28) float32
        label = int(self.labels[index])
        img = self.apply_transforms(img)
        return img, label

    def __len__(self) -> int:
        return len(self.labels)


# ---------------------------------------------------------------------------
# 内部辅助：读取 IDX 文件
# ---------------------------------------------------------------------------

def _open(path: str):
    """自动判断是否 gzip 压缩并返回文件对象。"""
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _load_mnist_images(path: str) -> np.ndarray:
    """返回形状 (N, 28, 28) 的 float32 数组，像素归一化至 [0, 1]。"""
    with _open(path) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"Invalid MNIST image file magic: {magic:#010x}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols).astype(np.float32) / 255.0


def _load_mnist_labels(path: str) -> np.ndarray:
    """返回形状 (N,) 的 uint8 数组。"""
    with _open(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"Invalid MNIST label file magic: {magic:#010x}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data
