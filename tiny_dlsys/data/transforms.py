from __future__ import annotations

import numpy as np


class RandomFlipHorizontal:
    """以概率 p 随机水平翻转图像。

    输入 img 形状为 (H, W) 或 (H, W, C)。
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            # 沿宽度轴（axis=-1 对应 W）翻转
            return np.flip(img, axis=-1).copy()
        return img


class RandomCrop:
    """对图像四周填充 padding 个像素（零填充），再随机裁剪回原始尺寸。

    输入 img 形状为 (H, W) 或 (H, W, C)。
    """

    def __init__(self, padding: int = 4):
        self.padding = padding

    def __call__(self, img: np.ndarray) -> np.ndarray:
        pad = self.padding
        if pad == 0:
            return img

        if img.ndim == 2:
            h, w = img.shape
            padded = np.pad(img, ((pad, pad), (pad, pad)), mode="constant")
        else:
            h, w = img.shape[:2]
            padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="constant")

        top = np.random.randint(0, 2 * pad + 1)
        left = np.random.randint(0, 2 * pad + 1)
        return padded[top : top + h, left : left + w]
