"""Step 11: 端到端 MNIST 训练集成测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import tiny_dlsys as tdl
import tiny_dlsys.nn as nn
import tiny_dlsys.optim as optim
from tiny_dlsys.data import DataLoader, MNISTDataset


def _resolve_mnist_paths() -> tuple[Path, Path, Path, Path]:
    """兼容两种常见 MNIST 文件命名。"""
    root = Path(__file__).resolve().parents[1] / "data"

    candidates = [
        (
            root / "mnist_data" / "train-images-idx3-ubyte.gz",
            root / "mnist_data" / "train-labels-idx1-ubyte.gz",
            root / "mnist_data" / "t10k-images-idx3-ubyte.gz",
            root / "mnist_data" / "t10k-labels-idx1-ubyte.gz",
        ),
        (
            root / "train-images.gz",
            root / "train-labels.gz",
            root / "test-images.gz",
            root / "test-labels.gz",
        ),
    ]

    for paths in candidates:
        if all(p.exists() for p in paths):
            return paths

    raise FileNotFoundError(
        "未找到 MNIST 数据文件。请检查 data/ 目录下是否存在 mnist_data/*.gz "
        "或 train-images.gz/train-labels.gz/test-images.gz/test-labels.gz。"
    )


class TestMNISTTraining(unittest.TestCase):
    """Step 11：验证完整 MNIST 训练链路可运行。"""

    def test_step11_mnist_end_to_end_training(self):
        np.random.seed(0)

        train_images, train_labels, _, _ = _resolve_mnist_paths()

        try:
            train_set = MNISTDataset(str(train_images), str(train_labels))
        except (EOFError, OSError, ValueError) as err:
            self.fail(
                "MNIST 训练数据读取失败，通常是 gzip 文件损坏或 IDX 头不合法。"
                f"失败文件: {train_images} / {train_labels}; 错误: {err}"
            )

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        device = tdl.cuda() if tdl.cuda().enabled() else tdl.cpu()

        model = nn.Sequential(
            nn.Linear(784, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 10, device=device),
        )
        loss_fn = nn.SoftmaxLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        params_before = [p.numpy().copy() for p in model.parameters()]
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        model.train()
        for x_np, y_np in train_loader:
            x_np = x_np.reshape((x_np.shape[0], -1)).astype(np.float32)
            y_np = y_np.astype(np.int32)

            x = tdl.Tensor(x_np, device=device, requires_grad=False)
            y = tdl.Tensor(y_np, device=device, requires_grad=False)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.numpy())
            pred = logits.numpy().argmax(axis=1)
            total_correct += int((pred == y_np).sum())
            total_seen += int(y_np.shape[0])

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / max(total_seen, 1)

        params_after = [p.numpy() for p in model.parameters()]
        has_param_update = any(
            not np.allclose(before, after) for before, after in zip(params_before, params_after)
        )

        self.assertTrue(np.isfinite(avg_loss), "训练后的平均 loss 必须是有限值。")
        self.assertGreater(total_seen, 0, "训练样本数必须大于 0。")
        self.assertTrue(0.0 <= avg_acc <= 1.0, "训练准确率必须在 [0, 1] 范围内。")
        self.assertTrue(has_param_update, "参数在一次完整训练后应发生更新。")


if __name__ == "__main__":
    unittest.main()
