"""极简 MNIST 训练示例。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import tiny_dlsys as tdl
import tiny_dlsys.nn as nn
import tiny_dlsys.optim as optim
from tiny_dlsys.data import DataLoader, MNISTDataset


def resolve_train_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1] / "data"
    candidates = [
        (
            root / "mnist_data" / "train-images-idx3-ubyte.gz",
            root / "mnist_data" / "train-labels-idx1-ubyte.gz",
            root / "mnist_data" / "t10k-images-idx3-ubyte.gz",
            root / "mnist_data" / "t10k-labels-idx1-ubyte.gz",
        ),
        (root / "train-images.gz", root / "train-labels.gz",
        root / "test-images.gz", root / "test-labels.gz"),
    ]
    for p in candidates:
        if all(x.exists() for x in p):
            return p
    raise FileNotFoundError("找不到 MNIST 训练数据文件。")


def main():
    np.random.seed(0)
    device = tdl.cuda() if tdl.cuda().enabled() else tdl.cpu()
    train_images, train_labels, eval_images, eval_labels = resolve_train_paths()

    train_dataset = MNISTDataset(str(train_images), str(train_labels))
    eval_dataset = MNISTDataset(str(eval_images), str(eval_labels))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=True)

    model = nn.Sequential(
        nn.Linear(784, 128, device=device),
        nn.ReLU(),
        nn.Linear(128, 10, device=device),
    )
    loss_fn = nn.SoftmaxLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total_loss = 0.0

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
        
        total_correct = 0
        total_seen = 0
        model.eval()

        for x_np, y_np in eval_loader:
            x_np = x_np.reshape((x_np.shape[0], -1)).astype(np.float32)
            y_np = y_np.astype(np.int32)
            x = tdl.Tensor(x_np, device=device, requires_grad=False)

            logits = model(x)
            pred = logits.numpy().argmax(axis=1)
            total_correct += int((pred == y_np).sum())
            total_seen += int(y_np.shape[0])

        print(
            f"epoch {epoch + 1}: "
            f"loss={total_loss / len(train_loader):.4f}, "
            f"acc={total_correct / total_seen:.4f}"
        )


if __name__ == "__main__":
    main()
