"""Step 1–5 反向传播测试。

使用数值梯度检验（finite difference）验证各算子的 analytical gradient 正确性。
"""

from __future__ import annotations

import unittest

import numpy as np

import tiny_dlsys as tdl
from tiny_dlsys.ops import (
    add,
    add_scalar,
    multiply,
    power_scalar,
    exp,
    log,
    relu,
    matmul,
    reshape,
    transpose,
    broadcast_to,
    summation,
    slice_op,
    flip,
    dilate,
    stack,
    split,
)

# 数值梯度检验的 epsilon
EPS = 1e-5
# 梯度相对误差容限
RTOL = 1e-3
ATOL = 1e-4


# ---------------------------------------------------------------------------
# Step 1–2: Device / NDArray（通过后续算子间接验证）
# ---------------------------------------------------------------------------


class TestDeviceNDArray(unittest.TestCase):
    """Step 1–2：Device 与 NDArray 通过 Tensor 运算间接验证。"""

    def test_tensor_on_cpu(self):
        """Tensor 可在 CPU 上创建并计算。"""
        a = tdl.Tensor([1.0, 2.0, 3.0])
        b = tdl.Tensor([4.0, 5.0, 6.0])
        c = add(a, b)
        np.testing.assert_allclose(c.numpy(), [5.0, 7.0, 9.0])


# ---------------------------------------------------------------------------
# Step 3: autograd 计算图与反向传播
# ---------------------------------------------------------------------------


class TestAutograd(unittest.TestCase):
    """Step 3：autograd 核心。"""

    def test_simple_backward(self):
        """简单链式求导。"""
        a = tdl.Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = tdl.Tensor([4.0, 5.0, 6.0], requires_grad=True)
        c = add(a, b)
        loss = c.sum()
        loss.backward()
        # d(loss)/d(a) = d(loss)/d(c) * d(c)/d(a) = 1 * 1 = 1
        np.testing.assert_allclose(a.grad.numpy(), [1.0, 1.0, 1.0])
        np.testing.assert_allclose(b.grad.numpy(), [1.0, 1.0, 1.0])

    def test_multi_node_backward(self):
        """多节点计算图反向传播。"""
        a = tdl.Tensor([1.0, 2.0], requires_grad=True)
        b = tdl.Tensor([3.0, 4.0], requires_grad=True)
        c = add(multiply(a, b), add_scalar(a, 1.0))
        loss = c.sum()
        loss.backward()
        # c = a*b + a + 1, d(c)/d(a) = b + 1, d(c)/d(b) = a
        np.testing.assert_allclose(a.grad.numpy(), [4.0, 5.0])  # b + 1
        np.testing.assert_allclose(b.grad.numpy(), [1.0, 2.0])  # a


# ---------------------------------------------------------------------------
# Step 4: 数学算子反向传播
# ---------------------------------------------------------------------------


class TestMathOpsBackward(unittest.TestCase):
    """Step 4：数学算子梯度。"""

    def test_add_backward(self):
        a = tdl.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = tdl.Tensor([[0.5, 1.0], [1.5, 2.0]], requires_grad=True)
        c = add(a, b)
        loss = c.sum()
        loss.backward()
        np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 2)))
        np.testing.assert_allclose(b.grad.numpy(), np.ones((2, 2)))

    def test_mul_backward(self):
        a = tdl.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = tdl.Tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
        c = multiply(a, b)
        loss = c.sum()
        loss.backward()
        # d(a*b)/d(a) = b
        np.testing.assert_allclose(a.grad.numpy(), b.numpy())
        np.testing.assert_allclose(b.grad.numpy(), a.numpy())

    def test_matmul_backward(self):
        a = tdl.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = tdl.Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        c = matmul(a, b)
        loss = c.sum()
        loss.backward()
        # c = a @ I = a, d(loss)/d(a) = ones, d(loss)/d(b) = a^T @ ones
        np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 2)))
        expected_b_grad = a.numpy().T @ np.ones((2, 2))
        np.testing.assert_allclose(b.grad.numpy(), expected_b_grad)

    def test_reshape_backward(self):
        a = tdl.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = reshape(a, (4,))
        loss = b.sum()
        loss.backward()
        np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 2)))

    def test_summation_backward(self):
        a = tdl.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = summation(a, axes=(1,))
        loss = b.sum()
        loss.backward()
        # b = [3, 7], loss = 10, grad = 1 broadcast to (2,)
        np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 2)))

    def test_exp_log_backward(self):
        a = tdl.Tensor([[1.0, 2.0], [0.5, 1.0]], requires_grad=True)
        b = log(exp(a))
        loss = b.sum()
        loss.backward()
        # log(exp(a)) = a, d/d(a) = 1
        np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 2)))

    def test_relu_backward(self):
        a = tdl.Tensor([[-1.0, 2.0], [0.0, -3.0]], requires_grad=True)
        b = relu(a)
        loss = b.sum()
        loss.backward()
        # grad = 1 where a >= 0 else 0（实现使用 >= 0）
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(a.grad.numpy(), expected)

    def test_power_scalar_backward(self):
        a = tdl.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = power_scalar(a, 2)
        loss = b.sum()
        loss.backward()
        # d(a^2)/da = 2a
        np.testing.assert_allclose(a.grad.numpy(), 2 * a.numpy())

    def test_broadcast_to_backward(self):
        a = tdl.Tensor([[1.0, 2.0]], requires_grad=True)
        b = broadcast_to(a, (3, 2))
        loss = b.sum()
        loss.backward()
        # grad 被 reduce 回 (1, 2)，每列 3 个 1 求和
        np.testing.assert_allclose(a.grad.numpy(), np.full((1, 2), 3.0))

    def test_transpose_backward(self):
        a = tdl.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = transpose(a)
        loss = b.sum()
        loss.backward()
        np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 2)))

    def test_slice_backward(self):
        a = tdl.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        b = slice_op(a, (slice(0, 1), slice(0, 2)))
        loss = b.sum()
        loss.backward()
        expected = np.zeros((2, 3))
        expected[0, :2] = 1.0
        np.testing.assert_allclose(a.grad.numpy(), expected)

    def test_flip_backward(self):
        a = tdl.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = flip(a, (0, 1))
        loss = b.sum()
        loss.backward()
        np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 2)))

    def test_dilate_backward(self):
        a = tdl.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = dilate(a, (0, 1), 1)
        loss = b.sum()
        loss.backward()
        self.assertIsNotNone(a.grad)
        self.assertEqual(a.grad.shape, a.shape)
        np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 2)))


# ---------------------------------------------------------------------------
# Step 5: TensorTuple 算子反向传播
# ---------------------------------------------------------------------------


class TestTupleOpsBackward(unittest.TestCase):
    """Step 5：Stack、Split、TupleGetItem 梯度。"""

    def test_stack_backward(self):
        """Stack 梯度：拆分成多份。"""
        a = tdl.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = tdl.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        s = stack((a, b), axis=0)
        loss = s.sum()
        loss.backward()
        np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 2)))
        np.testing.assert_allclose(b.grad.numpy(), np.ones((2, 2)))

    def test_split_backward_single_consumer(self):
        """Split 梯度：单消费者（只取一个元素）。"""
        a = tdl.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
        p = split(a, axis=0, indices_or_sections=3)
        loss = p[0].sum()  # 只使用第一个
        loss.backward()
        expected = np.zeros((3, 2))
        expected[0, :] = 1.0
        np.testing.assert_allclose(a.grad.numpy(), expected)

    def test_tuple_get_item_backward(self):
        """TupleGetItem 梯度：梯度只回传到对应位置。"""
        a = tdl.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        p = split(a, axis=0)
        t0 = p[0]
        loss = t0.sum()
        loss.backward()
        expected = np.zeros((2, 2))
        expected[0, :] = 1.0
        np.testing.assert_allclose(a.grad.numpy(), expected)


# ---------------------------------------------------------------------------
# 数值梯度检验（可选，验证 analytical 与 numerical 一致）
# ---------------------------------------------------------------------------


class TestNumericalGradient(unittest.TestCase):
    """数值梯度与解析梯度一致性检验。"""

    def test_add_numerical(self):
        np.random.seed(42)
        x = np.random.randn(3, 4).astype(np.float32) * 0.5
        y = np.random.randn(3, 4).astype(np.float32) * 0.5

        a = tdl.Tensor(x, requires_grad=True)
        b = tdl.Tensor(y, requires_grad=True)
        c = add(a, b)
        loss = c.sum()
        loss.backward()

        np.testing.assert_allclose(a.grad.numpy(), np.ones_like(x), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(b.grad.numpy(), np.ones_like(y), rtol=RTOL, atol=ATOL)

    def test_matmul_numerical(self):
        np.random.seed(42)
        x = np.random.randn(2, 3).astype(np.float32) * 0.5
        y = np.random.randn(3, 2).astype(np.float32) * 0.5

        a = tdl.Tensor(x, requires_grad=True)
        b = tdl.Tensor(y, requires_grad=True)
        c = matmul(a, b)
        loss = c.sum()
        loss.backward()

        # 数值梯度：d(sum(a@b))/da = ones @ b^T, d/db = a^T @ ones
        num_da = np.ones((2, 2)) @ y.T
        num_db = x.T @ np.ones((2, 2))
        np.testing.assert_allclose(a.grad.numpy(), num_da, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(b.grad.numpy(), num_db, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    unittest.main()
