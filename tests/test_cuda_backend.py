"""Step 15 — CUDA 后端基础验证。

CPU vs CUDA 对比测试：每类算子在两个设备上各跑一遍，
断言 numpy() 结果在浮点误差范围内相等。
"""

from __future__ import annotations

import functools
import unittest

import numpy as np

import tiny_dlsys as tdl
import tiny_dlsys.nn as nn
import tiny_dlsys.ops as ops
from tiny_dlsys.backend.device import cpu, cuda


def skip_if_no_cuda(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not cuda().enabled():
            self.skipTest("CUDA not available")
        return fn(self, *args, **kwargs)
    return wrapper


class TestCUDAOps(unittest.TestCase):
    """逐元素、规约、矩阵算子 CPU vs CUDA 对比。"""

    @skip_if_no_cuda
    def test_elementwise(self):
        np.random.seed(0)
        x = np.random.randn(4, 5).astype("float32")
        y = np.abs(np.random.randn(4, 5).astype("float32")) + 0.1

        cases = [
            ("add",    lambda a, b: a + b),
            ("mul",    lambda a, b: a * b),
            ("divide", lambda a, b: a / b),
            ("neg",    lambda a, b: -a),
            ("power",  lambda a, b: ops.power_scalar(a, 2)),
            ("relu",   lambda a, b: ops.relu(a)),
        ]
        for name, fn in cases:
            r_cpu = fn(tdl.Tensor(x, device=cpu()), tdl.Tensor(y, device=cpu())).numpy()
            r_gpu = fn(tdl.Tensor(x, device=cuda()), tdl.Tensor(y, device=cuda())).numpy()
            np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-5, err_msg=name)

    @skip_if_no_cuda
    def test_unary(self):
        np.random.seed(1)
        x = np.abs(np.random.randn(4, 5).astype("float32")) + 0.1

        cases = [
            ("exp",  ops.exp),
            ("log",  ops.log),
            ("tanh", ops.tanh),
        ]
        for name, fn in cases:
            r_cpu = fn(tdl.Tensor(x, device=cpu())).numpy()
            r_gpu = fn(tdl.Tensor(x, device=cuda())).numpy()
            np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-5, err_msg=name)

    @skip_if_no_cuda
    def test_reduce(self):
        np.random.seed(2)
        x = np.random.randn(4, 5).astype("float32")

        cases = [
            ("sum_axis0", lambda a: ops.summation(a, axes=(0,))),
            ("sum_axis1", lambda a: ops.summation(a, axes=(1,))),
            ("sum_all",   lambda a: ops.summation(a)),
        ]
        for name, fn in cases:
            r_cpu = fn(tdl.Tensor(x, device=cpu())).numpy()
            r_gpu = fn(tdl.Tensor(x, device=cuda())).numpy()
            np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-5, err_msg=name)

    @skip_if_no_cuda
    def test_matmul(self):
        np.random.seed(3)
        a = np.random.randn(4, 3).astype("float32")
        b = np.random.randn(3, 5).astype("float32")

        r_cpu = ops.matmul(tdl.Tensor(a, device=cpu()), tdl.Tensor(b, device=cpu())).numpy()
        r_gpu = ops.matmul(tdl.Tensor(a, device=cuda()), tdl.Tensor(b, device=cuda())).numpy()
        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-4)


class TestCUDADeviceMigration(unittest.TestCase):
    """设备迁移验证。"""

    @skip_if_no_cuda
    def test_to_cuda_and_back(self):
        x = np.random.randn(3, 4).astype("float32")
        t = tdl.Tensor(x, device=cpu())
        t_gpu = t.to(cuda())
        self.assertEqual(t_gpu.device, cuda())
        np.testing.assert_allclose(t_gpu.numpy(), x, atol=1e-6)

    @skip_if_no_cuda
    def test_from_numpy_cuda(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        t = tdl.Tensor(x, device=cuda())
        np.testing.assert_allclose(t.numpy(), x, atol=1e-6)


class TestCUDAAutograd(unittest.TestCase):
    """自动微分在 CUDA 上的验证。"""

    @skip_if_no_cuda
    def test_backward_device(self):
        x = np.random.randn(3, 4).astype("float32")
        t = tdl.Tensor(x, device=cuda(), requires_grad=True)
        ops.summation(t).backward()
        self.assertEqual(t.grad.device, cuda())
        np.testing.assert_allclose(t.grad.numpy(), np.ones((3, 4)), atol=1e-6)

    @skip_if_no_cuda
    def test_matmul_backward_cuda(self):
        np.random.seed(4)
        a_np = np.random.randn(2, 3).astype("float32")
        b_np = np.random.randn(3, 2).astype("float32")

        a = tdl.Tensor(a_np, device=cuda(), requires_grad=True)
        b = tdl.Tensor(b_np, device=cuda(), requires_grad=True)
        ops.summation(ops.matmul(a, b)).backward()

        self.assertEqual(a.grad.device, cuda())
        self.assertEqual(b.grad.device, cuda())

        a_c = tdl.Tensor(a_np, device=cpu(), requires_grad=True)
        b_c = tdl.Tensor(b_np, device=cpu(), requires_grad=True)
        ops.summation(ops.matmul(a_c, b_c)).backward()

        np.testing.assert_allclose(a.grad.numpy(), a_c.grad.numpy(), atol=1e-4)
        np.testing.assert_allclose(b.grad.numpy(), b_c.grad.numpy(), atol=1e-4)


class TestCUDADilate(unittest.TestCase):
    """dilate 手写实现正确性验证。"""

    @skip_if_no_cuda
    def test_dilate(self):
        x = np.arange(6, dtype="float32").reshape(2, 3)
        r_cpu = ops.dilate(tdl.Tensor(x, device=cpu()), axes=(0, 1), dilation=1).numpy()
        r_gpu = ops.dilate(tdl.Tensor(x, device=cuda()), axes=(0, 1), dilation=1).numpy()
        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-6)


class TestCUDAForwardStep(unittest.TestCase):
    """前向单步验证：CUDA 设备上跑一次 forward 无报错且输出无 NaN。"""

    @skip_if_no_cuda
    def test_forward_single_step(self):
        dev = cuda()
        model = nn.Sequential(
            nn.Linear(784, 128, device=dev),
            nn.ReLU(),
            nn.Linear(128, 10, device=dev),
        )
        x = tdl.Tensor(np.random.randn(8, 784).astype("float32"), device=dev)
        out = model(x)
        self.assertEqual(out.shape, (8, 10))
        self.assertFalse(np.any(np.isnan(out.numpy())))


if __name__ == "__main__":
    unittest.main()
