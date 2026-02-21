# tiny-dlsys API 设计

> 一个面向教学的微型深度学习系统，支持单机单卡（CPU / CUDA）。

---

## 目录

- [1. 整体架构](#1-整体架构)
- [2. 项目结构](#2-项目结构)
- [3. backend — 设备与存储](#3-backend--设备与存储)
- [4. autograd — 计算图与自动微分](#4-autograd--计算图与自动微分)
- [5. ops — 算子库](#5-ops--算子库)
- [6. nn — 神经网络模块](#6-nn--神经网络模块)
- [7. optim — 优化器](#7-optim--优化器)
- [8. data — 数据加载](#8-data--数据加载)
- [9. init — 参数初始化](#9-init--参数初始化)

---

## 1. 整体架构

系统分为三层，自顶向下依次是：

```
┌─────────────────────────────────────────────────────┐
│  用户代码层   Tensor + 运算符重载 (__add__ 等)        │
│              nn.Module / Optimizer / DataLoader      │
└──────────────────────┬──────────────────────────────┘
                       │  Tensor 运算委托给 TensorOp
                       ▼
┌─────────────────────────────────────────────────────┐
│  自动微分层   Op (TensorOp / TensorTupleOp)          │
│              compute(): NDArray → NDArray (前向)     │
│              gradient(): Tensor → Tensor  (反向)     │
└──────────────────────┬──────────────────────────────┘
                       │  compute() 内部调用 NDArray 方法
                       ▼
┌─────────────────────────────────────────────────────┐
│  后端计算层   NDArray                                 │
│              根据 self.device 分派到具体后端           │
│              ├── backend_numpy  (CPU)                │
│              └── backend_cuda   (CUDA)               │
└─────────────────────────────────────────────────────┘
```

### 调用链路示例（以 `c = a + b` 为例）

```
Tensor.__add__(a, b)
  → ops.EWiseAdd()(a, b)                              # 运算符重载 → TensorOp
    → Tensor.make_from_op(EWiseAdd, [a, b])           # 构建计算图节点
      → Value.realize_cached_data()                    # 惰性求值时触发
        → EWiseAdd.compute(a_ndarray, b_ndarray)       # Op 层：操作 NDArray
          → NDArray.__add__(a_nd, b_nd)                # NDArray 层
            → device.add(a_raw, b_raw)                 # Device 后端：numpy 或 CUDA
```

### 关键设计原则

| 层 | 操作的数据类型 | 职责 |
|----|--------------|------|
| **Op.compute** | NDArray → NDArray | 用 NDArray 的原子操作组合出前向逻辑，**设备无关** |
| **Op.gradient** | Tensor → Tensor | 构建梯度计算图节点（符号式），支持高阶微分 |
| **NDArray** | 原始数据 | 持有 `device` 引用，每个运算方法内部分派到对应后端 |
| **Device 后端** | 原始数组（numpy.ndarray / CUDA buffer） | 执行实际数值计算 |

Op **不直接接触 Device**，只通过 NDArray 的方法间接调用后端。这意味着：
- 同一个 `EWiseAdd.compute` 既能跑 CPU 也能跑 CUDA，设备差异由 NDArray 屏蔽。
- 新增后端（如 Metal、OpenCL）只需实现 NDArray 对应的 Device 后端，无需修改任何 Op。

---

## 2. 项目结构

```
tiny-dlsys/                        # 项目根目录
├── tiny_dlsys/                    # Python 包
│   ├── __init__.py
│   ├── autograd.py                # 计算图、Value、Tensor
│   ├── ops/
│   │   ├── __init__.py
│   │   ├── ops_math.py            # 数学算子（前向 + 反向）
│   │   └── ops_tuple.py           # TensorTuple 算子
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── device.py              # Device 抽象
│   │   ├── ndarray.py             # NDArray 统一接口
│   │   ├── backend_numpy.py       # CPU 后端（NumPy）
│   │   └── backend_cuda.py        # CUDA 后端（调用 src/ 编译产物）
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── module.py              # Module、Parameter
│   │   ├── linear.py              # Linear
│   │   ├── conv.py                # Conv2d
│   │   ├── norm.py                # BatchNorm、LayerNorm
│   │   ├── activation.py          # ReLU、Tanh、Sigmoid
│   │   ├── loss.py                # SoftmaxLoss、MSELoss
│   │   ├── container.py           # Sequential
│   │   └── dropout.py             # Dropout
│   ├── optim/
│   │   ├── __init__.py
│   │   ├── optimizer.py           # Optimizer 基类
│   │   ├── sgd.py                 # SGD
│   │   └── adam.py                # Adam
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # Dataset
│   │   └── dataloader.py          # DataLoader
│   └── init/
│       ├── __init__.py
│       └── initializers.py        # 参数初始化函数
├── src/                           # C++/CUDA 源码（pybind11 后端）
│   ├── ops.cu                     # CUDA kernel 实现
│   ├── ops.cuh                    # kernel 声明
│   └── bind.cpp                   # pybind11 绑定入口
├── doc/                           # 文档
├── tests/                         # 测试
└── setup.py                       # 包构建配置
```

---

## 3. backend — 设备与存储

### 3.1 Device

设备抽象。所有的 NDArray 都绑定到某个 Device 上。

```python
class Device:
    """设备基类"""
    def enabled(self) -> bool: ...
    def zeros(self, shape: tuple, dtype="float32") -> NDArray: ...
    def ones(self, shape: tuple, dtype="float32") -> NDArray: ...
    def randn(self, shape: tuple, dtype="float32") -> NDArray: ...
    def empty(self, shape: tuple, dtype="float32") -> NDArray: ...
    def from_numpy(self, np_array) -> NDArray: ...
    def to_numpy(self, nd_array) -> numpy.ndarray: ...

class CPUDevice(Device): ...
class CUDADevice(Device): ...

def cpu() -> CPUDevice: ...
def cuda() -> CUDADevice: ...
def default_device() -> Device: ...
```

### 3.2 NDArray

与设备绑定的多维数组，是 Tensor 底层的数据载体。后端使用 NumPy（CPU）或自定义 CUDA kernel（GPU）。

```python
class NDArray:
    """与设备绑定的多维数组"""

    # --- 属性 ---
    @property
    def shape(self) -> tuple: ...
    @property
    def dtype(self) -> str: ...
    @property
    def device(self) -> Device: ...

    # --- 设备迁移 ---
    def to(self, device: Device) -> "NDArray": ...
    def numpy(self) -> numpy.ndarray: ...

    # --- 底层运算（由 Device 后端实现）---
    # 逐元素：add, mul, div, neg, exp, log, pow, maximum
    # 规约  ：sum, max（沿指定 axis）
    # 矩阵  ：matmul
    # 形状  ：reshape, transpose, broadcast_to, slice, flip, pad
    # 比较  ：eq, ge
```

---

## 4. autograd — 计算图与自动微分

### 4.1 Op

所有算子的基类，定义前向计算（`compute`）和反向求梯度（`gradient`）。

```python
class Op:
    def compute(self, *args: Tuple[NDArray, ...]) -> NDArray:
        """前向：在 NDArray 层面执行计算"""
        raise NotImplementedError

    def gradient(self, out_grad: "Value", node: "Value") -> Union["Value", Tuple["Value", ...]]:
        """反向：给定输出梯度，返回各输入的梯度（Tensor 层面）"""
        raise NotImplementedError

class TensorOp(Op):
    """返回 Tensor 的算子"""
    def __call__(self, *args) -> "Tensor": ...

class TensorTupleOp(Op):
    """返回 TensorTuple 的算子"""
    def __call__(self, *args) -> "TensorTuple": ...
```

### 4.2 Value

计算图节点基类，是 Tensor 和 TensorTuple 的公共父类。

```python
class Value:
    op: Optional[Op]             # 产生该节点的算子，叶节点为 None
    inputs: List["Value"]        # 输入节点列表
    cached_data: NDArray         # 前向计算缓存
    requires_grad: bool

    def realize_cached_data(self) -> NDArray:
        """惰性求值：递归计算并缓存前向结果"""

    def is_leaf(self) -> bool: ...
    def detach(self) -> "Value": ...

    @classmethod
    def make_const(cls, data, *, requires_grad=False) -> "Value": ...
    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]) -> "Value": ...
```

### 4.3 Tensor

核心数据结构，标量/向量/矩阵/高维数组的统一抽象，支持自动微分。

```python
class Tensor(Value):
    grad: "Tensor"

    def __init__(self, array, *, device=None, dtype=None, requires_grad=True): ...

    # --- 属性 ---
    @property
    def shape(self) -> tuple: ...
    @property
    def dtype(self): ...
    @property
    def device(self) -> Device: ...

    @property
    def data(self) -> "Tensor":
        """返回 detach 后的 Tensor（共享数据，不参与计算图）"""

    @data.setter
    def data(self, value: "Tensor") -> None: ...

    # --- 设备迁移 ---
    def to(self, device: Device) -> "Tensor": ...

    # --- 反向传播 ---
    def backward(self, out_grad: "Tensor" = None) -> None:
        """从当前节点出发，反向传播计算梯度（拓扑排序 + 反向遍历）"""

    # --- 运算符重载（委托给 ops 中的 TensorOp）---
    def __add__(self, other) -> "Tensor": ...
    def __mul__(self, other) -> "Tensor": ...
    def __truediv__(self, other) -> "Tensor": ...
    def __neg__(self) -> "Tensor": ...
    def __pow__(self, scalar) -> "Tensor": ...
    def __matmul__(self, other) -> "Tensor": ...
    def __sub__(self, other) -> "Tensor": ...

    def reshape(self, shape) -> "Tensor": ...
    def transpose(self, axes=None) -> "Tensor": ...
    def sum(self, axes=None) -> "Tensor": ...

    # --- 静态创建方法 ---
    @staticmethod
    def from_numpy(np_array, device=None, requires_grad=False) -> "Tensor": ...
    def numpy(self) -> numpy.ndarray: ...
```

### 4.4 TensorTuple

持有多个 Tensor 的容器节点（用于多输出算子，如 split）。

```python
class TensorTuple(Value):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> "Tensor": ...
    def tuple(self) -> tuple: ...
```

### 4.5 反向传播

```python
def compute_gradient_of_variables(output_tensor: Tensor, out_grad: Tensor) -> None:
    """
    核心反向传播算法：
    1. 拓扑排序计算图（从 output_tensor 出发）
    2. 反向遍历，累加每个节点的梯度
    3. 将最终梯度写入叶节点的 .grad 属性
    """

def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """对计算图做拓扑排序（后序 DFS）"""
```

---

## 5. ops — 算子库

每个算子是一个 `TensorOp` 子类，实现 `compute` 和 `gradient`。
同时提供对应的函数式接口。

### 5.1 逐元素算子

| 算子类 | 函数接口 | 语义 |
|--------|---------|------|
| `EWiseAdd` | `add(a, b)` | 逐元素加 |
| `AddScalar` | `add_scalar(a, scalar)` | 张量 + 标量 |
| `EWiseMul` | `multiply(a, b)` | 逐元素乘 |
| `MulScalar` | `mul_scalar(a, scalar)` | 张量 × 标量 |
| `EWiseDiv` | `divide(a, b)` | 逐元素除 |
| `DivScalar` | `divide_scalar(a, scalar)` | 张量 ÷ 标量 |
| `PowerScalar` | `power_scalar(a, scalar)` | 张量的标量次幂 |
| `Negate` | `negate(a)` | 取负 |
| `Exp` | `exp(a)` | 逐元素 exp |
| `Log` | `log(a)` | 逐元素 log |
| `ReLU` | `relu(a)` | max(0, x) |
| `Tanh` | `tanh(a)` | 双曲正切 |

### 5.2 矩阵 / 形状算子

| 算子类 | 函数接口 | 语义 |
|--------|---------|------|
| `MatMul` | `matmul(a, b)` | 矩阵乘法（支持 batch） |
| `Reshape` | `reshape(a, shape)` | 改变形状 |
| `Transpose` | `transpose(a, axes)` | 转置 |
| `BroadcastTo` | `broadcast_to(a, shape)` | 广播 |
| `Summation` | `summation(a, axes)` | 沿指定轴求和 |
| `Slice` | `slice(a, slices)` | 切片 |
| `Flip` | `flip(a, axes)` | 沿轴翻转 |
| `Dilate` | `dilate(a, axes, dilation)` | 膨胀（用于转置卷积） |
| `Stack` | `stack(tensors, axis)` | 沿新轴拼接 |
| `Split` | `split(a, axis)` | 沿轴拆分（返回 TensorTuple） |

### 5.3 规约算子

| 算子类 | 函数接口 | 语义 |
|--------|---------|------|
| `Summation` | `summation(a, axes)` | 求和 |
| `Max` | `max(a, axes)` | 求最大值 |

---

## 6. nn — 神经网络模块

### 6.1 Module 与 Parameter

```python
class Parameter(Tensor):
    """可训练参数，本质上是 requires_grad=True 的 Tensor"""

class Module:
    training: bool

    def parameters(self) -> List[Parameter]:
        """递归收集所有 Parameter"""

    def train(self) -> "Module":
        """切换到训练模式"""

    def eval(self) -> "Module":
        """切换到评估模式"""

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)
```

### 6.2 线性层

```python
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype="float32"):
        """
        参数: weight (in_features, out_features), bias (out_features,)
        初始化: Kaiming uniform
        """
    def forward(self, x: Tensor) -> Tensor:
        # x @ weight + bias
```

### 6.3 卷积层

```python
class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 bias: bool = True, device=None, dtype="float32"):
        """
        参数: weight (kernel_size, kernel_size, in_channels, out_channels)
        使用 im2col 实现卷积
        """
    def forward(self, x: Tensor) -> Tensor:
        # x: (N, H, W, C_in)  NHWC 布局
        # return: (N, H_out, W_out, C_out)
```

### 6.4 归一化层

```python
class BatchNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device=None, dtype="float32"):
        """
        参数: weight (dim,), bias (dim,)
        状态: running_mean, running_var
        训练时用 batch 统计量，推理时用 running 统计量
        """
    def forward(self, x: Tensor) -> Tensor: ...

class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype="float32"):
        """
        参数: weight (dim,), bias (dim,)
        对最后一维做归一化
        """
    def forward(self, x: Tensor) -> Tensor: ...
```

### 6.5 激活函数

```python
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor: ...
```

### 6.6 Dropout

```python
class Dropout(Module):
    def __init__(self, p: float = 0.5):
        """训练时以概率 p 随机置零并缩放，推理时直通"""
    def forward(self, x: Tensor) -> Tensor: ...
```

### 6.7 容器

```python
class Sequential(Module):
    def __init__(self, *modules: Module): ...
    def forward(self, x: Tensor) -> Tensor:
        """依次调用各子模块"""
```

### 6.8 Embedding

```python
class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype="float32"):
        """参数: weight (num_embeddings, embedding_dim)"""
    def forward(self, x: Tensor) -> Tensor:
        """x: (seq_len,) 或 (batch, seq_len) 整数索引"""
```

### 6.9 损失函数

```python
class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        """
        logits: (batch, num_classes)
        y:      (batch,) 整数标签
        return: 标量，交叉熵损失的 batch 平均值
        实现:   log_sum_exp 技巧保证数值稳定
        """

class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """均方误差"""
```

---

## 7. optim — 优化器

### 7.1 Optimizer 基类

```python
class Optimizer:
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        """将所有参数的 .grad 置为 None"""

    def step(self) -> None:
        """执行一步参数更新"""
        raise NotImplementedError
```

### 7.2 SGD

```python
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        """
        支持动量和权重衰减。
        v_t = momentum * v_{t-1} + (grad + weight_decay * param)
        param = param - lr * v_t
        """
    def step(self) -> None: ...
```

### 7.3 Adam

```python
class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        """
        Adam 优化器，带偏差修正。
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)
        """
    def step(self) -> None: ...
```

---

## 8. data — 数据加载

### 8.1 Dataset

```python
class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def apply_transforms(self, x):
        """依次应用 self.transforms 中的变换"""
```

内置数据集示例：

```python
class MNISTDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str, transforms: Optional[List] = None): ...
    def __getitem__(self, index) -> Tuple[numpy.ndarray, int]: ...
    def __len__(self) -> int: ...
```

### 8.2 DataLoader

```python
class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False):
        """
        按 batch_size 迭代 dataset。
        shuffle=True 时每个 epoch 随机打乱顺序。
        """
    def __iter__(self) -> Iterator[Tuple[Tensor, ...]]: ...
    def __len__(self) -> int:
        """返回 batch 总数"""
```

### 8.3 Transforms

```python
class RandomFlipHorizontal:
    def __init__(self, p: float = 0.5): ...
    def __call__(self, img: numpy.ndarray) -> numpy.ndarray: ...

class RandomCrop:
    def __init__(self, padding: int = 4): ...
    def __call__(self, img: numpy.ndarray) -> numpy.ndarray: ...
```

---

## 9. init — 参数初始化

所有初始化函数就地写入 NDArray 并返回 Tensor。

```python
def zeros(*shape, device=None, dtype="float32", requires_grad=False) -> Tensor: ...
def ones(*shape, device=None, dtype="float32", requires_grad=False) -> Tensor: ...
def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False) -> Tensor: ...
def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False) -> Tensor: ...
def one_hot(n: int, i: Tensor, device=None, dtype="float32") -> Tensor:
    """返回 (len(i), n) 的 one-hot 矩阵"""

def xavier_uniform(fan_in: int, fan_out: int, gain=1.0, **kwargs) -> Tensor:
    """U(-a, a), a = gain * sqrt(6 / (fan_in + fan_out))"""

def xavier_normal(fan_in: int, fan_out: int, gain=1.0, **kwargs) -> Tensor:
    """N(0, std^2), std = gain * sqrt(2 / (fan_in + fan_out))"""

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity="relu", **kwargs) -> Tensor:
    """U(-bound, bound), bound = gain * sqrt(3 / fan_in)"""

def kaiming_normal(fan_in: int, fan_out: int, nonlinearity="relu", **kwargs) -> Tensor:
    """N(0, std^2), std = gain / sqrt(fan_in)"""
```

---

## 附：构建路线图

按模块依赖关系，推荐的实现顺序如下。核心依赖链为：

```
Device → NDArray → autograd → ops → init → nn → optim → data
```

- [x] **Step 1 — backend: Device 抽象**
  实现 `backend/device.py`：`Device` 基类、`CPUDevice`、`CUDADevice`，以及 `cpu()`、`cuda()`、`default_device()` 工厂函数。这是整个系统的地基，NDArray 绑定到具体 Device 上。

- [x] **Step 2 — backend: NDArray + CPU 后端**
  实现 `backend/ndarray.py`（NDArray 统一接口）和 `backend/backend_numpy.py`（用 NumPy 封装 CPU 算子：add / mul / matmul / reshape / sum / transpose 等）。后续所有 `Op.compute` 都通过 NDArray 调用这里。

- [x] **Step 3 — autograd: 计算图核心**
  完善 `autograd.py`：补全 `Tensor.__init__`（设备分派、dtype 处理）、`backward()`、以及核心算法 `compute_gradient_of_variables()`（拓扑排序 + 反向遍历累加梯度）和 `find_topo_sort()`。这是整个自动微分引擎。

- [x] **Step 4 — ops: 数学算子**
  实现 `ops/ops_math.py`，每个算子均需实现 `compute`（NDArray 层面前向）和 `gradient`（Tensor 层面反向）：
  - 逐元素：`EWiseAdd` / `AddScalar` / `EWiseMul` / `MulScalar` / `EWiseDiv` / `DivScalar` / `PowerScalar` / `Negate` / `Exp` / `Log` / `ReLU` / `Tanh`
  - 矩阵/形状：`MatMul` / `Reshape` / `Transpose` / `BroadcastTo` / `Summation` / `Slice` / `Flip` / `Dilate`

- [x] **Step 5 — ops: TensorTuple 算子**
  实现 `ops/ops_tuple.py`：`Stack`（沿新轴拼接多个 Tensor）和 `Split`（返回 TensorTuple），供 `Conv2d` im2col 及多输出场景使用。

- [x] **Step 6 — init: 参数初始化**
  实现 `init/initializers.py`：`zeros` / `ones` / `randn` / `rand` / `one_hot` / `xavier_uniform` / `xavier_normal` / `kaiming_uniform` / `kaiming_normal`。`nn` 层的参数初始化依赖此模块。

- [x] **Step 7 — nn: Module 与 Parameter 基类**
  实现 `nn/module.py`：`Parameter`（`requires_grad=True` 的 Tensor 子类）和 `Module`（`parameters()` 递归收集参数、`train()` / `eval()` 模式切换）。所有 nn 层的公共父类。

- [x] **Step 8 — nn: 各网络层**
  按依赖复杂度由低到高依次实现：
  - [ ] `activation.py` — `ReLU` / `Tanh` / `Sigmoid`
  - [ ] `linear.py` — `Linear`（Kaiming uniform 初始化）
  - [ ] `norm.py` — `BatchNorm` / `LayerNorm`
  - [ ] `dropout.py` — `Dropout`
  - [ ] `container.py` — `Sequential`
  - [ ] `loss.py` — `SoftmaxLoss`（log-sum-exp 数值稳定）/ `MSELoss`
  - [ ] `conv.py` — `Conv2d`（im2col 实现，依赖 `Dilate` / `Flip`）
  - [ ] `module.py` — `Embedding`

- [x] **Step 9 — optim: 优化器**
  - [ ] `optimizer.py` — `Optimizer` 基类（`zero_grad` / `step`）
  - [ ] `sgd.py` — SGD（支持 momentum + weight_decay）
  - [ ] `adam.py` — Adam（m/v 一阶/二阶矩 + 偏差修正）

- [x] **Step 10 — data: 数据加载**
  - [ ] `dataset.py` — `Dataset` 基类 + `MNISTDataset`
  - [ ] `dataloader.py` — `DataLoader`（支持 shuffle）
  - [ ] 数据增强变换：`RandomFlipHorizontal` / `RandomCrop`

- [x] **Step 11 — 端到端集成验证**
  用本文档"典型训练流程"跑通 MNIST 训练，验证前向传播、反向传播、参数更新全链路正确。此时 CPU 路径完整可用。

- [ ] **Step 12 — backend: CUDA 后端（可选）**
  实现 `backend/backend_cuda.py`，为 NDArray 的每个算子接入 CUDA kernel。由于 Device 抽象已屏蔽硬件差异，上层代码（Op / nn / optim）无需任何修改。

---

## 附：典型训练流程

```python
import tiny_dlsys as tdl
import tiny_dlsys.nn as nn
import tiny_dlsys.optim as optim
from tiny_dlsys.data import DataLoader, MNISTDataset

# 构建模型
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# 选择设备
device = tdl.cuda() if tdl.cuda().enabled() else tdl.cpu()

loss_fn = nn.SoftmaxLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 加载数据
train_set = MNISTDataset("data/train-images.gz", "data/train-labels.gz")
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# 训练循环
model.train()
for epoch in range(10):
    total_loss = 0.0
    for X, y in train_loader:
        X, y = tdl.Tensor(X, device=device), tdl.Tensor(y, device=device)

        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.numpy()
    print(f"epoch {epoch}, loss = {total_loss / len(train_loader):.4f}")
```

## 附：测试
* python3 -m unittest tests.test_backprop -v
