"""Extra teaching sections and auto-checked exercises for existing prerequisites.

The original prerequisite notebooks are compact "reference cards".  For
learners with no machine-learning background they need more guided practice;
these blocks are appended by ``build_prerequisites.py``.
"""

from __future__ import annotations

from exercise_kit import Exercise, code, exercise_block, md


# --------------------------------------------------------------------------- NumPy
NUMPY_EXERCISES = [
    Exercise(
        title="批量取出已选动作的 Q 值",
        prompt="实现 `select_actions(q, actions)`：`q` 为 `(B,A)`，`actions` 为 `(B,)` 整数，返回 `(B,)`，第 i 项是 `q[i, actions[i]]`。**不要用 for 循环。**",
        hint="`q[np.arange(len(actions)), actions]`。",
        stub='''
import numpy as np

def select_actions(q, actions):
    raise NotImplementedError
''',
        solution='''
import numpy as np

def select_actions(q, actions):
    return q[np.arange(len(actions)), actions]
''',
        test='''
import numpy as np

def test_select_actions():
    q = np.array([[1., 4., 2.], [5., 0., 3.], [2., 1., 6.], [0., 9., 8.]])
    out = select_actions(q, np.array([2, 0, 1, 1]))
    assert out.shape == (4,), f"输出应为 (4,)，得到 {out.shape}"
    assert np.allclose(out, [2., 5., 1., 9.]), f"应为 [2,5,1,9]，得到 {out}"
''',
    ),
    Exercise(
        title="one-hot 编码",
        prompt="实现 `one_hot(indices, n)`：输入 `(B,)` 整数数组，返回 `(B,n)` 的 float 数组，每行只有对应位置为 1。",
        hint="先 `out = np.zeros((len(indices), n))`，再用上一题的高级索引赋值为 1。",
        stub='''
import numpy as np

def one_hot(indices, n):
    raise NotImplementedError
''',
        solution='''
import numpy as np

def one_hot(indices, n):
    indices = np.asarray(indices)
    out = np.zeros((len(indices), n))
    out[np.arange(len(indices)), indices] = 1.0
    return out
''',
        test='''
import numpy as np

def test_one_hot():
    out = one_hot([2, 0], 3)
    assert out.shape == (2, 3), f"应为 (2,3)，得到 {out.shape}"
    assert np.array_equal(out, [[0, 0, 1], [1, 0, 0]]), f"结果不对：{out}"
''',
    ),
    Exercise(
        title="移动平均",
        prompt="实现 `moving_average(x, window)`：返回长度 `len(x)-window+1` 的数组，第 i 项是 `x[i:i+window]` 的平均。训练曲线几乎都要先平滑再看。",
        hint="`np.convolve(x, np.ones(window) / window, mode=\"valid\")`，或者用 `np.cumsum`。",
        stub='''
import numpy as np

def moving_average(x, window):
    raise NotImplementedError
''',
        solution='''
import numpy as np

def moving_average(x, window):
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(window) / window, mode="valid")
''',
        test='''
import numpy as np

def test_moving_average():
    out = moving_average([1, 2, 3, 4, 5], 2)
    assert np.allclose(out, [1.5, 2.5, 3.5, 4.5]), f"应为 [1.5,2.5,3.5,4.5]，得到 {out}"
    assert len(moving_average(np.arange(100), 10)) == 91, "长度应为 len(x)-window+1"
''',
    ),
    Exercise(
        title="只用训练集统计量做标准化",
        prompt="""
        实现 `fit_normalizer(train)`，返回函数 `normalize(x)`：按**列**（每个特征）减去训练集均值、除以训练集标准差（加 `1e-8` 防止除 0）。
        测试集必须用训练集的统计量，否则就是数据泄漏。
        """,
        stub='''
import numpy as np

def fit_normalizer(train):
    raise NotImplementedError
''',
        solution='''
import numpy as np

def fit_normalizer(train):
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8
    def normalize(x):
        return (x - mean) / std
    return normalize
''',
        test='''
import numpy as np

def test_fit_normalizer():
    rng = np.random.default_rng(0)
    train = rng.normal([5.0, -3.0], [2.0, 0.5], size=(1000, 2))
    norm = fit_normalizer(train)
    z = norm(train)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-6) and np.allclose(z.std(axis=0), 1, atol=1e-3), "训练集标准化后每列应均值 0、标准差 1"
    test = np.array([[5.0, -3.0]])
    assert np.allclose(norm(test), 0, atol=0.1), "测试数据应使用训练集的均值和标准差"
''',
    ),
]


# ---------------------------------------------------------------------------- Math
MATH_EXERCISES = [
    Exercise(
        title="矩阵乘法的 shape",
        prompt="实现 `matmul_shape(a, b)`：输入两个二维 shape 元组，可相乘时返回结果 shape，不可相乘时返回 `None`。不要真的创建数组。",
        stub='''
def matmul_shape(a, b):
    raise NotImplementedError
''',
        solution='''
def matmul_shape(a, b):
    return (a[0], b[1]) if a[1] == b[0] else None
''',
        test='''
def test_matmul_shape():
    assert matmul_shape((32, 4), (4, 2)) == (32, 2), "(B,F)@(F,A) → (B,A)"
    assert matmul_shape((32, 4), (2, 4)) is None, "内维 4 和 2 不同，不能相乘"
    assert matmul_shape((1, 512), (512, 6)) == (1, 6)
''',
    ),
    Exercise(
        title="数值稳定的 softmax",
        prompt="实现 `softmax(logits)`：支持一维或二维输入（二维时对最后一维计算）。`logits=[1000, 1000]` 时也必须返回 `[0.5, 0.5]`。",
        hint="减去最大值：`z = logits - logits.max(axis=-1, keepdims=True)`。",
        stub='''
import numpy as np

def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    raise NotImplementedError
''',
        solution='''
import numpy as np

def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
''',
        test='''
import numpy as np

def test_softmax():
    assert np.allclose(softmax([1000.0, 1000.0]), [0.5, 0.5]), "大 logits 也不能溢出"
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert p.shape == (2, 3) and np.allclose(p.sum(axis=1), 1.0), "二维输入应对每一行归一化"
    assert np.allclose(p[1], 1 / 3), "logits 全相同时概率应均匀"
''',
    ),
    Exercise(
        title="每一步的折扣回报",
        prompt="""
        实现 `discounted_returns(rewards, gamma)`，返回与 `rewards` 等长的列表，第 t 项是
        $G_t=r_t+\\gamma r_{t+1}+\\gamma^2 r_{t+2}+\\dots$。策略梯度（Chapter11）就需要每一步的 $G_t$。
        """,
        hint="从后往前：`G = r + gamma * G`，每步把 G 放进结果，最后反转。",
        stub='''
def discounted_returns(rewards, gamma):
    raise NotImplementedError
''',
        solution='''
def discounted_returns(rewards, gamma):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return out[::-1]
''',
        test='''
def test_discounted_returns():
    out = discounted_returns([0.0, 0.0, 1.0], 0.9)
    assert len(out) == 3, "长度应与 rewards 相同"
    assert all(abs(a - b) < 1e-9 for a, b in zip(out, [0.81, 0.9, 1.0])), f"应为 [0.81, 0.9, 1.0]，得到 {out}"
    assert discounted_returns([1.0, 1.0], 0.0) == [1.0, 1.0], "gamma=0 时只看即时奖励"
''',
    ),
]


# ------------------------------------------------------------------ Machine learning
ML_SECTIONS = [
    md(r"""
    ## 4. 分类：从“预测一个数”到“预测概率”

    回归输出任意实数；**分类**输出“属于每一类的概率”。最简单的分类模型是**逻辑回归**：

    $$p(y=1\mid x)=\sigma(xw+b),\qquad \sigma(z)=\frac{1}{1+e^{-z}}$$

    损失用**交叉熵**：正确类别的概率越低，惩罚越大。RL 中的离散策略 $\pi(a\mid s)$ 就是一个“多分类器”：
    输入状态，输出每个动作的概率——只是训练信号来自奖励，而不是标签。
    """),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
n = 200
x0 = rng.normal([-1.0, -1.0], 0.8, size=(n // 2, 2))
x1 = rng.normal([1.2, 1.0], 0.8, size=(n // 2, 2))
X = np.vstack([x0, x1]); y = np.r_[np.zeros(n // 2), np.ones(n // 2)]

w, b, lr = np.zeros(2), 0.0, 0.5
for _ in range(300):
    p = 1 / (1 + np.exp(-(X @ w + b)))
    w -= lr * X.T @ (p - y) / n          # 交叉熵对 w 的梯度
    b -= lr * np.mean(p - y)
accuracy = np.mean((p > 0.5) == y)
print("w:", np.round(w, 3), "b:", round(b, 3), "训练准确率:", accuracy)

xs = np.linspace(-3, 3, 50)
plt.figure(figsize=(4.5, 3.6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=12)
plt.plot(xs, -(w[0] * xs + b) / w[1], "k--", label="decision boundary p=0.5")
plt.ylim(-3.5, 3.5); plt.legend(); plt.grid(alpha=.25); plt.show(); plt.close()
"""),
    md(r"""
    ## 5. 亲眼看到过拟合

    用不同复杂度的多项式拟合 15 个带噪点。阶数太低：**欠拟合**（训练、验证误差都高）；
    阶数太高：**过拟合**（训练误差很低，验证误差变大）。在 RL 里，过拟合表现为“在训练用的随机种子或数据区间上很好，换一个就崩”。
    """),
    code(r"""
rng = np.random.default_rng(2)
x_tr = np.sort(rng.uniform(-1, 1, 15)); y_tr = np.sin(3 * x_tr) + rng.normal(0, .15, 15)
x_va = np.sort(rng.uniform(-1, 1, 100)); y_va = np.sin(3 * x_va) + rng.normal(0, .15, 100)

degrees = range(1, 11)
train_err, val_err = [], []
for d in degrees:
    coef = np.polyfit(x_tr, y_tr, d)
    train_err.append(np.mean((np.polyval(coef, x_tr) - y_tr) ** 2))
    val_err.append(np.mean((np.polyval(coef, x_va) - y_va) ** 2))

plt.figure(figsize=(6, 3.2))
plt.plot(list(degrees), train_err, "o-", label="train MSE"); plt.plot(list(degrees), val_err, "s-", label="validation MSE")
plt.yscale("log"); plt.xlabel("polynomial degree (model complexity)"); plt.legend(); plt.grid(alpha=.25); plt.show(); plt.close()
best = int(np.argmin(val_err)) + 1
print("验证误差最小的阶数:", best, " 10 阶的训练/验证误差:", round(train_err[-1], 4), round(val_err[-1], 2))
"""),
]

ML_EXERCISES = [
    Exercise(
        title="随机划分训练/验证索引",
        prompt="实现 `train_val_split(n, val_fraction, seed)`：返回 `(train_idx, val_idx)` 两个整数数组，二者不重叠、合起来正好是 `0..n-1`，验证集大小为 `int(n * val_fraction)`。",
        hint="`perm = np.random.default_rng(seed).permutation(n)`，前面的给验证集，剩下的给训练集。",
        stub='''
import numpy as np

def train_val_split(n, val_fraction, seed):
    raise NotImplementedError
''',
        solution='''
import numpy as np

def train_val_split(n, val_fraction, seed):
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_fraction)
    return perm[n_val:], perm[:n_val]
''',
        test='''
import numpy as np

def test_train_val_split():
    tr, va = train_val_split(100, 0.2, seed=0)
    assert len(va) == 20 and len(tr) == 80, "验证集应 20 个、训练集 80 个"
    assert len(set(tr) & set(va)) == 0, "训练集与验证集不能重叠"
    assert sorted(np.r_[tr, va].tolist()) == list(range(100)), "合起来应覆盖全部索引"
    tr2, va2 = train_val_split(100, 0.2, seed=0)
    assert np.array_equal(va, va2), "相同 seed 应得到相同划分"
''',
    ),
    Exercise(
        title="一步梯度下降",
        prompt="""
        对线性回归 $\\hat y = xw+b$、损失 $L=\\frac1N\\sum(\\hat y-y)^2$，实现 `gd_step(w, b, x, y, lr)`，返回更新后的 `(w, b)`。
        梯度：$\\partial L/\\partial w=\\frac2N\\sum(\\hat y-y)x$，$\\partial L/\\partial b=\\frac2N\\sum(\\hat y-y)$。
        """,
        stub='''
import numpy as np

def gd_step(w, b, x, y, lr):
    raise NotImplementedError
''',
        solution='''
import numpy as np

def gd_step(w, b, x, y, lr):
    error = w * x + b - y
    return w - lr * 2 * np.mean(error * x), b - lr * 2 * np.mean(error)
''',
        test='''
import numpy as np

def test_gd_step():
    x = np.array([1.0, 2.0]); y = np.array([2.0, 4.0])
    w, b = gd_step(0.0, 0.0, x, y, lr=0.1)
    assert abs(w - 1.0) < 1e-9 and abs(b - 0.6) < 1e-9, f"一步后应为 w=1.0, b=0.6，得到 {w}, {b}"
    for _ in range(2000):
        w, b = gd_step(w, b, x, y, lr=0.1)
    assert abs(w - 2.0) < 1e-3 and abs(b) < 1e-3, "多步后应收敛到 w=2, b=0"
''',
    ),
    Exercise(
        title="分类准确率",
        prompt="实现 `accuracy(probs, labels)`：`probs` 为 `(N,C)` 概率，`labels` 为 `(N,)` 整数类别，返回预测类别（概率最大者）等于标签的比例。",
        stub='''
import numpy as np

def accuracy(probs, labels):
    raise NotImplementedError
''',
        solution='''
import numpy as np

def accuracy(probs, labels):
    return float(np.mean(np.argmax(probs, axis=1) == np.asarray(labels)))
''',
        test='''
import numpy as np

def test_accuracy():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    assert abs(accuracy(probs, [0, 1, 1, 1]) - 0.75) < 1e-9, "4 个里对了 3 个，应为 0.75"
''',
    ),
]


# ------------------------------------------------------------------------- PyTorch
PYTORCH_SECTIONS = [
    md(r"""
    ## 5. 用 PyTorch 写一个分类 MLP（含 mini-batch）

    真实训练很少一次用全部数据，而是每次取一个 **mini-batch**。`TensorDataset + DataLoader` 负责打乱和分批，
    DQN 从 replay buffer 抽 batch 的思路与此相同。下面在一个“两个月牙”形状的二分类数据上训练 MLP。
    """),
    code(r"""
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

def make_moons(n, noise=0.15):
    t = torch.rand(n) * math.pi
    upper = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
    lower = torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], dim=1)
    x = torch.cat([upper, lower]) + noise * torch.randn(2 * n, 2)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    return x, y

x_train, y_train = make_moons(400)
x_val, y_val = make_moons(200)
loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()       # 输入 logits (B,C) 和整数标签 (B,)

for epoch in range(40):
    model.train()
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    if epoch % 10 == 9:
        model.eval()
        with torch.no_grad():
            acc = (model(x_val).argmax(dim=1) == y_val).float().mean().item()
        print(f"epoch {epoch+1:>2}: 最后一个 batch 的 loss {loss.item():.3f}, 验证准确率 {acc:.3f}")
assert acc > 0.9
"""),
    md(r"""
    ## 6. 保存与加载、target network 的“复制”

    - 保存模型通常只保存参数：`torch.save(model.state_dict(), path)`；加载时先建同结构模型，再 `load_state_dict`。
    - DQN 的 target network 就是“定期把在线网络的参数复制过去”：`target.load_state_dict(online.state_dict())`。
    - DDPG/SAC 用**软更新**：$\theta_{target}\leftarrow\tau\theta+(1-\tau)\theta_{target}$，每步只靠近一点点（本课练习 4）。
    """),
    code(r"""
import copy, os, tempfile

online = nn.Linear(4, 2)
target = copy.deepcopy(online)                 # 结构和参数都复制一份
with torch.no_grad():
    online.weight += 1.0                       # 模拟在线网络被训练改动
print("复制前差异:", (online.weight - target.weight).abs().max().item())
target.load_state_dict(online.state_dict())    # 硬更新（DQN 每 N 步一次）
print("复制后差异:", (online.weight - target.weight).abs().max().item())

path = os.path.join(tempfile.gettempdir(), "tiny_model.pt")
torch.save(online.state_dict(), path)
restored = nn.Linear(4, 2); restored.load_state_dict(torch.load(path, weights_only=True))
assert torch.equal(restored.weight, online.weight)
print("保存/加载成功:", path)
"""),
]

PYTORCH_EXERCISES = [
    Exercise(
        title="用 gather 取已选动作的 Q 值",
        prompt="实现 `chosen_q(q_values, actions)`：`q_values` 为 `(B,A)` 张量，`actions` 为 `(B,)` 的 `long` 张量，返回 `(B,)`。要求使用 `gather`，并保持梯度可以传回 `q_values`。",
        hint="`q_values.gather(1, actions.unsqueeze(1)).squeeze(1)`。",
        stub='''
import torch

def chosen_q(q_values, actions):
    raise NotImplementedError
''',
        solution='''
import torch

def chosen_q(q_values, actions):
    return q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
''',
        test='''
import torch

def test_chosen_q():
    q = torch.tensor([[1., 4., 2.], [5., 0., 3.]], requires_grad=True)
    out = chosen_q(q, torch.tensor([1, 2]))
    assert tuple(out.shape) == (2,), f"应为 (2,)，得到 {tuple(out.shape)}"
    assert torch.allclose(out, torch.tensor([4., 3.])), f"应为 [4, 3]，得到 {out}"
    out.sum().backward()
    assert torch.equal(q.grad, torch.tensor([[0., 1., 0.], [0., 0., 1.]])), "梯度只应流向被选中的位置"
''',
    ),
    Exercise(
        title="构建 MLP",
        prompt="实现 `make_mlp(sizes)`：例如 `sizes=[4, 64, 64, 2]` 返回 `Linear(4,64)→ReLU→Linear(64,64)→ReLU→Linear(64,2)`，**最后一层后面不加激活**（Q 值可以是负数）。",
        hint="循环 `zip(sizes[:-1], sizes[1:])` 收集层，除最后一层外都追加 `nn.ReLU()`，最后 `nn.Sequential(*layers)`。",
        stub='''
import torch.nn as nn

def make_mlp(sizes):
    raise NotImplementedError
''',
        solution='''
import torch.nn as nn

def make_mlp(sizes):
    layers = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(n_in, n_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
''',
        test='''
import torch
import torch.nn as nn

def test_make_mlp():
    net = make_mlp([4, 64, 64, 2])
    kinds = [type(m).__name__ for m in net]
    assert kinds == ["Linear", "ReLU", "Linear", "ReLU", "Linear"], f"层结构不对：{kinds}"
    assert tuple(net(torch.zeros(5, 4)).shape) == (5, 2), "输入 (5,4) 应输出 (5,2)"
''',
    ),
    Exercise(
        title="完整训练循环",
        prompt="""
        实现 `train_regressor(model, x, y, steps, lr)`：用 Adam 和 MSE 训练 `model`，返回**每一步的 loss 列表**（Python float）。
        必须包含完整五步：forward → loss → `zero_grad` → `backward` → `step`。
        """,
        stub='''
import torch

def train_regressor(model, x, y, steps, lr):
    raise NotImplementedError
''',
        solution='''
import torch

def train_regressor(model, x, y, steps, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history
''',
        test='''
import torch

def test_train_regressor():
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 64).unsqueeze(1); y = torch.sin(3 * x)
    model = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    history = train_regressor(model, x, y, steps=300, lr=0.01)
    assert len(history) == 300 and isinstance(history[0], float), "应返回 300 个 float"
    assert history[-1] < history[0] * 0.1, f"loss 应下降到初始的 10% 以下：{history[0]:.3f} → {history[-1]:.3f}"
''',
    ),
    Exercise(
        title="软更新 target network",
        prompt="实现 `soft_update(target, source, tau)`：对每个参数执行 $\\theta_t\\leftarrow\\tau\\theta_s+(1-\\tau)\\theta_t$，原地修改 `target`，不能产生梯度。",
        hint="`with torch.no_grad(): for t, s in zip(target.parameters(), source.parameters()): t.mul_(1 - tau).add_(tau * s)`。",
        stub='''
import torch

def soft_update(target, source, tau):
    raise NotImplementedError
''',
        solution='''
import torch

def soft_update(target, source, tau):
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(1 - tau).add_(tau * s)
''',
        test='''
import torch

def test_soft_update():
    source, target = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(1.0); source.bias.fill_(1.0); target.weight.fill_(0.0); target.bias.fill_(0.0)
    soft_update(target, source, 0.1)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.1)), "一次 tau=0.1 后应为 0.1"
    soft_update(target, source, 0.1)
    assert torch.allclose(target.bias, torch.tensor([0.19])), "两次后应为 0.1 + 0.1*(1-0.1) = 0.19"
    assert target.weight.grad is None, "软更新不应产生梯度"
''',
    ),
]


def numpy_extra() -> list[dict]:
    return exercise_block(NUMPY_EXERCISES)


MATH_SECTIONS = [
    md(r"""
    ## 4. 矩阵乘法一步步算（第一次接触请务必手算）

    $C=AB$ 中，$C_{ij}$ = **A 的第 i 行**与 **B 的第 j 列**逐项相乘再求和。所以 A 的列数必须等于 B 的行数：
    `(m, n) @ (n, p) → (m, p)`。

    在神经网络里：每一行是一个样本，每一列是一个特征；`X @ W` 就是“每个样本对每个输出神经元做一次加权求和”。
    下面用循环和 `@` 各算一次，确认二者一致。
    """),
    code(r"""
import numpy as np

A = np.array([[1., 2., 3.],
              [4., 5., 6.]])          # (2, 3)：2 个样本，3 个特征
B = np.array([[1., 0.],
              [0., 1.],
              [1., 1.]])              # (3, 2)：3 个输入 → 2 个输出
C_loop = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        C_loop[i, j] = sum(A[i, k] * B[k, j] for k in range(3))
print("循环结果:\n", C_loop)
print("A @ B:\n", A @ B)
assert np.allclose(C_loop, A @ B)
"""),
    md(r"""
    ## 5. 多变量函数的梯度：指向“上坡最陡”的方向

    对 $f(w_1,w_2)$，梯度 $\nabla f=(\partial f/\partial w_1,\ \partial f/\partial w_2)$ 是一个向量，指向函数增长最快的方向。
    梯度下降沿 **负梯度** 走一小步，函数值就会下降。下面在等高线图上画出梯度下降的轨迹。
    """),
    code(r"""
import matplotlib.pyplot as plt

f = lambda w: (w[0] - 1) ** 2 + 4 * (w[1] + 0.5) ** 2
grad = lambda w: np.array([2 * (w[0] - 1), 8 * (w[1] + 0.5)])

w = np.array([-2.0, 1.5]); path = [w.copy()]
for _ in range(30):
    w = w - 0.1 * grad(w)
    path.append(w.copy())
path = np.array(path)

g1, g2 = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-2, 2, 100))
plt.figure(figsize=(5.5, 3.6))
plt.contour(g1, g2, (g1 - 1) ** 2 + 4 * (g2 + 0.5) ** 2, levels=20, cmap="viridis")
plt.plot(path[:, 0], path[:, 1], "o-", color="red", markersize=3, label="gradient descent")
plt.scatter([1], [-0.5], marker="*", s=200, color="black", label="minimum")
plt.legend(); plt.title("follow the negative gradient"); plt.show(); plt.close()
print("终点:", np.round(path[-1], 4), " f =", round(f(path[-1]), 6))
"""),
]


def math_extra() -> list[dict]:
    return MATH_SECTIONS + exercise_block(MATH_EXERCISES)


def ml_extra() -> list[dict]:
    return ML_SECTIONS + exercise_block(ML_EXERCISES)


def pytorch_extra() -> list[dict]:
    return PYTORCH_SECTIONS + exercise_block(PYTORCH_EXERCISES)
