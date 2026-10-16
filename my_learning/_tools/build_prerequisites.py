"""Generate the zero-background prerequisite notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import foundations_bandit
import foundations_cnn
import foundations_neural_net
import foundations_probability
from foundations_extras import math_extra, ml_extra, numpy_extra, pytorch_extra


LEARNING_ROOT = Path(__file__).resolve().parent.parent
OUT = LEARNING_ROOT / "01_prerequisites"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


def save(name: str, cells: list[dict]) -> None:
    for index, cell in enumerate(cells):
        cell["id"] = f"{name.lower()}-{index:02d}"
    doc = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUT / f"{name}.ipynb").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def index_cells() -> list[dict]:
    return [
        md(r"""
        # Prerequisite 00 · 零基础预备篇导航

        [返回总目录](../README.md) · [长期学习计划](../05_reference/LONG_TERM_PLAN.md) · [完成后进入 Chapter01](../02_chapters/Chapter01.ipynb)

        预备篇不是让你先学完“所有数学和深度学习”，而是补齐进入强化学习的最小工具箱。每个 notebook 都围绕后面会真实出现的代码和公式。

        **给完全没有机器学习基础的你：**原书默认读者已经懂机器学习和 PyTorch。带 **B** 的 4 个 notebook（概率、神经网络从零、CNN/序列、老虎机）
        是专门为“机器学习零基础”补的，不要跳过。每个 notebook 末尾都有**自动批改练习**：看到 `[通过]` 才算完成。
        """),
        md(r"""
        ## 推荐顺序

        | 顺序 | Notebook | 学完能做什么 | 建议时长 |
        |---:|---|---|---:|
        | 1 | [Python 基础](Prerequisite01_Python.ipynb) | 读懂训练循环、函数和对象 | 2–6 h |
        | 2 | [NumPy](Prerequisite02_NumPy.ipynb) | 处理 state、batch、action indexing | 3–4 h |
        | 3 | [Matplotlib](Prerequisite03_Matplotlib.ipynb) | 画回报、移动平均和多 seed 置信带 | 1–2 h |
        | 4 | [数学：线性代数与微积分](Prerequisite04_Math.ipynb) | 理解矩阵 shape、梯度、softmax | 3–4 h |
        | 4B | [概率、采样与蒙特卡洛](Prerequisite04B_Probability.ipynb) | 理解期望、方差、条件概率、log 概率、重要性采样 | 3–5 h |
        | 5 | [机器学习基础](Prerequisite05_Machine_Learning.ipynb) | 理解 loss、梯度下降、分类、过拟合与数据划分 | 4–5 h |
        | 5B | [神经网络从零开始](Prerequisite05B_Neural_Network_From_Scratch.ipynb) | 手写前向/反向传播，理解 Q 网络是什么 | 4–6 h |
        | 6 | [PyTorch](Prerequisite06_PyTorch.ipynb) | 独立写出 MLP 训练循环、mini-batch、保存与 target 复制 | 4–6 h |
        | 6B | [卷积网络与序列模型速成](Prerequisite06B_CNN_and_Sequences.ipynb) | 读懂 Atari CNN 与文本 LSTM 的 shape 流 | 4–6 h |
        | 7 | [Gymnasium](Prerequisite07_Gymnasium.ipynb) | 正确运行 episode、处理终止和 wrapper | 2–3 h |
        | 7B | [从多臂老虎机走进 RL](Prerequisite07B_Bandit_Bridge.ipynb) | 增量更新、探索与利用、第一个 Q-learning | 3–4 h |
        | 8 | [入学测试](Prerequisite08_Readiness_Check.ipynb) | 判断是否可以进入 RL 主线 | 1–2 h |

        如果已经有基础，可以直接做入学测试；不会的题再回到对应预备章，不必机械重学全部内容。
        """),
        md(r"""
        ## 什么暂时不用学

        进入 RL 前不要求：复杂 Python 元编程、NumPy 全部 API、Matplotlib 美化、严格实分析证明、大型 CNN/Transformer、分布式训练或自定义 CUDA。

        当前目标只有三个：**看懂数据形状、看懂优化过程、看懂环境交互**。遇到知识缺口再按需扩展。
        Transformer/LLM 相关知识在 Chapter13、Chapter19 之前再补，见[长期学习计划](../05_reference/LONG_TERM_PLAN.md)中的“章节前置知识矩阵”。
        """),
        code(r"""
# 修改为 True，查看自己还缺哪一块。
finished = {
    "Python": False, "NumPy": False, "Matplotlib": False, "Math": False, "Probability (4B)": False,
    "Machine Learning": False, "Neural Net From Scratch (5B)": False, "PyTorch": False,
    "CNN & Sequences (6B)": False, "Gymnasium": False, "Bandit Bridge (7B)": False,
}
remaining = [name for name, done in finished.items() if not done]
print("已完成:", sum(finished.values()), "/", len(finished))
print("下一步:", remaining[0] if remaining else "进入 Prerequisite08 入学测试")
"""),
        md(r"""
        ## 时间建议

        - 会编程、但没有机器学习基础：每周 8–10 小时，约 **5–7 周**完成预备篇（合计约 40–55 小时）。
        - 编程也不熟：先用 [Python 中文官方教程](https://docs.python.org/zh-cn/3/tutorial/index.html) 补 2–4 周，再开始本篇。
        - 每章都要运行代码并完成自动批改练习；只阅读会制造“好像会了”的错觉。

        时间不是过关条件。[入学测试](Prerequisite08_Readiness_Check.ipynb)通过才进入主线。
        """),
    ]


def python_cells() -> list[dict]:
    return [
        md(r"""
        # Prerequisite 01 · Python：读懂训练循环

        [预备篇导航](Prerequisite00_Index.ipynb) · [下一章：NumPy](Prerequisite02_NumPy.ipynb)

        **目标：**能够阅读变量、容器、条件、循环、函数和类；能够把一段 agent–environment 交互拆成小函数。这里不讲与 RL 无关的高级语法。
        """),
        md(r"""
        ## 1. 数据：一个 transition 就是一条记录

        Python 常用容器在 RL 中各有职责：tuple 表示固定字段，list 收集轨迹，dict 保存配置或指标。
        """),
        code(r"""
transition = ("state_0", 1, 0.5, "state_1", False)
state, action, reward, next_state, done = transition
trajectory = [transition]
config = {"gamma": 0.99, "learning_rate": 1e-3}

print("action/reward:", action, reward)
print("trajectory length:", len(trajectory))
print("gamma:", config["gamma"])
"""),
        md(r"""
        ## 2. 控制流与函数

        训练循环本质上是：重复选动作、执行、保存经验，遇到终止就 reset。函数应只有清楚的输入和输出，避免依赖难追踪的全局变量。
        """),
        code(r"""
def discounted_return(rewards, gamma):
    total = 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
    return total

rewards = [0.0, 0.0, 1.0]
print("return:", discounted_return(rewards, gamma=0.9))
assert abs(discounted_return(rewards, 0.9) - 0.81) < 1e-9
"""),
        md(r"""
        ## 3. 类：把状态和行为放在一起

        环境和神经网络通常是对象。`__init__` 保存状态，方法读取或修改它；不要把“类”理解为新算法。
        """),
        code(r"""
class EpsilonGreedyAgent:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.steps = 0

    def act(self, q_values):
        self.steps += 1
        # 为便于演示，这里只写 greedy 分支。
        return max(range(len(q_values)), key=lambda i: q_values[i])

agent = EpsilonGreedyAgent()
print("chosen action:", agent.act([0.2, 0.8, 0.1]), "steps:", agent.steps)
"""),
        md(r"""
        ## 4. 初学者常见坑

        - `=` 是赋值，`==` 是比较。
        - list 是可变对象；复制时注意 `copy()` 与共享引用。
        - 函数没有显式 `return` 时返回 `None`。
        - 不要用 `from module import *`，变量来源会变得不清楚。
        - 异常信息应从最后一行开始读，再沿 traceback 找到自己的代码。

        **练习：**让 `discounted_return` 同时返回每个时刻的 return，例如 `[0.81, 0.9, 1.0]`。

        <details><summary>参考思路</summary>反向遍历 reward，每次把新的 `total` 插到列表开头，或最后再 reverse。</details>
        """),
        md(r"""
        ## 官方选读

        - [Python 中文官方教程](https://docs.python.org/zh-cn/3/tutorial/index.html)：重点看控制流、数据结构、函数、模块和类。

        **过关：**能不看答案写出 `discounted_return`，并能解释 tuple、list、dict 在训练代码中的不同用途。
        """),
    ]


def numpy_cells() -> list[dict]:
    return [
        md(r"""
        # Prerequisite 02 · NumPy：shape 是第一等公民

        [上一章：Python](Prerequisite01_Python.ipynb) · [下一章：Matplotlib](Prerequisite03_Matplotlib.ipynb)

        **目标：**熟练处理 state、batch 和 action 维度，理解广播与高级索引。RL 中大量“算法错误”其实是 shape 悄悄错了。
        """),
        md(r"""
        ## 1. 数组、shape、dtype

        一个状态可能是 `(features,)`，一批状态通常是 `(batch, features)`，Q 网络输出通常是 `(batch, actions)`。
        """),
        code(r"""
import numpy as np

states = np.array([[0.1, 0.2, -0.3], [0.0, 0.4, 0.5]], dtype=np.float32)
weights = np.array([[1.0, -1.0], [0.5, 0.2], [-0.3, 0.8]], dtype=np.float32)
q_values = states @ weights
print("states:", states.shape, states.dtype)
print("Q(s,a):", q_values.shape, "\n", q_values)
assert q_values.shape == (2, 2)
"""),
        md(r"""
        ## 2. axis 与动作索引

        `axis=1` 表示在每个样本的动作维上归约。训练 DQN 时，需要从每行 Q 值中取出实际执行动作对应的一项。
        """),
        code(r"""
actions = np.array([1, 0])
batch_index = np.arange(len(actions))
chosen_q = q_values[batch_index, actions]
greedy_actions = q_values.argmax(axis=1)

print("chosen Q:", chosen_q)
print("greedy actions:", greedy_actions)
print("max next Q:", q_values.max(axis=1))
"""),
        md(r"""
        ## 3. 广播：方便，也危险

        `(B,A)` 加 `(B,1)` 会把每个样本的标量复制到所有动作；加 `(B,)` 通常无法表达你想要的语义。先在纸上写 shape，再写代码。
        """),
        code(r"""
advantages = np.array([[1.0, 2.0], [-1.0, 3.0]])
values = np.array([[10.0], [20.0]])
dueling_q = values + advantages - advantages.mean(axis=1, keepdims=True)
print(dueling_q)
assert np.allclose(dueling_q.mean(axis=1), values[:, 0])
"""),
        md(r"""
        ## 4. 随机数与可复现

        推荐使用 `np.random.default_rng(seed)`，并把生成器显式传入实验边界。相同 seed 是调试工具，不代表一次运行就足以得出结论。
        """),
        code(r"""
rng_a = np.random.default_rng(42)
rng_b = np.random.default_rng(42)
sample_a = rng_a.normal(size=4)
sample_b = rng_b.normal(size=4)
print(sample_a)
assert np.allclose(sample_a, sample_b)
"""),
        md(r"""
        ## 练习与过关标准

        **练习：**给定 `q.shape == (4,3)` 和 `actions == [2,0,1,2]`，不用循环取出四个已选动作的 Q 值。

        <details><summary>答案</summary>`q[np.arange(4), actions]`。结果 shape 应为 `(4,)`。</details>

        **过关：**看到数组运算能说出输入、输出 shape；会使用 `reshape`、布尔 mask、`axis`、矩阵乘法和 action indexing。

        **官方选读：**[NumPy absolute basics](https://numpy.org/doc/stable/user/absolute_beginners)；需要更系统时读 [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)。
        """),
    ]


def matplotlib_cells() -> list[dict]:
    return [
        md(r"""
        # Prerequisite 03 · Matplotlib：读懂训练曲线

        [上一章：NumPy](Prerequisite02_NumPy.ipynb) · [下一章：数学](Prerequisite04_Math.ipynb)

        **目标：**画原始回报、移动平均、多随机种子均值和置信带，并知道曲线不能单独证明算法优劣。
        """),
        code(r"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
episodes = np.arange(1, 201)
raw_return = 1 - np.exp(-episodes/55) + rng.normal(0, .22, len(episodes))
window = 20
smooth = np.convolve(raw_return, np.ones(window)/window, mode="valid")

plt.figure(figsize=(7, 3.3))
plt.plot(episodes, raw_return, alpha=.25, label="raw return")
plt.plot(episodes[window-1:], smooth, linewidth=2, label=f"MA({window})")
plt.xlabel("episode"); plt.ylabel("return"); plt.title("Raw signal and moving average")
plt.grid(alpha=.25); plt.legend(); plt.show(); plt.close()
"""),
        md(r"""
        ## 1. 移动平均做了什么

        它降低可视噪声，但会引入滞后，且不能增加数据。窗口越大越平滑，也越容易掩盖突然坍缩。正式报告应说明窗口大小，并保留原始数据。
        """),
        code(r"""
# 五个 seed 才是五次独立实验；200 个 episode 不是 200 个独立 seed。
runs = np.stack([
    1 - np.exp(-episodes/(45 + 4*seed)) + np.random.default_rng(seed).normal(0, .18, len(episodes))
    for seed in range(5)
])
mean = runs.mean(axis=0)
sem = runs.std(axis=0, ddof=1) / np.sqrt(len(runs))

plt.figure(figsize=(7, 3.3))
plt.plot(episodes, mean, label="5-seed mean")
plt.fill_between(episodes, mean-1.96*sem, mean+1.96*sem, alpha=.2, label="95% CI")
plt.xlabel("episode"); plt.ylabel("return"); plt.grid(alpha=.25); plt.legend();
plt.show(); plt.close()
print("final mean:", round(mean[-20:].mean(), 3))
"""),
        md(r"""
        ## 2. 曲线阅读顺序

        1. 横轴是什么：episode、环境帧、gradient step 还是 wall-clock？
        2. 任务指标是什么：return、success rate、episode length？
        3. 优化指标是否对应：loss、entropy、KL、Q scale？
        4. 有几个 seed，阴影代表标准差、标准误还是置信区间？
        5. 曲线是训练策略还是关闭探索后的独立评估？

        **失败案例：**只画最好 seed；平滑窗口大到遮住坍缩；不同算法使用不同样本预算；只画 loss 不画任务回报。
        """),
        md(r"""
        ## 练习与过关标准

        **练习：**把窗口改为 5、50，解释变化；然后让一个 seed 在第 150 回合坍缩，观察均值和置信带。

        **过关：**能独立画 raw + moving average；能画多 seed 均值与置信带；能指出一张图还缺少哪些实验信息。

        **官方选读：**[Matplotlib tutorials](https://matplotlib.org/stable/tutorials/index.html)，优先看 Quick start 和 Pyplot tutorial。
        """),
    ]


def math_cells() -> list[dict]:
    return [
        md(r"""
        # Prerequisite 04 · 数学与概率：只补 RL 必需部分

        [上一章：Matplotlib](Prerequisite03_Matplotlib.ipynb) · [下一章：概率与采样](Prerequisite04B_Probability.ipynb)

        **目标：**理解向量、矩阵、期望、方差、导数、链式法则、log、softmax 和交叉熵，不追求严格数学证明。
        """),
        md(r"""
        ## 1. 向量、矩阵与加权和

        神经网络线性层是 $y=xW+b$；离散期望是加权和 $\mathbb E[X]=\sum_xp(x)x$。Bellman 方程也是“奖励 + 下一状态价值的概率加权和”。
        """),
        code(r"""
import numpy as np

x = np.array([2.0, -1.0])
W = np.array([[1.0, .5, -.2], [.3, -1.0, .8]])
b = np.array([.1, .2, .3])
print("linear output:", x @ W + b)

outcomes = np.array([-1.0, 0.0, 2.0])
probabilities = np.array([.2, .5, .3])
expectation = np.sum(probabilities * outcomes)
variance = np.sum(probabilities * (outcomes - expectation)**2)
print("E[X], Var[X]:", expectation, variance)
"""),
        md(r"""
        ## 2. 导数与链式法则

        导数描述“参数微小变化让目标怎样变化”。若 $a=wx$、$L=(a-y)^2$：

        $$\frac{\partial L}{\partial w}=2(a-y)x$$

        这就是把 $\partial L/\partial a$ 与 $\partial a/\partial w$ 相乘。深度学习的反向传播只是对大型计算图重复链式法则。
        """),
        code(r"""
x, y, w = 3.0, 7.0, 1.5
analytic = 2 * (w*x - y) * x
eps = 1e-5
loss = lambda weight: (weight*x - y)**2
numeric = (loss(w+eps) - loss(w-eps)) / (2*eps)
print("analytic / numerical gradient:", analytic, numeric)
assert abs(analytic - numeric) < 1e-6
"""),
        md(r"""
        ## 3. log、softmax 与交叉熵

        softmax 把 logits 变成和为 1 的概率：

        $$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}$$

        为数值稳定先减去最大 logit。分类交叉熵 $-\log p_y$ 会强烈惩罚给正确类别极低概率；策略梯度中的 `log_prob` 也来自对概率乘积取 log。
        """),
        code(r"""
logits = np.array([2.0, 1.0, -1.0])
stable = logits - logits.max()
probs = np.exp(stable) / np.exp(stable).sum()
target = 0
cross_entropy = -np.log(probs[target])
print("probabilities:", np.round(probs, 4), "sum:", probs.sum())
print("cross-entropy:", round(cross_entropy, 4))
"""),
        md(r"""
        ## 练习与过关标准

        **练习：**手算分布 `[0.25, 0.75]` 上变量 `[0, 2]` 的期望和方差；解释为什么方差不是“平均误差”。

        <details><summary>答案</summary>期望为 1.5；方差为 $0.25(0-1.5)^2+0.75(2-1.5)^2=0.75$。</details>

        **过关：**能读懂矩阵乘法的 shape；能用期望解释随机回报；能说出梯度符号决定更新方向；能计算 softmax 与 cross-entropy。
        """),
    ]


def ml_cells() -> list[dict]:
    return [
        md(r"""
        # Prerequisite 05 · 机器学习基础：训练不是背 API

        [上一章：概率与采样](Prerequisite04B_Probability.ipynb) · [下一章：神经网络从零开始](Prerequisite05B_Neural_Network_From_Scratch.ipynb)

        **目标：**理解模型、loss、优化、训练/验证/测试、过拟合和 baseline。RL 虽没有现成标签，但同样依赖这些训练原则。
        """),
        md(r"""
        ## 1. 监督学习的最小闭环

        给定输入 $x$ 和目标 $y$，模型产生 $\hat y=f_\theta(x)$，loss 衡量误差，梯度下降更新参数：

        $$\theta\leftarrow\theta-\alpha\nabla_\theta L(\theta)$$

        学习率 $\alpha$ 太小进展慢，太大会震荡或发散。RL 中 target 来自 reward 和 bootstrap，而不是人工标签，但优化器看到的仍是 loss。
        """),
        code(r"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)
x = rng.uniform(-2, 2, 160)
y = 2.5*x - .7 + rng.normal(0, .35, len(x))
indices = rng.permutation(len(x))
train_idx, val_idx = indices[:120], indices[120:]
w, b, lr = 0.0, 0.0, .08
train_loss, val_loss = [], []
for _ in range(100):
    error = w*x[train_idx] + b - y[train_idx]
    w -= lr * 2*np.mean(error*x[train_idx])
    b -= lr * 2*np.mean(error)
    train_loss.append(np.mean((w*x[train_idx]+b-y[train_idx])**2))
    val_loss.append(np.mean((w*x[val_idx]+b-y[val_idx])**2))

plt.figure(figsize=(7, 3.2)); plt.plot(train_loss, label="train"); plt.plot(val_loss, label="validation")
plt.xlabel("epoch"); plt.ylabel("MSE"); plt.grid(alpha=.25); plt.legend(); plt.show(); plt.close()
print("w/b:", round(w, 3), round(b, 3), "validation MSE:", round(val_loss[-1], 3))
"""),
        md(r"""
        ## 2. 为什么要分 train / validation / test

        - **train：**更新参数。
        - **validation：**选择模型和超参数。
        - **test：**所有决定完成后，只用于一次最终评估。

        在 test 上反复调参，相当于把 test 变成 validation。股票、文本世界和网页任务尤其容易因时间或相似样本泄漏而得到虚高结果。
        """),
        md(r"""
        ## 3. 过拟合、归一化和 baseline

        - 训练 loss 继续下降、validation loss 上升，是典型过拟合信号。
        - 归一化统计量只能从训练数据估计，再应用到 validation/test。
        - 更复杂模型必须和随机、规则或简单线性 baseline 比较。
        - 公平比较要固定数据、预算、指标与随机种子集合。

        RL 的训练分布会随策略变化，因此比监督学习更难；这些原则不是可以跳过，反而更重要。
        """),
        code(r"""
# 同一个错误尺度下，比较三个学习率的梯度下降轨迹。
def optimize(lr, steps=25):
    value, history = 4.0, []
    for _ in range(steps):
        value -= lr * 2*value  # d(value^2)/dvalue = 2*value
        history.append(value**2)
    return history

for lr in (.02, .2, 1.1):
    history = optimize(lr)
    print(f"lr={lr}: first={history[0]:.3f}, last={history[-1]:.3f}")
"""),
        md(r"""
        ## 练习与过关标准

        **练习：**解释为什么不能先对全部数据做标准化，再切分训练/测试；设计一个只比较学习率的公平实验。

        **过关：**能解释 loss 与任务指标的区别；能正确划分数据；能识别过拟合和泄漏；知道一次只改变一个因素。

        **选读：**[scikit-learn Getting Started](https://scikit-learn.org/stable/getting_started.html) 用于理解 fit/predict、预处理与模型评估；[Dive into Deep Learning 的线性模型部分](https://d2l.ai/chapter_linear-classification/)用于补充回归、分类和训练过程。
        """),
    ]


def pytorch_cells() -> list[dict]:
    return [
        md(r"""
        # Prerequisite 06 · PyTorch：从 tensor 到完整训练循环

        [上一章：神经网络从零开始](Prerequisite05B_Neural_Network_From_Scratch.ipynb) · [下一章：卷积网络与序列模型](Prerequisite06B_CNN_and_Sequences.ipynb)

        **目标：**理解 tensor、autograd、`nn.Module`、optimizer、`detach/no_grad`、device 和概率分布；独立写出一个小型训练循环。
        """),
        md(r"""
        ## 1. Tensor 与 NumPy 的联系

        Tensor 同样有 shape、dtype 和广播，还能记录计算图并在 CPU/GPU 上计算。RL 中通常把 NumPy observation 转成 float tensor，再送入网络。
        """),
        code(r"""
import torch

torch.manual_seed(6)
states = torch.tensor([[.1, .2, -.3], [0., .4, .5]], dtype=torch.float32)
linear = torch.nn.Linear(3, 2)
q_values = linear(states)
actions = torch.tensor([1, 0])
chosen_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
print("Q shape:", tuple(q_values.shape), "chosen Q shape:", tuple(chosen_q.shape))
"""),
        md(r"""
        ## 2. Autograd：计算图上的链式法则

        `loss.backward()` 把梯度累积到参数的 `.grad`；`optimizer.step()` 才真正修改参数。默认梯度会累加，所以每轮先 `zero_grad()`。
        """),
        code(r"""
w = torch.tensor(1.5, requires_grad=True)
x, y = torch.tensor(3.0), torch.tensor(7.0)
loss = (w*x - y)**2
loss.backward()
print("loss:", loss.item(), "dL/dw:", w.grad.item())
assert abs(w.grad.item() - 2*(1.5*3-7)*3) < 1e-6
"""),
        md(r"""
        ## 3. 完整训练循环

        顺序固定为：准备 batch → forward → loss → `zero_grad` → `backward` → `step`。评估时使用 `model.eval()` 与 `torch.no_grad()`。
        """),
        code(r"""
x = torch.linspace(-1, 1, 100).unsqueeze(1)
y = 2.5*x - .4 + .08*torch.randn_like(x)
model = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=.03)
history = []
for _ in range(120):
    prediction = model(x)
    loss = torch.nn.functional.mse_loss(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    history.append(loss.item())

model.eval()
with torch.no_grad():
    final_mse = torch.nn.functional.mse_loss(model(x), y).item()
print("first/final loss:", round(history[0], 4), round(final_mse, 4))
assert final_mse < history[0] * .1
"""),
        md(r"""
        ## 4. detach、no_grad 与概率分布

        - `detach()` 返回共享数据但不继续追踪当前计算图，常用于 TD target 或 advantage target。
        - `no_grad()` 在整个代码块中关闭梯度，适合 target network 和评估。
        - `torch.distributions.Categorical/Normal` 提供 `sample`、`log_prob`、`entropy`，是策略梯度的基础。
        """),
        code(r"""
logits = torch.tensor([.2, 1.0, -.4], requires_grad=True)
distribution = torch.distributions.Categorical(logits=logits)
action = distribution.sample()
objective = -distribution.log_prob(action) * torch.tensor(1.5)
objective.backward()
print("sample/log_prob/entropy:", action.item(), distribution.log_prob(action).item(), distribution.entropy().item())
print("logit gradient:", logits.grad)
"""),
        md(r"""
        ## 初学者常见坑与过关标准

        - 把 batch 维 `squeeze` 掉，单样本能跑、多样本失败。
        - 忘记 `zero_grad()`；对 target 路径忘记 `detach/no_grad()`。
        - NumPy 默认 float64，网络参数通常 float32，dtype 不一致。
        - 保存整个临时计算图导致内存持续增长。

        **练习：**把回归模型改成一层线性网络；比较 SGD 和 Adam，但保持数据、初始化、步数一致。

        **过关：**能不看模板写出完整训练循环；能解释何时使用 `detach`；会检查 tensor shape、dtype、device 和 gradient。

        **官方选读：**[PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)，依次看 Tensors、Build Model、Autograd、Optimization、Save/Load。
        """),
    ]


def gym_cells() -> list[dict]:
    return [
        md(r"""
        # Prerequisite 07 · Gymnasium：环境交互契约

        [上一章：卷积网络与序列模型](Prerequisite06B_CNN_and_Sequences.ipynb) · [下一章：从多臂老虎机走进 RL](Prerequisite07B_Bandit_Bridge.ipynb)

        **目标：**正确使用 `make/reset/step`，理解 spaces、episode、`terminated/truncated`、seed 和 wrapper。Gymnasium 是接口，不是强化学习算法。
        """),
        md(r"""
        ## 1. 最小随机 agent

        `reset()` 返回 observation 和 info；`step()` 返回五项。`terminated` 表示任务 MDP 的终止，`truncated` 通常表示时间限制或外部截断。
        """),
        code(r"""
import gymnasium as gym

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=7)
total_reward = 0.0
for step in range(500):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break
env.close()
print("observation shape:", observation.shape)
print("episode steps/return/end flags:", step+1, total_reward, terminated, truncated)
"""),
        md(r"""
        ## 2. Spaces 是数据契约

        - `Discrete(n)`：整数动作 `0..n-1`。
        - `Box(low, high, shape)`：连续向量或图像。
        - observation/action 必须满足对应 space 的 shape、dtype 与范围。

        wrapper 可以变换 observation、action 或 reward，但必须同步更新 space，并保持 reset/step 契约。
        """),
        code(r"""
env = gym.make("CartPole-v1")
print("observation space:", env.observation_space)
print("action space:", env.action_space)
obs, _ = env.reset(seed=123)
assert env.observation_space.contains(obs)
assert env.action_space.contains(0)
env.close()
"""),
        md(r"""
        ## 3. 结束 episode 与停止 bootstrap 不是同一问题

        采样循环通常在 `terminated or truncated` 时 reset。但构造价值 target 时：

        - `terminated=True`：任务真正结束，通常不 bootstrap。
        - `truncated=True`：只是时间限制，若下一状态仍有价值，通常应该 bootstrap。

        具体仍要检查环境语义和数据采集方式；不要把旧教程中的单一 `done` 无脑复制到新 API。
        """),
        code(r"""
def td_target(reward, next_value, gamma, terminated):
    return reward if terminated else reward + gamma*next_value

print("true terminal target:", td_target(1.0, 10.0, .9, True))
print("time-limit target:", td_target(1.0, 10.0, .9, False))
assert td_target(1.0, 10.0, .9, True) == 1.0
"""),
        md(r"""
        ## 练习与过关标准

        **练习：**运行 20 个随机 CartPole episode，记录 return 的均值和标准差；分别统计 `terminated` 与 `truncated` 次数。

        **过关：**能写完整 episode loop；会检查 space；能解释 reset 的 seed；能区分 episode reset 条件与 bootstrap mask。

        **官方选读：**[Gymnasium Basic Usage](https://gymnasium.farama.org/introduction/basic_usage/)；遇到旧版 `done` 代码时读 [Gymnasium Migration Guide](https://gymnasium.farama.org/main/introduction/migration_guide/)。
        """),
    ]


def readiness_cells() -> list[dict]:
    return [
        md(r"""
        # Prerequisite 08 · 强化学习入学测试

        [上一章：从多臂老虎机走进 RL](Prerequisite07B_Bandit_Bridge.ipynb) · [预备篇导航](Prerequisite00_Index.ipynb) · [进入 Chapter01](../02_chapters/Chapter01.ipynb)

        闭卷完成。不会的题不是失败，而是告诉你应该回到哪个预备章。建议概念题达到 12/15、代码测试全部通过后进入 RL 主线。
        """),
        md(r"""
        ## 概念题

        1. list、tuple、dict 各适合保存什么？
        2. `(B,F) @ (F,A)` 的输出 shape 是什么？
        3. `axis=1` 在 `(B,A)` 的 Q 值上表示什么？
        4. 移动平均窗口变大有什么代价？
        5. 期望和方差分别描述什么？
        6. 梯度下降为什么沿负梯度更新？
        7. train、validation、test 为什么不能混用？
        8. `zero_grad/backward/step` 各做什么？
        9. `detach` 和 `no_grad` 的使用场景是什么？
        10. `terminated` 和 `truncated` 为什么影响 bootstrap？
        11. 为什么用样本平均可以估计期望？样本数变成 4 倍，误差大约变成多少？
        12. $\pi(a\mid s)$ 中竖线左右分别是什么？为什么策略梯度要用 log 概率？
        13. 为什么神经网络必须有非线性激活？Q 表和 Q 网络有什么区别？
        14. `(4,84,84)` 的输入经过 `Conv2d(4,32,8,stride=4)` 后 shape 是多少？为什么 Atari 要堆叠 4 帧？
        15. 用一句话解释“新估计 = 旧估计 + 步长 × (目标 − 旧估计)”，并说出 ε=0 的贪心策略可能出什么问题。
        """),
        md(r"""
        <details><summary>参考要点</summary>

        1. 可变序列、固定记录、键值配置；2. `(B,A)`；3. 对每个样本的动作维操作；4. 更滞后并可能掩盖突变；5. 中心趋势与离散程度；6. 负梯度是局部下降最快方向；7. 防止选择偏差和泄漏；8. 清梯度、反传、更新；9. 切断局部图与关闭整个块的梯度；10. 真终止没有未来价值，时间截断可能仍有；11. 大数定律，误差约减半（$1/\sqrt n$）；12. 左边是结果（动作），右边是条件（状态）；乘积变求和、避免下溢、便于求导；13. 多层线性仍是线性；Q 表逐格存储、Q 网络用参数近似并能泛化；14. `(32,20,20)`，单帧看不出速度和方向；15. 用误差修正估计；可能过早锁定次优动作、永远不再探索。

        </details>
        """),
        md(r"""
        ## 代码题 1：NumPy batch 与动作索引
        """),
        code(r"""
import numpy as np

q = np.array([[1., 4., 2.], [5., 0., 3.], [2., 1., 6.]])
actions = np.array([1, 0, 2])
chosen = q[np.arange(len(actions)), actions]
assert np.allclose(chosen, [4., 5., 6.])
assert np.array_equal(q.argmax(axis=1), actions)
print("NumPy test passed:", chosen)
"""),
        md(r"""
        ## 代码题 2：PyTorch 训练循环
        """),
        code(r"""
import torch

torch.manual_seed(8)
x = torch.linspace(-1, 1, 80).unsqueeze(1)
y = -1.7*x + .3
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=.15)
first_loss = None
for _ in range(80):
    loss = torch.nn.functional.mse_loss(model(x), y)
    if first_loss is None: first_loss = loss.item()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
assert loss.item() < first_loss*.01
print("PyTorch test passed; loss:", round(loss.item(), 6))
"""),
        md(r"""
        ## 代码题 3：终止语义
        """),
        code(r"""
def target(reward, next_value, gamma, terminated):
    return reward + (0.0 if terminated else gamma*next_value)

assert target(1, 10, .9, True) == 1
assert target(1, 10, .9, False) == 10
print("terminal-mask test passed")
"""),
        md(r"""
        ## 代码题 4：CNN 输出 shape 与 Q-learning 更新
        """),
        code(r"""
import numpy as np
import torch
import torch.nn as nn

conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU())
with torch.no_grad():
    out = conv(torch.zeros(2, 4, 84, 84))
assert tuple(out.shape) == (2, 64, 9, 9), out.shape

Q = np.zeros((2, 2)); Q[1] = [0.0, 4.0]
s, a, r, s_next, alpha, gamma = 0, 1, 1.0, 1, 0.5, 0.9
Q[s, a] += alpha * (r + gamma * Q[s_next].max() - Q[s, a])
assert np.isclose(Q[0, 1], 2.3)
print("CNN shape 与 Q-learning 更新测试通过")
"""),
        md(r"""
        ## 判定

        - 概念 0–8：按顺序完成全部预备篇（包括带 B 的 4 个补课 notebook）。
        - 概念 9–11：只补错题对应章节，再重做代码题。
        - 概念 12–15 且代码全过：进入 [Chapter01](../02_chapters/Chapter01.ipynb)。

        通过后也不用背所有 API。真正重要的是：遇到 shape、梯度和环境问题时，知道怎样检查，而不是把它误认为 RL 算法失效。
        """),
    ]


def insert_before_last(cells: list[dict], extra: list[dict]) -> list[dict]:
    """Insert extra teaching/exercise cells before the closing "过关" cell."""
    return cells[:-1] + extra + cells[-1:]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    materials = {
        "Prerequisite00_Index": index_cells(),
        "Prerequisite01_Python": python_cells(),
        "Prerequisite02_NumPy": insert_before_last(numpy_cells(), numpy_extra()),
        "Prerequisite03_Matplotlib": matplotlib_cells(),
        "Prerequisite04_Math": insert_before_last(math_cells(), math_extra()),
        foundations_probability.NAME: foundations_probability.cells(),
        "Prerequisite05_Machine_Learning": insert_before_last(ml_cells(), ml_extra()),
        foundations_neural_net.NAME: foundations_neural_net.cells(),
        "Prerequisite06_PyTorch": insert_before_last(pytorch_cells(), pytorch_extra()),
        foundations_cnn.NAME: foundations_cnn.cells(),
        "Prerequisite07_Gymnasium": gym_cells(),
        foundations_bandit.NAME: foundations_bandit.cells(),
        "Prerequisite08_Readiness_Check": readiness_cells(),
    }
    for name, cells in materials.items():
        save(name, cells)
    print(f"Generated {len(materials)} prerequisite notebooks")


if __name__ == "__main__":
    main()
