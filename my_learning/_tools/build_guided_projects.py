"""Generate the hands-on environment guide and two fully worked RL projects."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


LEARNING_ROOT = Path(__file__).resolve().parent.parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


def save(folder: str, name: str, cells: list[dict]) -> None:
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
    target = LEARNING_ROOT / folder
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{name}.ipynb").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def environment_cells() -> list[dict]:
    return [
        md(r"""
        # 01 · 环境、Jupyter 与报错阅读

        [返回总目录](../README.md) · [零基础预备篇](../01_prerequisites/Prerequisite00_Index.ipynb)

        这不是算法课，而是避免初学者被工具问题挡住。完成后你应该会：确认解释器与依赖、理解 cell 状态、从 traceback 找到第一处自己的错误，并知道什么时候 Restart Kernel。
        """),
        md(r"""
        ## 1. 当前 Notebook 到底使用哪个 Python？

        终端里的 Python、IDE 里的 Python、Jupyter kernel 可能不是同一个。排查依赖问题时先输出解释器路径和版本，不要立刻重复安装。
        """),
        code(r"""
import platform
import sys

print("Python executable:", sys.executable)
print("Python version:", platform.python_version())
print("Working directory:", __import__("pathlib").Path.cwd())
"""),
        code(r"""
import numpy, matplotlib, torch, gymnasium

versions = {
    "numpy": numpy.__version__,
    "matplotlib": matplotlib.__version__,
    "torch": torch.__version__,
    "gymnasium": gymnasium.__version__,
}
for package, version in versions.items():
    print(f"{package:12s} {version}")
"""),
        md(r"""
        ## 2. Cell 有状态：能运行不等于可复现

        Jupyter 会记住已经执行过的变量。若先运行下面第二格、再回头修改第一格，屏幕上的代码和内存中的值可能不一致。完成一个实验后使用 **Restart Kernel and Run All**；从上到下仍能运行，才算可复现。
        """),
        code(r"""
learning_rate = 0.01
run_name = f"baseline-lr-{learning_rate}"
print(run_name)
"""),
        md(r"""
        ## 3. 怎样阅读 traceback

        1. 先看最后一行：异常类型和直接原因。
        2. 从下往上找第一个属于自己 notebook 的行。
        3. 打印该行输入的 type、shape、dtype、范围。
        4. 做最小复现，而不是一次修改五处。

        常见分类：`NameError` 是变量/执行顺序；`ModuleNotFoundError` 是解释器或环境；`ValueError` 常是 shape；`RuntimeError` 常是 PyTorch dtype/device/计算图。
        """),
        code(r"""
# 用检查函数让错误更早、更清楚地出现。
def check_batch(states, actions):
    assert states.ndim == 2, f"states 应为 (B,F)，实际 {states.shape}"
    assert actions.ndim == 1, f"actions 应为 (B,)，实际 {actions.shape}"
    assert len(states) == len(actions), "batch 大小不一致"

states = numpy.zeros((4, 3), dtype=numpy.float32)
actions = numpy.array([0, 1, 0, 1])
check_batch(states, actions)
print("batch contract passed")
"""),
        md(r"""
        ## 4. 新手工作习惯

        - 一个 notebook 只承担一个明确问题。
        - 参数集中在一个配置 cell，不散落魔法数字。
        - 代码运行前先写预期结果；运行后记录是否符合。
        - 不在调试时升级所有依赖。
        - 修改算法前先保存 baseline、seed 和原始曲线。
        - 遇到异常查看[排错手册](../05_reference/DEBUGGING_PLAYBOOK.md)。

        **过关：**重启 kernel 后 Run All 成功；能说出当前解释器路径；能根据一条 shape 错误定位到自己的代码。
        """),
    ]


def gridworld_cells() -> list[dict]:
    return [
        md(r"""
        # Guided Project 01 · 从零完成 GridWorld Q-learning

        [返回项目目录](README.md) · [对应 Chapter05](../02_chapters/Chapter05.ipynb) · [实验模板](Experiment_Template.ipynb)

        这是第一个必须真正完成的 RL 项目。我们会经历：定义问题 → 验证环境 → 随机基线 → Q-learning → 独立评估 → 曲线诊断 → 故障注入。
        """),
        md(r"""
        ## 1. 先把问题说清楚

        5×5 网格，起点左上、终点右下，中间有障碍。动作是上/下/左/右；每步奖励 -0.02，撞障碍额外 -0.08，到达终点奖励 +1。状态是格子编号，因此 Q 表 shape 为 `(25,4)`。

        $$Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma(1-d)\max_{a'}Q(s',a')-Q(s,a)]$$

        代码中的 `done` 必须和公式里的 $(1-d)$ 对应。
        """),
        code(r"""
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self):
        self.size = 5
        self.start, self.goal = (0, 0), (4, 4)
        self.obstacles = {(1, 1), (2, 1), (3, 3)}
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.reset()

    def encode(self, pos):
        return pos[0] * self.size + pos[1]

    def reset(self):
        self.pos = self.start
        return self.encode(self.pos)

    def step(self, action):
        row, col = self.pos; dr, dc = self.moves[action]
        candidate = (int(np.clip(row+dr, 0, 4)), int(np.clip(col+dc, 0, 4)))
        hit_obstacle = candidate in self.obstacles
        if not hit_obstacle:
            self.pos = candidate
        done = self.pos == self.goal
        reward = 1.0 if done else (-0.10 if hit_obstacle else -0.02)
        return self.encode(self.pos), reward, done

env = GridWorld()
assert env.reset() == 0 and env.encode(env.goal) == 24
print("states/actions:", 25, 4, "obstacles:", env.obstacles)
"""),
        md(r"""
        ## 2. 环境测试早于训练

        在学习算法之前，用固定动作检查边界、障碍和终点。如果环境错了，agent 可能非常成功地优化一个错误任务。
        """),
        code(r"""
env.reset()
state, reward, done = env.step(0)  # 在顶边向上
assert state == 0 and reward == -0.02 and not done
env.reset(); env.step(3)           # 到 (0,1)
state, reward, done = env.step(1)  # 向下撞 (1,1)
assert state == 1 and reward == -0.10 and not done
print("boundary and obstacle tests passed")
"""),
        md(r"""
        ## 3. 随机基线

        baseline 回答“学习算法至少要超过谁”。这里记录 success rate、return 和 episode length，而不是只看是否偶尔到达终点。
        """),
        code(r"""
def evaluate_policy(policy, episodes=200, seed=0):
    rng = np.random.default_rng(seed)
    returns, lengths, successes = [], [], []
    for _ in range(episodes):
        state, total = env.reset(), 0.0
        for step in range(100):
            action = policy(state, rng)
            state, reward, done = env.step(action)
            total += reward
            if done: break
        returns.append(total); lengths.append(step+1); successes.append(done)
    return np.asarray(returns), np.asarray(lengths), np.asarray(successes)

random_result = evaluate_policy(lambda state, rng: int(rng.integers(4)))
print("random success / return / length:",
      random_result[2].mean(), random_result[0].mean(), random_result[1].mean())
"""),
        md(r"""
        ## 4. 训练 Q-learning

        epsilon 负责收集不同动作的数据；学习率控制一次 TD error 改多少；折扣因子控制未来终点奖励能传多远。先固定三个参数建立基线，再逐个调整。
        """),
        code(r"""
def train_q_learning(seed=0, episodes=700, alpha=.2, gamma=.97, explore=True, initial_action0_bias=False):
    rng = np.random.default_rng(seed)
    q = np.zeros((25, 4), dtype=np.float64)
    if initial_action0_bias:
        q[:, 0] = 1.0
    returns, successes, lengths = [], [], []
    for episode in range(episodes):
        state, total, done = env.reset(), 0.0, False
        epsilon = max(.05, 1.0 - episode/500) if explore else 0.0
        for step in range(100):
            action = int(rng.integers(4)) if rng.random() < epsilon else int(q[state].argmax())
            next_state, reward, done = env.step(action)
            target = reward if done else reward + gamma*q[next_state].max()
            q[state, action] += alpha*(target - q[state, action])
            state, total = next_state, total + reward
            if done: break
        returns.append(total); successes.append(done); lengths.append(step+1)
    return q, np.asarray(returns), np.asarray(successes), np.asarray(lengths)

q, returns, successes, lengths = train_q_learning(seed=10)
window = 40
smooth_return = np.convolve(returns, np.ones(window)/window, mode="valid")
smooth_success = np.convolve(successes, np.ones(window)/window, mode="valid")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].plot(smooth_return); axes[0].set_title("moving-average return")
axes[1].plot(smooth_success); axes[1].set_title("moving success rate")
for ax in axes: ax.set_xlabel("episode"); ax.grid(alpha=.25)
plt.tight_layout(); plt.show(); plt.close()
print("last-100 success/return/length:", successes[-100:].mean(), returns[-100:].mean(), lengths[-100:].mean())
"""),
        md(r"""
        ## 5. 独立评估与路径检查

        评估时关闭 epsilon，使用新的 episode，不再更新 Q。成功率高仍不够：还要查看路径是否循环、是否明显长于最短路。
        """),
        code(r"""
greedy = lambda state, rng: int(q[state].argmax())
eval_returns, eval_lengths, eval_successes = evaluate_policy(greedy, episodes=100, seed=999)

state, path = env.reset(), [env.pos]
for _ in range(30):
    state, _, done = env.step(int(q[state].argmax()))
    path.append(env.pos)
    if done: break
print("evaluation success / mean length:", eval_successes.mean(), eval_lengths.mean())
print("greedy path:", path)
assert eval_successes.mean() > .95 and path[-1] == env.goal
"""),
        md(r"""
        ## 6. 故障注入：完全不探索

        如果初始估计偏向动作 0，同时 epsilon 从一开始就是 0，agent 会在顶边反复向上，永远看不到终点奖励。这不是学习率问题，而是数据覆盖问题。
        """),
        code(r"""
q_bad, bad_returns, bad_successes, _ = train_q_learning(
    seed=10, episodes=5, explore=False, initial_action0_bias=True
)
print("first-five no-exploration success:", bad_successes.mean())
assert bad_successes.sum() == 0
"""),
        md(r"""
        ## 7. 调参顺序与验收

        1. 固定 seed，确保环境测试和随机基线可信。
        2. 只调 epsilon schedule，先获得稳定成功轨迹。
        3. 比较 learning rate 的稳定性，再调 gamma。
        4. 用 5 个 seed 重复；报告均值和失败 seed。

        **项目验收：**评估成功率 >95%；能从空白写出 TD target；能解释无探索为何失败；能添加一个陷阱格并预测策略变化。

        **下一步：**先完成对应的 [Chapter05](../02_chapters/Chapter05.ipynb)，再进入 [DQN CartPole](Guided_Project02_DQN_CartPole.ipynb)。
        """),
    ]


def dqn_cells() -> list[dict]:
    return [
        md(r"""
        # Guided Project 02 · PyTorch DQN 训练 CartPole

        [返回项目目录](README.md) · [对应 Chapter06](../02_chapters/Chapter06.ipynb) · [DQN 扩展](../02_chapters/Chapter08.ipynb)

        这个项目把预备篇的 NumPy、Matplotlib、PyTorch、Gymnasium 与 DQN 公式真正连起来。目标不是追求最高分，而是理解一条 transition 如何变成一次网络更新。
        """),
        md(r"""
        ## 1. DQN 数据流

        ```text
        observation → online network → epsilon-greedy action → env.step
                    → replay buffer → random batch
                    → online Q(s,a) 与 target y → Huber loss → optimizer
        ```

        使用 Double DQN target：

        $$a^*=\arg\max_a Q_\theta(s',a),\qquad
        y=r+\gamma(1-d)Q_{\theta^-}(s',a^*)$$

        `terminated` 决定 $d$；时间截断结束 episode，但不自动把未来价值设为 0。
        """),
        code(r"""
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from torch import nn

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(1)
device = torch.device("cpu")

env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=SEED)
print("observation/action:", obs.shape, env.action_space)
env.close()
"""),
        md(r"""
        ## 2. 先测随机基线

        如果训练后的 agent 没有明显超过随机动作，就不能说“学会了”。CartPole 随机策略通常只能维持十几到几十步。
        """),
        code(r"""
def random_cartpole(episodes=20, seed=42):
    baseline_env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed); scores = []
    for episode in range(episodes):
        state, _ = baseline_env.reset(seed=seed+episode); total = 0.0
        for _ in range(500):
            state, reward, terminated, truncated, _ = baseline_env.step(int(rng.integers(2)))
            total += reward
            if terminated or truncated: break
        scores.append(total)
    baseline_env.close()
    return np.asarray(scores)

random_scores = random_cartpole()
print("random mean/min/max:", random_scores.mean(), random_scores.min(), random_scores.max())
"""),
        md(r"""
        ## 3. 网络和 replay buffer

        网络输入 `(B,4)`、输出 `(B,2)`。Replay buffer 保存原始 transition；采样时再转换成 tensor，便于检查 shape 和 dtype。
        """),
        code(r"""
class DQN(nn.Module):
    def __init__(self, obs_size=4, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_actions))
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=20_000):
        self.data = deque(maxlen=capacity)
    def append(self, transition): self.data.append(transition)
    def sample(self, batch_size, rng):
        indices = rng.choice(len(self.data), size=batch_size, replace=False)
        columns = list(zip(*(self.data[i] for i in indices)))
        return tuple(np.asarray(column) for column in columns)
    def __len__(self): return len(self.data)

test_net = DQN()
assert test_net(torch.zeros(3, 4)).shape == (3, 2)
print(test_net)
"""),
        md(r"""
        ## 4. 一次 batch 更新：逐行对应公式

        - `gather` 取得实际动作的 $Q(s,a)$。
        - online network 在下一状态选动作。
        - target network 评价该动作。
        - 只有真正 `terminated` 才移除 bootstrap。
        """),
        code(r"""
def update_batch(online, target, optimizer, batch, gamma=.99):
    states, actions, rewards, next_states, terminated = batch
    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=device)

    chosen_q = online(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = online(next_states_t).argmax(dim=1, keepdim=True)
        next_q = target(next_states_t).gather(1, next_actions).squeeze(1)
        expected_q = rewards_t + gamma*(~terminated_t).float()*next_q
    loss = nn.functional.smooth_l1_loss(chosen_q, expected_q)
    optimizer.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return loss.item(), chosen_q.detach().mean().item()
"""),
        md(r"""
        ## 5. 完整训练

        先收集 500 条经验再更新；epsilon 按环境步数下降；target 每 250 步同步。曲线同时记录 return、loss、Q 均值和 epsilon，避免只凭一条曲线判断。
        """),
        code(r"""
def train_dqn(seed=42, episodes=190):
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    online, target = DQN().to(device), DQN().to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=8e-4)
    buffer = ReplayBuffer()
    returns, losses, q_means, epsilons = [], [], [], []
    total_steps = 0
    for episode in range(episodes):
        state, _ = env.reset(seed=seed+episode)
        total = 0.0
        for _ in range(500):
            epsilon = max(.05, 1.0-total_steps/10_000)
            if rng.random() < epsilon:
                action = int(rng.integers(2))
            else:
                with torch.no_grad():
                    action = int(online(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).argmax())
            next_state, reward, terminated, truncated, _ = env.step(action)
            buffer.append((state, action, reward, next_state, terminated))
            state, total, total_steps = next_state, total+reward, total_steps+1
            if len(buffer) >= 500:
                loss, q_mean = update_batch(online, target, optimizer, buffer.sample(64, rng))
                losses.append(loss); q_means.append(q_mean)
            if total_steps % 250 == 0:
                target.load_state_dict(online.state_dict())
            if terminated or truncated: break
        returns.append(total); epsilons.append(epsilon)
    env.close()
    return online, np.asarray(returns), np.asarray(losses), np.asarray(q_means), np.asarray(epsilons)

online, returns, losses, q_means, epsilons = train_dqn()
window = 20
smooth_return = np.convolve(returns, np.ones(window)/window, mode="valid")
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes[0,0].plot(returns, alpha=.25); axes[0,0].plot(np.arange(window-1,len(returns)), smooth_return); axes[0,0].set_title("return")
axes[0,1].plot(losses, alpha=.5); axes[0,1].set_title("Huber loss per update")
axes[1,0].plot(q_means); axes[1,0].set_title("mean chosen Q")
axes[1,1].plot(epsilons); axes[1,1].set_title("epsilon")
for ax in axes.flat: ax.grid(alpha=.25)
plt.tight_layout(); plt.show(); plt.close()
print("last-20 training return:", round(returns[-20:].mean(), 1), "updates:", len(losses))
"""),
        md(r"""
        ## 6. 独立评估

        训练回报包含 epsilon 探索，不能直接当最终性能。评估使用新环境、greedy action、`eval()` 和 `no_grad()`，且不写 replay、不更新参数。
        """),
        code(r"""
def evaluate_dqn(model, episodes=20):
    env = gym.make("CartPole-v1")
    model.eval(); scores = []
    with torch.no_grad():
        for episode in range(episodes):
            state, _ = env.reset(seed=10_000+episode); total = 0.0
            for _ in range(500):
                action = int(model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).argmax())
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated: break
            scores.append(total)
    env.close()
    return np.asarray(scores)

evaluation = evaluate_dqn(online)
print("evaluation mean/min/max:", evaluation.mean(), evaluation.min(), evaluation.max())
assert evaluation.mean() > random_scores.mean() * 2
"""),
        md(r"""
        ## 7. 怎样诊断，而不是盲目调参

        - return 不升：先查 action、reward、terminal mask、target 与 replay 数据。
        - loss 降但 return 不升：target 可能自洽但错误，或评估/探索有问题。
        - Q 持续远超最大可能回报：检查终止 bootstrap、reward scale 和 target 同步。
        - epsilon 已很低但经验池没有成功轨迹：探索下降过快。
        - 单 seed 很好：不能证明稳定；正式报告至少跑 3–5 个 seed。

        **故障练习：**把 terminal mask 删除、target sync 改为每步、epsilon decay 改为 500 步。运行前先写下各自可能出现的曲线。
        """),
        md(r"""
        ## 8. 项目验收与下一步

        你应能不看代码回答：

        1. replay buffer 为什么让 SGD 更合理？
        2. online/target 分别在哪一步出现？
        3. `gather` 的输入输出 shape 是什么？
        4. 为什么 `terminated` 与 `truncated` 分开保存？
        5. 为什么训练 return 和评估 return 不同？

        完成后进入 [Chapter06](../02_chapters/Chapter06.ipynb)，再用 [Chapter08](../02_chapters/Chapter08.ipynb)逐项加入 Double、n-step、Dueling 或 PER。不要一次组合全部扩展。
        """),
    ]


def main() -> None:
    save("00_start_here", "01_Environment_and_Jupyter", environment_cells())
    save("03_guided_projects", "Guided_Project01_GridWorld", gridworld_cells())
    save("03_guided_projects", "Guided_Project02_DQN_CartPole", dqn_cells())
    print("Generated environment guide and 2 guided projects")


if __name__ == "__main__":
    main()
