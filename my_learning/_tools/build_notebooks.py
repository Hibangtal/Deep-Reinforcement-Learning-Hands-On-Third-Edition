"""Generate the beginner-friendly chapter notebooks in this directory.

The notebooks are intentionally compact: every chapter contains the essential
math, a small executable example, links to the full repository programs, and a
separate extension section.  Re-run this file after editing CHAPTERS to rebuild
all notebooks deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from training_content import TRAINING_GUIDE


LEARNING_ROOT = Path(__file__).resolve().parent.parent
ROOT = LEARNING_ROOT.parent
OUT = LEARNING_ROOT / "02_chapters"
REFERENCE_OUT = LEARNING_ROOT / "05_reference"


def md(text: str) -> dict:
    cleaned = dedent(text).strip()
    # Multiline values interpolated into an indented f-string can prevent
    # textwrap.dedent from seeing a common margin.  Remove the template margin
    # line by line so headings/tables are not accidentally rendered as code.
    # Fenced code blocks (pseudo-code) keep their own indentation; stripping it
    # there would flatten nested loops such as "for s: for a:".
    lines, in_fence = [], False
    for line in cleaned.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            line = line.lstrip()
        elif not in_fence and line.startswith("        "):
            line = line[8:]
        lines.append(line)
    cleaned = "\n".join(lines)
    return {"cell_type": "markdown", "metadata": {}, "source": cleaned + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


CHAPTERS = [
    {
        "title": "What Is Reinforcement Learning? / 什么是强化学习",
        "goals": ["分清监督学习与强化学习", "识别 agent、environment、state、action、reward", "读懂 MDP、策略与回报"],
        "map": "智能体观察 $S_t$，按策略 $\\pi(a|s)$ 选出 $A_t$；环境返回 $R_{t+1}$ 与 $S_{t+1}$。学习目标不是眼前奖励，而是长期累计回报。",
        "formula": r"""
### 核心公式

折扣回报：

$$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}$$

状态价值与动作价值：

$$V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s],\qquad Q^\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a]$$

其中 $0\le\gamma<1$ 控制“多看重未来”。MDP 用 $(\mathcal S,\mathcal A,P,R,\gamma)$ 描述；Markov 性表示预测未来只需要当前状态，而不需要完整历史。
""",
        "bridge": "下面先把回报公式直接翻译成倒序循环，再解一个两状态 Markov reward process。倒序计算能把复杂度从重复求和降到一次遍历。",
        "codes": [r"""
import numpy as np

def discounted_returns(rewards, gamma=0.9):
    # 返回每个时刻的 G_t；rewards[i] 对应 R_{i+1}。
    returns = np.zeros(len(rewards), dtype=float)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns

rewards = [1, 0, 2]
print("G_t:", discounted_returns(rewards, gamma=0.9))

# MRP 的 Bellman 方程 V = r + gamma P V，可整理为线性方程求解。
P = np.array([[0.7, 0.3], [0.2, 0.8]])
r = np.array([1.0, 2.0])
gamma = 0.9
V = np.linalg.solve(np.eye(2) - gamma * P, r)
print("V:", V)
"""],
        "links": [],
        "tips": ["先确定状态是否真的包含决策所需信息；缺失历史时问题可能是 POMDP。", "奖励是任务定义，价值是策略下对未来奖励的预测，两者不要混用。", r"$\gamma$ 越大不等于一定越好：有效时间跨度大约为 $1/(1-\gamma)$。"],
        "exercise": r"把奖励改为 `[0, 0, 10]`，分别使用 $\gamma=0,0.5,0.99$；解释第一个时刻的回报为何变化。",
        "extension": "持续任务常用平均奖励或 differential return；观测不满足 Markov 性时，可堆叠历史、使用 RNN，或显式建模 belief state。",
    },
    {
        "title": "OpenAI Gym API and Gymnasium / 环境接口",
        "goals": ["理解 Gymnasium 的 space、reset、step", "正确处理 terminated 与 truncated", "用 wrapper 改造环境而不改环境源码"],
        "map": "`reset()` 给出初始观测；`step(action)` 返回 `(obs, reward, terminated, truncated, info)`。一个 episode 在 `terminated or truncated` 时结束，但二者对 bootstrap 的含义不同。",
        "formula": r"""
### 交互循环与终止语义

策略采样：$A_t\sim\pi(\cdot\mid S_t)$，环境采样：$S_{t+1}\sim P(\cdot\mid S_t,A_t)$。

- `terminated=True`：MDP 的真正终止状态，通常 bootstrap 值为 0。
- `truncated=True`：时间上限或外部条件截断；若下一状态仍有定义，训练目标通常仍应 bootstrap。

因此 TD mask 通常应是 $m_t=1-\mathbb 1[\text{terminated}]$，而不是简单使用 `done`。
""",
        "bridge": "下面的循环限制最多 200 步，即使环境版本或随机策略发生变化也会快速结束。关闭环境是 notebook 中容易漏掉的小细节。",
        "codes": [r"""
try:
    import gymnasium as gym
except ImportError:
    gym = None

if gym is None:
    print("未安装 gymnasium；按项目 requirements.txt 安装后再运行本单元。")
else:
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    for step in range(200):
        action = env.action_space.sample()  # 随机策略只是接口演示
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
    print({"steps": step + 1, "return": total_reward, "last_obs_shape": obs.shape})
"""],
        "links": ["../Chapter02/01_agent_anatomy.py", "../Chapter02/02_cartpole_random.py", "../Chapter02/03_random_action_wrapper.py", "../Chapter02/04_cartpole_random_monitor.py"],
        "tips": ["固定 `reset(seed=...)` 只是复现实验的一部分，还要固定 NumPy/PyTorch 随机种子。", "训练时不要开实时渲染，它常常成为主要性能瓶颈。", "wrapper 的顺序会改变语义，尤其是奖励裁剪、帧堆叠与时间限制。"],
        "exercise": "写一个 `RewardWrapper` 把每步奖励乘 0.1，并检查 episode 长度是否保持不变。",
        "extension": "向量环境把多个环境的观测拼成 batch；录制视频需要 `render_mode='rgb_array'` 并使用 `RecordVideo`。",
    },
    {
        "title": "Deep Learning with PyTorch / PyTorch 深度学习基础",
        "goals": ["掌握 tensor、shape、device 与 dtype", "理解 autograd 与优化循环", "用 `nn.Module` 连接 RL 的函数逼近"],
        "map": r"神经网络 $f_\theta(x)$ 是带参数的可微函数。前向计算产生预测与损失，autograd 反向累积梯度，optimizer 更新参数。",
        "formula": r"""
### 从损失到参数更新

均方误差：$L(\theta)=\frac1N\sum_i(f_\theta(x_i)-y_i)^2$。

梯度下降：$\theta\leftarrow\theta-\alpha\nabla_\theta L(\theta)$。

链式法则把最终损失的梯度传回每层。PyTorch 的 `.backward()` **累积**到 `.grad`，所以每轮更新前需要 `optimizer.zero_grad()`。
""",
        "bridge": "这个线性回归例子完整展示 RL 训练也会复用的四步：前向、算损失、反向、更新。",
        "codes": [r"""
import torch
from torch import nn

torch.manual_seed(42)
x = torch.linspace(-1, 1, 100).unsqueeze(1)
y = 3 * x + 0.5 + 0.05 * torch.randn_like(x)

model = nn.Linear(1, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

for _ in range(200):
    prediction = model(x)
    loss = loss_fn(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("weight, bias:", model.weight.item(), model.bias.item())
print("final loss:", loss.item())
"""],
        "links": ["../Chapter03/01_modules.py", "../Chapter03/02_tensorboard.py", "../Chapter03/03_atari_gan.py", "../Chapter03/04_atari_gan_ignite.py"],
        "tips": ["始终在关键边界打印或断言 shape；多数 RL bug 是 batch 维度错误。", "评估时使用 `model.eval()` 与 `torch.no_grad()`。", "把 tensor 转 NumPy 前通常先 `.detach().cpu()`。"],
        "exercise": "把模型改成 `Linear(1, 16) → ReLU → Linear(16, 1)`，比较收敛速度和最终误差。",
        "extension": "TensorBoard 用于曲线监控；Ignite 封装事件循环；GAN 示例用于练习更复杂的训练组织，而不是本章 RL 主线。",
    },
    {
        "title": "The Cross-Entropy Method / 交叉熵方法",
        "goals": ["理解无梯度规划式的 elite 筛选", "把 episode 回报转成监督学习数据", "认识 CEM 在稀疏奖励下的局限"],
        "map": "采样一批 episode → 按回报选前若干百分位 → 用 elite 的 `(state, action)` 做分类训练 → 重复。这里的“交叉熵”指分类损失，不是直接计算策略梯度。",
        "formula": r"""
### Elite 筛选与分类目标

阈值 $q$ 是 batch 回报的百分位数；elite 集合 $E=\{\tau_i:R(\tau_i)\ge q\}$。

$$L(\theta)=-\frac1{|E|}\sum_{(s,a)\in E}\log\pi_\theta(a\mid s)$$

策略逐渐提高 elite 动作的概率。由于只看整条轨迹排名，奖励稀疏或 episode 很长时，信号会很弱。
""",
        "bridge": "先用一个小 batch 看 percentile 怎样留下高回报样本，再计算分类损失。",
        "codes": [r"""
import numpy as np
import torch
import torch.nn.functional as F

returns = np.array([3, 10, 6, 1, 8], dtype=float)
threshold = np.percentile(returns, 70)
elite_ids = np.flatnonzero(returns >= threshold)
print("threshold:", threshold, "elite episodes:", elite_ids.tolist())

# 假设 elite 轨迹收集到 4 个状态，其正确动作是 actions。
logits = torch.tensor([[2., 0.], [0., 2.], [1., 0.], [0., 1.]], requires_grad=True)
actions = torch.tensor([0, 1, 0, 1])
loss = F.cross_entropy(logits, actions)
loss.backward()
print("loss:", loss.item(), "gradient shape:", tuple(logits.grad.shape))
"""],
        "links": ["../Chapter04/01_cartpole.py", "../Chapter04/02_frozenlake_naive.py", "../Chapter04/03_frozenlake_tweaked.py", "../Chapter04/04_frozenlake_nonslippery.py"],
        "tips": ["percentile 太高会导致样本过少，太低则学习信号不纯。", "保留上一轮优秀轨迹能帮助稀疏奖励，但要避免数据越来越陈旧。", "先用非滑动 FrozenLake 检查逻辑，再增加随机转移。"],
        "exercise": "把 percentile 从 70 改为 50 和 90，观察 elite 数量；解释 batch 大小时阈值稳定性的差异。",
        "extension": "CEM 也可直接搜索参数分布：采样参数、评估、用 elite 重新估计均值和方差；这与第 17 章黑盒优化相呼应。",
    },
    {
        "title": "Tabular Learning and the Bellman Equation / 表格学习与 Bellman 方程",
        "goals": ["理解 Bellman 最优方程", "区分 value iteration 与 Q iteration", "实现小型 MDP 的动态规划"],
        "map": "Bellman 方程把“整段未来”递归拆成“一步奖励 + 下一状态价值”。表格法显式保存每个状态或状态-动作对的值。",
        "formula": r"""
### Bellman 最优方程

$$V_*(s)=\max_a\sum_{s'}P(s'\mid s,a)\left[R(s,a,s')+\gamma V_*(s')\right]$$

$$Q_*(s,a)=\sum_{s'}P(s'\mid s,a)\left[R(s,a,s')+\gamma\max_{a'}Q_*(s',a')\right]$$

已知模型 $P,R$ 时可反复应用 Bellman backup；未知模型时，第 6 章会改用采样更新。
""",
        "bridge": "下面是 3 状态、2 动作的确定性 MDP。终止状态索引为 2；迭代直到最大改变量很小。",
        "codes": [r"""
import numpy as np

# next_state[s, a] 与 reward[s, a]
next_state = np.array([[0, 1], [0, 2], [2, 2]])
reward = np.array([[0., 0.], [0., 1.], [0., 0.]])
terminal = np.array([False, False, True])
gamma = 0.9
V = np.zeros(3)

for iteration in range(100):
    q = reward + gamma * V[next_state] * (~terminal[next_state])
    new_V = q.max(axis=1)
    if np.max(np.abs(new_V - V)) < 1e-10:
        break
    V = new_V

policy = q.argmax(axis=1)
print("iterations:", iteration + 1)
print("V*:", V, "greedy actions:", policy)
"""],
        "links": ["../Chapter05/01_frozenlake_v_iteration.py", "../Chapter05/02_frozenlake_q_iteration.py"],
        "tips": ["终止状态不要 bootstrap。", "同步更新使用旧表计算整张新表；原地更新可能更快，但分析方式不同。", "价值相近时策略可能因浮点误差改变，调试时同时看 Q 值。"],
        "exercise": "把到达终点的奖励从 1 改为 0.1，并给留在状态 0 的动作奖励 0.02；观察最优策略如何变化。",
        "extension": "policy iteration 在策略评估与策略改进之间交替；异步 value iteration 只更新部分状态，优先 sweeping 会优先更新 Bellman 误差大的状态。",
    },
    {
        "title": "Deep Q-Networks / 深度 Q 网络",
        "goals": ["从 Q-learning 过渡到函数逼近", "理解 replay buffer 与 target network", "写出正确的 TD target 和 loss"],
        "map": "DQN 用神经网络一次输出所有动作的 Q 值。经验回放削弱样本相关性；目标网络让训练目标短期稳定。",
        "formula": r"""
### Q-learning 与 DQN

表格更新：
$$Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]$$

DQN 目标与损失：
$$y=r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a'),\qquad L(\theta)=\mathbb E[(Q_\theta(s,a)-y)^2]$$

$\theta^-$ 是延迟同步的 target network；计算 $y$ 时必须停止梯度。
""",
        "bridge": "小型 chain 环境让 Q-learning 的核心更新一目了然；完整 Pong DQN 请沿下方链接阅读。",
        "codes": [r"""
import numpy as np

rng = np.random.default_rng(42)
Q = np.zeros((5, 2))  # 动作 0=左，1=右；状态 4 为终点
alpha, gamma, epsilon = 0.2, 0.95, 0.2

for episode in range(1500):
    s = 0
    for _ in range(20):
        a = rng.integers(2) if rng.random() < epsilon else int(Q[s].argmax())
        s_next = max(0, s - 1) if a == 0 else min(4, s + 1)
        r = 1.0 if s_next == 4 else 0.0
        terminated = s_next == 4
        target = r if terminated else r + gamma * Q[s_next].max()
        Q[s, a] += alpha * (target - Q[s, a])
        s = s_next
        if terminated:
            break

print(np.round(Q, 3))
print("greedy policy:", Q.argmax(axis=1))
"""],
        "links": ["../Chapter06/01_frozenlake_q_learning.py", "../Chapter06/02_dqn_pong.py", "../Chapter06/03_dqn_play.py", "../Chapter06/lib/dqn_model.py", "../Chapter06/lib/wrappers.py"],
        "tips": ["只对实际执行的动作使用 `gather` 取 $Q(s,a)$。", "target 应位于 `torch.no_grad()` 中或显式 `.detach()`。", "先填充一段 replay buffer 再训练；否则初期样本高度相似。"],
        "exercise": "把 epsilon 设为 0，解释全零 Q 表的 `argmax` 为什么可能导致永远学不到终点。",
        "extension": r"Huber loss 比 MSE 对异常 TD error 更稳健；soft target update 使用 $\theta^-\leftarrow\tau\theta+(1-\tau)\theta^-$。",
    },
    {
        "title": "Higher-Level RL Libraries / 高层 RL 库",
        "goals": ["理解 selector、agent、experience source、replay buffer 的职责", "掌握 n-step experience", "能把 PTAN 抽象映射回普通 Python"],
        "map": "库的价值不是隐藏算法，而是稳定地处理动作选择、环境交互、episode 边界、回放采样和 target 同步。",
        "formula": r"""
### n-step transition

$$R_t^{(n)}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}$$

若中途未终止：$y_t=R_t^{(n)}+\gamma^nV(S_{t+n})$；若已终止，则不再 bootstrap。一个经验对象至少需要 `(state, action, reward, last_state)`，其中 `last_state=None` 可清楚表示真正终止。
""",
        "bridge": "用一个纯 Python 函数构造 n-step 回报，先理解数据语义，再阅读 PTAN 的封装。",
        "codes": [r"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Experience:
    state: int
    action: int
    reward: float
    last_state: int | None

def n_step_return(rewards, gamma=0.9):
    return sum((gamma ** k) * r for k, r in enumerate(rewards))

exp = Experience(state=2, action=1, reward=n_step_return([1, 0, 2]), last_state=5)
print(exp)
print("bootstrap discount gamma^3 =", 0.9 ** 3)
"""],
        "links": ["../Chapter07/01_actions.py", "../Chapter07/02_agents.py", "../Chapter07/03_exp_sources.py", "../Chapter07/04_replay_buf.py", "../Chapter07/05_target_net.py", "../Chapter07/06_cartpole.py"],
        "tips": ["先追踪一个 episode 的经验对象，再追踪 batch；这样最容易发现边界错误。", "库版本变化时先检查构造器签名，不要照抄旧教程。", "抽象层越高，越要用 shape 和终止语义的单元测试兜底。"],
        "exercise": "修改函数，使它同时返回累积回报和实际步数；说明提前终止时如何选择 bootstrap 折扣。",
        "extension": "Stable-Baselines3、TorchRL、Ray RLlib 的抽象层和适用规模不同；学习阶段建议至少手写一次核心 loss。",
    },
    {
        "title": "DQN Extensions / DQN 扩展",
        "goals": ["理解 Double、Dueling、Noisy、PER、n-step、Distributional DQN", "知道每项扩展解决哪个问题", "理解 Rainbow 的组合思路"],
        "map": "扩展分别针对过估计、表示、探索、采样效率、信用分配和回报分布；它们并非互斥，因此可组合成 Rainbow。",
        "formula": r"""
### 最常用的三个公式

Double DQN：
$$y=r+\gamma Q_{\theta^-}\!\left(s',\arg\max_a Q_\theta(s',a)\right)$$

Dueling：
$$Q(s,a)=V(s)+A(s,a)-\frac1{|\mathcal A|}\sum_{a'}A(s,a')$$

Prioritized replay：$p_i\propto(|\delta_i|+\varepsilon)^\alpha$，重要性权重 $w_i=(N P(i))^{-\beta}$。

[扩展部分](#extension) 补充 NoisyNet、categorical distribution 与组合选择。
""",
        "bridge": "下面用固定张量对比普通 DQN 与 Double DQN 的目标动作，并验证 Dueling 聚合后的 advantage 均值被消除。",
        "codes": [r"""
import torch

online_next = torch.tensor([[2.0, 3.0, 1.0]])
target_next = torch.tensor([[2.5, 2.0, 4.0]])
ordinary_target_q = target_next.max(dim=1).values
chosen_action = online_next.argmax(dim=1, keepdim=True)
double_target_q = target_next.gather(1, chosen_action).squeeze(1)
print("ordinary / double:", ordinary_target_q.item(), double_target_q.item())

value = torch.tensor([[5.0]])
advantage = torch.tensor([[1.0, -1.0, 2.0]])
q = value + advantage - advantage.mean(dim=1, keepdim=True)
print("dueling Q:", q, "mean Q:", q.mean().item())
"""],
        "links": ["../Chapter08/01_dqn_basic.py", "../Chapter08/02_dqn_n_steps.py", "../Chapter08/03_dqn_double.py", "../Chapter08/04_dqn_noisy_net.py", "../Chapter08/05_dqn_prio_replay.py", "../Chapter08/06_dqn_dueling.py", "../Chapter08/07_dqn_distrib.py", "../Chapter08/08_dqn_rainbow.py"],
        "tips": ["一次只加一种扩展并保留 baseline，才能判断收益来自哪里。", "PER 更新 priority 后要防止零概率，并处理重要性权重的 shape。", "distributional projection 最容易出现质量丢失，要断言每行概率和接近 1。"],
        "exercise": "构造 online 与 target Q 值，让普通 DQN 的目标比 Double DQN 高很多；说明这与过估计的关系。",
        "extension": "NoisyNet 把可学习噪声注入线性层；categorical DQN 在固定 support 上预测回报分布；Rainbow 常组合 n-step、Double、Dueling、PER、Noisy 与 distributional 方法。",
    },
    {
        "title": "Ways to Speed Up RL / RL 加速方法",
        "goals": ["区分采样瓶颈与训练瓶颈", "理解 batch、向量环境与多进程", "用测量而不是直觉做优化"],
        "map": "吞吐量由环境采样、预处理、主机到设备传输、前向/反向和日志共同决定。并行只对当前瓶颈有效。",
        "formula": r"""
### 吞吐量与利用率

粗略地，串行每步耗时 $T\approx T_{env}+T_{prep}+T_{net}+T_{log}$；吞吐量约为 $1/T$。若 $N$ 个环境能有效并行，环境部分可趋近 $T_{env}/N$，但通信和同步开销会增长。

训练时关注：environment steps/s、updates/s、GPU utilization、等待队列长度，以及达到目标回报所需的 wall-clock time。
""",
        "bridge": "下面比较逐样本前向和批量前向。微基准只解释方向，真实结论要在完整训练中用 profiler 验证。",
        "codes": [r"""
import time
import torch
from torch import nn

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 8)).eval()
x = torch.randn(2048, 128)

with torch.no_grad():
    t0 = time.perf_counter()
    loop_out = torch.cat([net(row[None]) for row in x])
    t_loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    batch_out = net(x)
    t_batch = time.perf_counter() - t0

print({"loop_s": round(t_loop, 4), "batch_s": round(t_batch, 4),
       "same": torch.allclose(loop_out, batch_out, atol=1e-6)})
"""],
        "links": ["../Chapter09/00_slow_grads.py", "../Chapter09/01_baseline.py", "../Chapter09/02_n_envs.py", "../Chapter09/03_parallel.py", "../Chapter09/04_wrappers_n_env.py", "../Chapter09/04_wrappers_parallel.py"],
        "tips": ["先记录 baseline 的回报曲线和 wall-clock，再优化。", "多进程要避免每步传大图像；尽量在 worker 侧预处理。", "更高 FPS 不等于更快收敛，算法的样本效率同样重要。"],
        "exercise": "把 batch 大小换成 32、512、8192，重复多次取中位数；解释为何非常小的计时不可靠。",
        "extension": "可进一步使用 pinned memory、异步采样、mixed precision、`torch.compile` 与共享内存，但每项都应以端到端指标验证。",
    },
    {
        "title": "Stocks Trading Using RL / 用 RL 做股票交易",
        "goals": ["把价格序列建模为环境", "理解收益、仓位与交易成本", "避免时间泄漏和不现实评估"],
        "map": "观测只能含当前及过去信息；动作决定仓位；奖励应反映净值变化并扣除成本。重点是正确评估，不是把训练收益当作可交易结论。",
        "formula": r"""
### 交易回报

简单收益 $r_t=(p_{t+1}-p_t)/p_t$，对数收益 $\ell_t=\log(p_{t+1}/p_t)$。

若仓位 $h_t\in\{-1,0,1\}$，一种简化奖励为：
$$R_{t+1}=h_t\ell_t-c|h_t-h_{t-1}|$$

$c$ 同时近似手续费与滑点。训练/验证/测试必须按时间切分；随机打乱会把未来信息泄漏到过去。
""",
        "bridge": "用合成价格展示持仓收益与换仓成本。这里不是投资策略，只用于验证环境记账。",
        "codes": [r"""
import numpy as np

prices = np.array([100., 102., 101., 104., 103.])
positions = np.array([0., 1., 1., -1.])  # 区间 [t, t+1) 的仓位
log_returns = np.diff(np.log(prices))
previous_positions = np.r_[0., positions[:-1]]
cost_rate = 0.001
rewards = positions * log_returns - cost_rate * np.abs(positions - previous_positions)
equity = np.exp(np.cumsum(rewards))
print("step rewards:", np.round(rewards, 5))
print("equity multiplier:", np.round(equity, 5))
"""],
        "links": ["../Chapter10/lib/data.py", "../Chapter10/lib/environ.py", "../Chapter10/lib/models.py", "../Chapter10/train_model.py", "../Chapter10/train_model_conv.py", "../Chapter10/run_model.py"],
        "tips": ["先用手算轨迹测试 cash、position、commission，再开始训练。", "标准化参数只能在训练集估计。", "报告最大回撤、换手率和成本敏感性，不能只报平均回报。"],
        "exercise": "把成本改为 0、0.001、0.01，比较最终净值；解释频繁换仓为何在高成本下失败。",
        "extension": "真实回测还需考虑 bid-ask spread、成交量、延迟、企业行动、幸存者偏差和风险约束；这超出本章教学环境。",
    },
    {
        "title": "Policy Gradients - an Alternative / 策略梯度",
        "goals": ["从最大化期望回报推导 REINFORCE", "理解 baseline 不改变期望梯度", "正确处理 log-prob、return 与 entropy"],
        "map": r"价值方法间接选动作；策略梯度直接调整 $\pi_\theta(a|s)$，让高回报动作更可能出现。",
        "formula": r"""
### Policy gradient theorem

$$J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G(\tau)]$$

$$\nabla_\theta J(\theta)=\mathbb E\left[\sum_t\nabla_\theta\log\pi_\theta(A_t|S_t)(G_t-b(S_t))\right]$$

训练时最小化 $L=-\log\pi_\theta(a_t|s_t)\,\hat A_t$。baseline 降低方差；entropy bonus 鼓励探索。
""",
        "bridge": "下面用一批人为 advantage 做一次策略更新：正 advantage 提高所选动作概率，负 advantage 降低它。",
        "codes": [r"""
import torch
from torch import nn

torch.manual_seed(1)
policy = nn.Linear(3, 2)
states = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
actions = torch.tensor([0, 1, 0])
advantages = torch.tensor([1.0, -0.5, 0.8])

dist = torch.distributions.Categorical(logits=policy(states))
policy_loss = -(dist.log_prob(actions) * advantages.detach()).mean()
entropy = dist.entropy().mean()
loss = policy_loss - 0.01 * entropy
loss.backward()
print("policy loss:", policy_loss.item(), "entropy:", entropy.item())
print("gradient norm:", policy.weight.grad.norm().item())
"""],
        "links": ["../Chapter11/01_cartpole_dqn.py", "../Chapter11/02_cartpole_reinforce.py", "../Chapter11/03_cartpole_reinforce_baseline.py", "../Chapter11/04_cartpole_pg.py", "../Chapter11/05_pong_pg.py"],
        "tips": ["advantage 通常要 `.detach()`，否则会出现不期望的二阶耦合。", "对 batch advantage 做标准化常能稳定训练。", "不要对概率再手动 `log`；优先使用 `Categorical(logits=...)`。"],
        "exercise": "记录更新前后三个样本所选动作的概率，验证 advantage 正负与概率变化方向。",
        "extension": "off-policy policy gradient 需要 importance sampling；连续动作常用 Gaussian policy，并对标准差作正值约束。",
    },
    {
        "title": "Actor-Critic Methods - A2C and A3C / Actor-Critic",
        "goals": ["理解 actor 与 critic 的分工", "写出 A2C 联合损失", "区分同步 A2C 与异步 A3C"],
        "map": "critic 估计价值并形成低方差 advantage；actor 用 advantage 更新策略。A2C 同步汇总多个环境，A3C 由异步 worker 更新共享参数。",
        "formula": r"""
### 一步 actor-critic

$$\hat A_t=R_{t+1}+\gamma(1-d_t)V_\phi(S_{t+1})-V_\phi(S_t)$$

$$L_{actor}=-\log\pi_\theta(A_t|S_t)\,\operatorname{stopgrad}(\hat A_t)$$

$$L_{critic}=\tfrac12\hat A_t^2,\qquad L=L_{actor}+c_vL_{critic}-c_eH(\pi)$$

实际代码常用 n-step return 替代一步 TD。
""",
        "bridge": "单元格展示联合 loss 的 shape 与梯度路径。终止 mask 只屏蔽真正 terminal 的 bootstrap。",
        "codes": [r"""
import torch

log_prob = torch.tensor([-0.2, -1.0, -0.5], requires_grad=True)
value = torch.tensor([0.4, 0.2, 0.1], requires_grad=True)
next_value = torch.tensor([0.5, 0.3, 0.0])
reward = torch.tensor([1.0, 0.0, -0.2])
terminated = torch.tensor([False, False, True])
gamma = 0.99

target = reward + gamma * (~terminated) * next_value
advantage = target - value
actor_loss = -(log_prob * advantage.detach()).mean()
critic_loss = 0.5 * advantage.square().mean()
loss = actor_loss + 0.5 * critic_loss
loss.backward()
print({"target": target.detach().tolist(), "loss": loss.item()})
"""],
        "links": ["../Chapter12/01_cartpole_pg.py", "../Chapter12/02_pong_a2c.py", "../Chapter12/02_pong_a2c_tune.py", "../Chapter12/03_a3c_grad.py"],
        "tips": ["actor 和 critic 共用 backbone 时，两个 loss 都会更新共享层。", "value loss 系数过大可能压制策略学习。", "A3C 的并发 bug 难复现；学习时先让同步版本正确。"],
        "exercise": "去掉 `advantage.detach()`，查看 value 的梯度怎样变化，并解释为何这通常不是想要的 actor 更新。",
        "extension": "GAE 用指数加权 TD residual 平衡偏差与方差；V-trace 可修正分布式 actor 的 off-policy 数据。",
    },
    {
        "title": "The TextWorld Environment / 文本世界",
        "goals": ["把文本游戏理解为部分可观测问题", "处理 token、命令与动作掩码", "理解文本状态的 DQN/Transformer 表示"],
        "map": "观测是文本，动作也是命令；同一句话的顺序和历史可能重要。可行命令集合提供 action mask，避免在大量无效动作上浪费探索。",
        "formula": r"""
### POMDP 与动作掩码

历史 $h_t=(o_1,a_1,\ldots,o_t)$ 可编码为隐藏状态 $z_t=f_\theta(h_t)$，再估计 $Q(z_t,a)$。

若合法动作掩码 $m_a\in\{0,1\}$，可在 softmax 前设
$$\tilde z_a=\begin{cases}z_a&m_a=1\\-\infty&m_a=0\end{cases}$$
从而使非法动作概率为 0。
""",
        "bridge": "这里用简化 tokenizer 和 masked softmax 解释数据流；项目脚本负责真实 TextWorld/Hugging Face 集成。",
        "codes": [r"""
import re
import torch

def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())

observation = "You are in a kitchen. A red key is on the table."
tokens = tokenize(observation)
vocab = {word: i + 1 for i, word in enumerate(sorted(set(tokens)))}
token_ids = torch.tensor([vocab[w] for w in tokens])
print("tokens:", tokens)
print("ids:", token_ids.tolist())

logits = torch.tensor([2.0, 1.0, 3.0, -1.0])
legal = torch.tensor([True, False, True, False])
masked_logits = logits.masked_fill(~legal, -torch.inf)
print("masked probabilities:", torch.softmax(masked_logits, dim=0))
"""],
        "links": ["../Chapter13/train_basic.py", "../Chapter13/train_preproc.py", "../Chapter13/train_tr.py", "../Chapter13/lib/preproc.py", "../Chapter13/lib/model.py", "../Chapter13/chatgpt_interactive.py"],
        "tips": ["缓存 tokenizer 输出能明显减少训练开销。", "padding token 必须在 attention/pooling 中正确屏蔽。", "记录原始文本、可行动作与所选动作，比只记录 reward 更容易调试。"],
        "exercise": "给 logits 增加第五个动作并设为非法；验证无论 logit 多大，其概率仍为 0。",
        "extension": "可用 pretrained language model 编码状态，但环境交互成本、上下文长度与微调稳定性会成为新瓶颈。",
    },
    {
        "title": "Web Navigation / Web 导航",
        "goals": ["定义网页观测与点击动作", "理解 demonstrations 与 action discretization", "识别自动化环境的安全边界"],
        "map": "网页任务可用截图、DOM 特征或两者作为观测；动作可离散为元素选择，也可连续表示坐标。演示数据能在稀疏奖励前提供起点。",
        "formula": r"""
### 行为克隆与 RL 微调

演示数据 $D=\{(s_i,a_i)\}$ 的行为克隆损失：
$$L_{BC}=-\mathbb E_{(s,a)\sim D}[\log\pi_\theta(a|s)]$$

它只学到演示分布内行为，遇到自身错误导致的新状态会累积偏差；之后可用 RL 环境奖励继续微调。
""",
        "bridge": "连续点击坐标通常要映射到离散网格或候选元素。下面演示网格编码及反解。",
        "codes": [r"""
def encode_click(x, y, width, height, cols=8, rows=6):
    col = min(cols - 1, int(x / width * cols))
    row = min(rows - 1, int(y / height * rows))
    return row * cols + col

def decode_click(action, width, height, cols=8, rows=6):
    row, col = divmod(action, cols)
    # 返回网格中心，减少边界点击歧义
    return ((col + 0.5) * width / cols, (row + 0.5) * height / rows)

action = encode_click(420, 180, width=800, height=600)
print("action:", action, "decoded center:", decode_click(action, 800, 600))
"""],
        "links": ["../Chapter14/record_demo.py", "../Chapter14/wob_click_train.py", "../Chapter14/wob_click_play.py", "../Chapter14/wob_click_mm_train.py", "../Chapter14/lib/wob.py", "../Chapter14/lib/demos.py"],
        "tips": ["训练只应在授权的 sandbox 网站上进行。", "DOM 与截图可能异步变化，记录时间戳和页面状态。", "点击坐标要按 viewport 尺寸归一化，否则分辨率变化会破坏策略。"],
        "exercise": "测试页面四个角和中心的编码/解码；说明网格分辨率与动作空间大小的权衡。",
        "extension": "候选元素动作比固定网格更语义化；多模态策略可融合 DOM token 与截图特征，但数据管线更复杂。",
    },
    {
        "title": "Continuous Action Space / 连续动作空间",
        "goals": ["理解连续策略与动作缩放", "掌握 DDPG actor/critic 与 target update", "认识 D4PG 的 distributional critic"],
        "map": "连续动作无法逐个枚举。随机策略常输出 Gaussian 参数；DDPG 用确定性 actor 直接产生动作，并由 critic 提供梯度。",
        "formula": r"""
### DDPG 核心

critic 目标：$y=r+\gamma(1-d)Q_{\phi^-}(s',\mu_{\theta^-}(s'))$。

actor 目标：$J(\theta)=\mathbb E_s[Q_\phi(s,\mu_\theta(s))]$，训练时最小化 $-J$。

soft update：
$$\theta^-\leftarrow\tau\theta+(1-\tau)\theta^-$$

网络常输出 $u\in[-1,1]$，再映射到环境区间 $[l,h]$：$a=l+(u+1)(h-l)/2$。
""",
        "bridge": "动作缩放和 target soft update 都是短小但常出错的基础操作。",
        "codes": [r"""
import torch

def scale_action(unit_action, low, high):
    return low + (unit_action + 1.0) * 0.5 * (high - low)

unit = torch.tensor([-1.0, 0.0, 1.0])
print("scaled:", scale_action(unit, low=-2.0, high=4.0))

online = torch.nn.Linear(2, 1)
target = torch.nn.Linear(2, 1)
target.load_state_dict(online.state_dict())
with torch.no_grad():
    online.weight.add_(1.0)  # 模拟一次训练更新
    tau = 0.05
    for target_p, online_p in zip(target.parameters(), online.parameters()):
        target_p.mul_(1 - tau).add_(online_p, alpha=tau)
print("soft-updated target weight:", target.weight)
"""],
        "links": ["../Chapter15/01_check_env.py", "../Chapter15/02_train_a2c.py", "../Chapter15/03_play_a2c.py", "../Chapter15/04_train_ddpg.py", "../Chapter15/05_play_ddpg.py", "../Chapter15/06_train_d4pg.py"],
        "tips": ["同时检查 environment action bounds 和策略输出范围。", "critic 更新时 target actor/critic 都应停止梯度。", "探索噪声的尺度要相对于动作区间设定。"],
        "exercise": "把动作区间改为 `[2, 5]`，验证 -1、0、1 分别映射到哪里。",
        "extension": "D4PG 使用分布式 critic、n-step return 与分布式采样；TD3 用双 critic 和延迟 actor 更新降低过估计。",
    },
    {
        "title": "Trust Regions - PPO, TRPO, ACKTR, and SAC / 信赖域方法",
        "goals": ["理解策略更新不能迈太大步", "掌握 PPO clipped objective", "区分 TRPO、ACKTR 与 SAC 的核心想法"],
        "map": "旧策略收集的数据只适合有限幅度的新策略更新。TRPO 施加 KL 约束；PPO 用 clipping 近似；ACKTR 用曲率信息；SAC 最大化回报与熵。",
        "formula": r"""
### PPO 与 SAC

$r_t(\theta)=\pi_\theta(a_t|s_t)/\pi_{old}(a_t|s_t)$。

$$L^{clip}=\mathbb E\left[\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)\right]$$

SAC 的最大熵目标：
$$J(\pi)=\mathbb E\sum_t\gamma^t\left(R_{t+1}+\alpha\mathcal H(\pi(\cdot|S_t))\right)$$

PPO 不是严格 KL 约束；仍应监控 approximate KL 与 clip fraction。
""",
        "bridge": "用正负 advantage 观察 PPO clipping：clip 的作用方向取决于 advantage 符号。",
        "codes": [r"""
import torch

old_logp = torch.tensor([-0.7, -0.7, -0.7, -0.7])
new_logp = torch.tensor([-0.6, -0.2, -1.3, -0.8], requires_grad=True)
advantage = torch.tensor([1.0, 1.0, -1.0, -1.0])
ratio = (new_logp - old_logp).exp()
epsilon = 0.2
unclipped = ratio * advantage
clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
loss = -torch.minimum(unclipped, clipped).mean()
loss.backward()
print("ratio:", ratio.detach().round(decimals=3))
print("PPO loss:", loss.item(), "approx KL:", (old_logp - new_logp).mean().item())
"""],
        "links": ["../Chapter16/01_train_a2c.py", "../Chapter16/03_train_ppo.py", "../Chapter16/04_train_trpo.py", "../Chapter16/05_train_acktr.py", "../Chapter16/06_train_sac.py", "../Chapter16/lib/trpo.py", "../Chapter16/lib/kfac.py"],
        "tips": ["缓存 old log-prob，多个 epoch 内不要重新计算成当前策略。", "advantage 标准化与 value clipping 会显著影响 PPO。", "clip loss 正常不代表更新安全，仍要设置 target KL 或早停。"],
        "exercise": "逐项打印 unclipped 和 clipped surrogate；找出哪些样本真正触发保守边界。",
        "extension": r"TRPO 通过共轭梯度近似自然梯度；ACKTR 用 K-FAC 近似 Fisher；SAC 是 off-policy 并可自动调温度 $\alpha$。",
    },
    {
        "title": "Black-Box Optimizations in RL / 黑盒优化",
        "goals": ["理解 evolution strategies 与 genetic algorithms", "无需反向传播估计搜索方向", "认识参数空间搜索的优缺点"],
        "map": "黑盒方法只需要“参数 → 适应度”，不要求环境或策略可微。它们易并行、对长时依赖直接，但通常样本开销很大。",
        "formula": r"""
### Evolution Strategies (ES)

采样 $\epsilon_i\sim\mathcal N(0,I)$，评估 $F(\theta+\sigma\epsilon_i)$，梯度估计：

$$\nabla_\theta\mathbb E_\epsilon[F(\theta+\sigma\epsilon)]\approx\frac1{n\sigma}\sum_{i=1}^nF_i\epsilon_i$$

antithetic sampling 同时评估 $+\epsilon$ 与 $-\epsilon$ 可降低方差；fitness rank normalization 可降低尺度敏感性。
""",
        "bridge": "用 ES 最大化二维凹二次函数，观察参数逐渐靠近最优点。",
        "codes": [r"""
import numpy as np

rng = np.random.default_rng(7)
theta = np.array([4.0, -3.0])
optimum = np.array([1.5, 2.0])
sigma, lr, population = 0.2, 0.08, 200

def fitness(x):
    return -np.sum((x - optimum) ** 2, axis=-1)

for _ in range(120):
    noise = rng.normal(size=(population, 2))
    scores = fitness(theta + sigma * noise)
    scores = (scores - scores.mean()) / (scores.std() + 1e-8)
    grad = (scores[:, None] * noise).mean(axis=0) / sigma
    theta += lr * grad

print("theta:", np.round(theta, 3), "fitness:", round(float(fitness(theta)), 5))
"""],
        "links": ["../Chapter17/01_cartpole_es.py", "../Chapter17/02_cheetah_es.py", "../Chapter17/03_cartpole_ga.py", "../Chapter17/04_cheetah_ga.py"],
        "tips": ["多个随机 seed 复评 elite，避免选中纯噪声高分。", "worker 只需共享随机 seed/参数扰动时，通信成本可很低。", "先在低维函数验证符号和尺度，再跑昂贵环境。"],
        "exercise": "把 population 改成 10 和 1000，比较不同 seed 下的稳定性与评估次数。",
        "extension": "GA 使用选择、交叉、变异；novelty search 奖励行为差异而非任务分数，可缓解 deceptive reward。",
    },
    {
        "title": "Advanced Exploration / 高级探索",
        "goals": ["理解 epsilon-greedy 的局限", "掌握 count bonus、NoisyNet 与 RND 思路", "区分外在奖励和内在奖励"],
        "map": "简单随机动作在深度探索任务中效率很低。高级方法用访问次数、预测误差或参数噪声形成一致的探索驱动力。",
        "formula": r"""
### 内在奖励

count bonus：$r_i(s)=\beta/\sqrt{N(s)+1}$。

RND：固定随机 target 网络 $f$，训练 predictor $\hat f_\theta$；
$$r_i(s)=\|\hat f_\theta(s)-f(s)\|_2^2$$

总奖励常写成 $r=r_e+\eta r_i$。内在奖励需要归一化，否则尺度会随训练快速变化。
""",
        "bridge": "离散计数例子展示：新状态奖金大，反复访问后奖金衰减。",
        "codes": [r"""
from collections import defaultdict
import math

counts = defaultdict(int)
beta = 0.5
trajectory = ["A", "A", "B", "A", "C", "C"]
bonuses = []
for state in trajectory:
    # 先按已有访问次数计算“新奇度”，再登记本次访问
    bonus = beta / math.sqrt(counts[state] + 1)
    bonuses.append(bonus)
    counts[state] += 1

print("bonuses:", [round(x, 3) for x in bonuses])
print("counts:", dict(counts))
"""],
        "links": ["../Chapter18/riverswim.py", "../Chapter18/mcar_dqn.py", "../Chapter18/mcar_ppo.py", "../Chapter18/atari_dqn.py", "../Chapter18/atari_ppo.py", "../Chapter18/lib/dqn_extra.py"],
        "tips": ["分别记录 external 与 intrinsic return。", "随机电视问题会让预测误差方法沉迷于不可预测噪声。", "评估策略时通常关闭 exploration bonus/noise。"],
        "exercise": "让轨迹只在 A/B 循环，再首次访问 C；比较当时的 bonus，并说明它为何鼓励离开熟悉区域。",
        "extension": "pseudo-count 将密度模型推广到大状态空间；episodic memory、ICM 和 disagreement ensembles 是其他常见路线。",
    },
    {
        "title": "Reinforcement Learning with Human Feedback / RLHF",
        "goals": ["理解偏好标签与 reward model", "掌握 Bradley-Terry preference loss", "认识 KL 约束和 reward hacking"],
        "map": "先收集轨迹片段与人类偏好，再训练 reward model，最后让策略优化预测奖励，同时用 KL 或其他约束避免偏离参考策略太远。",
        "formula": r"""
### 偏好模型

两个片段 $\sigma^A,\sigma^B$ 的预测总分分别为 $R_A,R_B$：
$$P(A\succ B)=\frac{e^{R_A}}{e^{R_A}+e^{R_B}}=\sigma(R_A-R_B)$$

二元交叉熵训练 reward model。策略阶段常使用：
$$r_{total}=r_\phi-\beta\big(\log\pi_\theta(a|s)-\log\pi_{ref}(a|s)\big)$$

reward model 的高分不等于真实人类满意度，必须保留独立评估。
""",
        "bridge": "下面直接计算一批成对偏好的 stable logistic loss；label=1 表示偏好 A。",
        "codes": [r"""
import torch
import torch.nn.functional as F

score_a = torch.tensor([2.0, -0.5, 1.0], requires_grad=True)
score_b = torch.tensor([1.0, 0.5, 1.5], requires_grad=True)
labels = torch.tensor([1.0, 0.0, 1.0])

logits = score_a - score_b
loss = F.binary_cross_entropy_with_logits(logits, labels)
loss.backward()
print("P(A preferred):", torch.sigmoid(logits).detach())
print("preference loss:", loss.item())
"""],
        "links": ["../Chapter19/01_a2c.py", "../Chapter19/01_play.py", "../Chapter19/02_label_ui.py", "../Chapter19/03_reward_train.py", "../Chapter19/lib/rlhf.py", "../Chapter19/lib/ui_tools.py"],
        "tips": ["训练、验证偏好对必须按轨迹或来源分组，避免近重复泄漏。", "标签界面要随机交换 A/B 位置，减少位置偏差。", "持续抽查策略找到的 reward-model 高分样本，这是发现 hacking 的关键。"],
        "exercise": "交换 score_a 与 score_b，同时把 labels 改为 `1-labels`，验证 loss 应保持一致。",
        "extension": "LLM RLHF 还涉及 SFT、token-level KL、PPO/DPO 等；本章项目以可视化环境演示同一核心闭环。",
    },
    {
        "title": "AlphaGo Zero and MuZero / AlphaGo Zero 与 MuZero",
        "goals": ["理解 self-play + MCTS + policy/value network", "掌握 PUCT 选择公式", "区分已知规则模型与 MuZero 学习模型"],
        "map": "搜索用网络先验引导，搜索访问次数形成更强策略目标；self-play 产生训练数据。MuZero 学习 representation、dynamics、prediction 三个函数，不直接重建原始观测。",
        "formula": r"""
### PUCT 与训练目标

选择边：
$$a=\arg\max_a\left[Q(s,a)+c_{puct}P(s,a)\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}\right]$$

AlphaZero 风格损失：$L=(z-v)^2-\pi^T\log p+c\|\theta\|^2$。

MuZero 再学习隐藏状态转移 $s_{k+1}=g(s_k,a_k)$ 与预测 reward，并在 unroll 多步上累计 value/policy/reward loss。
""",
        "bridge": "下面实现单个节点的 PUCT 选择，直观看到高先验的未访问动作如何获得 exploration bonus。",
        "codes": [r"""
import numpy as np

def puct_scores(q, prior, visits, c=1.5):
    parent_visits = max(1, visits.sum())
    exploration = c * prior * np.sqrt(parent_visits) / (1 + visits)
    return q + exploration

q = np.array([0.6, 0.5, 0.1])
prior = np.array([0.2, 0.3, 0.5])
visits = np.array([10, 4, 0])
scores = puct_scores(q, prior, visits)
print("PUCT scores:", np.round(scores, 3), "selected action:", scores.argmax())
"""],
        "links": ["../Chapter20/lib/game.py", "../Chapter20/lib/mcts.py", "../Chapter20/lib/model.py", "../Chapter20/lib/muzero.py", "../Chapter20/train.py", "../Chapter20/train-mu.py", "../Chapter20/play.py"],
        "tips": ["MCTS value 回传时要按当前玩家视角正确翻转符号。", "训练 target policy 来自 visit count，不是单次 argmax。", "先对游戏规则和胜负判定写单元测试，再调网络。"],
        "exercise": "把第三个动作 prior 降到 0.05 并重新归一化，观察选择；增加该动作访问次数后再比较。",
        "extension": "根节点 Dirichlet noise 增加 self-play 多样性；temperature 控制 visit count 到动作概率的尖锐程度；MuZero 的 reanalyse 可用新网络重算旧数据目标。",
    },
    {
        "title": "RL in Discrete Optimization / 离散优化中的 RL",
        "goals": ["把魔方/组合问题表示成状态图", "理解 value-guided search", "区分学习启发式与搜索保证"],
        "map": "离散优化可看成图上的最短路：动作产生邻居，目标状态奖励最高。网络学习距离或价值，搜索算法用它排序节点。",
        "formula": r"""
### 学习启发式搜索

若每步代价为 1，可令网络预测到目标的剩余代价 $h_\theta(s)$。A* 排序：
$$f(s)=g(s)+h_\theta(s)$$

或用价值 $V(s)\approx\gamma^{d(s)}$ 推断距离。若学习到的 $h$ 会高估，标准 A* 的最优性保证可能失效；可通过加权 A* 或保守校准权衡速度与保证。
""",
        "bridge": "在一个小整数状态图上做 beam search；启发式只决定优先级，合法动作仍由环境定义。",
        "codes": [r"""
def neighbors(x):
    return {x - 1, x + 1, x * 2}

def beam_search(start, goal, width=3, max_depth=10):
    frontier = [(start, [start])]
    seen = {start}
    for _ in range(max_depth):
        candidates = []
        for state, path in frontier:
            for nxt in neighbors(state):
                if nxt == goal:
                    return path + [nxt]
                if -20 <= nxt <= 40 and nxt not in seen:
                    seen.add(nxt)
                    # 这里用 |goal-nxt| 代替学习到的 h_theta。
                    candidates.append((abs(goal - nxt), nxt, path + [nxt]))
        frontier = [(s, p) for _, s, p in sorted(candidates)[:width]]
    return None

print("path:", beam_search(1, 15))
"""],
        "links": ["../Chapter21/gen_cubes.py", "../Chapter21/train.py", "../Chapter21/solver.py", "../Chapter21/libcube/model.py", "../Chapter21/libcube/mcts.py", "../Chapter21/libcube/cubes/cube2x2.py", "../Chapter21/libcube/cubes/cube3x3.py"],
        "tips": ["对动作逆操作与状态编码写 round-trip 测试。", "训练集要覆盖不同打乱深度，否则启发式在深层状态上失真。", "报告解出率、解长度、展开节点数和时间，而不只是网络 loss。"],
        "exercise": "把 beam width 改为 1 和 10，比较是否找到路径及展开规模；解释贪心搜索的风险。",
        "extension": "curriculum learning 可逐渐增加打乱深度；bidirectional search、MCTS 与 transposition table 可进一步降低搜索量。",
    },
    {
        "title": "Multi-Agent RL / 多智能体强化学习",
        "goals": ["理解 Markov game 与联合动作", "认识非平稳性、合作与竞争", "区分 independent learning 与 CTDE"],
        "map": "每个智能体都在学习时，其他智能体的策略变化会让环境看起来非平稳。训练时可集中使用全局信息，执行时再让每个 agent 只用局部观测。",
        "formula": r"""
### Markov game

联合策略下 agent $i$ 的目标：
$$J_i(\pi_1,\ldots,\pi_n)=\mathbb E\left[\sum_t\gamma^t r_i(s_t,a_t^1,\ldots,a_t^n)\right]$$

Independent Q-learning 把其他 agent 当环境一部分；CTDE 的 centralized critic 可用联合状态/动作估计 $Q_i(s,a^1,\ldots,a^n)$，actor 执行时只用本地观测。
""",
        "bridge": "这个协调博弈展示多个同样最优均衡：双方动作相同时得 1，否则得 0。独立 Q-learning 的结果会受探索与随机种子影响。",
        "codes": [r"""
import numpy as np

rng = np.random.default_rng(3)
q1 = np.zeros(2)
q2 = np.zeros(2)
alpha, epsilon = 0.1, 0.1

for _ in range(3000):
    a1 = rng.integers(2) if rng.random() < epsilon else q1.argmax()
    a2 = rng.integers(2) if rng.random() < epsilon else q2.argmax()
    reward = float(a1 == a2)
    q1[a1] += alpha * (reward - q1[a1])
    q2[a2] += alpha * (reward - q2[a2])

print("agent 1 Q:", np.round(q1, 3), "policy:", q1.argmax())
print("agent 2 Q:", np.round(q2, 3), "policy:", q2.argmax())
"""],
        "links": ["../Chapter22/forest_random.py", "../Chapter22/forest_tigers_dqn.py", "../Chapter22/forest_both_dqn.py", "../Chapter22/battle_dqn.py", "../Chapter22/lib/model.py", "../Chapter22/lib/data.py"],
        "tips": ["明确 reward 是共享、个体还是混合，否则“合作”只是口号。", "评估要固定对手池并覆盖不同强度，避免只会打当前对手。", "并行环境中 agent id、alive mask、padding 最容易造成错位。"],
        "exercise": "把不协调奖励改为 -1，比较学习速度；用多个 seed 观察最终选择 0 还是 1。",
        "extension": "value decomposition（VDN/QMIX）、self-play、population-based training、opponent modeling 与 communication learning 是常见扩展。",
    },
]


# The second-pass teaching layer.  The compact chapter cards above keep the
# factual outline; this layer adds the learning path that a beginner needs:
# motivation, notation, derivation, a hand calculation, formula/code mapping,
# interpretation, and a recall check.
V2_GUIDE = {
    1: {
        "prereq": [r"会读简单求和符号 $\\sum$", "知道概率和期望的直观含义", "会运行 Python/NumPy 单元格"],
        "question": "如果今天拿到 1 分，或者等两步后拿到 10 分，智能体应怎样比较？强化学习的全部困难，几乎都从“动作会改变未来”这件事开始。",
        "symbols": [
            ("$S_t$", "时刻 $t$ 的状态", "作决定时已有的信息"),
            ("$A_t$", "智能体采取的动作", "由策略选择"),
            ("$R_{t+1}$", "执行动作后得到的奖励", "下标提醒它在动作之后到来"),
            (r"$\\pi(a|s)$", "策略", "在状态 $s$ 选动作 $a$ 的概率"),
            (r"$\\gamma$", "折扣因子", "未来奖励的权重，范围通常为 $[0,1)$"),
            ("$G_t$", "从时刻 $t$ 开始的回报", "未来奖励的折扣和"),
        ],
        "derivation": r"""
### 1. 从“未来一共能拿多少分”开始

把未来奖励依次写出：

$$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots$$

把第一项以外的部分提出一个 $\gamma$：

$$G_t=R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+\cdots)$$

括号内恰好是下一时刻的回报 $G_{t+1}$，所以得到最重要的递归式：

$$\boxed{G_t=R_{t+1}+\gamma G_{t+1}}$$

### 2. 从回报到价值

同一状态下，环境和策略都可能有随机性，因此单次 $G_t$ 会波动。对所有可能未来取条件期望：

$$V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]$$

把回报递归式代入：

$$V^\pi(s)=\mathbb E_\pi[R_{t+1}+\gamma V^\pi(S_{t+1})\mid S_t=s]$$

这就是 Bellman expectation equation。它不是凭空规定，而是“回报递归 + 条件期望”的直接结果。

### 3. 为什么矩阵解是 $(I-\gamma P)^{-1}r$

对有限 Markov reward process，可写成 $V=r+\gamma PV$。移项得到
$$(I-\gamma P)V=r,$$
若矩阵可逆，就有 $V=(I-\gamma P)^{-1}r$。代码中应使用 `solve`，不要显式计算逆矩阵。
""",
        "hand": r"""
奖励序列为 $[1,0,2]$，$\gamma=0.9$。从后向前算：

- $G_2=2$
- $G_1=0+0.9\times2=1.8$
- $G_0=1+0.9\times1.8=2.62$

先手算再运行代码；预期输出应为 `[2.62, 1.8, 2.0]`。这也是倒序循环中 `running = reward + gamma * running` 的来源。
""",
        "mapping": [
            ("$R_{t+1}$", "`rewards[i]`", "当前位置的即时奖励"),
            ("$G_{t+1}$", "更新前的 `running`", "已经算好的后续回报"),
            ("$G_t$", "更新后的 `running`", "当前时刻回报"),
            ("$P$", "`P`，shape `(n, n)`", "状态转移概率矩阵"),
            (r"$(I-\\gamma P)V=r$", "`np.linalg.solve(I - gamma * P, r)`", "直接解线性方程"),
        ],
        "read": r"第一段输出必须与手算一致。第二段的两个价值都大于单步奖励，因为价值包含长期累积奖励。把 $\gamma$ 改小，价值会更接近即时奖励。",
        "takeaways": ["RL 的核心对象是长期回报，不是单步奖励。", r"$G_t=R_{t+1}+\gamma G_{t+1}$ 是后续 Bellman、TD、DQN、actor-critic 的共同起点。", "价值是随机未来回报的条件期望。", "状态必须包含预测未来所需的信息，才满足 Markov 性。"],
        "check": ("不看上文：为什么回报要倒序计算？奖励和价值的区别是什么？", "因为 $G_t$ 依赖已经知道的 $G_{t+1}$，倒序一次遍历即可。奖励是环境某一步给出的反馈；价值是某策略下从当前状态出发的未来累计奖励期望。"),
    },
    2: {
        "prereq": ["理解第 1 章的 agent/environment 交互", "会写 `for` 循环", "知道 tuple 解包"],
        "question": "算法怎样在不关心游戏内部实现的情况下，同时操作 CartPole、Atari 和机器人？答案是一份严格的交互契约。",
        "symbols": [
            ("`observation`", "策略能看到的信息", "不一定等于完整状态"),
            ("`action_space`", "合法动作集合", "也负责随机采样"),
            ("`terminated`", "MDP 自然终止", "例如失败、成功"),
            ("`truncated`", "外部截断", "例如时间上限"),
            ("`info`", "调试/评估附加信息", "通常不直接输入策略"),
        ],
        "derivation": r"""
### 1. 一次交互对应 MDP 的哪一部分

给定 $S_t=s,A_t=a$，环境按照转移分布产生下一状态和奖励：

$$S_{t+1},R_{t+1}\sim P(\cdot,\cdot\mid s,a)$$

Gymnasium 把这一步实现为：

```python
next_obs, reward, terminated, truncated, info = env.step(action)
```

### 2. 为什么要区分 terminated 和 truncated

TD 目标需要判断下一状态是否还有未来价值：

$$y_t=R_{t+1}+\gamma m_tV(S_{t+1})$$

自然终止时 $m_t=0$；时间限制截断不一定意味着状态没有未来，所以常取

$$m_t=1-\mathbb 1[\text{terminated}]$$

但 episode 控制流必须在 `terminated or truncated` 时 reset。这说明“停止采样”和“是否 bootstrap”是两个不同问题。

### 3. Wrapper 为什么是组合而不是复制环境

Wrapper 保持相同的 `reset/step` 接口，只变换 observation、action 或 reward，因此可一层层组合。顺序不同会得到不同 MDP，例如“先裁剪奖励再累计”和“先累计再裁剪”并不等价。
""",
        "hand": r"""
假设最后一步奖励为 1，$\gamma=0.9$，critic 认为下一状态价值为 5：

- 真正终止：$y=1+0.9\times0\times5=1$
- 仅时间截断：$y=1+0.9\times1\times5=5.5$

把 truncated 错当 terminated，会系统性低估时间上限附近的价值。
""",
        "mapping": [
            ("$S_t$", "`obs`", "当前观测"),
            ("$A_t$", "`action`", "传给 `env.step` 的动作"),
            ("$R_{t+1}$", "`reward`", "动作之后获得"),
            ("$S_{t+1}$", "新的 `obs`", "下一轮策略输入"),
            ("episode 结束", "`terminated or truncated`", "控制流 reset 条件"),
            ("bootstrap mask", "`not terminated`", "价值目标中的 mask"),
        ],
        "read": "随机策略的 return 每次可能不同，这是环境与动作采样共同造成的。重点不是分数，而是确认 observation shape、返回值顺序和结束条件。",
        "takeaways": ["`reset` 和 `step` 是环境的最小契约。", "episode 结束条件与 bootstrap mask 不能混为一谈。", "训练时关闭渲染，评估时再录制。", "Wrapper 的组合顺序是算法定义的一部分。"],
        "check": ("`truncated=True, terminated=False` 时，循环是否停止？TD target 是否一定把下一状态价值置零？", "循环应停止并 reset；但 TD target 通常仍 bootstrap，因为截断不等于 MDP 自然终止。具体仍要结合环境定义。"),
    },
    3: {
        "prereq": ["会用 NumPy 数组", "理解函数、类与参数", "知道导数表示局部变化率"],
        "question": "DQN 为什么能从像素预测每个动作的长期价值？先不谈 RL，本章只解决一件事：如何让参数根据误差自动改变。",
        "symbols": [
            ("$x,y$", "输入与目标", "监督信号"),
            (r"$f_\theta(x)$", "带参数的网络", r"$\theta$ 包含权重和偏置"),
            (r"$L(\theta)$", "标量损失", "越小代表当前目标拟合越好"),
            (r"$\nabla_\theta L$", "损失对参数的梯度", "指向损失增大最快方向"),
            (r"$\alpha$", "学习率", "一次更新的步长"),
        ],
        "derivation": r"""
### 1. 从线性模型开始

$\hat y=wx+b$。单样本平方误差取

$$L=(\hat y-y)^2=(wx+b-y)^2$$

由链式法则：

$$\frac{\partial L}{\partial w}=2(wx+b-y)x,\qquad
\frac{\partial L}{\partial b}=2(wx+b-y)$$

梯度告诉我们参数微小增大时损失怎样变化。要让损失下降，就走反方向：

$$w\leftarrow w-\alpha\frac{\partial L}{\partial w}$$

### 2. 神经网络只是更长的复合函数

若 $z=Wx+b, h=\operatorname{ReLU}(z),\ \hat y=Uh+c$，则梯度沿计算图从 $L$ 反向应用链式法则。`.backward()` 完成这件事，但只负责计算梯度；`optimizer.step()` 才修改参数。

### 3. 为什么每轮要清梯度

PyTorch 默认执行 `param.grad += new_grad`，这是为了支持多分支或梯度累积。因此标准训练顺序是：前向 → loss → `zero_grad` → `backward` → `step`。
""",
        "hand": r"""
取 $x=2,y=7,w=1,b=0$：$\hat y=2$，$L=25$。

$$\partial L/\partial w=2(2-7)\times2=-20$$

学习率 0.1 时，$w_{new}=1-0.1(-20)=3$。新预测为 6，误差从 5 降为 1。这个单步例子说明负梯度为何会让权重增大。
""",
        "mapping": [
            (r"$f_\theta$", "`model`", "`nn.Module` 及其参数"),
            (r"$\hat y$", "`prediction = model(x)`", "前向结果，shape 与目标一致"),
            ("$L$", "`loss_fn(prediction, y)`", "标量 tensor"),
            ("清空旧梯度", "`optimizer.zero_grad()`", "防止非预期累积"),
            (r"$\nabla_\theta L$", "`loss.backward()`", "写入每个参数的 `.grad`"),
            (r"$\theta\leftarrow\theta-\alpha g$", "`optimizer.step()`", "由优化器更新参数"),
        ],
        "read": "训练后 weight 应接近 3、bias 接近 0.5，loss 接近 0。不是精确等于，因为数据含噪声且优化只运行有限步。",
        "takeaways": ["网络训练始终围绕前向、损失、反向、更新四步。", "梯度是局部方向，不保证任意大学习率都稳定。", "shape、dtype、device 是 PyTorch 调试三件套。", "RL 只是在目标数据的产生方式上比监督学习更复杂。"],
        "check": ("`.backward()` 会直接更新模型权重吗？为什么训练循环需要 `zero_grad()`？", "不会；它只计算并累积梯度。`step()` 才更新权重。因为 PyTorch 默认累积 `.grad`，每个独立 batch 前通常要清零。"),
    },
    4: {
        "prereq": ["理解 episode 与 episode return", "会读分类交叉熵", "知道 percentile/分位数"],
        "question": "如果我们暂时不会计算策略梯度，能不能只模仿自己表现最好的那些轨迹？CEM 就是这个朴素想法的系统版本。",
        "symbols": [
            (r"$\tau$", "一条完整轨迹", "状态、动作、奖励序列"),
            (r"$R(\tau)$", "轨迹总回报", "用于给 episode 排名"),
            (r"$q_\rho$", r"$\rho$ 分位阈值", "决定谁是 elite"),
            ("$E$", "elite 轨迹集合", "高于阈值的样本"),
            (r"$\pi_\theta(a|s)$", "待训练的分类策略", "输出动作概率"),
        ],
        "derivation": r"""
### 1. 把好轨迹变成监督数据

采样 batch $\{\tau_i\}$，选择回报达到分位阈值的集合：

$$E=\{\tau_i:R(\tau_i)\ge q_\rho\}$$

我们希望新策略提高 elite 中动作的似然：

$$\max_\theta\prod_{(s,a)\in E}\pi_\theta(a|s)$$

乘积容易数值下溢，取对数把乘积变成求和：

$$\max_\theta\sum_{(s,a)\in E}\log\pi_\theta(a|s)$$

优化器通常做最小化，于是加负号：

$$\boxed{L=-\frac1{|E|}\sum_{(s,a)\in E}\log\pi_\theta(a|s)}$$

这恰好是多分类交叉熵。

### 2. 为什么可能学不动

若所有轨迹回报都相同，elite 与普通样本没有区别；若成功极少，batch 里可能一个成功样本也没有。CEM 依赖“一个 batch 内已经出现可区分的好行为”。
""",
        "hand": r"""
回报 `[3,10,6,1,8]` 的 70 分位阈值约为 7.6，因此只有回报 10 和 8 的 episode 进入 elite。假设它们共提供 4 个 `(state, action)`，交叉熵只在这 4 个动作上训练。

思考：若把分位数提高到 95，可能只剩一个 episode，梯度方差会变大；降到 20，则大量平庸动作被当成正确标签。
""",
        "mapping": [
            (r"$R(\tau_i)$", "`returns`", "每个 episode 一个标量"),
            (r"$q_\rho$", "`np.percentile(returns, 70)`", "elite 阈值"),
            ("$E$", "`elite_ids`", "满足阈值的轨迹索引"),
            ("策略 logits", "`logits`，shape `(samples, actions)`", "尚未 softmax 的动作分数"),
            (r"$-\log\pi(a|s)$", "`F.cross_entropy(logits, actions)`", "稳定地完成 log-softmax + NLL"),
        ],
        "read": "先确认 elite 索引是回报 10 和 8 对应的位置。梯度 shape 与 logits 相同；每行所选动作的梯度会推动该动作概率上升。",
        "takeaways": ["CEM 把 RL 暂时转成 elite 行为的监督学习。", "分位数控制样本质量与数量的平衡。", "它适合短 episode、稠密反馈和容易偶然成功的任务。", "稀疏奖励下可保留历史 elite 或改用更合适的 RL 方法。"],
        "check": ("为什么 CEM 的 loss 是交叉熵？分位数越高是否一定越好？", "因为它最大化 elite 动作在策略下的对数似然，与分类交叉熵等价。分位数过高会让样本过少、方差很大，因此不一定更好。"),
    },
    5: {
        "prereq": ["掌握第 1 章的回报递归", "会做离散概率加权平均", "理解 `max` 与 `argmax` 的区别"],
        "question": "如果环境规则完全已知，我们能否不采集成千上万条轨迹，直接通过规则算出最优行为？这就是动态规划。",
        "symbols": [
            ("$P(s'|s,a)$", "转移概率", "动作后到达下一状态的概率"),
            ("$R(s,a,s')$", "一步奖励", "可依赖前后状态与动作"),
            ("$V_*(s)$", "最优状态价值", "从 $s$ 出发能达到的最大期望回报"),
            ("$Q_*(s,a)$", "最优动作价值", "先做 $a$，以后最优"),
            ("Bellman backup", "用旧价值计算新价值", "把未来信息向前传播一步"),
        ],
        "derivation": r"""
### 1. 固定策略时

由 $G_t=R_{t+1}+\gamma G_{t+1}$：

$$V^\pi(s)=\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)[R(s,a,s')+\gamma V^\pi(s')]$$

外层平均来自策略随机选动作，内层平均来自环境随机转移。

### 2. 从固定策略到最优策略

最优价值允许在当前状态选择最好的动作，因此把策略加权平均替换为最大值：

$$\boxed{V_*(s)=\max_a\sum_{s'}P(s'|s,a)[R+\gamma V_*(s')]}$$

若把“尝试动作 $a$ 后的期望”单独命名为 $Q_*(s,a)$：

$$Q_*(s,a)=\sum_{s'}P(s'|s,a)[R+\gamma\max_{a'}Q_*(s',a')]$$

### 3. 为什么反复迭代能传播奖励

初始 $V_0=0$ 时，一次 backup 看到一步内奖励；两次看到两步内奖励。对 $\gamma<1$，Bellman optimality operator 是 contraction，误差最多乘 $\gamma$，因此有限 MDP 中会收敛到唯一不动点。
""",
        "hand": r"""
小环境中，状态 1 选择右动作会立即到终点并得 1，所以第一次 backup 得 $V_1(1)=1$。状态 0 选择右动作到状态 1，但旧价值还是 0，因此第一次仍为 0；第二次才得到

$$V_2(0)=0+0.9\times V_1(1)=0.9$$

这就是“奖励逐轮向前传播”。
""",
        "mapping": [
            ("$s,a$", "`reward[s, a]`", "表格的行列索引"),
            ("$s'$", "`next_state[s, a]`", "确定性转移；随机环境需概率加权"),
            (r"$R+\gamma V(s')$", "`reward + gamma * V[next_state]`", "所有动作的一步 lookahead"),
            (r"$\max_a$", "`q.max(axis=1)`", "得到每个状态的最优价值"),
            (r"$\arg\max_a$", "`q.argmax(axis=1)`", "得到贪心动作"),
            ("收敛条件", "`max(abs(new_V - V)) < tol`", "Bellman residual 的简化监控"),
        ],
        "read": "状态 1 的贪心动作应直达终点；状态 0 的价值约为 0.9。终止状态的未来价值必须屏蔽，否则可能重复获得不存在的奖励。",
        "takeaways": ["Bellman 方程来自回报的一步递归。", "期望处理环境随机性，max 处理最优决策。", "value iteration 是不断应用 Bellman backup。", "`max` 给价值，`argmax` 给动作，二者不要混淆。"],
        "check": ("为什么状态 0 的价值要到第二轮才出现？`q.max` 和 `q.argmax` 分别返回什么？", "同步迭代第一轮使用旧的全零价值，所以还看不到两步后的奖励；第二轮才由状态 1 传播回来。`max` 返回最佳数值，`argmax` 返回最佳动作索引。"),
    },
    6: {
        "prereq": ["理解第 5 章 Bellman 最优方程", "掌握 PyTorch 的前向/损失/反向", "理解 epsilon-greedy"],
        "question": "现实环境没有转移概率表，而且状态可能是图像。怎样用实际经历替代模型，用神经网络替代表格？",
        "symbols": [
            (r"$Q_\theta(s,a)$", "online network 的预测", "当前正在更新"),
            (r"$Q_{\theta^-}$", "target network", "短时间保持不变"),
            (r"$\delta$", "TD error", "目标减预测"),
            (r"$\mathcal D$", "replay buffer", "保存 transition"),
            ("$d$", "终止标志", "真正终止时为 1"),
        ],
        "derivation": r"""
### 1. 从 Bellman 方程到采样更新

Bellman 最优方程中的期望需要知道 $P$。未知模型时，用一次真实 transition $(s,a,r,s')$ 作为随机样本：

$$y=r+\gamma(1-d)\max_{a'}Q(s',a')$$

预测与目标之差为

$$\delta=y-Q(s,a)$$

表格 Q-learning 更新 $Q\leftarrow Q+\alpha\delta$。

### 2. 把表格换成神经网络

网络无法只修改一个格子，因此把 TD error 变成回归损失：

$$L(\theta)=\mathbb E_{(s,a,r,s')\sim\mathcal D}\left[\ell(Q_\theta(s,a),y)\right]$$

其中 $\ell$ 可用 MSE 或 Huber loss。

### 3. 为什么需要 target network

若 $y$ 也由同一个快速变化的网络产生，就像一边追逐一边移动的目标。DQN 用延迟参数 $\theta^-$ 计算

$$y=r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a')$$

并在计算 target 时停止梯度。这是 semi-gradient：只对左侧预测求导。

### 4. replay buffer 解决什么

连续帧高度相关，而且只反映最近策略。随机抽取旧 transition 能打散时间相关性并复用样本，但也使数据变成 off-policy；Q-learning 的目标形式允许这种复用。
""",
        "hand": r"""
假设 $r=1,\gamma=0.9,d=0$，target network 在下一状态输出 `[2,3]`，当前 $Q(s,a)=2.5$：

$$y=1+0.9\times3=3.7,\qquad \delta=3.7-2.5=1.2$$

若该 transition 终止，$y=1$，不能再加下一状态价值。先手算这个例子，再去读实际 batch 代码。
""",
        "mapping": [
            (r"$Q_\theta(s,\cdot)$", "`net(states)`，shape `(B, A)`", "每个状态的所有动作价值"),
            (r"$Q_\theta(s,a)$", "`.gather(1, actions[:, None])`", "只取实际动作列"),
            (r"$\max_{a'}Q_{\theta^-}(s',a')$", "`target_net(next_states).max(1).values`", "下一状态贪心价值"),
            ("$1-d$", "`~terminated`", "终止样本不 bootstrap"),
            ("$y$", "`rewards + gamma * mask * next_q`", "必须停止梯度"),
            ("$L$", "`smooth_l1_loss(q_sa, target)`", "对 batch 平均的 TD loss"),
        ],
        "read": "当前 notebook 先用表格 chain 验证采样更新。输出策略应在非终止状态选择右移。随后阅读项目 DQN 脚本时，按上表逐行寻找 network、gather、target、mask 与 loss。",
        "takeaways": ["Q-learning 是 Bellman 最优方程的采样近似。", "DQN 用网络替代表格，用回归 loss 替代单格更新。", "replay buffer 打散相关性并复用经验。", "target network 与停止梯度让目标更稳定。"],
        "check": ("为什么 target 要停止梯度？为什么 `gather` 只取执行动作，而不是对当前 Q 也取 max？", "target 是监督信号，DQN 的 semi-gradient 不沿 target 分支更新。行为数据告诉我们实际执行的是哪个动作，因此只应回归该动作的 Q；对当前 Q 取 max 会训练错误的动作。"),
    },
}


V2_GUIDE.update({
    7: {
        "prereq": ["理解 DQN 的 transition 与 target network", "知道 Python iterator 和 dataclass", "理解 n-step return 的目的"],
        "question": "完整 RL 代码为什么容易写成几百行？因为动作选择、环境交互、经验拼接、回放和训练纠缠在一起。高层库的核心价值是把这些职责拆开。",
        "symbols": [
            ("selector", "把分数变成动作", "argmax、epsilon-greedy、概率采样"),
            ("agent", "观测到动作的可调用对象", "组合网络与 selector"),
            ("experience source", "驱动环境并产生经验", "负责 episode 边界"),
            ("replay buffer", "保存并随机采样经验", "训练数据仓库"),
            ("target net", "在线/目标网络管理器", "同步或 soft update"),
        ],
        "derivation": r"""
### 1. n-step return 从哪里来

反复展开 $G_t=R_{t+1}+\gamma G_{t+1}$：

$$G_t=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^nG_{t+n}$$

把最后未知的真实回报替换成价值估计，得到 n-step target：

$$y_t=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^nV(S_{t+n})$$

若中途真正终止，最后一项消失，而且实际 $n$ 可能小于设定值。

### 2. 为什么要拆成组件

数据流可以写成：

```text
observation → agent → action → environment
     ↑                         ↓
     └──── experience source ← transition
                    ↓
              replay buffer → trainer
```

每个组件只有一个主要变化原因，便于单独测试。高层库不是替你理解算法，而是把已经理解的重复工程模式封装起来。
""",
        "hand": r"""
奖励 `[1,0,2]`、$\gamma=0.9$ 的 3-step reward 为
$1+0.9\times0+0.9^2\times2=2.62$。

若第 3 步后未终止且 $V(S_{t+3})=4$，target 还要加 $0.9^3\times4=2.916$，总计 5.536。若第 3 步终止，则不加。
""",
        "mapping": [
            (r"$\pi$ 或 $Q$", "agent 内的 model", "把 observation 变成 logits/Q"),
            ("动作选择规则", "selector", "网络输出到具体动作"),
            ("$(s,a,r,s')$", "`Experience`", "跨组件传递的最小数据结构"),
            ("$R_t^{(n)}$", "`n_step_return(rewards, gamma)`", "折扣奖励前缀"),
            ("终止", "`last_state is None`", "清楚表达禁止 bootstrap"),
            (r"$\theta^-$", "`TargetNet`", "管理目标参数同步"),
        ],
        "read": r"示例 `Experience` 是不可变数据对象，方便排查谁修改了经验。注意 `reward` 已经可以是 n-step 聚合值，此时 bootstrap 必须使用 $\gamma^n$ 而不是 $\gamma$。",
        "takeaways": ["高层库拆分工程职责，不改变算法数学。", "n-step return 来自反复展开一步回报递归。", "终止状态要在经验表示中无歧义。", "先理解纯 Python 数据流，再使用 PTAN 等库。"],
        "check": (r"3-step experience 为什么 bootstrap 要乘 $\gamma^3$？若第 2 步提前终止怎么办？", "因为从当前时刻到 bootstrap 状态跨了 3 次折扣。若第 2 步终止，只累计实际两步奖励，不再 bootstrap，并记录真实步数或用 `last_state=None`。"),
    },
    8: {
        "prereq": ["能完整解释基础 DQN", "理解 TD error", "熟悉概率分布和期望的直觉"],
        "question": "DQN 不是一个不可改变的整体。过估计、探索、样本利用和表示能力分别有不同的病因，也应使用不同的药。",
        "symbols": [
            (r"$\delta_i$", "第 $i$ 条经验的 TD error", "PER priority 的依据"),
            ("$V(s)$", "状态本身的价值", "Dueling value stream"),
            ("$A(s,a)$", "动作相对平均水平的优势", "Dueling advantage stream"),
            ("$Z(s,a)$", "随机回报分布", "distributional RL 的预测对象"),
            ("$n$", "n-step 长度", "信用传播速度与方差的折中"),
        ],
        "derivation": r"""
### 1. Double DQN：把“选择”与“评价”拆开

同一组带噪估计既选最大动作又报告其数值，会偏向被高估的动作。Double DQN 用 online network 选择：

$$a^*=\arg\max_a Q_\theta(s',a)$$

再用 target network 评价：

$$y=r+\gamma Q_{\theta^-}(s',a^*)$$

它不能消除所有误差，但减少了 max 带来的系统性正偏差。

### 2. Dueling：解决动作难区分时的表示问题

直接写 $Q=V+A$ 不可辨识：给 $V$ 加常数、给所有 $A$ 减同一常数，Q 不变。减去 advantage 均值固定了一个规范：

$$Q(s,a)=V(s)+A(s,a)-\frac1{|\mathcal A|}\sum_{a'}A(s,a')$$

### 3. PER：把训练预算花在误差大的样本上

$P(i)=p_i^\alpha/\sum_jp_j^\alpha$ 改变了采样分布，因此用
$w_i=(NP(i))^{-\beta}$ 部分校正偏差。$\alpha=0$ 退化为均匀采样，$\beta\to1$ 时校正更充分。

### 4. Distributional：不只预测均值

Bellman distributional target 处理随机回报分布 $Z$。Categorical DQN 把概率质量投影到固定 support；Q 值只是该分布的期望 $\sum_i p_i z_i$。
""",
        "hand": r"""
online 下一状态 Q 为 `[2,3,1]`，所以选择动作 1；target Q 为 `[2.5,2,4]`。

- 普通 DQN 用 target 自己的最大值 4
- Double DQN 在 online 选择的动作 1 上读取 target，得到 2

这个刻意构造的例子显示“谁负责选、谁负责评”会改变目标。
""",
        "mapping": [
            (r"$a^*=\arg\max Q_\theta$", "`online_next.argmax(...)`", "online 只负责选择"),
            (r"$Q_{\theta^-}(s',a^*)$", "`target_next.gather(..., chosen_action)`", "target 负责评价"),
            ("$V(s)$", "dueling 的 value head，shape `(B,1)`", "所有动作共享"),
            ("$A(s,a)$", "advantage head，shape `(B,A)`", "动作差异"),
            ("去均值", "`advantage.mean(dim=1, keepdim=True)`", "保持 batch/action 维正确"),
            ("$p_i,w_i$", "priority sampler 与 loss weight", "采样概率和偏差校正"),
        ],
        "read": "示例中 ordinary target=4、double target=2。Dueling 聚合后 Q 的动作均值等于 value；这正是减去平均 advantage 的效果。",
        "takeaways": ["每个 DQN 扩展都应对应明确问题。", "Double 分离动作选择和价值评价。", "Dueling 分离状态价值和相对动作优势。", "PER 改变采样分布，需要重要性权重。", "Rainbow 是多项兼容扩展的组合，不是一个神秘新 loss。"],
        "check": ("Double DQN 为什么仍然需要 target network？Dueling 中为什么要减 advantage 的均值？", "Double 仍需 target network 提供相对稳定的评价。减均值用于解决 $V+A$ 分解的不唯一性，并让 value 对应各动作 Q 的平均基线。"),
    },
    9: {
        "prereq": ["能运行并计时 PyTorch 代码", "理解 batch 前向", "知道 CPU/GPU 和进程的基本区别"],
        "question": "训练慢时，到底该增加环境、换 GPU，还是减少日志？不先测量瓶颈，优化往往只是移动等待位置。",
        "symbols": [
            ("$T_{env}$", "环境采样时间", "step、模拟、预处理"),
            ("$T_{net}$", "网络计算时间", "前向与反向"),
            ("throughput", "单位时间 environment steps", "工程速度指标"),
            ("sample efficiency", "达到回报所需样本数", "算法效率指标"),
            ("wall-clock", "达到目标回报的实际时间", "最终常用指标"),
        ],
        "derivation": r"""
### 1. 先写出时间预算

串行循环近似为

$$T_{step}=T_{env}+T_{prep}+T_{transfer}+T_{net}+T_{log}$$

吞吐量约为 $1/T_{step}$。优化某一项的收益受其他项限制。

### 2. Amdahl 定律提醒我们上限

若可并行部分比例为 $p$，使用 $N$ 个 worker 的理论加速上限：

$$S(N)=\frac1{(1-p)+p/N}$$

若只有 60% 时间在环境，哪怕环境并行到无限快，总加速也不超过 $1/(1-0.6)=2.5$ 倍。

### 3. batching 为什么有效

逐样本调用网络重复产生 Python、kernel launch 和内存访问开销。batch 把相同矩阵运算合并，使硬件并行单元得到更充分利用。但 batch 太大会增加延迟并让策略数据更旧。

### 4. 工程速度不等于学习速度

最终比较应是达到相同评估回报的 wall-clock，而不是只看 FPS。一个 FPS 更高但样本效率下降的版本可能更慢收敛。
""",
        "hand": r"""
假设一次迭代：环境 60 ms、网络 30 ms、其他 10 ms，总计 100 ms。把网络加速 3 倍后，总计 $60+10+10=80$ ms，只加速 1.25 倍，而不是 3 倍。

这说明 profiler 的第一问题应是“时间花在哪里”，不是“哪个组件听起来最昂贵”。
""",
        "mapping": [
            ("逐样本网络调用", "`[net(row[None]) for row in x]`", "大量小调用"),
            ("batch 网络调用", "`net(x)`", "一次矩阵批处理"),
            ("$T$", "`time.perf_counter()` 差值", "墙钟时间"),
            ("正确性", "`torch.allclose(loop_out, batch_out)`", "优化前后先验证等价"),
            ("吞吐量", "steps / elapsed seconds", "应在完整训练管线测量"),
        ],
        "read": "通常 batch 前向更快，但小规模 CPU 测量可能波动。单次结果不能作为可靠 benchmark；应 warm-up、多次重复、取中位数，并在目标硬件测量。",
        "takeaways": ["先测量、再优化、最后重新测量。", "Amdahl 定律决定局部优化的收益上限。", "batching、向量环境和多进程针对不同瓶颈。", "报告 wall-clock 学习曲线，而不只报告 FPS。"],
        "check": ("如果 GPU 利用率很低，是否能直接断定网络太小？", "不能。GPU 可能在等待环境、数据传输、Python 调度或同步；需要 profiler 和时间分解确定真正瓶颈。"),
    },
    10: {
        "prereq": ["理解 Gym 环境建模", "知道简单收益和复利", "会做按时间切分的数据实验"],
        "question": "一个回测赚钱，是策略真的学到了规律，还是偷看未来、忽略成本或记账出错？交易环境首先是数据与评估问题。",
        "symbols": [
            ("$p_t$", "时刻 $t$ 的价格", "观测只能使用当时可知值"),
            ("$h_t$", "区间 $[t,t+1)$ 的仓位", "正/零/负表示多仓/空仓/空头"),
            ("$r_t$", "资产收益", "价格相对变化"),
            ("$c$", "单位换仓成本", "手续费、点差、滑点近似"),
            ("equity", "累计净值", "逐步奖励复合后的结果"),
        ],
        "derivation": r"""
### 1. 从价格到可加的奖励

简单收益为 $(p_{t+1}-p_t)/p_t$。对数收益

$$\ell_t=\log p_{t+1}-\log p_t=\log(p_{t+1}/p_t)$$

具有可加性：$\sum_t\ell_t=\log(p_T/p_0)$，适合作为逐步奖励。

### 2. 仓位怎样影响收益

在区间 $[t,t+1)$ 持有仓位 $h_t$，忽略杠杆细节时毛收益近似 $h_t\ell_t$。从 $h_{t-1}$ 换到 $h_t$ 产生交易量 $|h_t-h_{t-1}|$，所以

$$R_{t+1}=h_t\ell_t-c|h_t-h_{t-1}|$$

注意时间顺序：先根据时刻 $t$ 可见信息决定 $h_t$，再获得 $t\to t+1$ 的收益。

### 3. 为什么不能随机划分数据

市场分布随时间变化，特征窗口也含相邻数据。随机划分会让未来时期进入训练集，甚至让高度重叠的窗口分到训练和测试两侧。正确顺序是 train → validation → test，并只在训练集拟合标准化参数。
""",
        "hand": r"""
价格从 100 到 102，对数收益约 $\log(1.02)=0.0198$。若从空仓换到多仓，$c=0.001$，该步净奖励约 $0.0188$。

若下一步仍持多仓，不再支付换仓成本；若从 +1 直接换到 -1，交易量是 2，成本是 $2c$，不是 $c$。
""",
        "mapping": [
            (r"$\ell_t$", "`np.diff(np.log(prices))`", "每个相邻价格区间一个值"),
            ("$h_t$", "`positions`", "长度与收益区间相同"),
            ("$h_{t-1}$", "`np.r_[0., positions[:-1]]`", "初始假设空仓"),
            ("换仓成本", "`cost_rate * abs(position - previous)`", "仓位变化绝对值"),
            ("累计净值", "`np.exp(np.cumsum(rewards))`", "对数奖励还原成净值倍数"),
        ],
        "read": "检查每一步奖励，而不只看最终净值。示例中换仓时出现成本扣减；从多头切到空头成本最大。这个逐步审计应成为交易环境的单元测试。",
        "takeaways": ["动作、持仓和收益区间的时间对齐最重要。", "交易成本与仓位变化量相关。", "训练/验证/测试必须按时间顺序。", "教学回测不等于可部署投资策略。"],
        "check": ("为什么从仓位 +1 变成 -1 的成本按 2 计算？标准化均值能否用全部数据估计？", "仓位变化绝对值是 $|-1-1|=2$，相当于先平多仓再开空仓。标准化参数只能用训练期估计，否则泄漏未来分布信息。"),
    },
    11: {
        "prereq": ["理解概率分布与期望", "会求对数和链式法则", "掌握 episode return"],
        "question": "Q-learning 先估计每个动作的价值再选择；能否直接让带来高回报的动作变得更可能？Policy gradient 给出了可微答案。",
        "symbols": [
            (r"$\tau$", "整条轨迹", "由状态、动作、奖励组成"),
            (r"$p_\theta(\tau)$", "策略生成轨迹的概率", "环境概率不依赖策略参数"),
            (r"$J(\theta)$", "期望回报目标", "希望最大化"),
            (r"$\log\pi_\theta(a|s)$", "动作对数概率", "其梯度提供更新方向"),
            (r"$\hat A_t$", "advantage 估计", "动作比基线好多少"),
        ],
        "derivation": r"""
### 1. 从期望回报出发

$$J(\theta)=\sum_\tau p_\theta(\tau)R(\tau)$$

直接求导：

$$\nabla J=\sum_\tau \nabla p_\theta(\tau)R(\tau)$$

使用 log-derivative trick：$\nabla p=p\nabla\log p$，得到

$$\nabla J=\mathbb E_{\tau\sim p_\theta}[R(\tau)\nabla\log p_\theta(\tau)]$$

### 2. 轨迹概率为什么化成动作 log-prob 之和

$$p_\theta(\tau)=p(s_0)\prod_t\pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)$$

环境转移 $P$ 不含 $\theta$，因此

$$\nabla\log p_\theta(\tau)=\sum_t\nabla\log\pi_\theta(a_t|s_t)$$

再利用过去奖励不受当前动作影响，可把整条轨迹回报换成 reward-to-go $G_t$，得到 REINFORCE。

### 3. baseline 为什么不引入偏差

对只依赖状态的 $b(s)$：

$$\mathbb E_{a\sim\pi}[b(s)\nabla\log\pi(a|s)]
=b(s)\sum_a\pi(a|s)\nabla\log\pi(a|s)
=b(s)\nabla\sum_a\pi(a|s)=0$$

所以减去 baseline 不改变期望，只降低方差。
""",
        "hand": r"""
某状态有两个动作，当前概率 `[0.6,0.4]`。如果采样到动作 0 且 advantage 为正，最小化 $-\log0.6\times A$ 会提高动作 0 概率；若 advantage 为负，则会降低它。

因此记忆口诀是：**正 advantage 奖励所选动作，负 advantage 惩罚所选动作**。
""",
        "mapping": [
            (r"$\pi_\theta(\cdot|s)$", "`Categorical(logits=policy(states))`", "稳定地产生离散策略分布"),
            (r"$\log\pi(a_t|s_t)$", "`dist.log_prob(actions)`", "shape `(B,)`"),
            (r"$\hat A_t$", "`advantages`", "通常标准化并停止梯度"),
            (r"$-\log\pi\hat A$", "`-(log_prob * advantages.detach()).mean()`", "梯度下降等价于提升目标"),
            (r"$H(\pi)$", "`dist.entropy().mean()`", "鼓励策略保持探索"),
        ],
        "read": "代码没有环境，只隔离展示策略 loss 的方向。检查 gradient norm 非零；随后把 `advantages` 全部改成 0，梯度应只剩 entropy 项。",
        "takeaways": ["likelihood-ratio trick 让不可微环境不妨碍策略求梯度。", "轨迹 log-prob 是各动作 log-prob 之和。", "reward-to-go 和 baseline 都用于降低方差。", "实现时对 advantage 停止梯度。"],
        "check": ("环境转移不可微，为什么 policy gradient 仍能训练？baseline 为什么不能依赖当前采样动作而随意设置？", "梯度只通过策略的 log-prob 计算，环境只负责提供采样和回报。一般 baseline 需保证相关期望项为零；只依赖状态的 baseline 满足这一性质，任意依赖动作的 baseline 可能引入偏差。"),
    },
    12: {
        "prereq": ["掌握 REINFORCE 与 baseline", "理解一步 TD target", "能区分 policy 与 value 输出"],
        "question": "REINFORCE 要等整局结束且方差很大。能否边玩边学习，并让一个网络告诉 actor‘这一步比预期好多少’？",
        "symbols": [
            ("actor", r"策略 $\pi_\theta$", "决定动作"),
            ("critic", r"价值 $V_\phi$", "评价状态"),
            (r"$\delta_t$", "TD residual", "一步 surprise"),
            (r"$\hat A_t$", "advantage 估计", "actor 的加权信号"),
            ("$c_v,c_e$", "value/entropy 系数", "平衡联合 loss"),
        ],
        "derivation": r"""
### 1. 用 critic 构造低方差 baseline

理想 advantage 为 $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$。未知 $Q$ 时，用一步样本近似：

$$Q^\pi(s_t,a_t)\approx R_{t+1}+\gamma V_\phi(s_{t+1})$$

所以

$$\hat A_t=R_{t+1}+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t)$$

这同时是 TD error。

### 2. actor 与 critic 各学什么

actor 最大化 $\log\pi(a_t|s_t)\hat A_t$，实现中最小化其负数。critic 回归 bootstrap target：

$$L_V=\tfrac12(V_\phi(s_t)-y_t)^2=\tfrac12\hat A_t^2$$

联合目标常写成

$$L=L_{actor}+c_vL_V-c_eH(\pi)$$

actor 使用 `advantage.detach()`，避免策略项通过 advantage 意外更新 critic。

### 3. A2C 与 A3C 的差别

A2C 同步等待多个环境收集一段 rollout，再组成 batch 更新；A3C 的 worker 异步对共享模型施加梯度。核心 actor-critic 数学相同，主要区别是并发与数据时效。
""",
        "hand": r"""
$r=1,\gamma=0.9,V(s)=0.4,V(s')=0.5$，未终止：

$$y=1+0.9\times0.5=1.45,\quad \hat A=1.45-0.4=1.05$$

正 advantage 表示结果比 critic 原先预期好，actor 应提高该动作概率。若真正终止，则 $y=1$，advantage 为 0.6。
""",
        "mapping": [
            ("$V(s_t)$", "`value`", "critic 当前预测，需梯度"),
            ("$V(s_{t+1})$", "`next_value`", "构造 target 时停止梯度"),
            ("$y_t$", "`reward + gamma * (~terminated) * next_value`", "bootstrap target"),
            (r"$\hat A_t$", "`target - value`", "actor 权重兼 critic residual"),
            ("$L_{actor}$", "`-(log_prob * advantage.detach()).mean()`", "不经 advantage 回传"),
            ("$L_V$", "`0.5 * advantage.square().mean()`", "更新 critic"),
        ],
        "read": "输出 target 第三个样本应等于 reward，因为它终止。去掉 `detach` 虽然代码仍运行，但 actor loss 会额外沿 value 分支传播，改变我们定义的优化目标。",
        "takeaways": ["actor 决策，critic 评价。", "一步 TD error 可作为 advantage 的低方差估计。", "联合 loss 的各项更新不同参数路径。", "A2C/A3C 的数学主线相同，系统实现不同。"],
        "check": (r"为什么 critic loss 可以写成 $\tfrac12\hat A^2$？actor loss 中为什么 detach advantage？", "因为 advantage 正是 bootstrap target 减当前 value，即 critic 的回归残差。detach 用于让策略项只更新 log-prob 路径，critic 由自己的 value loss 更新。"),
    },
})


V2_GUIDE.update({
    13: {
        "prereq": ["理解 POMDP 的直觉", "知道 token、embedding 和 sequence", "掌握动作价值或策略输出"],
        "question": "看到‘你在厨房里’，智能体还需要记住刚才拿过钥匙吗？文本游戏把语言理解、记忆和决策压缩进同一个任务。",
        "symbols": [
            ("$O_t$", "当前文本观测", "可能不包含完整世界状态"),
            ("$H_t$", "交互历史", "观测与动作序列"),
            ("$z_t$", "历史的向量表示", "RNN/Transformer/池化结果"),
            ("$m_a$", "合法动作 mask", "1 合法、0 非法"),
            ("vocabulary", "token 到整数的映射", "决定模型输入 id"),
        ],
        "derivation": r"""
### 1. 为什么当前文本不一定是 Markov 状态

相同房间描述可能对应“钥匙已经拿走”和“钥匙仍在桌上”两种隐藏状态。若观测不足，应让表示依赖历史：

$$z_t=f_\theta(O_1,A_1,\ldots,O_t)$$

然后估计 $Q(z_t,a)$ 或 $\pi(a|z_t)$。简单任务可拼接最近若干观测，长依赖更适合 recurrent/attention 模型。

### 2. token 怎样进入网络

文本经过 tokenizer 得到 id 序列 $(i_1,\ldots,i_L)$，embedding 查表得到向量，再池化或编码：

$$e_j=E[i_j],\qquad z=\operatorname{Encoder}(e_1,\ldots,e_L)$$

padding 只是凑 batch 长度，必须用 attention mask 排除。

### 3. action mask 为什么在 logits 上操作

令非法动作 logit 为 $-\infty$：

$$\tilde z_a=\begin{cases}z_a&m_a=1\\-\infty&m_a=0\end{cases}$$

softmax 后 $e^{-\infty}=0$，非法动作概率严格为零。事后把概率乘 0 再归一化也能做，但 logits mask 更稳定、实现更直接。
""",
        "hand": r"""
logits `[2,1,3,-1]`，只有动作 0、2 合法。mask 后为 `[2,-∞,3,-∞]`。合法动作 softmax 概率约为 `[0.269,0.731]`，非法动作均为 0。

注意：mask 不能把所有动作都屏蔽，否则 softmax 会产生 NaN。数据管线应断言每个状态至少有一个合法动作。
""",
        "mapping": [
            ("文本 $O_t$", "`observation`", "原始字符串"),
            ("token 序列", "`tokens = tokenize(observation)`", "规范化后的词元"),
            ("token id", "`vocab[word]`", "embedding 的整数索引"),
            ("$z_a$", "`logits`，shape `(num_actions,)`", "每个候选命令分数"),
            ("$m_a$", "`legal` bool tensor", "合法命令位置"),
            (r"$-\infty$ mask", "`masked_fill(~legal, -torch.inf)`", "softmax 前屏蔽"),
        ],
        "read": "示例 tokenizer 故意简单，只讲数据形状。真实项目的 preprocessing、预训练编码器和 action candidates 分别查看链接文件，不要把示例 tokenizer 当成生产方案。",
        "takeaways": ["文本观测往往是部分可观测的，需要历史。", "tokenizer、padding mask 和 action mask 都属于算法输入定义。", "非法动作应在采样前屏蔽。", "先保证小词表/小模型的数据管线正确，再接预训练模型。"],
        "check": ("padding mask 和 action mask 分别屏蔽什么？为什么不能让一个状态的所有 action 都非法？", "padding mask 屏蔽序列中补齐位置，action mask 屏蔽当前状态不可执行的命令。全非法会让 softmax 分母为零等价情形，产生 NaN。"),
    },
    14: {
        "prereq": ["理解 Gym 环境和图像/DOM 观测", "了解行为克隆", "会处理二维坐标"],
        "question": "网页按钮的位置、大小和 DOM 会变化，奖励又常常只在任务完成时出现。智能体怎样从少量演示学会第一步？",
        "symbols": [
            ("$s$", "截图、DOM 或融合观测", "只能来自授权 sandbox 页面"),
            ("$a$", "点击/输入/滚动动作", "可按元素或坐标定义"),
            ("$D$", "演示数据集", "人类或脚本产生的 $(s,a)$"),
            ("$L_{BC}$", "行为克隆损失", "模仿演示动作"),
            ("viewport", "浏览器可视区域", "坐标归一化的基准"),
        ],
        "derivation": r"""
### 1. 行为克隆就是动作的最大似然

给定演示 $D=\{(s_i,a_i)\}$，最大化演示动作概率：

$$\max_\theta\prod_i\pi_\theta(a_i|s_i)$$

取负对数得到分类或连续回归形式的行为克隆 loss：

$$L_{BC}=-\frac1N\sum_i\log\pi_\theta(a_i|s_i)$$

它提供稀疏奖励之前的合理初始策略，但只覆盖演示访问过的状态。

### 2. covariate shift 为什么会累积错误

执行时，策略的一个小错误会进入演示中没出现过的页面状态；在那里预测更差，又导致更大偏移。因此常结合 DAgger 式重新标注或 RL 微调。

### 3. 坐标离散化

对于宽 $W$、高 $H$ 的页面和 $C\times R$ 网格：

$$col=\left\lfloor xC/W\right\rfloor,\quad row=\left\lfloor yR/H\right\rfloor$$

动作编号为 $a=row\cdot C+col$。反解使用格子中心能减少边界歧义。分辨率越高，定位越准，但动作空间更大、探索更难。
""",
        "hand": r"""
页面 $800\times600$，网格 $8\times6$，点击 $(420,180)$：

- `col=floor(420/800*8)=4`
- `row=floor(180/600*6)=1`
- `action=1*8+4=12`

动作 12 的格子中心为 $(450,150)$。编码后再解码不会恢复原点，而是恢复所属格子中心，这是量化误差。
""",
        "mapping": [
            ("$(x,y)$", "`x, y`", "像素点击坐标"),
            ("$W,H$", "`width, height`", "viewport 尺寸"),
            ("$C,R$", "`cols, rows`", "动作网格分辨率"),
            ("$a$", "`row * cols + col`", "二维到一维动作 id"),
            ("演示 $D$", "`lib/demos.py` 中的轨迹数据", "监督预训练来源"),
            ("环境交互", "`lib/wob.py`", "页面状态与动作执行"),
        ],
        "read": "示例验证坐标转换，而非真实浏览器控制。先测试四角、中心和越界策略；真实项目再处理截图同步、元素变化与任务 reset。",
        "takeaways": ["行为克隆提供起点，但会遭遇分布偏移。", "动作表示决定学习难度。", "坐标必须相对 viewport 归一化。", "网页 RL 只能在明确授权的隔离环境运行。"],
        "check": ("提高网格分辨率有什么好处和代价？行为克隆训练 loss 很低为何仍可能执行失败？", "更高分辨率定位更准，但动作数增多、样本更稀疏。低训练 loss 只说明拟合演示分布；策略自己的错误会进入未见状态并累积。"),
    },
    15: {
        "prereq": ["理解 actor-critic", "会对复合函数使用链式法则", "了解 replay buffer 与 target network"],
        "question": "方向盘角度有无限多种可能，无法像 DQN 一样对所有动作取 max。怎样让网络直接给出最优连续动作？",
        "symbols": [
            (r"$\mu_\theta(s)$", "确定性 actor", "直接输出连续动作"),
            (r"$Q_\phi(s,a)$", "critic", "评价状态-动作对"),
            (r"$\theta^-,\phi^-$", "target 参数", "缓慢跟随在线网络"),
            (r"$\tau$", "soft-update 比例", "通常远小于 1"),
            ("$[l,h]$", "环境动作范围", "网络输出必须映射到该范围"),
        ],
        "derivation": r"""
### 1. critic 的目标

DDPG 是 off-policy actor-critic。给定 replay transition：

$$y=r+\gamma(1-d)Q_{\phi^-}(s',\mu_{\theta^-}(s'))$$

critic 最小化 $(Q_\phi(s,a)-y)^2$，形式与 DQN 相似，只是下一动作由 target actor 产生。

### 2. actor 怎样得到梯度

actor 希望最大化 critic 对其动作的评分：

$$J(\theta)=\mathbb E_s[Q_\phi(s,\mu_\theta(s))]$$

链式法则给出

$$\nabla_\theta J=\mathbb E[\nabla_aQ_\phi(s,a)|_{a=\mu(s)}\nabla_\theta\mu_\theta(s)]$$

自动微分会穿过 critic 的动作输入回到 actor；实现时最小化 `-critic(s, actor(s)).mean()`。

### 3. soft target update

$$\theta^-\leftarrow(1-\tau)\theta^-+\tau\theta$$

$\tau=1$ 等价于硬同步；小 $\tau$ 是指数移动平均，使 target 平滑变化。

### 4. 动作缩放

若 actor 用 `tanh` 输出 $u\in[-1,1]$，映射到 $[l,h]$：
$a=l+(u+1)(h-l)/2$。若各动作维度范围不同，`low/high` 必须是向量。
""",
        "hand": r"""
动作范围 $[-2,4]$：区间中心为 1，半宽为 3。因此 $u=-1,0,1$ 映射为 $-2,1,4$。

target 参数原为 2，online 参数更新到 4，$\tau=0.05$：新 target 为 $0.95\times2+0.05\times4=2.1$，只移动一小步。
""",
        "mapping": [
            (r"$u\in[-1,1]$", "actor 的 `tanh` 输出", "归一化动作"),
            (r"$a\in[l,h]$", "`scale_action(unit, low, high)`", "环境实际动作"),
            (r"$\theta,\theta^-$", "`online`, `target`", "在线与目标模块"),
            (r"$(1-\tau)\theta^-+\tau\theta$", "`target_p.mul_(1-tau).add_(online_p, alpha=tau)`", "原地 soft update"),
            ("actor loss", "`-critic(states, actor(states)).mean()`", "经 critic 动作梯度更新 actor"),
        ],
        "read": "缩放结果应为 `[-2,1,4]`。soft update 后 target 只吸收 online 变化的 5%。项目代码中再寻找 critic target、actor loss 与 exploration noise。",
        "takeaways": ["连续动作不能枚举，actor 直接产生动作。", "critic 学 Bellman target，actor 沿 critic 对动作的梯度上升。", "target actor/critic 都要停止 target 分支梯度。", "动作范围和探索噪声必须按每个维度校准。"],
        "check": ("训练 actor 时为什么不能把 critic 输出 detach？target actor 的输出为什么又要停止梯度？", "actor 需要梯度穿过 critic 的动作输入才能知道如何改变动作；target 分支只构造稳定监督信号，不应参与当前反向传播。"),
    },
    16: {
        "prereq": ["掌握 actor-critic 与 advantage", "理解概率比值", "知道 KL divergence 的直觉"],
        "question": "策略梯度的一次更新如果太大，新策略会突然忘掉旧行为。怎样在提升回报的同时限制策略变化？",
        "symbols": [
            (r"$\pi_{old}$", "采集当前 batch 的旧策略", "一个 PPO update 周期内固定"),
            (r"$r_t(\theta)$", "新旧动作概率比", "衡量该动作概率变化"),
            (r"$\epsilon$", "PPO clip 范围", "常见如 0.1 或 0.2"),
            ("KL", "新旧策略差异", "监控更新幅度"),
            (r"$\alpha$", "SAC entropy 温度", "探索与回报权衡"),
        ],
        "derivation": r"""
### 1. 为什么出现概率比

数据来自旧策略，但目标想评估新策略。importance sampling 给出

$$\mathbb E_{a\sim\pi_\theta}[f(a)]
=\mathbb E_{a\sim\pi_{old}}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}f(a)\right]$$

令 $r_t(\theta)=\exp(\log\pi_\theta-\log\pi_{old})$，unclipped surrogate 为 $r_t\hat A_t$。

### 2. PPO 为什么取两个项的 minimum

$$L^{clip}=\mathbb E[\min(r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]$$

- $\hat A>0$：不再奖励把好动作概率提高到 $1+\epsilon$ 以上。
- $\hat A<0$：不再奖励把坏动作概率降低到 $1-\epsilon$ 以下。

`min` 与 advantage 符号共同形成分段行为；不能简单理解为“把 ratio 永远截断”。

### 3. TRPO/ACKTR/SAC 放在同一章的原因

TRPO 显式近似求解 KL 约束优化；ACKTR 用 K-FAC 近似自然梯度曲率；SAC 则是 off-policy 最大熵 actor-critic，目标包含 $+\alpha H(\pi)$。它们都关注稳定更新，但数据分布和优化形式不同。
""",
        "hand": r"""
设 $\epsilon=0.2$。

- $A=+1,r=1.5$：unclipped=1.5，clipped=1.2，取 1.2。
- $A=-1,r=0.5$：unclipped=-0.5，clipped=-0.8，取 -0.8。
- $A=+1,r=0.7$：取 0.7；PPO 不阻止“朝错误方向”的变化，而是让梯度继续纠正它。

这三个数值例子比只背公式更能理解 clipping。
""",
        "mapping": [
            (r"$\log\pi_{old}$", "`old_logp`", "rollout 时缓存并固定"),
            (r"$\log\pi_\theta$", "`new_logp`", "当前网络重新计算"),
            ("$r_t$", "`(new_logp - old_logp).exp()`", "用 log 差避免直接除小概率"),
            (r"$r_t\hat A$", "`unclipped`", "原始 surrogate"),
            ("clip 项", "`ratio.clamp(1-eps, 1+eps) * advantage`", "保守边界"),
            ("最大化 $L^{clip}$", "`loss = -minimum(...).mean()`", "优化器执行最小化"),
        ],
        "read": "逐项比较 `unclipped` 与 `clipped`，不要只看平均 loss。approximate KL 只是样本估计；实践中还应记录 clip fraction，并在 KL 过大时提前停止 epoch。",
        "takeaways": ["PPO ratio 来自 importance sampling。", "clipping 的作用必须结合 advantage 正负理解。", "old log-prob 在同一轮多 epoch 更新中保持固定。", "PPO、TRPO、ACKTR、SAC 都关注稳定性，但不是同一种算法。"],
        "check": ("为什么 PPO 存 log-prob 而不是概率？clipping 是否保证 KL 一定很小？", "log 概率数值更稳定，ratio 可由 log 差取 exp 得到。clipping 只限制采样动作上的 surrogate 激励，并不严格保证整体 KL，因此仍要监控 KL。"),
    },
    17: {
        "prereq": ["理解参数向量和随机采样", "知道 Gaussian 分布", "会计算标准化分数"],
        "question": "如果环境不可微、奖励离散、甚至策略里包含复杂规则，还能优化吗？黑盒方法只问一个问题：这组参数得分多少？",
        "symbols": [
            (r"$\theta$", "当前策略参数", "搜索中心"),
            (r"$\epsilon$", "Gaussian 扰动", "探索方向"),
            (r"$\sigma$", "扰动尺度", "搜索半径"),
            (r"$F(\theta)$", "适应度/episode return", "无需可微"),
            ("population", "每代候选数量", "方差与计算量权衡"),
        ],
        "derivation": r"""
### 1. 优化一个平滑后的目标

ES 实际考虑 Gaussian 平滑目标

$$J(\theta)=\mathbb E_{\epsilon\sim\mathcal N(0,I)}[F(\theta+\sigma\epsilon)]$$

利用 score-function identity，可得

$$\nabla_\theta J=\frac1\sigma\mathbb E[F(\theta+\sigma\epsilon)\epsilon]$$

Monte Carlo 近似：

$$\hat g=\frac1{n\sigma}\sum_{i=1}^nF_i\epsilon_i$$

直觉是：高分扰动方向获得正权重，低分方向获得负权重，中心朝更可能高分的方向移动。

### 2. 为什么标准化 fitness

给所有 fitness 加常数理论上不应改变期望梯度，因为 $\mathbb E[\epsilon]=0$，但有限 population 会有噪声。减均值、除标准差或 rank transform 可降低尺度和异常值影响。

### 3. 与 policy gradient 的差别

policy gradient 在动作空间采样并利用每步 log-prob；ES 在参数空间采样并只用整条轨迹 fitness。ES 易于跨 worker 并行，但高维参数下通常需要大量环境评估。
""",
        "hand": r"""
一维情况下，若正扰动 $+0.2$ 得分 10，负扰动 $-0.2$ 得分 4，那么加权扰动和为正，参数会向正方向移动。antithetic sampling 总是成对评估 $+\epsilon,-\epsilon$，直接比较两个方向，常能降低方差。
""",
        "mapping": [
            (r"$\epsilon_i$", "`noise`，shape `(population, dim)`", "每个候选的随机方向"),
            (r"$\theta+\sigma\epsilon_i$", "`theta + sigma * noise`", "候选参数"),
            ("$F_i$", "`scores`", "批量 fitness"),
            ("标准化 $F_i$", "`(scores-mean)/(std+eps)`", "降低尺度影响"),
            (r"$\hat g$", "`(scores[:,None] * noise).mean(0) / sigma`", "ES 梯度估计"),
            ("参数更新", "`theta += lr * grad`", "因为这里直接最大化 fitness"),
        ],
        "read": "二维凹函数的最优点是 `[1.5,2.0]`，最终 theta 应接近它。不同 seed 会有波动；比较 population 时应同时报告总 fitness evaluations。",
        "takeaways": ["ES 不需要对环境或策略求导。", "它优化 Gaussian 平滑后的 fitness。", "fitness 加权的扰动平均给出搜索方向。", "并行简单不等于样本高效。"],
        "check": ("为什么代码使用 `theta +=` 而神经网络 loss 通常使用梯度下降？ES 的主要成本是什么？", "这里估计的是 fitness 上升方向，所以直接加；神经网络通常定义要最小化的 loss。ES 主要成本是大量候选都要完成环境评估。"),
    },
    18: {
        "prereq": ["理解 epsilon-greedy", "知道外在奖励与状态访问次数", "掌握基本神经网络预测误差"],
        "question": "在迷宫里随机抖动不等于有目的地探索。怎样让智能体真正偏好‘没见过’或‘还预测不好’的状态？",
        "symbols": [
            ("$r_e$", "extrinsic reward", "环境任务给出的奖励"),
            ("$r_i$", "intrinsic reward", "探索机制产生的新奇奖励"),
            ("$N(s)$", "状态访问次数", "离散 count bonus"),
            ("$f(s)$", "固定随机 target 特征", "RND 中不训练"),
            (r"$\hat f_\theta(s)$", "predictor", "努力拟合 target"),
        ],
        "derivation": r"""
### 1. count bonus

希望首次访问奖励大、重复访问逐渐衰减，可取

$$r_i(s)=\frac{\beta}{\sqrt{N(s)+1}}$$

分母加 1 避免首次访问除零；平方根让衰减比 $1/N$ 更缓慢。

### 2. 从计数到 RND

高维图像几乎不会精确重复，无法直接建字典。RND 固定一个随机 target 网络 $f$，训练 predictor $\hat f_\theta$ 拟合已经见过的状态：

$$L_{RND}=\|\hat f_\theta(s)-f(s)\|^2$$

熟悉状态预测误差下降，新状态误差通常较大，因此把同一误差作为 intrinsic reward。

### 3. 合并奖励时的尺度问题

$$r_{total}=r_e+\eta\,\operatorname{normalize}(r_i)$$

RND error 会随训练下降且不同观测尺度差异很大，因此常用运行均值/方差归一化。评估时通常关闭 intrinsic reward，只看真实任务表现。

### 4. 随机电视问题

若观测含无法预测的随机噪声，prediction-error agent 会持续获得高奖励并沉迷其中。这说明“不可预测”不总等于“有用的新奇”。
""",
        "hand": r"""
$\beta=0.5$ 时，同一状态第 1、2、3 次访问 bonus 分别为 $0.5,0.354,0.289$。一个从未访问的新状态仍是 0.5，因此在外在奖励相同的情况下更有吸引力。
""",
        "mapping": [
            ("$N(s)$", "`counts[state]`", "访问前的累计次数"),
            ("$N(s)+1$", "`counts[state] + 1`", "避免除零"),
            (r"$\beta/\sqrt{N+1}$", "`beta / math.sqrt(...)`", "当前 intrinsic bonus"),
            ("更新计数", "`counts[state] += 1`", "应在计算本次 bonus 后登记"),
            ("RND error", "predictor 与 frozen target 的 MSE", "高维 pseudo-novelty"),
        ],
        "read": "输出中 A 的 bonus 随重复访问下降，首次 B/C 的 bonus 相同。实际训练要分别记录 extrinsic return、intrinsic return 和总 reward，避免只看到合成指标。",
        "takeaways": ["epsilon-greedy 随机，但不记得哪里已经探索过。", "count bonus 用访问频率表示新奇。", "RND 用预测误差近似高维状态新奇度。", "内在奖励必须归一化，并警惕随机电视问题。"],
        "check": ("为什么 RND target network 不能一起训练？为什么评估时不应把 intrinsic reward 当任务成绩？", "若 target 也追着 predictor 变化，误差不再稳定表示新奇。intrinsic reward 是训练辅助信号，不是环境真实任务目标。"),
    },
})


V2_GUIDE.update({
    19: {
        "prereq": ["掌握 actor-critic/PPO 主线", "理解二分类交叉熵", "知道 KL 正则的直觉"],
        "question": "很多目标无法写成可靠规则：‘动作自然不自然’、‘回答有没有帮助’。能否让人只比较两个片段，再由模型学出奖励？",
        "symbols": [
            (r"$\sigma^A,\sigma^B$", "两个待比较轨迹片段", "包含若干 observation/action"),
            (r"$r_\phi$", "reward model 的逐步评分", "由偏好数据学习"),
            ("$R_A,R_B$", "片段总预测分", "逐步评分求和"),
            (r"$\pi_{ref}$", "参考策略", "限制新策略漂移"),
            (r"$\beta$", "KL 惩罚强度", "对齐收益与保守程度权衡"),
        ],
        "derivation": r"""
### 1. 从相对偏好到概率

人类标签只说 A 比 B 好，不给绝对分数。Bradley-Terry 模型假设

$$P(A\succ B)=\frac{e^{R_A}}{e^{R_A}+e^{R_B}}
=\sigma(R_A-R_B)$$

只依赖分数差，所以给两个分数同时加常数不改变偏好概率。

### 2. preference loss

标签 $y=1$ 表示偏好 A，二元交叉熵为

$$L=-y\log\sigma(R_A-R_B)-(1-y)\log[1-\sigma(R_A-R_B)]$$

实现应使用 `binary_cross_entropy_with_logits(R_A-R_B, y)`，避免先 sigmoid 后 log 的数值不稳定。

### 3. 为什么策略不能只追 reward model

reward model 只在有限标注分布内可靠。策略会主动搜索其漏洞，产生人类不喜欢但预测高分的行为。常加入参考策略约束：

$$r_{total}=r_\phi-\beta[\log\pi_\theta(a|s)-\log\pi_{ref}(a|s)]$$

并持续让人检查策略找到的高分样本，形成迭代数据闭环。

### 4. 一个完整闭环

```text
初始策略 → 收集片段 → 人类比较 → 训练 reward model
    ↑                                  ↓
独立评估 ← 受 KL 约束的策略优化 ← 预测奖励
```
""",
        "hand": r"""
若 $R_A=2,R_B=1$，则 $P(A\succ B)=\sigma(1)\approx0.731$。若标签确实偏好 A，loss 为 $-\log0.731\approx0.313$。

若人类偏好 B，同一预测的 loss 为 $-\log(1-0.731)\approx1.313$，梯度会推动 $R_A-R_B$ 下降。
""",
        "mapping": [
            ("$R_A,R_B$", "`score_a`, `score_b`", "每个 pair 一个总分"),
            ("$R_A-R_B$", "`logits = score_a - score_b`", "偏好 logit"),
            ("$y$", "`labels`，0/1 float", "人类偏好方向"),
            ("preference BCE", "`binary_cross_entropy_with_logits`", "稳定的 sigmoid + BCE"),
            (r"$\log\pi_\theta-\log\pi_{ref}$", "策略与参考的 token/action log-ratio", "样本级 KL 惩罚估计"),
        ],
        "read": "第一和第三对样本的标签偏好 A，但第三对中 A 分数更低，因此第三对会贡献较大纠正梯度。交换 A/B 并翻转标签，loss 应保持一致，这是很好的单元测试。",
        "takeaways": ["偏好模型学习相对顺序，不需要绝对奖励标签。", "Bradley-Terry 概率只取决于片段分数差。", "策略会利用 reward model 漏洞，因此需要约束和独立人评。", "RLHF 是持续收集、标注、训练、审计的闭环。"],
        "check": ("为什么给 $R_A,R_B$ 同时加 100 不改变偏好概率？KL 惩罚能否保证完全没有 reward hacking？", "概率只依赖差值 $R_A-R_B$，共同平移会抵消。KL 只能限制漂移、降低风险，不能证明 reward model 没有漏洞，仍需数据迭代和独立评估。"),
    },
    20: {
        "prereq": ["理解 value/policy network", "知道树搜索节点与边", "熟悉 self-play 的概念"],
        "question": "网络擅长模式识别，搜索擅长向前推演。能否让二者互相改进，最后不依赖人类棋谱？",
        "symbols": [
            ("$P(s,a)$", "网络给出的动作先验", "引导搜索初期"),
            ("$N(s,a)$", "边访问次数", "搜索积累的证据"),
            ("$Q(s,a)$", "边的平均回传价值", "利用项"),
            (r"$\pi$", "MCTS visit-count 策略", "训练 policy head 的目标"),
            ("$z$", "最终比赛结果", "训练 value head 的目标"),
        ],
        "derivation": r"""
### 1. PUCT 在利用与探索之间平衡

$$a=\arg\max_a[Q(s,a)+U(s,a)]$$

其中

$$U(s,a)=c_{puct}P(s,a)\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}$$

高 $Q$ 鼓励利用已知好动作；高 prior、低访问次数带来大 $U$，鼓励探索。随着某边被反复访问，分母增大，exploration bonus 下降。

### 2. 搜索怎样变成训练数据

根节点各动作访问次数产生更强策略目标

$$\pi(a|s)\propto N(s,a)^{1/\tau}$$

self-play 最终结果 $z$ 作为 value target。网络同时最小化

$$L=(z-v_\theta(s))^2-\pi^T\log p_\theta(s)+c\|\theta\|^2$$

新网络再为下一轮搜索提供更好先验与叶节点价值，形成 policy iteration 风格闭环。

### 3. MuZero 改变了什么

AlphaZero 搜索使用已知游戏规则。MuZero 学习三个函数：representation $h(o_{1:t})\to s_0$、dynamics $g(s_k,a_k)\to(s_{k+1},r_k)$、prediction $f(s_k)\to(p_k,v_k)$。隐藏状态只需保留对预测 reward/value/policy 有用的信息，不要求重建像素。

### 4. 回传时的玩家视角

零和轮流行动游戏中，子节点对下一玩家的正价值，对当前玩家通常是负价值。忘记在层级间翻转符号会让搜索偏好帮助对手。
""",
        "hand": r"""
三个动作的 $Q=[0.6,0.5,0.1]$，第三个动作虽然当前 Q 低，但 prior=0.5 且从未访问，其探索项最大，PUCT 可能优先尝试它。

访问后 $N$ 增加，若回传价值仍差，$U$ 下降，搜索会逐渐转向更可靠的动作。这就是“先验提出候选，搜索用证据修正”。
""",
        "mapping": [
            ("$Q(s,a)$", "`q`", "当前边平均价值"),
            ("$P(s,a)$", "`prior`", "policy head 先验概率"),
            ("$N(s,a)$", "`visits`", "每条边访问次数"),
            ("$U(s,a)$", "`c * prior * sqrt(parent_visits)/(1+visits)`", "探索项"),
            ("PUCT 选择", "`scores.argmax()`", "选择 Q+U 最大动作"),
            (r"$\pi$ target", "root visit counts 归一化", "不是网络自身 argmax"),
        ],
        "read": "示例应选择高 prior 且未访问的第三个动作。修改它的访问次数与 Q，观察选择如何转向。完整项目中重点追踪 selection、expansion、backup 和 root policy 四段。",
        "takeaways": ["网络提供先验和叶节点价值，MCTS 用模拟改进决策。", "visit counts 是 policy target，最终胜负是 value target。", "PUCT 的 Q 是利用项，U 是探索项。", "MuZero 学习用于规划的隐藏动态，而不是完整世界模拟。"],
        "check": ("为什么训练 policy head 使用访问次数而不是搜索选出的单个动作？MuZero dynamics 是否必须还原下一帧？", "访问分布包含搜索对多个动作的相对证据，信息比单个 argmax 更丰富。MuZero 隐藏动态只需支持 reward/value/policy 预测，不要求生成真实观测。"),
    },
    21: {
        "prereq": ["理解状态图与最短路", "知道 BFS/A* 的基本思想", "能区分真实代价和启发式估计"],
        "question": "魔方状态数量巨大，盲目搜索会爆炸。神经网络能否学会判断‘哪个状态离目标更近’，再把判断交给搜索算法？",
        "symbols": [
            ("$g(s)$", "从起点到 $s$ 的已知代价", "搜索已经付出的步数"),
            (r"$h_\theta(s)$", "到目标的估计剩余代价", "学习到的启发式"),
            ("$f(s)$", "搜索优先级", "常取 $g+h$"),
            ("branching factor", "每个状态的合法后继数", "决定搜索树增长速度"),
            ("scramble depth", "从目标打乱的步数", "可用于 curriculum"),
        ],
        "derivation": r"""
### 1. 从组合优化到图搜索

状态是节点，合法动作是有向边，每步代价通常为 1。目标是找到从初态到目标的最短路径。

BFS 按 $g(s)$ 排序，保证单位代价下最短，但展开节点约随 $b^d$ 指数增长。A* 加入启发式：

$$f(s)=g(s)+h(s)$$

### 2. admissible heuristic 与最优性

若 $0\le h(s)\le h^*(s)$，即从不高估真实剩余代价，标准 A* 可保证最优。学习网络通常没有天然不高估保证，因此可能更快但失去严格最优性。

加权 A* 使用 $f=g+wh$，$w>1$ 更贪心；beam search 每层只保留固定数量候选，内存可控但可能丢掉正确路径。

### 3. 价值与距离的联系

若只在达到目标时得 1、每步折扣 $\gamma$，距目标 $d$ 的理想价值约为

$$V(s)=\gamma^{d(s)}$$

所以可由 $d(s)\approx\log V(s)/\log\gamma$ 建立启发式直觉。但模型误差、循环和多路径会使关系不再精确。

### 4. curriculum 产生训练数据

从已解状态施加 $k$ 次随机动作可得到已知“至多 $k$ 步可解”的样本。逐渐增加 $k$，让网络先学近目标结构，再覆盖深层状态。
""",
        "hand": r"""
整数玩具图从 1 到 15，动作是 `-1,+1,*2`。纯贪心可能沿局部看起来接近的路径前进；beam width=3 同时保留三个候选，降低过早丢失好路径的风险。

这里 `abs(goal-next)` 是人工启发式；在魔方项目中相同位置由网络预测替代。
""",
        "mapping": [
            ("状态邻居", "`neighbors(state)`", "应用所有合法动作"),
            ("启发式 $h$", "`abs(goal - nxt)`", "玩具示例中的剩余距离估计"),
            ("beam 候选", "`candidates`", "当前深度所有后继"),
            ("按优先级排序", "`sorted(candidates)`", "小 $h$ 优先"),
            ("beam width", "`[:width]`", "限制每层保留节点数"),
            ("去重", "`seen`", "避免重复状态和循环"),
        ],
        "read": "运行后确认路径每相邻状态确实由合法动作连接。把 width=1/3/10 比较：更宽通常更可靠，但展开和内存更多。",
        "takeaways": ["离散优化可统一成状态图搜索。", "网络常作为启发式，而搜索负责组合与验证。", "学习启发式快但通常没有 admissibility 保证。", "解出率、路径长度和展开节点数要一起报告。"],
        "check": ("为什么学习到的 heuristic 可能破坏 A* 最优性？beam search 的 width 增大有什么变化？", "网络可能高估真实剩余代价，使最优路径被错误延后或剪枝。更大 width 保留更多候选、通常提高解出率，但时间和内存增长。"),
    },
    22: {
        "prereq": ["掌握单智能体 Q-learning/actor-critic", "理解联合概率与博弈收益", "知道局部观测"],
        "question": "当环境里的其他角色也在学习时，昨天有效的策略今天可能失效。多智能体 RL 的难点不是简单复制多个 DQN，而是学习对象本身在变化。",
        "symbols": [
            (r"$a^1,\ldots,a^n$", "所有 agent 的联合动作", "共同决定转移和奖励"),
            ("$r_i$", "agent $i$ 的奖励", "可共享、竞争或混合"),
            ("$o_i$", "agent $i$ 的局部观测", "执行时通常只能看到它"),
            ("centralized critic", "训练时使用全局信息的评价器", "缓解非平稳性"),
            ("CTDE", "集中训练、分散执行", "训练/部署信息边界不同"),
        ],
        "derivation": r"""
### 1. Markov game 扩展了 MDP

转移依赖联合动作：

$$P(s'|s,a^1,\ldots,a^n)$$

agent $i$ 最大化自己的目标

$$J_i=\mathbb E\left[\sum_t\gamma^t r_i(s_t,a_t^1,\ldots,a_t^n)\right]$$

若所有 $r_i$ 相同是完全合作；若一方收益恰为另一方负数是零和；现实常是混合博弈。

### 2. 为什么 independent learning 非平稳

agent 1 看到的有效转移还取决于 agent 2 的策略：

$$P_{eff}(s'|s,a^1)=\sum_{a^2}\pi_2(a^2|o_2)P(s'|s,a^1,a^2)$$

当 $\pi_2$ 学习并变化时，$P_{eff}$ 也变化，破坏单智能体算法假设的固定环境。

### 3. CTDE 怎样帮助

训练 critic 时使用全局状态和联合动作：

$$Q_i(s,a^1,\ldots,a^n)$$

它能解释奖励究竟由谁的动作导致；执行 actor 时仍只输入 $o_i$，满足分散部署限制。必须严格防止训练可见信息泄漏进执行网络。

### 4. credit assignment

共享团队奖励告诉大家成败，却不说明哪个 agent 贡献最大。value decomposition、counterfactual baseline 等方法试图把团队价值分解为可学习的个体信用。
""",
        "hand": r"""
协调博弈：两 agent 都选 0 或都选 1 时各得 1，否则得 0。存在两个同样好的均衡 `(0,0)` 和 `(1,1)`。

独立 Q-learning 可能因早期随机波动收敛到任意一个均衡；若两个 agent 各自偏向不同动作，短期会互相得到 0，协调变得困难。
""",
        "mapping": [
            ("$Q_1(a^1),Q_2(a^2)$", "`q1`, `q2`", "两个独立 learner 的估计"),
            ("$a^1,a^2$", "`a1`, `a2`", "各自 epsilon-greedy 动作"),
            ("共享 $r$", "`float(a1 == a2)`", "协调成功奖励"),
            ("独立更新", "`q1[a1] += ...`, `q2[a2] += ...`", "彼此当作环境"),
            ("联合信息", "centralized critic 输入", "示例未实现，项目/扩展继续学习"),
        ],
        "read": "两个 Q 表最终应偏向同一个动作，但多个 seed 可能选择不同均衡。这个现象不是 bug，而是对称博弈的多均衡性质。",
        "takeaways": ["联合动作共同决定转移与奖励。", "其他 agent 学习会造成非平稳性。", "CTDE 训练时看全局、执行时只看局部。", "多智能体评估必须覆盖不同队友、对手和随机种子。"],
        "check": ("为什么不能直接把 centralized critic 的全局状态也喂给执行 actor？independent Q-learning 为什么仍可能在简单任务有效？", "部署时单个 agent 往往拿不到全局信息，这会造成训练/执行不一致。简单任务中其他策略变化有限、共享奖励清晰时，独立学习仍可能近似有效。"),
    },
})


# Final-edition layer: explicit book-section coverage, end-to-end algorithms,
# real repository source routes, a second executable check, and exercise
# solutions.  Long engineering details are kept in collapsible blocks by the
# notebook builder so that the main reading path stays calm.
FINAL_GUIDE = {
    1: {
        "coverage": [
            ("Supervised / unsupervised / reinforcement learning", "问题引入与心智模型", "对比理解"),
            ("Complications in RL", "常见坑与扩展", "延迟奖励、非平稳、探索"),
            ("RL formalisms: reward, agent, environment, action, observation", "符号表与交互递归", "详细"),
            ("Markov process / MRP / MDP", "分步推导、矩阵求值实验", "详细"),
            ("Policy", "价值定义与后续章节连接", "主线"),
        ],
        "algorithm": r"""
```text
给定策略 π 和环境：
1. 从初始状态 S0 开始。
2. 按 π(·|St) 选择动作 At。
3. 环境产生 Rt+1、St+1。
4. 重复直到终止，得到一条 trajectory。
5. 从后向前计算 Gt = Rt+1 + γGt+1。
6. 多条轨迹中，同一状态的 Gt 平均值就是 Vπ(s) 的 Monte Carlo 估计。
```

已知 MRP 转移矩阵时无需采样，可直接解 $(I-\gamma P)V=r$；未知环境时只能通过轨迹估计。这一分叉会贯穿全书：**model-based 使用模型，model-free 使用样本**。
""",
        "source": [],
        "deep": "本章没有仓库脚本，原因是它负责建立语言而不是训练模型。建议把本章代码当成后续实现的单元测试基准：任何 TD target 最终都应能追溯到 `reward + gamma * future_value`。",
        "extra_code": r"""
import numpy as np

# 验证 Bellman 方程的残差：正确解应使 r + gamma P V - V 接近 0。
P = np.array([[0.7, 0.3], [0.2, 0.8]])
r = np.array([1.0, 2.0])
gamma = 0.9
V = np.linalg.solve(np.eye(2) - gamma * P, r)
residual = r + gamma * P @ V - V
print("Bellman residual:", residual)
assert np.allclose(residual, 0.0)
""",
        "solution": r"对 `[0,0,10]`：$\gamma=0$ 时第一个回报为 0；$\gamma=0.5$ 时为 2.5；$\gamma=0.99$ 时为 9.801。折扣越接近 1，越重视远期奖励。注意这不意味着总应选择大 $\gamma$：持续任务的方差和有效时间跨度也会变大。",
        "next": "第 2 章把这里抽象的 agent/environment 交互落成 Gymnasium API；第 5 章会把回报递归正式变成 Bellman 最优方程。",
    },
    2: {
        "coverage": [
            ("The anatomy of the agent", "交互契约与算法流程", "详细"),
            ("Hardware and software requirements", "索引的环境说明", "实用"),
            ("Gymnasium spaces / environment / creation / CartPole", "符号表、代码实验", "详细"),
            ("The random CartPole agent", "第一段可运行实验", "完整小例子"),
            ("Wrappers / rendering / extra API", "源码路线、常见坑、扩展", "详细"),
        ],
        "algorithm": r"""
```text
obs, info = env.reset(seed=seed)
while True:
    action = policy(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    store(obs, action, reward, terminated, next_obs)
    obs = next_obs
    if terminated or truncated:
        break
env.close()
```

存储 transition 时建议保留 `terminated` 与 `truncated`，不要过早合成一个 `done`。控制流用二者的 OR；bootstrap mask 通常只看 `terminated`。
""",
        "source": [
            ("../Chapter02/01_agent_anatomy.py", "`Environment`, `Agent.step`", "先看自制最小接口，理解控制权在哪一侧"),
            ("../Chapter02/02_cartpole_random.py", "主循环", "对应 notebook 的 reset/step 生命周期"),
            ("../Chapter02/03_random_action_wrapper.py", "`RandomActionWrapper.action`", "理解 ActionWrapper 只变换动作"),
            ("../Chapter02/04_cartpole_random_monitor.py", "视频/监控", "理解评估记录与训练循环分离"),
        ],
        "deep": "源码阅读顺序应从 `01_agent_anatomy.py` 开始，因为它把 Gymnasium 隐藏的协议显式写出；再看随机 agent；最后看 wrapper。`RandomActionWrapper` 的 `action()` 在底层环境收到动作前拦截它，因此 epsilon 随机性属于环境包装层，而不是策略网络。",
        "extra_code": r"""
# 同一 next_value，在自然终止与时间截断下产生不同 TD target。
reward, gamma, next_value = 1.0, 0.9, 5.0
cases = [(True, False), (False, True), (False, False)]
for terminated, truncated in cases:
    target = reward + gamma * (not terminated) * next_value
    stop_episode = terminated or truncated
    print({"terminated": terminated, "truncated": truncated,
           "stop": stop_episode, "target": target})
""",
        "solution": "继承 `gym.RewardWrapper` 并实现 `reward(self, reward): return 0.1 * reward`。用相同 seed 分别运行原环境和包装环境；episode 步数应一致，总回报应缩小到约十分之一。若步数变化，说明 wrapper 意外改了动作、观测或终止条件。",
        "next": "第 3 章学习用 PyTorch 表示可训练策略；第 4 章第一次把 Gym trajectory 变成监督训练数据。",
    },
    3: {
        "coverage": [
            ("Tensors / tensor creation / operations / GPU", "符号表、训练实验、源码精读", "详细"),
            ("Gradients and autograd", "链式法则推导与第二实验", "详细"),
            ("NN building blocks / custom layers", "`OurModule` 源码路线", "详细"),
            ("Loss functions and optimizers", "训练循环映射", "详细"),
            ("TensorBoard", "源码路线与工程说明", "实用"),
            ("GAN on Atari images", "折叠源码精读", "结构理解"),
            ("PyTorch Ignite", "事件驱动训练说明", "结构理解"),
        ],
        "algorithm": r"""
```text
for batch in data:
    prediction = model(inputs)       # 建立计算图
    loss = loss_fn(prediction, target)
    optimizer.zero_grad()            # 清除旧梯度
    loss.backward()                  # 链式法则
    optional_clip_grad_norm()
    optimizer.step()                 # 修改参数
    logger.add_scalar(...)           # 日志不参与梯度
```

GAN 有两个优化问题：discriminator 区分真实/生成图像，generator 让生成图像被判为真实。训练时必须明确哪个阶段冻结哪组参数。Ignite 把 `for batch` 外壳和事件回调抽离，数学 loss 没有变化。
""",
        "source": [
            ("../Chapter03/01_modules.py", "`OurModule.forward`", "看 `nn.Module`、子层注册和 shape 流"),
            ("../Chapter03/02_tensorboard.py", "`SummaryWriter` 调用", "看 scalar 与 graph 日志"),
            ("../Chapter03/03_atari_gan.py", "`InputWrapper`, `Discriminator`, `Generator`, `iterate_batches`", "看完整双优化器循环"),
            ("../Chapter03/04_atari_gan_ignite.py", "`process_batch` 与事件 handler", "对比手写循环与 Ignite"),
        ],
        "deep": "读 GAN 时按数据 shape 走：Atari observation → `InputWrapper` 归一化/换轴 → discriminator 图像输入；随机 latent → generator → 假图。先标出两个 loss 各自更新谁，再看 `detach` 在哪里切断 generator 梯度。Ignite 版的 `process_batch` 相当于普通训练循环的一次迭代，handler 只负责日志、采样与终止条件。",
        "extra_code": r"""
import torch

# 用 autograd 验证手算梯度：x=2, y=7, w=1, b=0。
x = torch.tensor(2.0)
y = torch.tensor(7.0)
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
loss = (w * x + b - y) ** 2
loss.backward()
print("dL/dw:", w.grad.item(), "dL/db:", b.grad.item())
assert w.grad.item() == -20.0 and b.grad.item() == -10.0
""",
        "solution": "MLP 可写成 `nn.Sequential(nn.Linear(1,16), nn.ReLU(), nn.Linear(16,1))`。相同数据下它也能拟合直线，但参数更多，不保证比线性模型更快；比较时固定 seed，并记录每 20 步 loss，而不是只看最后一次。",
        "next": "第 4 章将网络输出解释为动作 logits；第 6 章将同一训练循环用于拟合 TD target。",
    },
    4: {
        "coverage": [
            ("Taxonomy of RL methods", "心智模型与章节连接", "概览"),
            ("CEM in practice", "最大似然推导", "详细"),
            ("CEM on CartPole", "算法流程、源码路线", "详细"),
            ("CEM on FrozenLake", "失败原因、改进与源码路线", "详细"),
            ("Theoretical background", "elite MLE / 交叉熵解释", "主线"),
        ],
        "algorithm": r"""
```text
repeat:
    batch = []
    while len(batch) < batch_size:
        用当前策略采样完整 episode
        保存每步 (observation, action) 和 episode return
    reward_bound = percentile(returns, percentile)
    elite = 所有 return >= reward_bound 的 episode
    states, actions = 展平 elite transitions
    loss = cross_entropy(policy(states), actions)
    更新策略
    若平均回报达到阈值则停止
```

FrozenLake 难点是成功轨迹稀少且随机转移使好动作标签含噪。书中版本通过 one-hot observation、折扣/保留 elite、非滑动环境等方式逐步诊断，而不是把失败简单归因于网络。
""",
        "source": [
            ("../Chapter04/01_cartpole.py", "`iterate_batches`, `filter_batch`", "完整 CEM 数据生成与 elite 过滤"),
            ("../Chapter04/02_frozenlake_naive.py", "naive baseline", "观察稀疏奖励为何失败"),
            ("../Chapter04/03_frozenlake_tweaked.py", "折扣/elite 改进", "比较数据选择变化"),
            ("../Chapter04/04_frozenlake_nonslippery.py", "非滑动环境", "隔离环境随机性"),
        ],
        "deep": "先精读 `EpisodeStep` 与 `Episode`，确认回报属于整条 episode，而监督标签属于每一步。再看 `filter_batch`：它先算 percentile，再把 elite 轨迹展平。最容易出现的 bug 是把每步 reward 当 episode return，或忘记把离散状态 one-hot 后输入线性层。",
        "extra_code": r"""
import numpy as np

returns = np.array([3., 10., 6., 1., 8.])
for percentile in (50, 70, 90):
    bound = np.percentile(returns, percentile)
    elite = returns[returns >= bound]
    print(percentile, "bound=", round(bound, 2), "elite=", elite.tolist())
""",
        "solution": "50 分位阈值为 6，保留 `[10,6,8]`；70 分位约 7.6，保留 `[10,8]`；90 分位约 9.2，只保留 `[10]`。batch 小时高分位会让样本量剧烈波动，因此应同时监控 elite 数量。",
        "next": "第 5 章转向已知模型的动态规划；第 11 章会给出不依赖 hard elite 筛选的策略梯度。",
    },
    5: {
        "coverage": [
            ("Value, state, and optimality", "价值定义与手算", "详细"),
            ("Bellman equation of optimality", "分步推导", "详细"),
            ("Value of the action", "V/Q 对照", "详细"),
            ("Value iteration method", "算法伪代码与收敛解释", "详细"),
            ("Value iteration in practice", "第一个与第二实验", "完整"),
            ("Q-iteration for FrozenLake", "源码路线", "完整"),
        ],
        "algorithm": r"""
```text
初始化 V(s)=0
先用随机交互估计 transition counts 与 reward table
repeat:
    对每个状态 s：
        对每个动作 a：
            q[a] = Σs' P_hat(s'|s,a) [R_hat + γV(s')]
        V_new(s) = max_a q[a]
    V = V_new
直到 Bellman 变化足够小
执行时 action = argmax_a q[a]
```

仓库示例并非直接读取 FrozenLake 内部模型，而是用随机步估计转移概率与奖励，因此兼具 model-based 学习和动态规划。
""",
        "source": [
            ("../Chapter05/01_frozenlake_v_iteration.py", "`play_n_random_steps`, `calc_action_value`, `value_iteration`", "先估模型，再更新 V"),
            ("../Chapter05/02_frozenlake_q_iteration.py", "`select_action`, `value_iteration`", "直接存 Q，减少重复计算"),
        ],
        "deep": "在 V-iteration 脚本中，`transits[(state, action)]` 保存下一状态计数，`rewards[(state, action, next_state)]` 保存观察奖励；`calc_action_value` 把计数归一化成概率。Q-iteration 把这一 lookahead 结果缓存为表。两者最优策略相同，但状态存储和计算位置不同。",
        "extra_code": r"""
import numpy as np

# 随机转移的一次 Bellman expectation：动作可能到两个下一状态。
probs = np.array([0.75, 0.25])
rewards = np.array([0.0, 1.0])
next_values = np.array([2.0, 0.0])
gamma = 0.9
q_value = np.sum(probs * (rewards + gamma * next_values))
print("Q(s,a) =", q_value)
assert np.isclose(q_value, 1.6)
""",
        "solution": r"修改奖励后应分别比较‘停留动作的无限小奖励’与‘到终点的一次奖励’。若状态 0 可永远获得 0.02，其价值约为 $0.02/(1-0.9)=0.2$；通往终点的两步价值约为 $0.9\times0.1=0.09$，因此循环可能反而最优。这正说明 reward specification 会改变任务。",
        "next": "第 6 章在不知道转移概率时，用单条 transition 的 TD target 近似这里的期望，并用网络替代表格。",
    },
    6: {
        "coverage": [
            ("Real-life value iteration", "从模型到采样的过渡", "详细"),
            ("Tabular Q-learning", "TD 更新与 chain 实验", "详细"),
            ("Deep Q-learning", "loss、replay、target network 推导", "详细"),
            ("DQN on Pong: wrappers/model/training/performance/play", "完整流程与真实源码路线", "详细"),
            ("Things to try", "练习与扩展", "实践"),
        ],
        "algorithm": r"""
```text
初始化 online Qθ、target Qθ-、replay D
for frame in 1..N:
    以 epsilon-greedy 从 Qθ(obs) 选 action
    与环境交互，存 transition 到 D
    若 D 尚未 warm up：continue
    batch = D.sample(B)
    predicted = Qθ(states).gather(actions)
    with no_grad:
        next = max_a Qθ-(next_states, a)
        target = rewards + γ * non_terminal * next
    loss = Huber(predicted, target)
    backward + optimizer.step
    每 K 帧同步 θ- ← θ；逐步降低 epsilon
```
""",
        "source": [
            ("../Chapter06/lib/wrappers.py", "`ImageToPyTorch`, `BufferWrapper`, `make_env`", "像素预处理、帧堆叠与环境创建"),
            ("../Chapter06/lib/dqn_model.py", "`DQN.forward`", "卷积输出尺寸与动作 Q"),
            ("../Chapter06/02_dqn_pong.py", "`ExperienceBuffer`, `Agent.play_step`, `batch_to_tensors`, `calc_loss`", "完整训练数据流"),
            ("../Chapter06/03_dqn_play.py", "评估循环", "加载模型、贪心动作与渲染"),
        ],
        "deep": "精读顺序不要从 `main` 开始：先确认 wrapper 输出 shape/dtype，再算 `DQN` 卷积后的展平维度，然后跟踪 `Agent.play_step` 如何产生 terminal transition，最后进入 `calc_loss`。在 loss 中逐项核对 `gather`、terminal mask、target network 和 `no_grad`；这四处决定数学是否正确。",
        "extra_code": r"""
import torch
import torch.nn.functional as F

# 一个完整的 batch DQN loss，显式展示 shape。
q_all = torch.tensor([[1., 2.], [3., 1.]], requires_grad=True)   # (B=2, A=2)
actions = torch.tensor([1, 0])                                  # (B,)
rewards = torch.tensor([1., -1.])
terminated = torch.tensor([False, True])
target_next_all = torch.tensor([[4., 2.], [5., 6.]])
q_sa = q_all.gather(1, actions[:, None]).squeeze(1)
with torch.no_grad():
    target = rewards + 0.9 * (~terminated) * target_next_all.max(1).values
loss = F.smooth_l1_loss(q_sa, target)
loss.backward()
print("q_sa:", q_sa.detach(), "target:", target, "loss:", loss.item())
assert q_sa.shape == target.shape == (2,)
""",
        "solution": "epsilon=0 且全零 Q 时，NumPy `argmax` 总返回第一个动作；chain 示例的动作 0 向左，agent 会一直停在起点，永远看不到终点奖励。可用随机 tie-breaking、非零 epsilon 或 optimistic initialization 解决。",
        "next": "第 7 章把这里的 selector、agent、experience、buffer、target net 拆成复用组件；第 8 章逐项修正基础 DQN 的弱点。",
    },
    7: {
        "coverage": [
            ("Why RL libraries?", "组件化动机", "详细"),
            ("PTAN action selectors / agents", "数据流与源码路线", "详细"),
            ("Experience sources / first-last", "n-step 推导与实验", "详细"),
            ("Replay buffers / TargetNet / Ignite helpers", "组件职责表", "详细"),
            ("PTAN CartPole solver", "端到端伪代码与源码", "完整"),
            ("Other RL libraries", "扩展", "概览"),
        ],
        "algorithm": r"""
```text
selector = EpsilonGreedyActionSelector(epsilon)
agent = DQNAgent(net, selector)
exp_source = ExperienceSourceFirstLast(env, agent, gamma, steps_count=n)
buffer = ExperienceReplayBuffer(exp_source, capacity)
target_net = TargetNet(net)
repeat:
    buffer.populate(1)
    若 buffer 太小：continue
    batch = buffer.sample(B)
    states, actions, rewards, last_states = unpack(batch)
    计算 n-step DQN loss
    更新 online net；周期同步 target_net
```
""",
        "source": [
            ("../Chapter07/01_actions.py", "action selectors", "从分数到动作"),
            ("../Chapter07/02_agents.py", "`DQNNet`, `PolicyNet` 与 agent", "从观测到动作分数"),
            ("../Chapter07/03_exp_sources.py", "experience source 示例", "环境驱动和 n-step 拼接"),
            ("../Chapter07/04_replay_buf.py", "replay buffer", "存储与随机采样"),
            ("../Chapter07/05_target_net.py", "TargetNet", "同步语义"),
            ("../Chapter07/06_cartpole.py", "`unpack_batch` 与训练循环", "把组件装回完整 DQN"),
        ],
        "deep": r"用 `Chapter07/lib.py` 的 `ToyEnv` 和 `DullAgent` 先观察 experience source 的边界行为，因为真实 CartPole 会掩盖数据错误。重点打印 `ExperienceFirstLast.reward` 是否已包含 n-step 折扣和 `last_state=None` 是否只用于自然终止。进入 solver 后，`unpack_batch` 必须使用 $\gamma^n$ bootstrap。",
        "extra_code": r"""
def n_step_target(rewards, gamma, bootstrap=None):
    total = sum((gamma ** i) * r for i, r in enumerate(rewards))
    if bootstrap is not None:
        total += (gamma ** len(rewards)) * bootstrap
    return total

print("terminated:", n_step_target([1, 0, 2], 0.9, bootstrap=None))
print("non-terminal:", n_step_target([1, 0, 2], 0.9, bootstrap=4.0))
assert abs(n_step_target([1, 0, 2], 0.9) - 2.62) < 1e-9
""",
        "solution": "让函数返回 `(total, len(rewards))`。提前终止时 `bootstrap=None`；未终止时使用 `gamma ** actual_steps`，不是固定使用设定的 n，因为 episode 边界或数据切片可能让实际步数更短。",
        "next": "第 8 章继续使用同一 PTAN 数据管线，只替换 loss、模型或 buffer；这正体现组件化的价值。",
    },
}


FINAL_GUIDE.update({
    8: {
        "coverage": [
            ("Basic and tuned DQN", "baseline 与公平比较", "详细"),
            ("N-step DQN", "第 7 章推导与源码", "详细"),
            ("Double DQN", "选择/评价分离", "详细"),
            ("Noisy networks", "参数噪声精读", "详细"),
            ("Prioritized replay", "概率与重要性权重", "详细"),
            ("Dueling DQN", "可辨识分解", "详细"),
            ("Categorical DQN", "分布投影说明与实验", "详细"),
            ("Combining everything / Rainbow", "组合表与源码路线", "完整"),
        ],
        "algorithm": r"""
```text
Rainbow 风格一次更新：
1. Noisy network 产生带参数噪声的 Q 分布，无需固定 epsilon schedule。
2. 从 prioritized buffer 按 P(i) 抽取 batch，并取得 importance weights。
3. 使用 n-step reward 构造更远期 target。
4. online network 选动作（Double）；target network 评价该动作。
5. categorical projection 把平移后的目标分布投影回固定 atoms。
6. 以 weighted cross-entropy 更新网络。
7. 用新的 per-sample loss/TD error 更新 priorities。
8. 周期同步 target，并 reset noisy layers。
```

组合前必须先有同一 baseline、同一评估协议和逐项 ablation，否则无法知道改进来自算法还是超参数。
""",
        "source": [
            ("../Chapter08/lib/common.py", "`Hyperparams`, `calc_loss_dqn`, `EpsilonTracker`", "所有实验共享基线"),
            ("../Chapter08/lib/dqn_extra.py", "`NoisyDQN`, `PrioReplayBuffer`, distributional/dueling 模型", "扩展核心实现"),
            ("../Chapter08/02_dqn_n_steps.py", "n-step 配置", "看数据源而非 loss 如何变化"),
            ("../Chapter08/03_dqn_double.py", "`calc_loss_double_dqn`", "online argmax + target gather"),
            ("../Chapter08/05_dqn_prio_replay.py", "priority/weight 更新", "采样偏差校正"),
            ("../Chapter08/07_dqn_distrib.py", "categorical projection", "概率质量搬运"),
            ("../Chapter08/08_dqn_rainbow.py", "组合训练", "确认哪些扩展真正组合"),
        ],
        "deep": r"Categorical projection 是本章最容易被略过的核心：对每个 target atom 计算 $Tz=\operatorname{clip}(r+\gamma z)$，映射到浮点位置 $b=(Tz-v_{min})/\Delta z$，再把质量按距离分给相邻整数 atom。边界刚好落在整数上时要避免质量丢失。投影后每个样本概率和应为 1。NoisyNet 则在权重中维护可学习的 $\mu,\sigma$，每次 resample 噪声改变整段策略，而非每一步独立随机动作。",
        "extra_code": r"""
import numpy as np

# PER：从 priority 得到采样概率和归一化 importance weights。
priorities = np.array([1.0, 2.0, 4.0])
alpha, beta = 0.6, 0.4
probs = priorities ** alpha
probs /= probs.sum()
weights = (len(probs) * probs) ** (-beta)
weights /= weights.max()
print("P(i):", np.round(probs, 3), "weights:", np.round(weights, 3))
assert np.isclose(probs.sum(), 1.0)
""",
        "solution": "例如 online=`[0,10]`、target=`[100,0]`。普通 DQN target 取 100；Double DQN 先由 online 选动作 1，再在 target 读取 0。这个极端例子清楚显示普通 max 可能选择 target 网络偶然高估的动作。",
        "next": "第 9 章不再改变 Bellman 数学，而是优化同一算法的数据生成与计算吞吐；第 18 章会重新深入探索机制。",
    },
    9: {
        "coverage": [
            ("Why speed matters / baseline", "时间预算与基准", "详细"),
            ("PyTorch computation graph", "无用梯度与 detach", "详细"),
            ("Several environments", "批量环境", "详细"),
            ("Playing and training in separate processes", "生产者/消费者架构", "详细"),
            ("Tweaking wrappers", "预处理位置与复制成本", "详细"),
            ("Benchmark results", "实验设计与解释", "完整"),
        ],
        "algorithm": r"""
```text
建立 baseline：固定算法、seed 集合、硬件与评估协议
profile 得到 env/preprocess/transfer/forward/backward/log 时间
按瓶颈选择一种改动：
  - 网络小调用多 → batching
  - 环境慢 → vector env / 多进程 actor
  - 图像传输重 → actor 侧预处理或共享内存
  - 梯度图过大 → target/no-action 路径 no_grad
每次只改一项，确认数值结果等价
比较 steps/s、updates/s、达到相同回报的 wall-clock
```
""",
        "source": [
            ("../Chapter09/00_slow_grads.py", "慢梯度示例", "识别不必要 computation graph"),
            ("../Chapter09/01_baseline.py", "baseline trainer", "所有优化的对照"),
            ("../Chapter09/02_n_envs.py", "多环境采样", "观察 batch/等待变化"),
            ("../Chapter09/03_parallel.py", "并行 actor/trainer", "队列与进程边界"),
            ("../Chapter09/04_wrappers_parallel.py", "worker 侧 wrappers", "减少跨进程数据量"),
            ("../Chapter09/04_wrappers_n_env.py", "非并行对照", "隔离 wrapper 优化收益"),
        ],
        "deep": "并行版本应沿队列追踪数据：worker 持有环境和 actor 网络快照，产生 experience；trainer 消费 batch 并更新主网络；参数同步频率控制策略陈旧度。图像若在预处理前跨进程传输，IPC 成本可能吞掉并行收益。比较 benchmark 时同时记录 CPU 核数、环境数、batch、同步频率和评估回报。",
        "extra_code": r"""
def amdahl_speedup(parallel_fraction, workers):
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / workers)

for workers in (1, 2, 4, 16, 10_000):
    print(workers, round(amdahl_speedup(0.6, workers), 3))
print("theoretical ceiling:", 1 / (1 - 0.6))
""",
        "solution": "可靠微基准应先 warm-up，重复多次并取中位数；batch=32 可能被 Python/调度开销主导，512 更能利用矩阵并行，8192 可能受内存带宽或缓存限制。最终仍需在完整训练中比较 wall-clock 学习曲线。",
        "next": "第 10 章展示另一个常见瓶颈：环境与数据正确性往往比网络算力更重要。",
    },
    10: {
        "coverage": [
            ("Why trading / problem decisions", "状态、动作、奖励定义", "详细"),
            ("Data", "OHLC 读取、切分、特征", "源码精读"),
            ("Trading environment", "记账与终止", "详细"),
            ("Models", "feed-forward / convolution", "结构对比"),
            ("Training code", "端到端流程", "完整"),
            ("Results: feed-forward / convolution", "评估解读", "详细"),
            ("Things to try", "练习与扩展", "实践"),
        ],
        "algorithm": r"""
```text
按时间加载 OHLC 数据 → 只用训练期拟合预处理
env.reset: 选择起点、清空仓位与收益
每一步：
  obs = 最近 bars 特征 + 当前仓位
  action ∈ {skip, buy, close}（以仓库定义为准）
  根据价格变化更新持仓 PnL
  若仓位改变，扣 commission
  到达窗口末尾/破产/数据末尾则结束
DQN 收集 replay 并训练；定期在独立验证区间评估
最终只在从未用于选择模型的测试区间报告结果
```
""",
        "source": [
            ("../Chapter10/lib/data.py", "价格读取与 `Prices` 数据", "先确认字段和时间顺序"),
            ("../Chapter10/lib/environ.py", "`State`, `StocksEnv`", "逐步审计观测、reward、commission"),
            ("../Chapter10/lib/models.py", "前馈/卷积模型", "比较输入 shape 与 receptive field"),
            ("../Chapter10/lib/validation.py", "验证函数", "区分训练 reward 与回测指标"),
            ("../Chapter10/train_model.py", "训练入口", "数据切分、env、DQN 配置"),
            ("../Chapter10/tests/test_environ.py", "环境测试", "记账逻辑的可执行规范"),
        ],
        "deep": "先读测试再读环境：测试能告诉你作者认为什么是正确的 price move、position 和 reward。特别核对动作发生在 bar 的哪个价格、commission 在开仓还是平仓扣除、观测窗口是否包含下一根 bar。模型部分最后读，因为错误记账无法靠更强网络弥补。书中卷积模型只是改变时间特征提取，不解决非平稳、成本和数据泄漏。",
        "extra_code": r"""
import numpy as np

# 时间切分：任何 rolling 特征都只能向过去看。
prices = np.arange(20.0)
train, valid, test = prices[:12], prices[12:16], prices[16:]
train_mean = train.mean()
print("ranges:", (train[0], train[-1]), (valid[0], valid[-1]), (test[0], test[-1]))
print("train-only mean:", train_mean, "leaky all-data mean:", prices.mean())
assert train[-1] < valid[0] < test[0]
""",
        "solution": "成本为 0 时只计算仓位乘价格收益；0.001 时每次换仓扣费；0.01 时频繁换仓很可能吞掉所有优势。应打印每步 `gross_return`、`turnover_cost` 和 `net_reward`，确认差值精确等于成本项。",
        "next": "第 11 章转入策略方法；交易环境也可用 actor-critic，但数据泄漏和评估原则完全不变。",
    },
    11: {
        "coverage": [
            ("Values and policy", "价值法与策略法对照", "详细"),
            ("REINFORCE", "完整 likelihood-ratio 推导", "详细"),
            ("REINFORCE issues", "方差、baseline、credit assignment", "详细"),
            ("CartPole policy gradients", "渐进源码路线", "完整"),
            ("Pong policy gradients", "像素与长轨迹工程", "源码精读"),
        ],
        "algorithm": r"""
```text
收集一批完整 trajectories：保存 state、action、reward、log_prob
对每条轨迹从后向前计算 reward-to-go Gt
可选：估计 baseline b(st)，advantage = Gt - b(st)
合并 batch 并标准化 advantage
policy_loss = -mean(log_prob(action|state) * stopgrad(advantage))
entropy_bonus = mean(H(policy(state)))
更新策略；若有 baseline，再用 MSE 单独更新 value
```
""",
        "source": [
            ("../Chapter11/01_cartpole_dqn.py", "DQN 对照", "同任务比较价值法"),
            ("../Chapter11/02_cartpole_reinforce.py", "REINFORCE baseline", "最小完整策略梯度"),
            ("../Chapter11/03_cartpole_reinforce_baseline.py", "reward baseline", "看方差变化"),
            ("../Chapter11/04_cartpole_pg.py", "entropy / batch 改进", "逐项增加稳定技巧"),
            ("../Chapter11/05_pong_pg.py", "Pong 策略梯度", "高维观测和长 episode"),
            ("../Chapter11/lib/common.py", "共享 loss/数据工具", "变量语义集中处"),
        ],
        "deep": "阅读 CartPole 系列时不要只比较最终分数，要做差分阅读：每个脚本相对前一个增加了什么、它改变了 bias 还是 variance。Pong 中回报常按局内得分边界重置/归一化，原因是不同回合间的动作不应互相分配信用。确认 `log_prob` 与对应 action/state 来自同一策略采样时刻。",
        "extra_code": r"""
import numpy as np

# 数值验证 baseline 项期望为 0：sum_a pi(a) * grad log pi(a) = 0。
pi = np.array([0.6, 0.4])
# 对二动作 softmax 的第一个 logit 求导：d log pi_a / dz0 = 1[a=0]-pi0
grad_logp = np.array([1 - pi[0], -pi[0]])
expectation = np.sum(pi * grad_logp)
print("E[grad log pi] =", expectation)
assert np.isclose(expectation, 0.0)
""",
        "solution": "应在更新前保存 `dist.probs.gather(...)`，执行一次优化，再重新前向。正 advantage 样本的所选动作概率通常上升，负 advantage 样本下降；但共享参数会造成样本间耦合，所以不要要求每个样本都严格单调，重点检查总体 loss 方向和单样本隔离测试。",
        "next": "第 12 章用 critic 替代整局 Monte Carlo baseline，使策略能以较低方差、更频繁地更新。",
    },
    12: {
        "coverage": [
            ("Variance reduction / CartPole variance", "baseline 与实验直觉", "详细"),
            ("Advantage Actor-Critic (A2C)", "TD advantage、联合 loss", "详细"),
            ("A2C on Pong / tuning", "源码路线", "完整"),
            ("Asynchronous A3C", "梯度并行与同步", "详细"),
        ],
        "algorithm": r"""
```text
同步 A2C：
并行 N 个环境收集 T 步 rollout
若最后状态未终止，用 critic bootstrap
从后向前计算 n-step returns 或 GAE advantages
一次前向得到 policy logits 与 values
actor_loss = -log_prob * stopgrad(advantage)
critic_loss = (return - value)^2
loss = actor + cv*critic - ce*entropy
对 N*T 样本更新共享网络

A3C：多个 worker 各自采样/算梯度，将梯度异步施加到共享参数。
```
""",
        "source": [
            ("../Chapter12/01_cartpole_pg.py", "CartPole actor-critic", "低维验证 advantage"),
            ("../Chapter12/02_pong_a2c.py", "Pong A2C", "n-step rollout 与联合网络"),
            ("../Chapter12/02_pong_a2c_tune.py", "调参入口", "评价协议与搜索空间"),
            ("../Chapter12/03_a3c_grad.py", "梯度 worker", "参数/梯度跨进程传递"),
            ("../Chapter12/lib/common.py", "unpack/loss 工具", "公式到 tensor 的集中映射"),
        ],
        "deep": "A2C 关键不是‘两个网络’，而是 bootstrap return 如何与 rollout 边界对齐。跟踪每个环境最后状态的 `last_value`，真正终止为 0，截断或 rollout 切片仍应估值。A3C 重点检查 worker 梯度是否对应某个参数快照、主进程何时清梯度，以及异步延迟是否可接受。",
        "extra_code": r"""
import numpy as np

# GAE: delta_t 的指数加权和；lambda=0 近似一步 TD，lambda→1 接近长回报。
deltas = np.array([1.0, 0.5, -0.2])
gamma, lam = 0.99, 0.95
gae = np.zeros_like(deltas)
running = 0.0
for t in range(len(deltas) - 1, -1, -1):
    running = deltas[t] + gamma * lam * running
    gae[t] = running
print("GAE:", np.round(gae, 4))
""",
        "solution": "去掉 detach 后，actor loss 会通过 `advantage = target - value` 给 value 额外梯度；它可能推动 critic 仅为减小策略项而改变，而不是回归 return。比较 `value.grad` 可看到联合梯度与纯 critic loss 梯度不同。标准实现让 actor 与 critic 各自遵循定义好的目标。",
        "next": "第 15 章把 actor-critic 扩展到连续动作；第 16 章继续解决策略更新步子过大的稳定性问题。",
    },
    13: {
        "coverage": [
            ("Interactive fiction / environment", "POMDP 与 TextWorld", "详细"),
            ("Deep NLP basics", "token、embedding、序列模型", "详细"),
            ("Baseline DQN", "文本编码到动作价值", "完整流程"),
            ("Tweaking observations", "预处理与状态信息", "详细"),
            ("Transformers", "预训练编码器源码路线", "结构理解"),
            ("ChatGPT", "交互/自动 agent 入口", "应用与限制"),
        ],
        "algorithm": r"""
```text
TextWorld reset → textual observation + admissible commands
preprocessor:
  normalize text → tokenize → ids → pad/mask
model:
  embedding/encoder → state vector
  对候选动作编码或输出固定动作 Q/logits
用 legal-action mask 排除不可执行命令
DQN/策略方法选择命令并 step
把文本、候选命令、所选命令、reward 一起记录到 replay
```
""",
        "source": [
            ("../Chapter13/lib/preproc.py", "tokenization / preprocessing", "先看文本怎样变 tensor"),
            ("../Chapter13/lib/model.py", "文本模型", "跟踪 embedding、hidden、动作输出 shape"),
            ("../Chapter13/train_basic.py", "baseline DQN", "完整交互和 replay"),
            ("../Chapter13/train_preproc.py", "改进 observation", "比较输入信息"),
            ("../Chapter13/train_tr.py", "transformer 版本", "预训练编码器接入"),
            ("../Chapter13/tests/test_preproc.py", "预处理测试", "边界与 padding 的规范"),
            ("../Chapter13/chatgpt_interactive.py", "交互式 agent", "提示、状态与外部模型边界"),
        ],
        "deep": "先读 `test_preproc.py` 再读预处理实现，确认大小写、未知词、padding 和长度截断。随后画出 model 输入 shape：`(batch, seq_len)` ids → embedding `(batch, seq_len, dim)` → mask pooling/encoder → Q/logits。Transformer 版的主要变化是表示层，不改变 RL transition。ChatGPT 入口依赖外部服务，应把响应失败、成本、非确定性与安全边界当作环境工程问题。",
        "extra_code": r"""
import torch

# 带 padding mask 的平均池化；padding id 对表示不应有贡献。
emb = torch.tensor([[[1., 0.], [3., 2.], [99., 99.]]])
mask = torch.tensor([[1., 1., 0.]])
pooled = (emb * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
print("masked mean:", pooled)
assert torch.allclose(pooled, torch.tensor([[2., 1.]]))
""",
        "solution": "新增第五个动作并将其 mask 设为 False，即使 logit=100，`masked_fill(~legal, -inf)` 后概率仍为 0。还应增加 `assert legal.any()`，避免所有动作非法导致 softmax NaN。",
        "next": "第 14 章把语言与视觉/DOM 结合到网页环境；共同难点仍是部分可观测、动作候选和昂贵交互。",
    },
    14: {
        "coverage": [
            ("Evolution / browser automation / challenges", "问题建模与安全边界", "详细"),
            ("MiniWoB / MiniWoB++", "环境接口与源码路线", "详细"),
            ("Simple clicking approach", "坐标离散与模型", "完整"),
            ("Adding text description", "多模态输入", "详细"),
            ("Human demonstrations", "行为克隆推导与数据", "详细"),
            ("Things to try", "练习与扩展", "实践"),
        ],
        "algorithm": r"""
```text
在授权 MiniWoB sandbox reset 任务
获取 screenshot + DOM/text + viewport metadata
候选动作：网格点击或元素点击
可选先用 demonstrations 做 behavior cloning
在线阶段：策略选动作 → 浏览器执行 → 等待页面稳定 → 读取 reward/termination
记录 trajectory；用 PG/A2C/DQN 微调
评估时按任务、seed、成功率和平均操作数报告
```
""",
        "source": [
            ("../Chapter14/lib/wob.py", "环境封装", "浏览器 observation/action 生命周期"),
            ("../Chapter14/lib/model.py", "点击模型", "图像/文本输入与动作输出"),
            ("../Chapter14/record_demo.py", "演示录制", "人类动作怎样进入数据集"),
            ("../Chapter14/lib/demos.py", "演示读取/拼接", "行为克隆 batch"),
            ("../Chapter14/wob_click_train.py", "点击训练", "简单视觉策略"),
            ("../Chapter14/wob_click_mm_train.py", "多模态训练", "文本描述加入后的数据流"),
        ],
        "deep": "网页状态异步变化，因此一次 step 不只是‘发点击’：还要等待 DOM/截图稳定并确认任务是否结束。精读 `wob.py` 时标出等待和异常路径。演示数据必须同时保存 viewport、observation 和动作坐标，否则回放时无法正确归一化。多模态模型要确认图像 batch 与文本 batch 来自同一页面时刻。",
        "extra_code": r"""
# 检查坐标归一化能否跨 viewport 保持相对位置。
def normalize_xy(x, y, width, height):
    return x / width, y / height

def denormalize_xy(nx, ny, width, height):
    return nx * width, ny * height

n = normalize_xy(400, 300, 800, 600)
print("normalized:", n, "on 1920x1080:", denormalize_xy(*n, 1920, 1080))
assert n == (0.5, 0.5)
""",
        "solution": "四角要注意右/下边界：应使用 `min(cols-1, int(...))`。中心落在中间格。网格越细量化误差越小，但动作类别数按 `rows*cols` 增长，演示和探索样本更稀疏；元素级动作能减少无效区域，但依赖 DOM 可用性。",
        "next": "第 15 章处理真正连续动作；网页坐标虽然可连续化，书中先用离散网格是为了控制探索难度。",
    },
})


FINAL_GUIDE.update({
    15: {
        "coverage": [
            ("Why continuous space?", "离散/连续动作差异", "详细"),
            ("A2C method", "Gaussian policy 与连续采样", "详细"),
            ("Deep deterministic policy gradients", "DDPG 推导与源码", "详细"),
            ("Distributional policy gradients", "D4PG 投影与源码", "详细"),
            ("Things to try", "练习与扩展", "实践"),
        ],
        "algorithm": r"""
```text
DDPG/D4PG：
初始化 actor μθ、critic Qφ 及 target 副本、replay
与环境交互：a = scale(μθ(s) + exploration_noise)
采样 batch
with no_grad:
    a' = μθ-(s')
    y = r_n + γ^n * non_terminal * Qφ-(s', a')
critic: 回归 y（D4PG 改为 categorical distribution projection）
actor: 最小化 -Qφ(s, μθ(s))
soft-update 两个 target
```

连续 A2C 则让 actor 输出 Gaussian 的均值和方差，on-policy 采样动作；DDPG 是 deterministic、off-policy，并依赖 replay 和探索噪声。
""",
        "source": [
            ("../Chapter15/01_check_env.py", "环境检查", "先确认 observation/action bounds"),
            ("../Chapter15/02_train_a2c.py", "连续 A2C", "Gaussian 分布、log_prob、entropy"),
            ("../Chapter15/04_train_ddpg.py", "DDPG trainer", "replay、双 target、actor/critic 更新"),
            ("../Chapter15/06_train_d4pg.py", "D4PG trainer", "n-step 与分布投影"),
            ("../Chapter15/lib/model.py", "actor/critic/distributional critic", "输出范围与 shape"),
            ("../Chapter15/lib/common.py", "unpack_batch_a2c / unpack_batch_ddqn", "episode 边界与 bootstrap 的 batch 整理"),
        ],
        "deep": r"连续 A2C 要把每个动作维度的 log-prob 求和，entropy 也按维度汇总；标准差应通过 `softplus`/exp 等保证为正。DDPG 训练 actor 时 critic 参数可以参与前向但不应被 actor optimizer 更新。D4PG 的 categorical projection 与第 8 章相似，但 target atom 加上 n-step reward 后还要乘 $\gamma^n$，并处理连续动作的 target actor。",
        "extra_code": r"""
import numpy as np

# 简化 categorical projection：把标量 2.3 的质量分到相邻 atoms 2 和 3。
atoms = np.arange(0., 5.)
value = 2.3
lower, upper = int(np.floor(value)), int(np.ceil(value))
projected = np.zeros_like(atoms)
projected[lower] += upper - value
projected[upper] += value - lower
print("atoms:", atoms, "projected mass:", projected)
assert np.isclose(projected.sum(), 1.0)
""",
        "solution": "区间 `[2,5]` 的中心是 3.5、半宽 1.5，因此 unit action -1/0/1 映射到 2/3.5/5。实现应支持向量 low/high，并在送入环境前断言所有维度位于 bounds 内。",
        "next": "第 16 章继续使用连续控制环境，对比 PPO、TRPO、ACKTR 和 SAC 的稳定更新方式。",
    },
    16: {
        "coverage": [
            ("Environments / A2C baseline", "公平基线与评估", "详细"),
            ("PPO", "ratio、clip、epoch 流程", "详细"),
            ("TRPO", "KL 约束、共轭梯度、线搜索", "详细"),
            ("ACKTR", "自然梯度与 K-FAC", "结构理解"),
            ("SAC", "最大熵、双 Q、温度", "详细"),
            ("Overall results", "同协议比较", "详细"),
        ],
        "algorithm": r"""
```text
PPO：冻结 rollout 时的 old_log_prob 和 old_value
计算 returns / advantages
for epoch in 1..K:
  shuffle 成 minibatches
  new_log_prob, value, entropy = model(batch)
  ratio = exp(new-old)
  policy = -mean(min(ratio*A, clip(ratio)*A))
  value_loss = regression(value, return)（可 value clip）
  更新；若 approximate KL 超阈值则提前结束 epoch

TRPO：用 Fisher-vector product + conjugate gradient 求受 KL 约束方向，再 line search。
SAC：从 replay 更新 twin Q、随机 actor 和 temperature α，并 soft-update targets。
```
""",
        "source": [
            ("../Chapter16/01_train_a2c.py", "共同 baseline", "环境/评估协议"),
            ("../Chapter16/03_train_ppo.py", "PPO epoch/minibatch", "old log-prob 与 clip loss"),
            ("../Chapter16/lib/trpo.py", "共轭梯度、线搜索", "TRPO 数值核心"),
            ("../Chapter16/04_train_trpo.py", "TRPO trainer", "KL 约束如何接入 rollout"),
            ("../Chapter16/lib/kfac.py", "K-FAC optimizer", "ACKTR 曲率近似"),
            ("../Chapter16/06_train_sac.py", "SAC", "off-policy replay、双 critic、entropy"),
        ],
        "deep": r"PPO 精读时确认 `old_log_prob` 来自采样时策略并跨多个 epoch 固定；若每轮重算，ratio 会退化。TRPO 的 `conjugate_gradients` 不显式构造巨大 Fisher 矩阵，而只需要矩阵-向量积；line search 检查实际 KL 与 surrogate 改善。SAC 与前面三者不同：它是 off-policy，通常用 twin Q 的较小值抑制过估计，并可能自动优化 $\alpha$ 使 entropy 接近目标。",
        "extra_code": r"""
import numpy as np

eps = 0.2
cases = [(1.5, 1.0), (0.5, -1.0), (0.7, 1.0), (1.3, -1.0)]
for ratio, adv in cases:
    raw = ratio * adv
    clipped = np.clip(ratio, 1-eps, 1+eps) * adv
    surrogate = min(raw, clipped)
    print({"ratio": ratio, "A": adv, "raw": raw,
           "clip_term": clipped, "used": surrogate})
""",
        "solution": "打印四个样本的 `raw` 与 `clip_term`，只有当过度改变会让 surrogate 更乐观时，`min` 才选择保守项。正 advantage 且 ratio>1.2、负 advantage 且 ratio<0.8 是典型触发；朝错误方向的变化仍保留梯度纠正。",
        "next": "第 17 章完全绕开反向传播；第 18 章比较这些策略方法与价值方法在探索任务上的表现。",
    },
    17: {
        "coverage": [
            ("Black-box methods", "适用边界", "详细"),
            ("Evolution strategies", "梯度估计推导与 CartPole/HalfCheetah", "详细"),
            ("Genetic algorithms", "选择、变异、deep GA、novelty", "详细"),
            ("GA on CartPole / HalfCheetah", "源码路线", "完整"),
        ],
        "algorithm": r"""
```text
ES：
初始化参数中心 θ
repeat:
  生成成对噪声 ±εi
  并行评估 F(θ+σεi), F(θ-σεi)
  对 fitness 做 centered rank/标准化
  g = mean(Fi εi) / σ
  θ += learning_rate * optimizer(g)

GA：
评估 population → 选 elite → 复制/交叉/变异 → 新一代
可用 novelty score 替代或混合任务 fitness
```
""",
        "source": [
            ("../Chapter17/01_cartpole_es.py", "CartPole ES", "参数扰动、fitness 与更新"),
            ("../Chapter17/02_cheetah_es.py", "并行 HalfCheetah ES", "共享噪声/worker 评估"),
            ("../Chapter17/03_cartpole_ga.py", "CartPole GA", "elite 与变异"),
            ("../Chapter17/04_cheetah_ga.py", "HalfCheetah GA", "高维参数和并行"),
            ("../Chapter17/lib/common.py", "参数向量化工具", "state_dict 与 flat vector 往返"),
        ],
        "deep": "黑盒方法的首要工程问题是把模型参数稳定地展平和恢复，参数顺序必须固定。ES worker 可只接收 noise seed 而不是整组参数，降低通信量；主进程用相同 seed 重建扰动。GA 的 elite 复用容易降低多样性，novelty search 用行为描述符鼓励不同策略，但描述符选择会决定什么叫‘新颖’。",
        "extra_code": r"""
import numpy as np

rng = np.random.default_rng(0)
eps = rng.normal(size=5)
# 一维目标 F(theta)=-(theta-2)^2，在 theta=0 比较 antithetic pairs。
theta, sigma = 0.0, 0.1
f_plus = -((theta + sigma*eps) - 2)**2
f_minus = -((theta - sigma*eps) - 2)**2
grad = np.mean((f_plus - f_minus) * eps) / (2*sigma)
print("antithetic gradient estimate:", grad, "true gradient:", 4.0)
""",
        "solution": "population=10 评估少但梯度噪声大，不同 seed 可能偏离最优；1000 更稳定但每代环境成本高 100 倍。比较时必须用相同总 evaluation budget，而不能只比较迭代轮数。",
        "next": "第 18 章的 NoisyNet 也在参数中注入噪声，但目的和更新方式不同：它仍通过 TD/策略梯度学习。",
    },
    18: {
        "coverage": [
            ("Problems with epsilon-greedy / importance", "深度探索动机", "详细"),
            ("Noisy / count / prediction-based exploration", "三类机制", "详细"),
            ("MountainCar experiments", "DQN/PPO 各变体比较", "源码路线"),
            ("Atari experiments", "可扩展性与结果", "源码路线"),
        ],
        "algorithm": r"""
```text
RND 训练循环：
固定随机 target network f，不加入 optimizer
训练 predictor f_hat
每个 observation：
  target_feature = f(obs)             # no_grad
  prediction = f_hat(obs)
  intrinsic = squared_error(prediction, target_feature)
  用运行统计归一化 observation 与 intrinsic reward
  total_reward = external + eta * intrinsic
  RL 算法用 total_reward 更新
  predictor 以同一误差更新
评估只报告 external reward
```
""",
        "source": [
            ("../Chapter18/riverswim.py", "RiverSwim", "深度探索的可控小环境"),
            ("../Chapter18/mcar_dqn.py", "MountainCar DQN variants", "epsilon/noisy/count 对照"),
            ("../Chapter18/mcar_ppo.py", "MountainCar PPO variants", "策略法探索对照"),
            ("../Chapter18/atari_dqn.py", "Atari DQN", "高维扩展"),
            ("../Chapter18/atari_ppo.py", "Atari PPO", "RND/Noisy 等组合"),
            ("../Chapter18/lib/dqn_extra.py", "NoisyDQN / MountainCarNoisyNetDQN", "参数噪声探索的网络实现"),
            ("../Chapter18/lib/common.py", "PseudoCountRewardWrapper / NetworkDistillationRewardWrapper", "计数奖励与蒸馏奖励如何接入环境"),
            ("../Chapter18/lib/ppo.py", "batch_generator_distill / MountainCarNetDistillery", "内外奖励、RND 与 PPO rollout"),
        ],
        "deep": "MountainCar 的关键是连续多步向反方向加速才能爬坡，逐步独立 epsilon 动作很难形成一致行为；参数噪声能在一个状态区域保持动作偏好。count-based 对低维状态需离散化，bin 太粗会把不同状态当相同，太细又几乎不重复。RND 要冻结 target、归一化 observation/error，并防止 predictor 在 replay 频率上过拟合。",
        "extra_code": r"""
import torch
from torch import nn

torch.manual_seed(0)
target = nn.Linear(2, 4)
predictor = nn.Linear(2, 4)
for p in target.parameters():
    p.requires_grad_(False)
states = torch.tensor([[0., 0.], [1., 1.]])
with torch.no_grad():
    target_features = target(states)
intrinsic = (predictor(states) - target_features).square().mean(1)
print("intrinsic rewards:", intrinsic)
assert all(p.grad is None for p in target.parameters())
""",
        "solution": r"A/B 循环后其 count 增大，bonus 下降；首次 C 仍获得 $\beta$。这促使 agent 离开熟悉区域。但若 C 是不可控噪声状态，持续奖励可能有害，因此探索信号必须与可学习、可控制的新奇性结合。",
        "next": "第 19 章也学习一个奖励模型，但其信号来自人类偏好而不是新奇度；两者都存在代理目标被利用的问题。",
    },
    19: {
        "coverage": [
            ("Reward functions in complex environments", "为什么需要偏好", "详细"),
            ("Theoretical method / RLHF and LLMs", "Bradley-Terry、KL", "详细"),
            ("Initial A2C training", "实验闭环起点", "源码路线"),
            ("Labeling process", "UI、pair 采样与偏差", "详细"),
            ("Reward model training", "loss、验证", "详细"),
            ("A2C + reward model / multi-round fine-tuning", "完整闭环", "详细"),
            ("Overall results", "独立评价", "详细"),
        ],
        "algorithm": r"""
```text
1. 训练/准备初始 policy，收集 trajectory clips。
2. 成对展示 clips；记录 A/B/tie/skip 标签并随机左右位置。
3. reward model 对每帧/步评分，片段内求和。
4. 用 BCEWithLogits(score_A-score_B, preference) 训练并在独立 pairs 验证。
5. RL 阶段使用 predicted reward，并加入对 reference policy 的 KL penalty。
6. 收集新策略产生的片段，重点标注 reward-model 高分/不确定样本。
7. 重训 reward model 与 policy；用未参与训练的人类评价报告结果。
```
""",
        "source": [
            ("../Chapter19/01_a2c.py", "初始 policy", "真实环境 reward baseline"),
            ("../Chapter19/01_play.py", "片段生成/播放", "标注材料来源"),
            ("../Chapter19/02_label_ui.py", "标注 UI", "pair 顺序、标签存储"),
            ("../Chapter19/03_reward_train.py", "reward model trainer", "train/validation 与 checkpoint"),
            ("../Chapter19/lib/rlhf.py", "偏好数据与 loss", "Bradley-Terry 实现"),
            ("../Chapter19/lib/ui_tools.py", "UI 辅助", "展示与输入边界"),
        ],
        "deep": "检查 pair 数据切分必须按原始 trajectory 分组，避免同一片段的近重复同时出现在训练和验证。UI 应随机交换左右位置并支持 tie/无法判断。reward model 输出的绝对平移不可辨识，关注 pair accuracy/calibration 而非分数绝对值。策略优化后必须重新采样标注，因为数据分布已经改变。",
        "extra_code": r"""
import torch
import torch.nn.functional as F

a = torch.tensor([2.0, -0.5, 1.0])
b = torch.tensor([1.0, 0.5, 1.5])
y = torch.tensor([1.0, 0.0, 1.0])
loss_ab = F.binary_cross_entropy_with_logits(a-b, y)
loss_ba = F.binary_cross_entropy_with_logits(b-a, 1-y)
print("swap-invariant losses:", loss_ab.item(), loss_ba.item())
assert torch.allclose(loss_ab, loss_ba)
""",
        "solution": "交换 A/B 后同时把标签变为 `1-label`，logit 从 $R_A-R_B$ 变为其负数，BCE 完全相同。若不翻转标签，语义就变成相反偏好，loss 通常不同。",
        "next": "第 20 章的 self-play 也形成数据生成—训练—改进闭环，但监督信号来自搜索和游戏胜负，而非人类偏好。",
    },
    20: {
        "coverage": [
            ("Model-based vs model-free / board games", "概念对比", "详细"),
            ("AlphaGo Zero: MCTS/self-play/training", "PUCT 与闭环", "详细"),
            ("Connect 4 AlphaZero implementation/results", "游戏、MCTS、模型、训练源码", "完整"),
            ("MuZero high-level model/training", "三模型与 unroll loss", "详细"),
            ("Connect 4 MuZero", "节点、模型、搜索、数据", "完整源码路线"),
            ("MuZero results / Atari", "结果与扩展", "结构理解"),
        ],
        "algorithm": r"""
```text
AlphaZero self-play：
对每个局面运行 MCTS：selection(PUCT) → expansion → network evaluation → backup
根访问次数形成 policy target π；按温度采样动作
终局结果 z 回填该局所有状态
训练网络最小化 value MSE + policy cross-entropy + regularization
用新网络继续 self-play，并通过评估决定是否替换

MuZero：representation 得初始 hidden；MCTS 中 dynamics 展开 action 得 reward/next hidden；
prediction 给 policy/value；训练时对 K 步 unroll 累计 reward/value/policy losses。
```
""",
        "source": [
            ("../Chapter20/lib/game.py", "状态编码、合法动作、胜负", "先确保游戏规则"),
            ("../Chapter20/tests/test_game.py", "规则测试", "可执行规范"),
            ("../Chapter20/lib/mcts.py", "`find_leaf`, `search_batch`, `get_policy_value`", "AlphaZero 搜索"),
            ("../Chapter20/lib/model.py", "policy/value network", "训练输出"),
            ("../Chapter20/train.py", "self-play + train", "AlphaZero 闭环"),
            ("../Chapter20/lib/muzero.py", "`ReprModel`, `DynamicsModel`, `PredModel`, `run_mcts`, `sample_batch`", "MuZero 全流程"),
            ("../Chapter20/train-mu.py", "MuZero trainer", "unroll targets 与同步"),
        ],
        "deep": "先跑游戏测试再读搜索。AlphaZero `MCTS` 要检查同一状态在不同当前玩家视角下的 value 符号；batched search 还使用虚拟损失避免多个 leaf 请求重复路径。MuZero 中 `MCTSNode` 保存 prior、visit、value/reward；`expand_node` 调 dynamics/prediction，`backpropagate` 累积折扣 reward/value。训练 sample 必须将 episode 中的 action、reward、root policy 和最终 value 按 unroll 对齐。",
        "extra_code": r"""
import numpy as np

visits = np.array([30, 10, 0], dtype=float)
def visit_policy(visits, temperature):
    if temperature == 0:
        out = np.zeros_like(visits); out[visits.argmax()] = 1; return out
    weights = visits ** (1.0 / temperature)
    return weights / weights.sum()

for t in (1.0, 0.5, 0):
    print("temperature", t, "policy", visit_policy(visits, t))
""",
        "solution": "降低第三动作 prior 会减小其初始探索项；增加其 visit 又增大分母，使探索项继续下降。除非回传 Q 足够高，它会更不容易被选择。PUCT 中 prior 主要影响搜索早期，长期证据由 Q 与访问次数主导。",
        "next": "第 21 章同样把网络价值与搜索组合，但目标从对抗棋局变成单智能体离散优化。",
    },
    21: {
        "coverage": [
            ("Rubik's cube / God's number", "问题与最优性", "详细"),
            ("Actions / states / solving approaches", "环境编码与合法变换", "源码精读"),
            ("Training architecture/process", "curriculum 与 value/policy targets", "详细"),
            ("Model application/results", "搜索与指标", "详细"),
            ("Code outline: env/train/search", "真实源码路线", "完整"),
            ("2x2 / 3x3 experiments", "配置、结果与限制", "结构理解"),
            ("Further improvements", "扩展", "地图"),
        ],
        "algorithm": r"""
```text
训练：
从 solved cube 生成不同 scramble depth 的状态
编码 sticker/cubie state
对每个状态枚举合法动作的后继
用当前网络/已知深度构造 value 与 policy targets
训练双头网络；逐步增加 scramble depth

求解：
以待解状态为根
用网络 value/policy 引导 MCTS/启发式搜索
避免立即逆动作和重复状态
找到 solved state 后输出动作路径，并逐步验证变换合法
```
""",
        "source": [
            ("../Chapter21/libcube/cubes/_env.py", "`CubeEnv`, `scramble_cube`, `explore_state`", "统一魔方环境接口"),
            ("../Chapter21/libcube/cubes/cube2x2.py", "`Action`, `transform`, `encode_inplace`", "先读较小魔方"),
            ("../Chapter21/tests/libcube/cubes/test_cube2x2.py", "逆动作/变换测试", "规则正确性"),
            ("../Chapter21/libcube/model.py", "`Net`, `encode_states`, `sample_batch`", "训练数据与双头模型"),
            ("../Chapter21/libcube/mcts.py", "`search`, `_expand_leaves`, `_backup_leaf`, `find_solution`", "网络引导搜索"),
            ("../Chapter21/train.py", "训练入口", "curriculum/config"),
            ("../Chapter21/solver.py", "求解与统计", "解出率、深度、时间"),
        ],
        "deep": "先用 2x2 测试理解 `transform(action)` 与 `inverse_action`，确保动作后接逆动作恢复原状态；状态编码要固定颜色/方向约定。`sample_batch` 不是普通监督数据加载器，它会探索后继并根据 value target 方法构造标签。MCTS 使用 virtual loss 做 batch leaf expansion。求解报告要区分 search depth、solution length、expanded nodes 和 timeout。",
        "extra_code": r"""
from heapq import heappush, heappop

# A* 玩具实现：priority = 已付代价 g + heuristic h。
def astar(start, goal):
    heap = [(abs(goal-start), 0, start, [start])]
    best = {start: 0}
    while heap:
        _, g, s, path = heappop(heap)
        if s == goal: return path
        for nxt in {s-1, s+1, s*2}:
            if not -20 <= nxt <= 40: continue
            ng = g + 1
            if ng < best.get(nxt, 10**9):
                best[nxt] = ng
                heappush(heap, (ng + abs(goal-nxt), ng, nxt, path+[nxt]))
print("A* path:", astar(1, 15))
""",
        "solution": "width=1 是贪心 beam，内存小但容易剪掉正确路径；width=10 保留更多候选，解出率通常更高但展开量增加。记录是否找到、路径长度和候选数量，不能只比较运行时间。",
        "next": "第 22 章从单一求解器进入多个同时行动的 agent；状态、动作和经验的对齐问题会更复杂。",
    },
    22: {
        "coverage": [
            ("What is multi-agent RL?", "Markov game 与非平稳性", "详细"),
            ("MAgent environment/install/random setup", "环境接口与随机 baseline", "源码路线"),
            ("DQN for tigers", "共享网络与 group experience", "详细"),
            ("Tiger collaboration", "团队奖励与协作", "详细"),
            ("Training tigers and deer", "双群体同时学习", "完整流程"),
            ("Battle environment", "竞争/self-play", "源码路线"),
        ],
        "algorithm": r"""
```text
parallel env.reset → dict(agent_id → observation)
按 agent group 选择共享 policy/network
每步为所有存活 agent 生成 action dict
env.step(action_dict) → obs/reward/termination/truncation dicts
experience source 按 agent_id 维护各自 n-step deque
写入 experience 时附带 group；死亡 agent 正确结束轨迹
从 buffer 按 group 过滤 batch，分别更新各网络
定期在固定对手池评估捕获数、存活数、团队回报
```
""",
        "source": [
            ("../Chapter22/lib/data.py", "`ForestEnv`, `BattleEnv`, `MAgentExperienceSourceFirstLast`", "环境和多 agent 经验核心"),
            ("../Chapter22/lib/model.py", "`DQNModel`, `unpack_batch`, `calc_loss_dqn`", "共享 DQN 与 group batch"),
            ("../Chapter22/forest_random.py", "随机 baseline", "先理解 observation/action dict"),
            ("../Chapter22/forest_tigers_dqn.py", "单群体学习", "tigers 对固定 deer"),
            ("../Chapter22/forest_both_dqn.py", "双群体学习", "两个网络和 batch filter"),
            ("../Chapter22/battle_dqn.py", "battle/self-play", "对手非平稳与评估"),
        ],
        "deep": "`MAgentExperienceSourceFirstLast` 是本章关键：它为每个 agent id 维护独立历史，同时 agent 会出生/死亡，不能按数组位置假设身份稳定。`agent_group` 决定由哪个 policy 产生动作，经验中的 group 用于训练时过滤。双群体脚本应检查两个 optimizer、target net 和 replay 是否正确隔离；评估对手要冻结，避免测试环境也非平稳。",
        "extra_code": r"""
import numpy as np

# 协调博弈的联合 payoff 表：对角线为共享奖励 1。
payoff = np.eye(2)
print("joint payoff:\n", payoff)
for a1 in range(2):
    for a2 in range(2):
        print((a1, a2), "reward", payoff[a1, a2])
assert payoff[0, 0] == payoff[1, 1] == 1
""",
        "solution": "不协调奖励改为 -1 后，错误配对惩罚更强，可能加快已有偏好的一致化，也可能让早期噪声更难恢复。多个 seed 最终可能落在 `(0,0)` 或 `(1,1)`，两者都是对称最优均衡；应报告协调成功率而非把某个动作编号当唯一答案。",
        "next": "这是全书终点：MDP 被扩展为 Markov game，但回报、价值、bootstrap、函数逼近、replay 和策略优化仍是前面章节建立的同一套语言。",
    },
})


def bullet(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def link_list(paths: list[str]) -> str:
    if not paths:
        return "本章是理论起点，仓库没有独立 `Chapter01` 脚本；上面的数值例子就是本章配套实现。"
    return "\n".join(f"- [`{Path(p).name}`]({p.replace('../', '../../', 1)})" for p in paths)


def table_cell(value: str) -> str:
    """Keep formulas containing pipes from breaking Markdown tables."""
    return value.replace("|", r"\mid ").replace("\n", "<br>")


def symbol_table(rows: list[tuple[str, str, str]]) -> str:
    body = "\n".join(f"| {table_cell(a)} | {table_cell(b)} | {table_cell(c)} |" for a, b, c in rows)
    return "| 符号/名称 | 含义 | 阅读时怎么想 |\n|---|---|---|\n" + body


def mapping_table(rows: list[tuple[str, str, str]]) -> str:
    body = "\n".join(f"| {table_cell(a)} | {table_cell(b)} | {table_cell(c)} |" for a, b, c in rows)
    return "| 数学对象/步骤 | 代码对应 | 作用与形状提示 |\n|---|---|---|\n" + body


def coverage_table(rows: list[tuple[str, str, str]]) -> str:
    body = "\n".join(f"| {table_cell(a)} | {table_cell(b)} | {table_cell(c)} |" for a, b, c in rows)
    return "| 原书主题 | 本 notebook 对应位置 | 深度 |\n|---|---|---|\n" + body


def source_table(rows: list[tuple[str, str, str]]) -> str:
    if not rows:
        return "本章没有配套源码目录；两个数值实验就是本章的完整可执行部分。"
    body = "\n".join(
        f"| [`{Path(path).name}`]({path.replace('../', '../../', 1)}) | {table_cell(symbols)} | {table_cell(purpose)} |"
        for path, symbols, purpose in rows
    )
    return "| 文件 | 重点类/函数 | 阅读目的 |\n|---|---|---|\n" + body


def make_notebook(number: int, chapter: dict) -> dict:
    title = chapter["title"]
    guide = V2_GUIDE[number]
    final = FINAL_GUIDE[number]
    training = TRAINING_GUIDE[number]
    check_question, check_answer = guide["check"]
    cells = [
        md(f"""
        <a id="chapter-{number:02d}"></a>

        # Chapter {number:02d} · {title}

        [← 学习索引](../README.md) · [项目主页](../../README.md) · [跳到本章扩展](#extension)

        > 学习方式：按顺序读到代码单元，先预测输出再运行。正文只保留理解主线；非必要的证明、变体与工程细节放在[扩展](#extension)。

        **学完本章，你应该能：**

        {bullet(chapter['goals'])}

        **开始前最好已经会：**

        {bullet(guide['prereq'])}
        """),
        md(f"""
        ## 原书覆盖地图

        {coverage_table(final['coverage'])}

        “详细”表示正文包含推导或完整数据流；“源码路线”表示正文解释职责并链接真实实现；次要变体保留在扩展区。这样可以明确看到本章没有把原书哪个主题悄悄跳过。
        """),
        md(f"""
        ## 0. 先想一个问题

        {guide['question']}

        > 不必立刻回答。读完本章后再回到这里，尝试用两三句话作答。
        """),
        md(f"""
        ## 1. 心智模型：先知道信息怎样流动

        {chapter['map']}

        学习时始终追问三件事：**当前已知什么？要估计什么？哪个量会被更新？** 这比先背算法名称更重要。
        """),
        md(f"""
        ## 2. 符号表：先统一语言

        {symbol_table(guide['symbols'])}

        第一次阅读只需认识这些符号；看到公式时再回来查，不要求一次背完。
        """),
        md("## 3. 分步推导：公式不是凭空出现的\n\n" + guide["derivation"]),
        md(f"""
        ## 4. 手算一个最小例子

        {guide['hand']}

        > 手算的目的不是练算术，而是给公式建立可检查的直觉。代码输出与手算矛盾时，先停下来找原因。
        """),
        md("## 5. 本章公式卡\n\n" + chapter["formula"] + "\n\n这张卡用于复习，不替代上一节的推导。"),
        md(f"""
        ## 6. 公式 ↔ 代码逐项对照

        {mapping_table(guide['mapping'])}

        **从公式过渡到代码时：**{chapter['bridge']}
        """),
        md("""
        ## 7. 动手实验

        运行前先做三件事：

        1. 根据手算例子预测关键输出的方向或数量级；
        2. 找到上一张表中的每个代码变量；
        3. 运行后只改一个变量，观察结果是否符合公式。
        """),
    ]
    cells.extend(code(c) for c in chapter["codes"])
    cells.extend([
        md(f"""
        ### 怎样读运行结果

        {guide['read']}

        如果结果不符合描述，先检查随机种子、终止 mask、数组 shape 和依赖版本，而不是马上调超参数。
        """),
        md("## 8. 完整算法：把前面的零件装起来\n\n" + final["algorithm"]),
        md("""
        ## 9. 第二个验证实验

        第一个实验负责建立直觉；下面这个实验专门验证一个容易写错的性质。它仍然保持轻量，目的是让你能用断言判断实现是否符合公式。
        """),
        code(final["extra_code"]),
        md(f"""
        ## 10. 项目源码精读路线

        {source_table(final['source'])}

        <details>
        <summary>为什么按这个顺序读</summary>

        {final['deep']}

        </details>

        notebook 的实验负责隔离原理，仓库脚本负责完整环境、日志和训练。阅读时拿着第 6 节对照表寻找同一数学对象；变量名可以不同，数据语义不能不同。
        """),
        md(f"""
        ## 11. 训练曲线、调参与失败复盘

        **本章主诊断指标：**{training['metric']}

        下面的曲线来自可重复运行的微型训练，不是手工填写的数据。它用于学习“该看什么、怎样判断”；原书规模的训练应使用上一节链接的项目脚本，并同时保存配置、随机种子、环境步数和 wall-clock。

        > 曲线阅读顺序：先看任务指标，再看优化指标，最后看资源指标。单次最好成绩不能代替多随机种子均值与波动。
        """),
        code(training["code"]),
        md(f"""
        ### 曲线怎么读

        {training['read']}

        ### 调参过程：一次只改变一个因素

        {bullet(training['tuning'])}

        ### 失败案例复盘

        {bullet(training['failures'])}

        遇到失败时先保存配置和原始曲线，复现后再修；不要一边改环境、一边改网络、一边改学习率，否则无法知道修复来自哪里。
        """),
        md(f"""
        ## 12. 实用技巧与常见坑

        {bullet(chapter['tips'])}

        这些不是附属知识：RL 结果方差大，很多“算法没学会”其实是终止语义、shape、数据泄漏或评估方式错误。
        """),
        md(f"""
        ## 13. 动手练习

        {chapter['exercise']}

        <details>
        <summary>完成标准（做完再展开）</summary>

        不只给出结果；还应说明改了哪个变量、对应哪条公式、变化是否符合预期，并写出至少一个边界情况。若能用 `assert` 固化结论更好。
        </details>
        """),
        md(f"""
        ## 14. 练习参考答案

        <details>
        <summary>完成练习后再展开</summary>

        {final['solution']}

        </details>
        """),
        md(f"""
        ## 15. 合上 notebook 后，脑中应留下什么

        {bullet(guide['takeaways'])}

        回到开头的那个问题。现在请不用公式，向一个没学过本章的人解释答案；能讲清楚，才算真正理解。
        """),
        md(f"""
        ## 16. 两分钟自测

        **问题：**{check_question}

        <details>
        <summary>展开答案</summary>

        {check_answer}
        </details>
        """),
        md(f"""
        ## 17. 与前后章节连接

        {final['next']}

        学习强化学习最怕把算法记成孤岛。每章结束时，都应能说明本章复用了前面的什么，又为后面准备了什么。
        """),
        md(f"""
        <a id="extension"></a>

        ## 扩展：主线学完后再深挖

        {chapter['extension']}

        扩展内容用于建立知识地图，不要求第一次学习就掌握。先确保能完成上面的两分钟自测，再决定是否继续。

        [↑ 回到开头](#chapter-{number:02d}) · [← 学习索引](../README.md)
        """),
    ])
    # nbformat 4.5+ recommends a stable, unique id for every cell.
    for index, cell in enumerate(cells):
        cell["id"] = f"ch{number:02d}-cell-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_readme() -> str:
    rows = []
    for i, ch in enumerate(CHAPTERS, 1):
        zh = ch["title"].split(" / ", 1)[-1]
        rows.append(f"| {i:02d} | [{zh}](02_chapters/Chapter{i:02d}.ipynb) | `{ch['title'].split(' / ')[0]}` |")
    return dedent("""
    # 我的强化学习笔记

    **Python薄弱、没有机器学习基础：请先打开[零基础陪学入口](START_HERE.md)。** 新路线把每天做什么、具体视频选段、公式手算、练习反馈和补课位置串起来，并用 `python my_learning/coach.py` 保存跨次学习进度。

    这是一套从零基础到能够独立实现、训练和诊断强化学习算法的学习系统，以 *Deep Reinforcement Learning Hands-On, Third Edition* 的 22 章为主线，并链接当前仓库的真实源码。

    完整的逐章、逐主题核对结果见 [原书覆盖审计](05_reference/COVERAGE.md)。

    ## 只走这一条学习路线

    1. 打开[START_HERE：日常入口](START_HERE.md)，再进入[六阶段目录](06_lessons/README.md)中的当前阶段。
    2. 从阶段目录打开当前学习章节，按“本章导学 → 视频 → 手算 → 实践 → 反馈 → 本章总结”学习。
    3. 跟章节底部的“下一步”走；阶段结束先做综合复盘，不直接跳进高级章节。
    4. 章节引用到哪份notebook或源码，就只打开对应材料；不需要把所有目录按文件顺序读完。

    **P是前置基础、R是强化学习主线、O是选修。** 它们是保存进度的稳定课号，不是原书章号。课卡已归入六个中文阶段目录；旧进度和测验命令仍然有效。

    > 这些是原创学习笔记和小型实现，不复制原书正文。公式符号尽量保持教材常用约定；若与具体脚本变量名不同，以语义为准。

    ## 配套资料（不是另一套学习顺序）

    | 材料 | 用途 |
    |---|---|
    | [入门操作资料](00_start_here/README.md) | 章节引用时查环境、第一周操作和公式慢讲 |
    | [长期学习计划](05_reference/LONG_TERM_PLAN.md) | 已有编程基础的旧版预算与参考矩阵；不替代零基础日常路线 |
    | [零基础预备篇](01_prerequisites/Prerequisite00_Index.ipynb) | Python、NumPy、绘图、数学、概率、ML、神经网络从零、PyTorch、CNN/序列、Gymnasium、老虎机与入学测试 |
    | [00 · 学习路线与入门诊断](00_start_here/00_Study_Roadmap.ipynb) | 摸底、选择路线、阶段过关标准 |
    | [阶段考试](04_assessments/Milestone_Exams.ipynb) | 检查能否闭卷推导、实现和排错 |
    | [可复现实验模板](03_guided_projects/Experiment_Template.ipynb) | 多随机种子、置信带、调参记录与报告规范 |
    | [综合项目](03_guided_projects/Capstone_Projects.ipynb) | GridWorld、DQN、PPO/SAC 三档项目验收 |
    | [完整 GridWorld 项目](03_guided_projects/Guided_Project01_GridWorld.ipynb) | 第一次端到端完成 Q-learning |
    | [完整 DQN CartPole 项目](03_guided_projects/Guided_Project02_DQN_CartPole.ipynb) | 从 replay 到独立评估的深度 RL 项目 |
    | [掌握清单](05_reference/LEARNING_CHECKLIST.md) | 用能力证据而不是阅读进度打卡 |
    | [排错手册](05_reference/DEBUGGING_PLAYBOOK.md) | 从环境契约到评估协议的排查顺序 |
    | [中英术语表](05_reference/GLOSSARY.md) | 快速查找常用概念及一句话解释 |

    ## 原书伴读索引（按当前学习章节引用查阅）

    | 章 | Notebook | 原书英文主题 |
    |---:|---|---|
    """).strip() + "\n" + "\n".join(rows) + dedent("""

    原书章节号用于查教材；学习顺序以[六阶段目录](06_lessons/README.md)为准。例如PyTorch先在P11学习，Q-learning与DQN则拆成R05、R06两章。选修不阻塞主线。

    ## 目录结构

    ```text
    START_HERE.md      唯一日常学习入口
    06_lessons         六阶段 → 学习章节（导学、步骤、反馈、小结）
    00_start_here       入门操作与环境/公式辅助资料
    01_prerequisites    零基础预备篇（含自动批改练习）与入学测试
    02_chapters         原书 22 章伴读
    03_guided_projects  完整引导项目、实验模板、开放项目
    04_assessments      阶段考试
    05_reference        覆盖审计、排错、术语和掌握清单
    _tools              生成与验证脚本
    ```

    ## 维护说明

    学习者不必打开 `_tools`、`course.json` 或 `IMPLEMENTATION_PLAN.md`。生成器会重写课程文件，个人作业请另存；维护方式见[_tools说明](_tools/README.md)。
    """)


def make_coverage_audit() -> str:
    sections = [dedent("""
    # 原书覆盖审计

    本文件按原书目录的顶层主题逐项核对 22 个 notebook。它用于回答“哪些内容讲了、讲到什么深度、去哪里看”，避免仅凭文件数量宣称完成。

    - **详细：**正文含定义、推导、手算或完整数据流。
    - **完整/完整小例子：**含端到端算法流程和可执行验证。
    - **源码路线：**解释真实仓库文件、类和函数的阅读顺序。
    - **训练诊断：**每章都有真实运行的微型曲线、逐步调参顺序和失败案例；重型环境的正式成绩仍需运行原项目脚本。
    - **结构理解/概览/地图：**作为非主线内容放在源码精读或扩展中。
    """).strip()]
    for number, (chapter, final) in enumerate(zip(CHAPTERS, (FINAL_GUIDE[i] for i in range(1, 23))), 1):
        rows = coverage_table(final["coverage"])
        source_count = len(final["source"])
        sections.append(
            f"## [{number:02d} · {chapter['title']}](../02_chapters/Chapter{number:02d}.ipynb)\n\n"
            f"{rows}\n\n源码精读路线包含 **{source_count}** 个重点文件。"
        )
    return "\n\n".join(sections) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REFERENCE_OUT.mkdir(parents=True, exist_ok=True)
    for number, chapter in enumerate(CHAPTERS, 1):
        path = OUT / f"Chapter{number:02d}.ipynb"
        path.write_text(json.dumps(make_notebook(number, chapter), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (LEARNING_ROOT / "README.md").write_text(make_readme(), encoding="utf-8")
    (REFERENCE_OUT / "COVERAGE.md").write_text(make_coverage_audit(), encoding="utf-8")
    print(f"Generated {len(CHAPTERS)} notebooks in {OUT}")


if __name__ == "__main__":
    main()
