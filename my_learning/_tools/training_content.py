"""Reproducible curve, tuning, and failure-analysis labs for every chapter.

The experiments deliberately use tiny environments or datasets.  They expose
the same diagnostic signal as the full repository programs while remaining
fast enough to run inside a teaching notebook.
"""

from __future__ import annotations

from textwrap import dedent


PLOTTER = r"""
import numpy as np
import matplotlib.pyplot as plt

def plot_curve(series, title, ylabel, *, window=1):
    # Plot raw measurements and an optional moving average.
    values = np.asarray(series, dtype=float)
    x = np.arange(1, len(values) + 1)
    plt.figure(figsize=(7, 3.2))
    plt.plot(x, values, alpha=0.35, label="raw")
    if window > 1 and len(values) >= window:
        smooth = np.convolve(values, np.ones(window) / window, mode="valid")
        plt.plot(x[window - 1:], smooth, linewidth=2, label=f"MA({window})")
    plt.title(title)
    plt.xlabel("iteration / episode")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()
    plt.close()
"""


def lab(body: str) -> str:
    return dedent(PLOTTER + "\n" + body).strip()


TRAINING_GUIDE = {
    1: {
        "metric": "Bellman residual（相邻两次价值估计的最大变化）",
        "code": lab(r"""
gamma = 0.9
# 固定策略下的 3 状态 MRP；最后一个状态会持续获得奖励 1。
P = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
r = np.array([0.0, 0.0, 1.0])
v = np.zeros(3)
residuals = []
for _ in range(80):
    new_v = r + gamma * P @ v
    residuals.append(np.max(np.abs(new_v - v)))
    v = new_v

plot_curve(residuals, "Policy evaluation convergence", "Bellman residual")
print("estimated V:", np.round(v, 3), "final residual:", f"{residuals[-1]:.2e}")
"""),
        "read": "残差应快速下降后逐渐贴近 0；这表示反复 Bellman backup 正在逼近固定点。价值本身上升并不等于发散，要看残差是否收敛。",
        "tuning": [r"先固定 $\gamma=0.9$ 建立基线。", r"把 $\gamma$ 改为 0.5、0.99，只比较收敛速度和价值尺度。", "设停止条件 `residual < tolerance`，再权衡精度与迭代次数。"],
        "failures": [r"只看价值变大就误判发散：高 $\gamma$ 本来就会放大长期价值。", "原地更新与同步更新混用，导致结果顺序相关。", r"持续任务使用 $\gamma=1$，回报和价值可能没有有限解。"],
    },
    2: {
        "metric": "简化平衡环境中的最佳 episode length",
        "code": lab(r"""
rng = np.random.default_rng(2)

def evaluate(gain, episodes=20):
    scores = []
    for start in np.linspace(-0.18, 0.18, episodes):
        angle, velocity = start, 0.0
        for step in range(200):
            action = -np.sign(gain[0] * angle + gain[1] * velocity)
            velocity += 0.025 * action + 0.01 * angle
            angle += velocity
            if abs(angle) > 0.5:
                break
        scores.append(step + 1)
    return np.mean(scores)

gain = np.array([1.0, 0.0])
best = evaluate(gain)
history = []
for iteration in range(60):
    candidate = gain + rng.normal(0, 0.25, size=2)
    score = evaluate(candidate)
    if score > best:
        gain, best = candidate, score
    history.append(best)

plot_curve(history, "Wrapper-safe agent tuning", "best episode length")
print("best gain:", np.round(gain, 3), "score:", best)
"""),
        "read": "阶梯状上升是正常的：随机搜索只有找到更好的控制参数才更新 best。若曲线始终不动，先检查 `reset/step`、终止条件和动作语义，而不是增加网络。",
        "tuning": ["先让随机 agent 跑通，记录 episode length 基线。", "固定初始状态集合，再调控制增益，避免评估噪声掩盖变化。", "最后换多组初始状态验证，防止只记住单一 reset。"],
        "failures": ["把 `terminated` 和 `truncated` 都忽略，episode 永不结束。", "wrapper 改了 observation，却忘记同步 `observation_space`。", "用同一随机 episode 既调参又报告成绩，产生选择偏差。"],
    },
    3: {
        "metric": "监督训练 loss（用于验证 PyTorch 训练循环）",
        "code": lab(r"""
import torch

torch.manual_seed(3)
x = torch.linspace(-1, 1, 96).unsqueeze(1)
y = 2.5 * x - 0.4 + 0.08 * torch.randn_like(x)
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.12)
losses = []
for _ in range(80):
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plot_curve(losses, "PyTorch training loop", "MSE loss")
print("weight/bias:", round(model.weight.item(), 3), round(model.bias.item(), 3))
"""),
        "read": "loss 应先快后慢下降，参数接近 2.5 和 -0.4。若 loss 完全不变，优先检查梯度是否存在、optimizer 是否拿到模型参数、`step()` 是否执行。",
        "tuning": ["先用小模型和 SGD 验证数据、梯度、loss 能下降。", "学习率按 10 倍尺度尝试：0.01、0.1、1.0，找稳定下降区间。", "再加入隐藏层或 Adam；每次只改变一个因素。"],
        "failures": ["忘记 `zero_grad()`，梯度跨 batch 意外累加。", "在 forward 中调用 `.detach()`，计算图被切断。", "只看最后一个 batch 的 loss，把随机波动误当整体趋势。"],
    },
    4: {
        "metric": "CEM 策略选择最优动作的概率",
        "code": lab(r"""
rng = np.random.default_rng(4)
true_rewards = np.array([0.25, 0.75])
logits = np.zeros(2)
history = []
for iteration in range(45):
    probs = np.exp(logits - logits.max()); probs /= probs.sum()
    actions = rng.choice(2, size=120, p=probs)
    rewards = (rng.random(120) < true_rewards[actions]).astype(float)
    elite = actions[rewards >= np.quantile(rewards, 0.7)]
    counts = np.bincount(elite, minlength=2) + 1.0  # smoothing 防止 log(0)
    logits = np.log(counts / counts.sum())
    history.append(np.exp(logits[1]) / np.exp(logits).sum())

plot_curve(history, "CEM policy improvement", "P(best action)", window=5)
print("final action probabilities:", np.round(np.exp(logits) / np.exp(logits).sum(), 3))
"""),
        "read": "曲线总体应向 1 移动，但会因采样奖励上下波动。一次跌落不代表算法坏了；持续坍缩到错误动作才需要检查 elite 数量和样本量。",
        "tuning": ["先用较大 batch 保证 elite 中有信号。", "在 60%～90% 间调整 percentile；越高选择压力越强、方差也越大。", "困难环境保留多轮 elite 或对 episode 按折扣回报排序。"],
        "failures": ["percentile 太高导致 elite 几乎为空。", "稀疏奖励下所有 episode 同分，CEM 没有学习信号。", "只报告最好一次运行，没有跨随机种子比较。"],
    },
    5: {
        "metric": "Value iteration 的最大状态值变化量",
        "code": lab(r"""
gamma = 0.95
# 0--1--2--3(goal)，左右动作；进入 goal 获得 1。
n_states, n_actions = 4, 2
v = np.zeros(n_states)
deltas = []
for _ in range(60):
    q = np.zeros((n_states, n_actions))
    for s in range(n_states - 1):
        left, right = max(0, s - 1), min(n_states - 1, s + 1)
        q[s, 0] = gamma * v[left]
        q[s, 1] = (1.0 if right == n_states - 1 else 0.0) + gamma * v[right]
    new_v = q.max(axis=1)
    deltas.append(np.max(np.abs(new_v - v)))
    v = new_v

plot_curve(deltas, "Value iteration", "max |V_new - V|")
print("V*:", np.round(v, 3), "greedy actions:", q.argmax(axis=1))
"""),
        "read": "误差传播有明显的逐状态传递阶段，之后残差降到 0。若策略正确但数值有小差异，检查奖励是在离开状态还是进入状态时给出。",
        "tuning": ["先用确定性小 MDP 手算并和代码逐项对齐。", "用 tolerance 控制停止，而不是固定迭代次数。", "再引入随机转移，确认期望对所有 next state 求和。"],
        "failures": ["漏乘转移概率，把期望写成简单求和。", "terminal state 仍 bootstrap，重复计算终点奖励。", "只检查最终策略，不检查 Bellman residual，隐藏实现错误。"],
    },
    6: {
        "metric": "Q-learning 在链式环境中的 episode return",
        "code": lab(r"""
rng = np.random.default_rng(6)
q = np.zeros((6, 2))
returns = []
for episode in range(180):
    s, total = 0, 0.0
    epsilon = max(0.05, 1.0 - episode / 140)
    for _ in range(30):
        a = rng.integers(2) if rng.random() < epsilon else np.argmax(q[s])
        ns = max(0, s - 1) if a == 0 else min(5, s + 1)
        reward, done = (1.0 if ns == 5 else -0.01), ns == 5
        target = reward if done else reward + 0.95 * q[ns].max()
        q[s, a] += 0.2 * (target - q[s, a])
        total += reward; s = ns
        if done: break
    returns.append(total)

plot_curve(returns, "Q-learning on a tiny chain", "episode return", window=20)
print("greedy actions:", q.argmax(axis=1), "last-20 mean:", np.mean(returns[-20:]))
"""),
        "read": "原始回报噪声很大，移动平均应逐渐上升并稳定；同时 greedy action 应在非终止状态选择向右。只看 loss 下降不能证明策略变好。",
        "tuning": ["固定 replay、target、网络结构，先让 epsilon 从 1 线性下降。", "若回报不升，依次检查终止 mask、target、学习率，再动网络。", "基线稳定后才分别调整 replay warm-up、batch size、target sync。"],
        "failures": ["终止 transition 仍加入 bootstrap，Q 值系统性偏高。", "epsilon 降得太快，buffer 中几乎只有早期坏策略数据。", "训练回报上升但评估仍使用 epsilon，误判模型性能。"],
    },
    7: {
        "metric": "1-step 与 3-step Q-learning 的回报传播速度",
        "code": lab(r"""
rng = np.random.default_rng(7)

def train(n_steps):
    q = np.zeros((7, 2)); history = []
    for episode in range(120):
        s, trajectory, total = 0, [], 0.0
        for _ in range(35):
            a = rng.integers(2) if rng.random() < 0.2 else np.argmax(q[s])
            ns = max(0, s - 1) if a == 0 else min(6, s + 1)
            r, done = (1.0 if ns == 6 else -0.01), ns == 6
            trajectory.append((s, a, r)); total += r; s = ns
            if done: break
        for t, (state, action, _) in enumerate(trajectory):
            end = min(t + n_steps, len(trajectory))
            ret = sum((0.95 ** (k - t)) * trajectory[k][2] for k in range(t, end))
            if end < len(trajectory): ret += 0.95 ** (end - t) * q[trajectory[end][0]].max()
            q[state, action] += 0.2 * (ret - q[state, action])
        history.append(total)
    return np.convolve(history, np.ones(15) / 15, mode="valid")

one, three = train(1), train(3)
plt.figure(figsize=(7, 3.2)); plt.plot(one, label="1-step"); plt.plot(three, label="3-step")
plt.xlabel("episode"); plt.ylabel("moving-average return"); plt.grid(alpha=.25); plt.legend(); plt.show(); plt.close()
print("final means:", round(one[-1], 3), round(three[-1], 3))
"""),
        "read": "3-step 往往更快把终点奖励传回早期状态，但曲线也可能更抖。重点不是宣称 3-step 永远更好，而是理解传播速度与方差的交换。",
        "tuning": ["先保持 action selector、agent、buffer 不变，只改 `steps_count`。", "比较 1、3、5 step 的多随机种子均值和方差。", r"记得把 bootstrap 折扣改为 $\gamma^n$，否则比较无效。"],
        "failures": [r"改了 n-step 却仍乘单步 $\gamma$。", "episode 截断后错误地从下一局 bootstrap。", "PTAN experience 字段顺序理解错误，state/last_state 对调。"],
    },
    8: {
        "metric": "普通 max target 与 Double target 的过估计偏差",
        "code": lab(r"""
rng = np.random.default_rng(8)
ordinary, double = [], []
q_online = np.zeros(8); q_target = np.zeros(8)
for step in range(240):
    # 所有动作真实价值都是 0，只观测到零均值噪声。
    sample_online = rng.normal(0, 1, size=8)
    sample_target = rng.normal(0, 1, size=8)
    q_online += 0.06 * (sample_online - q_online)
    q_target += 0.06 * (sample_target - q_target)
    ordinary.append(q_target.max())
    chosen = q_online.argmax()
    double.append(q_target[chosen])

plt.figure(figsize=(7, 3.2)); plt.plot(ordinary, alpha=.65, label="ordinary max")
plt.plot(double, alpha=.65, label="Double: select/evaluate split")
plt.axhline(0, color="black", linewidth=1); plt.xlabel("update"); plt.ylabel("estimated target")
plt.grid(alpha=.25); plt.legend(); plt.show(); plt.close()
print("last-60 bias:", round(np.mean(ordinary[-60:]), 3), round(np.mean(double[-60:]), 3))
"""),
        "read": "真实价值为 0，普通 max 曲线通常保持正偏；Double 曲线围绕 0 波动。它展示的是统计偏差，不代表每个时间点 Double 都更低。",
        "tuning": ["先复现 tuned DQN baseline，再逐个加入 Double、n-step、PER。", "每次改动都保持训练帧数、评估 epsilon 和种子一致。", "Rainbow 组合后做 ablation，确认每个组件在当前环境确有贡献。"],
        "failures": ["online 和 target 都用同一个 `max`，代码名字叫 Double 但算法没变。", "PER 忘记 importance weights，比较结果带采样偏差。", "distributional projection 后概率和不为 1，却继续训练。"],
    },
    9: {
        "metric": "并行环境数增加后的成功经验累积速度",
        "code": lab(r"""
rng = np.random.default_rng(9)

def vector_training(n_envs, updates=100):
    positions = np.zeros(n_envs, dtype=int)
    successes, total = [], 0
    skill = 0.55
    for update in range(updates):
        move_right = rng.random(n_envs) < skill
        positions += np.where(move_right, 1, -1)
        positions = np.clip(positions, 0, 5)
        reached = positions == 5
        total += reached.sum()
        positions[reached] = 0
        skill = min(0.9, skill + 0.002 * reached.sum())
        successes.append(total)
    return successes

one, eight = vector_training(1), vector_training(8)
plt.figure(figsize=(7, 3.2)); plt.plot(one, label="1 env"); plt.plot(eight, label="8 envs")
plt.xlabel("learner update"); plt.ylabel("cumulative successes"); plt.grid(alpha=.25); plt.legend(); plt.show(); plt.close()
print("final successes:", one[-1], eight[-1])
"""),
        "read": "按 learner update 计，更多环境通常更快收集成功经验；但真实系统还要同时看 samples/s、GPU utilization 与 policy lag，环境数不是越多越好。",
        "tuning": ["先测单环境 samples/s 和 GPU 利用率，找到瓶颈。", "逐步增加环境数或 worker 数，记录吞吐和回报随 wall-clock 的变化。", "吞吐不再增加或 policy lag 变大时停止扩容。"],
        "failures": ["只按 gradient step 比较，忽略各方案消耗的环境样本不同。", "worker 使用过旧参数，吞吐上升但样本质量下降。", "多进程重复随机种子，收集到高度相关轨迹。"],
    },
    10: {
        "metric": "扣除交易成本后的 episode PnL",
        "code": lab(r"""
rng = np.random.default_rng(10)
returns_pattern = np.array([.02, .01, -.015, -.02, .012, .018, -.01, .006])
q = np.zeros((3, 3))  # state: down/flat/up, action: short/flat/long
pnl_history = []
for episode in range(180):
    returns = np.resize(np.roll(returns_pattern, episode % len(returns_pattern)), 64)
    returns = returns + rng.normal(0, .004, 64)
    position, total = 1, 0.0
    for t in range(len(returns) - 1):
        state = int(np.sign(returns[t])) + 1
        action = rng.integers(3) if rng.random() < max(.05, 1 - episode / 130) else np.argmax(q[state])
        cost = .002 * (action != position)
        reward = (action - 1) * returns[t + 1] - cost
        next_state = int(np.sign(returns[t + 1])) + 1
        q[state, action] += .12 * (reward + .9 * q[next_state].max() - q[state, action])
        position, total = action, total + reward
    pnl_history.append(total)

plot_curve(pnl_history, "Toy trading agent", "episode PnL after costs", window=20)
print("policy [down, flat, up]:", q.argmax(axis=1) - 1, "last-20 PnL:", np.mean(pnl_history[-20:]))
"""),
        "read": "应看扣费后 PnL 的移动平均、波动和换手率，而非只看训练 loss。若无费用时盈利、加入费用后转负，说明策略学到的是频繁交易而非稳健信号。",
        "tuning": ["先固定时间切分和 buy-and-hold 基线，禁止随机打乱未来数据。", "先调 reward/交易成本和 observation，再调网络与学习率。", "只在验证集选择超参数，最终测试集只使用一次。"],
        "failures": ["归一化使用了未来数据，造成 look-ahead leakage。", "训练和测试跨越同一时间窗口，成绩虚高。", "忽略手续费、滑点和持仓约束，策略不可执行。"],
    },
    11: {
        "metric": "REINFORCE 二臂 bandit 的平均回报与策略概率",
        "code": lab(r"""
rng = np.random.default_rng(11)
theta, baseline = 0.0, 0.0
rewards, probabilities = [], []
for episode in range(400):
    p = 1 / (1 + np.exp(-theta))
    action = float(rng.random() < p)
    reward = float(rng.random() < (0.25 if action == 0 else 0.75))
    baseline = 0.95 * baseline + 0.05 * reward
    theta += 0.08 * (reward - baseline) * (action - p)
    rewards.append(reward); probabilities.append(p)

plot_curve(rewards, "REINFORCE bandit", "reward", window=40)
print("P(best action):", round(probabilities[-1], 3), "last-40 reward:", np.mean(rewards[-40:]))
"""),
        "read": "单回合奖励仍是 0/1，真正的学习趋势在移动平均和 `P(best action)`。baseline 不改变期望梯度，却通常让曲线更平稳。",
        "tuning": ["先不加 baseline 验证梯度方向，再加入 running baseline 比较方差。", "学习率过小时概率不动，过大时过早饱和；按 3 倍左右尺度试验。", "最后再调 entropy coefficient，观察探索与收敛速度。"],
        "failures": ["对采样动作使用 `argmax`，失去策略梯度所需探索。", "忘记 `detach` return/advantage 的目标部分，引入错误梯度路径。", "entropy 符号写反，训练主动变得更确定或更随机。"],
    },
}


TRAINING_GUIDE.update({
    12: {
        "metric": "Actor-Critic 的平均回报和 critic 估值",
        "code": lab(r"""
rng = np.random.default_rng(12)
theta, value = 0.0, 0.0
rewards, values = [], []
for episode in range(450):
    p = 1 / (1 + np.exp(-theta))
    action = float(rng.random() < p)
    reward = float(rng.random() < (0.3 if action == 0 else 0.8))
    advantage = reward - value
    value += 0.08 * advantage
    theta += 0.06 * advantage * (action - p)
    rewards.append(reward); values.append(value)

plot_curve(rewards, "One-step actor-critic", "reward", window=45)
print("critic V:", round(value, 3), "P(best action):", round(1 / (1 + np.exp(-theta)), 3))
"""),
        "read": "回报移动平均应接近最优臂的成功率 0.8，critic 也应靠近当前策略的平均回报。critic loss 下降但 reward 不升，通常说明 actor 数据流或 advantage 符号有问题。",
        "tuning": ["先分别记录 policy loss、value loss、entropy、episode return。", "让 critic 学习率略高于 actor，先确认 value 能跟上策略分布。", "再调 n-step/GAE 和 entropy；一次只改一项并至少跑多种子。"],
        "failures": ["advantage 没有从 critic 图中分离，actor 更新反向改变 value target。", "A3C worker 梯度没有正确复制到共享模型。", "只看 critic loss 很小，却没有检查策略回报。"],
    },
    13: {
        "metric": "简化文本 observation 分类器的交叉熵",
        "code": lab(r"""
rng = np.random.default_rng(13)
# 词袋特征：[door, key, monster, treasure]；目标是选择 explore(0) / take(1)。
x = np.array([[1,0,0,0], [1,1,0,0], [0,1,0,1], [0,0,1,0], [0,0,0,1], [1,0,1,0]], float)
y = np.array([0,1,1,0,1,0])
w = rng.normal(0, .1, (4, 2)); losses = []
for _ in range(180):
    logits = x @ w; logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-9).mean(); losses.append(loss)
    grad = x.T @ (probs - np.eye(2)[y]) / len(y)
    w -= .35 * grad

plot_curve(losses, "Text feature learning", "cross-entropy")
print("predictions:", (x @ w).argmax(axis=1), "targets:", y)
"""),
        "read": "loss 下降且训练预测正确，只说明表示能拟合这个小数据集；TextWorld 中还必须看任务完成率和未见游戏泛化，不能把语言模型 loss 当成 RL 成功。",
        "tuning": ["先固定游戏集合和 tokenizer，建立随机/规则 baseline。", "比较原始文本、结构化实体特征和 transformer 表示，保持 RL 预算相同。", "在未见 seed/游戏上选模型，防止记住房间描述。"],
        "failures": ["词表在测试数据上拟合，泄漏未见文本信息。", "padding mask 错误，模型主要学习序列长度。", "训练 loss 很低但 agent 没有探索到完成任务的轨迹。"],
    },
    14: {
        "metric": "点击演示行为克隆的 loss 与命中率",
        "code": lab(r"""
rng = np.random.default_rng(14)
# 页面坐标 -> 4x4 点击格；RBF 模拟视觉特征，再做行为克隆。
xy = rng.random((320, 2))
labels = (np.minimum(3, (xy[:, 1] * 4).astype(int)) * 4 +
          np.minimum(3, (xy[:, 0] * 4).astype(int)))
grid = (np.arange(4) + .5) / 4
centers = np.array([(cx, cy) for cy in grid for cx in grid])
features = np.exp(-45 * ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
x = np.c_[features, np.ones(len(features))]
w = rng.normal(0, .03, (17, 16)); losses, accuracies = [], []
for _ in range(300):
    logits = x @ w; logits -= logits.max(1, keepdims=True)
    probs = np.exp(logits); probs /= probs.sum(1, keepdims=True)
    losses.append(-np.log(probs[np.arange(len(labels)), labels] + 1e-9).mean())
    accuracies.append((logits.argmax(1) == labels).mean())
    w -= 1.2 * x.T @ (probs - np.eye(16)[labels]) / len(x)

plot_curve(accuracies, "Click behavior cloning", "training accuracy", window=10)
print("final loss/accuracy:", round(losses[-1], 3), round(accuracies[-1], 3))
"""),
        "read": "分类准确率上升说明坐标到点击格的数据管线可学，但不能证明网页任务完成；真正评估还要记录有效点击率、页面成功率和分布外页面表现。",
        "tuning": ["先用演示做 behavior cloning，确认 observation/action 对齐。", "在固定页面集调网格大小和学习率，再换未见页面评估。", "RL fine-tuning 时同时画 imitation loss 与环境 success rate，防止灾难性遗忘。"],
        "failures": ["截图与动作时间戳错一帧，模型学习错误目标。", "坐标缩放忽略浏览器 viewport，点击整体偏移。", "只按训练页面准确率选模型，换布局后完全失效。"],
    },
    15: {
        "metric": "连续策略在二次奖励上的回报",
        "code": lab(r"""
target_action = 1.25

def train_actor(lr):
    theta, history = -1.2, []
    for _ in range(90):
        action = 2 * np.tanh(theta)
        reward = -(action - target_action) ** 2
        # d reward / d theta，展示 deterministic policy gradient 的最小形式。
        grad = -2 * (action - target_action) * 2 * (1 - np.tanh(theta) ** 2)
        theta += lr * grad
        history.append(reward)
    return history, 2 * np.tanh(theta)

stable, action_ok = train_actor(.035)
fast, action_fast = train_actor(1.5)
plt.figure(figsize=(7, 3.2)); plt.plot(stable, label="lr=.035"); plt.plot(fast, label="lr=1.5")
plt.xlabel("update"); plt.ylabel("reward"); plt.grid(alpha=.25); plt.legend(); plt.show(); plt.close()
print("final actions:", round(action_ok, 3), round(action_fast, 3), "target:", target_action)
"""),
        "read": "稳定学习率应平滑靠近最优回报 0；大学习率可能震荡或在 tanh 饱和区停滞。连续控制要同时观察 action 分布和 critic Q，不能只看 episode return。",
        "tuning": ["先检查动作缩放是否覆盖环境 bounds。", "从 critic loss/Q target 稳定性开始，再分别调 actor/critic learning rate。", r"最后调探索噪声、target soft-update $\tau$ 和 n-step，记录多种子。"],
        "failures": ["actor 输出 `tanh` 后没有缩放到真实 action range。", "target action 没有裁剪，环境收到非法动作。", "critic Q 值不断增大而回报不升，属于 extrapolation/bootstrapping 失稳。"],
    },
    16: {
        "metric": "PPO 更新中的回报代理指标与 KL",
        "code": lab(r"""
rng = np.random.default_rng(16)
theta, returns, kls = -0.8, [], []
for update in range(55):
    old_p = 1 / (1 + np.exp(-theta))
    actions = (rng.random(256) < old_p).astype(float)
    rewards = np.where(actions == 1, rng.normal(.8, .3, 256), rng.normal(.2, .3, 256))
    advantages = rewards - rewards.mean()
    old_prob = np.where(actions == 1, old_p, 1 - old_p)
    for _ in range(5):
        p = 1 / (1 + np.exp(-theta))
        prob = np.where(actions == 1, p, 1 - p)
        ratio = prob / old_prob
        active = ((advantages >= 0) & (ratio < 1.2)) | ((advantages < 0) & (ratio > .8))
        grad_logp = actions - p
        theta += .08 * np.mean(active * ratio * advantages * grad_logp)
    new_p = 1 / (1 + np.exp(-theta))
    kl = old_p*np.log((old_p+1e-9)/(new_p+1e-9)) + (1-old_p)*np.log((1-old_p+1e-9)/(1-new_p+1e-9))
    returns.append(.2 + .6 * new_p); kls.append(kl)

plt.figure(figsize=(7, 3.2)); plt.plot(returns, label="expected return"); plt.plot(kls, label="KL per update")
plt.xlabel("PPO update"); plt.grid(alpha=.25); plt.legend(); plt.show(); plt.close()
print("final expected return / max KL:", round(returns[-1], 3), round(max(kls), 4))
"""),
        "read": "期望回报应上升，而每次更新 KL 保持较小。若 surrogate objective 变好但 KL 突增、回报坍缩，说明 epoch、学习率或 clip 过于激进。",
        "tuning": ["固定 rollout length，先调 actor learning rate 使 KL 落在稳定范围。", "再调 PPO epochs/minibatch；epoch 增加会提高样本复用也会推高 KL。", r"最后调 clip、GAE $\lambda$、entropy，并同时报告 explained variance。"],
        "failures": ["每个 minibatch 重算 old log-prob，ratio 永远接近 1。", "advantage 未标准化且尺度剧烈变化。", "只监控 clipped loss，不监控 KL、entropy 与实际回报。"],
    },
    17: {
        "metric": "Evolution Strategies 搜索到的最佳目标值",
        "code": lab(r"""
rng = np.random.default_rng(17)
mean = np.array([4.0, -3.0, 2.0])
sigma, history = 1.2, []
for generation in range(90):
    noise = rng.normal(size=(80, len(mean)))
    candidates = mean + sigma * noise
    fitness = -np.sum(candidates ** 2, axis=1)
    elite_idx = np.argsort(fitness)[-12:]
    mean = candidates[elite_idx].mean(axis=0)
    sigma = max(.03, sigma * .97)
    history.append(fitness[elite_idx].max())

plot_curve(history, "Evolution strategy on sphere", "best fitness", window=5)
print("solution:", np.round(mean, 3), "distance to optimum:", round(np.linalg.norm(mean), 3))
"""),
        "read": "fitness 应趋近 0（本问题最大值），但 sigma 太快缩小会过早停滞。黑盒优化没有 policy/value loss，重点看 best/median fitness、种群多样性和 wall-clock。",
        "tuning": ["先用固定 sigma 建立种群大小与吞吐基线。", "再加入 sigma decay；若 median 很早停滞，减慢衰减。", "并行扩大 population 时按 wall-clock 和总评估次数双重比较。"],
        "failures": ["只保留单个最好个体，种群多样性迅速消失。", "不同 worker 使用相同噪声种子。", "与梯度法比较时忽略 ES 使用了更多环境评估。"],
    },
    18: {
        "metric": "随机探索与计数奖励探索的状态覆盖率",
        "code": lab(r"""
rng = np.random.default_rng(18)

def explore(use_bonus, steps=260):
    position, counts, coverage = 0, np.zeros(24), []
    for _ in range(steps):
        candidates = np.array([max(0, position - 1), min(23, position + 1)])
        if use_bonus:
            scores = 1 / np.sqrt(counts[candidates] + 1)
            action = rng.choice(np.flatnonzero(scores == scores.max()))
        else:
            action = rng.integers(2)
        position = candidates[action]; counts[position] += 1
        coverage.append(np.count_nonzero(counts))
    return coverage

random_curve, bonus_curve = explore(False), explore(True)
plt.figure(figsize=(7, 3.2)); plt.plot(random_curve, label="random"); plt.plot(bonus_curve, label="count bonus")
plt.xlabel("environment step"); plt.ylabel("unique states"); plt.grid(alpha=.25); plt.legend(); plt.show(); plt.close()
random_to_20 = np.flatnonzero(np.asarray(random_curve) >= 20)[0] + 1
bonus_to_20 = np.flatnonzero(np.asarray(bonus_curve) >= 20)[0] + 1
print("final coverage:", random_curve[-1], bonus_curve[-1], "/ 24",
      "steps-to-20:", random_to_20, bonus_to_20)
"""),
        "read": "count bonus 通常更快扩大覆盖，但覆盖率不是最终任务回报。正式实验应同时画 external return、intrinsic return、unique states，并在评估时关闭 bonus。",
        "tuning": ["先分别记录外在和内在奖励，绝不只看两者之和。", "调 bonus scale，使早期能影响行为、后期又不淹没任务奖励。", "RND 再调 observation/error normalization 与 predictor 更新频率。"],
        "failures": ["评估时仍加入 intrinsic reward，报告的任务成绩虚高。", "RND target 网络被 optimizer 更新，奖励定义不断漂移。", "随机电视等不可预测噪声持续产生高 bonus，agent 原地沉迷。"],
    },
    19: {
        "metric": "偏好奖励模型的 pairwise loss 与准确率",
        "code": lab(r"""
rng = np.random.default_rng(19)
features = rng.normal(size=(260, 4))
true_w = np.array([1.2, -.8, .5, .2])
pairs = rng.integers(0, len(features), size=(700, 2))
delta = features[pairs[:, 0]] - features[pairs[:, 1]]
labels = ((delta @ true_w + rng.normal(0, .35, len(delta))) > 0).astype(float)
w = np.zeros(4); losses, accuracies = [], []
for _ in range(160):
    logits = delta @ w; probs = 1 / (1 + np.exp(-logits))
    losses.append(-np.mean(labels*np.log(probs+1e-9) + (1-labels)*np.log(1-probs+1e-9)))
    accuracies.append(np.mean((probs > .5) == labels))
    w -= .25 * delta.T @ (probs - labels) / len(labels)

plt.figure(figsize=(7, 3.2)); plt.plot(losses, label="pairwise loss"); plt.plot(accuracies, label="accuracy")
plt.xlabel("reward-model update"); plt.grid(alpha=.25); plt.legend(); plt.show(); plt.close()
print("final accuracy:", round(accuracies[-1], 3), "cosine to true w:", round(w@true_w/(np.linalg.norm(w)*np.linalg.norm(true_w)), 3))
"""),
        "read": "pairwise loss 下降、held-out preference accuracy 上升才说明 reward model 在学。之后策略回报上升仍可能是 reward hacking，必须人工抽查高分轨迹。",
        "tuning": ["先按 trajectory/source 分割训练验证集，避免相似片段泄漏。", "调 reward-model 容量和正则，依据 held-out preference accuracy 选择。", "策略优化时逐步增加 KL penalty/奖励尺度，并周期补充新偏好数据。"],
        "failures": ["同一轨迹相邻片段跨训练/验证集，准确率虚高。", "reward normalization 在训练中漂移，PPO 目标尺度突然改变。", "只优化 learned reward，不检查真实任务指标和高分样本。"],
    },
    20: {
        "metric": "MCTS 根节点选择最佳动作的访问比例",
        "code": lab(r"""
rng = np.random.default_rng(20)
true_values = np.array([.35, .7, .5])
visits = np.zeros(3); value_sum = np.zeros(3); confidence = []
for simulation in range(1, 301):
    if (visits == 0).any():
        action = np.flatnonzero(visits == 0)[0]
    else:
        q = value_sum / visits
        action = np.argmax(q + 1.4 * np.sqrt(np.log(simulation) / visits))
    outcome = float(rng.random() < true_values[action])
    visits[action] += 1; value_sum[action] += outcome
    confidence.append(visits[true_values.argmax()] / simulation)

plot_curve(confidence, "MCTS root search", "visit fraction of best action", window=15)
print("visits:", visits.astype(int), "Q estimates:", np.round(value_sum / visits, 3))
"""),
        "read": "搜索初期要探索所有动作，之后最佳动作访问比例总体上升。AlphaZero 训练还应同时看 policy loss、value loss、self-play 胜率和树搜索改进幅度。",
        "tuning": ["先用规则游戏测试合法动作、胜负和对称变换。", "固定网络后调 simulations 与 exploration constant，记录搜索后策略相对先验的改进。", "再调 replay window、temperature 和训练/自博弈比率。"],
        "failures": ["backup 时玩家视角符号没有交替，价值方向全错。", "非法动作未 mask，policy 概率泄漏。", "只和当前自己对弈，模型退化时没有历史 checkpoint 检测。"],
    },
    21: {
        "metric": "离散路径问题的 episode steps（越低越好）",
        "code": lab(r"""
rng = np.random.default_rng(21)
q = np.zeros((16, 4)); steps_history = []
moves = [(-1,0), (1,0), (0,-1), (0,1)]
for episode in range(220):
    state = 0
    for step in range(80):
        action = rng.integers(4) if rng.random() < max(.05, 1-episode/160) else np.argmax(q[state])
        row, col = divmod(state, 4); dr, dc = moves[action]
        nr, nc = np.clip(row+dr, 0, 3), np.clip(col+dc, 0, 3)
        ns = int(nr*4+nc); done = ns == 15; reward = 1.0 if done else -.02
        target = reward if done else reward + .95 * q[ns].max()
        q[state, action] += .2 * (target - q[state, action])
        state = ns
        if done: break
    steps_history.append(step + 1)

plot_curve(steps_history, "Discrete path optimization", "steps to goal", window=20)
print("last-20 steps:", np.mean(steps_history[-20:]), "shortest possible: 6")
"""),
        "read": "步数移动平均应从截断上限降向最短路径 6。求解类任务还应报告 solved rate、search nodes 与最优性 gap，不能只报告训练 reward。",
        "tuning": ["先用可验证最短路的小实例检查状态编码和逆动作。", "先调探索和 replay，使 solved rate 上升，再增加搜索宽度。", "按 scramble depth 分层报告，不让简单样本掩盖困难样本。"],
        "failures": ["数据生成只含浅层 scramble，深层测试完全失败。", "状态哈希或对称化不一致，重复状态无法合并。", "MCTS/搜索找到解但路径包含互相抵消动作，未报告解长质量。"],
    },
    22: {
        "metric": "两个独立 Q-learning agent 的协作回报",
        "code": lab(r"""
rng = np.random.default_rng(22)
q1, q2, rewards = np.zeros(2), np.zeros(2), []
# 协调博弈：(1,1) 得 1；(0,0) 得 .6；不一致得 0。
payoff = np.array([[.6, 0.0], [0.0, 1.0]])
for episode in range(650):
    eps = max(.03, 1 - episode/450)
    a1 = rng.integers(2) if rng.random() < eps else np.argmax(q1)
    a2 = rng.integers(2) if rng.random() < eps else np.argmax(q2)
    reward = payoff[a1, a2]
    q1[a1] += .08 * (reward - q1[a1]); q2[a2] += .08 * (reward - q2[a2])
    rewards.append(reward)

plot_curve(rewards, "Independent learners in a coordination game", "team reward", window=50)
print("policies:", q1.argmax(), q2.argmax(), "last-50 reward:", np.mean(rewards[-50:]))
"""),
        "read": "移动平均上升表示形成协作，但独立学习可能收敛到次优的 (0,0)。多智能体曲线必须跨种子报告：环境对每个 agent 都因其他 agent 学习而非平稳。",
        "tuning": ["先训练随机/固定对手 baseline，再训练双方学习。", "分别记录 team reward、各 agent reward、胜率和策略熵。", "调 epsilon/对手池/target sync 时用多种子统计收敛到不同均衡的比例。"],
        "failures": ["把其他 agent 当作静态环境，replay 中旧数据严重过时。", "只报告最好种子，隐藏收敛到次优均衡的概率。", "共享奖励索引或 agent id 处理错误，经验归属错位。"],
    },
})


assert set(TRAINING_GUIDE) == set(range(1, 23))
