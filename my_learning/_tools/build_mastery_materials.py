"""Generate the cross-chapter mastery materials for the learning notebooks."""

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


def notebook(folder: str, name: str, cells: list[dict]) -> None:
    for index, cell in enumerate(cells):
        cell["id"] = f"{name.lower().replace('_', '-')}-{index:02d}"
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


def roadmap() -> list[dict]:
    return [
        md(r"""
        # 00 · 学习路线与入门诊断

        [返回总目录](../README.md) · [阶段考试](../04_assessments/Milestone_Exams.ipynb) · [综合项目](../03_guided_projects/Capstone_Projects.ipynb)

        这个 notebook 解决三个问题：**从哪里开始、什么时候可以进入下一阶段、怎样避免“看懂了但不会做”**。

        完全没有 Python、NumPy、机器学习或 PyTorch 基础时，请先完成[零基础预备篇](../01_prerequisites/Prerequisite00_Index.ipynb)，通过入学测试后再回到这里。

        > 学习规则：每章先手算、再运行、再改一个变量；阶段关卡不过，不靠继续堆章节解决。
        """),
        md(r"""
        ## 1. 十分钟摸底

        对每项按 `0/1/2` 评分：0=不会，1=看提示能做，2=能独立解释或实现。

        1. 能写出折扣回报并解释 $gamma$。
        2. 能从 Bellman 方程写出一次 value update。
        3. 能解释 terminal transition 为什么不能 bootstrap。
        4. 能写出 PyTorch 的 `zero_grad → backward → step`。
        5. 能解释 replay buffer 和 target network 各解决什么。
        6. 能从 $J(\theta)$ 推到 `log_prob * advantage`。
        7. 能解释 actor、critic 和 advantage 的职责。
        8. 能同时阅读 return、loss、entropy 和 KL 曲线。
        9. 能设计 train/validation/test，避免数据泄漏。
        10. 能用多个随机种子报告均值与波动。
        """),
        code(r"""
# 按上面的 10 项修改；默认 0 表示从基础路线开始。
scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
assert len(scores) == 10 and all(s in (0, 1, 2) for s in scores)

groups = {
    "数学与 Bellman": scores[:3],
    "PyTorch 与价值方法": scores[3:5],
    "策略方法": scores[5:8],
    "实验能力": scores[8:10],
}
for name, values in groups.items():
    print(f"{name}: {sum(values)}/{2*len(values)}")

total = sum(scores)
route = "完整路线（01 开始）" if total < 10 else "核心路线（01、05、06、11、12、16）" if total < 16 else "项目路线（先做阶段考试）"
print("总分:", total, "/20；建议:", route)
"""),
        md(r"""
        ## 2. 四阶段路线

        | 阶段 | 章节 | 要形成的能力 | 过关证据 |
        |---|---|---|---|
        | A 基础 | 01–05 | MDP、回报、Bellman、表格方法 | 手算小 MDP；从零写 value/Q iteration |
        | B 价值方法 | 06–10 | DQN、replay、target、Double/PER、实验效率 | 训练一个 DQN；解释每条曲线和一次失败 |
        | C 策略方法 | 11–18 | PG、A2C、PPO、连续控制、探索 | 从 loss 公式写更新；比较两个超参数 |
        | D 高级专题 | 19–22 | 偏好、搜索、规划、多智能体 | 画清数据闭环；完成一个专题小项目 |

        每阶段结束先做[阶段考试](../04_assessments/Milestone_Exams.ipynb)。正确率不是唯一标准：还必须能在不看答案时写出核心数据流。
        """),
        md(r"""
        ## 3. 建议节奏（每周 6–8 小时）

        - 第 1–2 周：01–05；重点是手算，不追求神经网络。
        - 第 3–4 周：06–10；完成 DQN 项目和一次故障注入。
        - 第 5–6 周：11–16；比较 REINFORCE、A2C、PPO 的数据使用方式。
        - 第 7 周：17–18；理解无梯度优化和探索。
        - 第 8 周：19–22；建立高级主题地图。
        - 第 9–10 周：完成一个[综合项目](../03_guided_projects/Capstone_Projects.ipynb)，按[实验模板](../03_guided_projects/Experiment_Template.ipynb)出报告。

        **每次学习闭环：**20 分钟阅读 → 20 分钟手算 → 30 分钟运行与改参 → 10 分钟闭卷复述 → 记录一个仍不理解的问题。
        """),
        md(r"""
        ## 4. 什么时候算“掌握”

        一个知识点至少经过四层：

        1. **识别：**看到公式知道各符号含义。
        2. **解释：**不用公式说清它解决的问题。
        3. **实现：**从空函数写出核心更新，并通过小测试。
        4. **诊断：**曲线异常时能提出有顺序的排查假设。

        只完成前两层叫“理解”，四层都完成才叫“掌握”。用[掌握清单](../05_reference/LEARNING_CHECKLIST.md)持续打卡。
        """),
    ]


def exams() -> list[dict]:
    return [
        md(r"""
        # 阶段考试 · 不靠熟悉感判断掌握程度

        [返回总目录](../README.md) · [学习路线](../00_start_here/00_Study_Roadmap.ipynb)

        每个关卡建议闭卷完成。先把答案写在纸上或新 cell 中，再展开参考要点。代码题应从空函数开始，不复制章节代码。
        """),
        md(r"""
        ## 关卡 A：MDP 与 Bellman（01–05）

        1. 推导 $G_t=R_{t+1}+\gamma G_{t+1}$。
        2. 写出 $V^\pi$ 与 $Q^\pi$ 的区别。
        3. 两状态 MDP 中手算一次 value iteration。
        4. 解释 `terminated` 与 `truncated` 对 bootstrap 的影响。
        5. 从零实现下面的 Bellman update。
        """),
        code(r"""
import numpy as np

def bellman_optimality(v, transitions, gamma):
    # transitions[s][a] 是 (probability, next_state, reward, done) 列表。
    new_v = np.zeros_like(v, dtype=float)
    for state, actions in enumerate(transitions):
        action_values = []
        for outcomes in actions:
            value = sum(p * (r + (0.0 if done else gamma * v[ns]))
                        for p, ns, r, done in outcomes)
            action_values.append(value)
        new_v[state] = max(action_values)
    return new_v

toy = [
    [[(1.0, 0, 0.0, False)], [(1.0, 1, 1.0, True)]],
    [[(1.0, 1, 0.0, True)], [(1.0, 1, 0.0, True)]],
]
result = bellman_optimality(np.zeros(2), toy, 0.9)
assert np.allclose(result, [1.0, 0.0])
print("关卡 A 代码测试通过:", result)
"""),
        md(r"""
        <details><summary>关卡 A 参考要点</summary>

        回报递推来自把无穷和的第一项拆出；$V$ 在策略下对动作求期望，$Q$ 固定了首个动作；最优 Bellman backup 对动作取 max；真正终止不 bootstrap，时间截断通常仍可 bootstrap，但要结合环境语义。

        </details>

        **过关线：**概念题至少 4/5；代码不看答案可写出；能构造一个 terminal mask 错误案例。
        """),
        md(r"""
        ## 关卡 B：DQN 与工程数据流（06–10）

        1. 画出 environment → buffer → batch → online/target → loss 的箭头图。
        2. 解释为什么相关的连续 transition 不适合直接做 SGD。
        3. 写出 Double DQN 中“选择”和“评价”分别由谁完成。
        4. 如果 loss 下降但 return 不升，按顺序列出五个检查项。
        5. 设计一个只改变 target sync frequency 的公平实验。

        <details><summary>参考要点</summary>

        优先检查评估策略、reward/termination、action 与 observation shape、target 构造、数据分布，再检查优化器。公平实验需固定帧预算、网络、buffer、探索、种子和评估协议，并报告多个种子。

        </details>
        """),
        md(r"""
        ## 关卡 C：策略梯度与连续控制（11–18）

        1. 从 likelihood-ratio trick 解释 `log_prob * return`。
        2. baseline 为什么通常降方差但不改变期望梯度？
        3. 比较 REINFORCE、A2C、PPO 的数据复用与稳定手段。
        4. PPO 中 ratio、clip、KL 各自告诉你什么？
        5. SAC 为什么同时使用 entropy、两个 critic 和 target network？

        <details><summary>参考要点</summary>

        REINFORCE 用完整 return、方差大；A2C 用 critic 构造 advantage；PPO 固定 old policy 并限制多 epoch 更新幅度。clip 是优化机制，KL 是实际移动距离的诊断，两者不能互相替代。SAC 用最大熵促进探索，双 critic 抑制高估，target 稳定 bootstrap。

        </details>
        """),
        md(r"""
        ## 关卡 D：高级主题（19–22）

        1. 画出偏好数据 → reward model → policy optimization → 新数据的闭环。
        2. MCTS 的 selection、expansion、evaluation、backup 各修改什么？
        3. AlphaZero 与 MuZero 对环境模型的假设有什么不同？
        4. 多智能体训练为什么非平稳？
        5. 分别给出 reward hacking、搜索价值符号错误、次优均衡的检测办法。

        **过关线：**能从数据来源一直讲到评估指标；能指出至少一种“优化指标变好、真实目标变差”的方式。
        """),
        md(r"""
        ## 总评分

        - 0–40%：返回对应章节，重新做手算和最小实验。
        - 40–70%：概念已形成，集中补代码题和失败诊断。
        - 70–85%：可以进入综合项目，但必须使用实验模板。
        - 85% 以上：尝试修改环境或算法，而不只是复现已有结果。

        分数不是目的。最可靠的标准是：面对一条异常曲线，你能否提出可证伪的假设，并设计只改变一个因素的实验。
        """),
    ]


def experiment_template() -> list[dict]:
    return [
        md(r"""
        # 可复现实验模板

        [返回总目录](../README.md) · [排错手册](../05_reference/DEBUGGING_PLAYBOOK.md)

        这个模板演示 RL 实验的最小可信单位：**配置、多个随机种子、原始曲线、聚合统计、失败记录**。示例使用二臂 bandit，方法可以原样迁移到 DQN/PPO。
        """),
        code(r"""
from dataclasses import asdict, dataclass
import random
import numpy as np
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Config:
    episodes: int = 400
    learning_rate: float = 0.08
    baseline_rate: float = 0.05
    seeds: tuple = (0, 1, 2, 3, 4)

config = Config()
print(asdict(config))
"""),
        md(r"""
        ## 1. 单次运行必须是纯函数式边界

        输入配置和 seed，返回逐 episode 指标。不要让上一轮模型、全局 RNG 或 replay buffer 泄漏到下一轮。
        """),
        code(r"""
def run_one(seed, cfg):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    theta, baseline = 0.0, 0.0
    rewards = []
    for _ in range(cfg.episodes):
        p = 1 / (1 + np.exp(-theta))
        action = float(rng.random() < p)
        reward = float(rng.random() < (0.25 if action == 0 else 0.75))
        advantage = reward - baseline
        baseline += cfg.baseline_rate * advantage
        theta += cfg.learning_rate * advantage * (action - p)
        rewards.append(reward)
    return np.asarray(rewards)

runs = np.stack([run_one(seed, config) for seed in config.seeds])
assert runs.shape == (len(config.seeds), config.episodes)
print("raw result shape:", runs.shape)
"""),
        md(r"""
        ## 2. 先聚合每个 seed，再跨 seed 统计

        不要把所有 episode 当成独立样本计算置信区间；实验重复单位是 seed。下面展示均值和 95% 近似置信带。
        """),
        code(r"""
window = 30
smoothed = np.stack([
    np.convolve(run, np.ones(window) / window, mode="valid") for run in runs
])
mean = smoothed.mean(axis=0)
sem = smoothed.std(axis=0, ddof=1) / np.sqrt(len(config.seeds))
x = np.arange(window, config.episodes + 1)

plt.figure(figsize=(7, 3.4))
plt.plot(x, mean, label="mean return")
plt.fill_between(x, mean - 1.96*sem, mean + 1.96*sem, alpha=.2, label="95% CI")
plt.axhline(.75, color="black", linestyle="--", linewidth=1, label="optimal expectation")
plt.xlabel("episode"); plt.ylabel("moving-average reward")
plt.grid(alpha=.25); plt.legend(); plt.show(); plt.close()
print("final per-seed means:", np.round(runs[:, -50:].mean(axis=1), 3))
"""),
        md(r"""
        ## 3. 调参记录表

        | run id | 唯一改动 | 事前假设 | 主要指标 | 结果 | 下一步 |
        |---|---|---|---|---|---|
        | baseline | 无 | 建立可复现基线 | return / entropy | 待填写 | — |
        | lr-low | learning rate ↓ | 更稳但更慢 | return / KL | 待填写 | 接受或否定假设 |
        | lr-high | learning rate ↑ | 更快但方差大 | return / KL | 待填写 | 检查崩溃种子 |

        正式项目还应保存：代码版本、依赖版本、设备、环境版本、总环境步数、wall-clock、最佳与最后 checkpoint。
        """),
        md(r"""
        ## 4. 最小报告结构

        1. **问题与假设：**为什么做这次实验？
        2. **不可变条件：**哪些配置、预算和评估规则保持一致？
        3. **结果：**所有 seed 的聚合曲线和最终指标表。
        4. **失败：**至少记录一个无效尝试，以及它排除了什么解释。
        5. **结论边界：**结论适用于哪些环境、预算和种子？

        如果报告中只有最好曲线、没有 seed、没有评估协议，它只能算演示，不能算可信实验。
        """),
    ]


def capstones() -> list[dict]:
    return [
        md(r"""
        # 综合项目 · 从“看懂”走到“能独立完成”

        [返回总目录](../README.md) · [实验模板](Experiment_Template.ipynb) · [阶段考试](../04_assessments/Milestone_Exams.ipynb)

        三个项目按难度递增。不要三个一起做：先完成项目 A，再选择 B 或 C。每个项目都要求基线、曲线、失败复盘和独立评估。
        """),
        md(r"""
        ## 项目 A：表格 GridWorld（基础必做）

        **目标：**不使用 RL 库，实现随机策略评估、value iteration、Q-learning，并比较 model-based 与 model-free。

        **里程碑：**

        1. 自己定义 observation、action、reward、termination。
        2. 用手算的 3×3 地图验证 Bellman update。
        3. 实现 value iteration，残差低于 $10^{-6}$。
        4. 实现 Q-learning，画 return、success rate、episode length。
        5. 故意删除 terminal mask，记录错误曲线并解释。

        **验收：**三个随机种子成功率均超过 90%；greedy 路径不长于最短路两步；报告至少一个失败案例。
        """),
        md(r"""
        ## 项目 B：DQN 控制任务（常用算法必做）

        **目标：**使用 Chapter06/08 的数据流训练 DQN，然后加入 Double DQN 做公平比较。

        **最小实现清单：**network、epsilon selector、replay、warm-up、batch gather、terminal mask、target sync、独立 evaluation。

        **必须画的曲线：**训练 return、无探索评估 return、TD loss、Q 均值、epsilon、buffer size、steps/s。

        **消融实验：**只改变一种因素：target sync、n-step 或 Double。至少 3 个 seed，固定总环境帧数。

        **故障注入：**任选一个——去掉 target network、终止状态继续 bootstrap、epsilon 过快降到 0。先预测曲线，再运行验证。

        **验收：**能从公式重新写出 `gather` 和 target；能解释 loss 与 return 不同步；报告不只展示最好 seed。
        """),
        md(r"""
        ## 项目 C：PPO 或 SAC 连续控制（进阶二选一）

        **PPO 路线：**实现 rollout、return/GAE、old log-prob、ratio、clip、value loss、entropy，多 epoch minibatch 更新。

        **SAC 路线：**实现 stochastic actor、reparameterization、双 critic、target critic、temperature/entropy 项和 replay。

        **必须监控：**evaluation return、actor/critic loss、entropy、动作均值/标准差；PPO 额外看 KL/clip fraction，SAC 额外看 Q target/temperature。

        **验收：**至少 5 个 seed；能识别一次策略坍缩；比较最佳 checkpoint 与最后 checkpoint；说明样本预算和 wall-clock。
        """),
        md(r"""
        ## 项目交付物

        每个项目提交五样东西：

        1. `README`：环境、算法、运行命令和预期耗时。
        2. `config`：所有超参数，不藏在代码常量中。
        3. `train/eval` 分离的代码。
        4. 多 seed 曲线和最终指标表。
        5. 一页复盘：成功原因、失败原因、结论边界、下一次实验。

        可直接套用[可复现实验模板](Experiment_Template.ipynb)。遇到异常按[排错手册](../05_reference/DEBUGGING_PLAYBOOK.md)从数据契约开始检查。
        """),
        code(r"""
# 自评量表：每项 0=没有，1=部分做到，2=可复现且能解释。
rubric = {
    "公式与实现逐项对应": 0,
    "随机/规则基线": 0,
    "独立评估协议": 0,
    "多个随机种子": 0,
    "曲线与资源指标": 0,
    "消融实验": 0,
    "失败案例复盘": 0,
    "他人可复现": 0,
}
assert all(score in (0, 1, 2) for score in rubric.values())
total = sum(rubric.values())
print("项目得分:", total, "/", 2*len(rubric))
print("建议:", "可进入新环境" if total >= 13 else "先补齐低于 2 分的项目")
"""),
    ]


def main() -> None:
    notebook("00_start_here", "00_Study_Roadmap", roadmap())
    notebook("04_assessments", "Milestone_Exams", exams())
    notebook("03_guided_projects", "Experiment_Template", experiment_template())
    notebook("03_guided_projects", "Capstone_Projects", capstones())
    print("Generated 4 cross-chapter mastery notebooks")


if __name__ == "__main__":
    main()
