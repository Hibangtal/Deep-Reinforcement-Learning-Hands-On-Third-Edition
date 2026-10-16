# 长期学习计划：机器学习零基础怎样学完这本书

[返回总目录](../README.md) · [预备篇导航](../01_prerequisites/Prerequisite00_Index.ipynb) · [学习路线与入门诊断](../00_start_here/00_Study_Roadmap.ipynb)

这份计划回答四个问题：**要多久、先学什么、哪些章节可以不学、遇到瓶颈去哪里补**。

## 1. 先说清楚：这本书对零基础意味着什么

原书的目标读者是“熟悉 Python、微积分和机器学习概念”的人。它的强项是**完整可运行的工程代码**，弱项是**机器学习基础几乎不讲**：

- 第 3 章只用一章篇幅介绍 PyTorch，第 6 章就用卷积网络训练 Atari；
- 第 11 章以后的策略梯度、PPO、SAC 需要你熟练理解期望、方差、log 概率和梯度；
- 第 13、14、19 章用到 LSTM、预训练语言模型和 LLM 接口；
- 很多实验在 CPU 上要跑几个小时甚至几天。

所以本仓库的策略是：**补齐地基 → 主线精学 → 专题按需**。不要追求“22 章全部读完”，而要追求“主线算法能从零写出、能诊断失败”。

## 2. 总时间预估

按**每周 8–10 小时**估算（会编程、但没有机器学习基础）：

| 阶段 | 内容 | 周数 | 累计 |
|---|---|---:|---:|
| 0 地基 | [预备篇](../01_prerequisites/Prerequisite00_Index.ipynb) 00–08，含 04B/05B/06B/07B | 5–7 | 约 1.5 个月 |
| A 表格方法 | Chapter01–05 + [GridWorld 项目](../03_guided_projects/Guided_Project01_GridWorld.ipynb) | 3–4 | 约 2.5 个月 |
| B 价值方法 | Chapter06–09 + [DQN 项目](../03_guided_projects/Guided_Project02_DQN_CartPole.ipynb) | 6–7 | 约 4 个月 |
| C 策略方法 | Chapter11、12、15、16 | 4–5 | 约 5 个月 |
| 综合项目 | [Capstone](../03_guided_projects/Capstone_Projects.ipynb) 任选一档 | 3–4 | **约 6 个月** |
| D 专题 | 10、13、14、17–22 按兴趣 | 每章 1–3 | 按需 |

如果每周只有 4–5 小时，所有周数乘以 2，核心路线约一年。**进度慢不是问题，跳过过关证据才是问题。**

## 3. 主线、选读与专题

| 章 | 主题 | 定位 | 额外前置知识 | 外部依赖/算力提醒 |
|---:|---|---|---|---|
| 01 | 什么是 RL | 主线 | [7B 老虎机](../01_prerequisites/Prerequisite07B_Bandit_Bridge.ipynb) | — |
| 02 | Gym/Gymnasium API | 主线 | [07 Gymnasium](../01_prerequisites/Prerequisite07_Gymnasium.ipynb) | — |
| 03 | PyTorch 深度学习 | 主线 | [5B](../01_prerequisites/Prerequisite05B_Neural_Network_From_Scratch.ipynb)、[06](../01_prerequisites/Prerequisite06_PyTorch.ipynb)、[6B](../01_prerequisites/Prerequisite06B_CNN_and_Sequences.ipynb) | GAN 示例偏重，可只读不跑 |
| 04 | 交叉熵方法 | 主线 | [4B 概率](../01_prerequisites/Prerequisite04B_Probability.ipynb) | CPU 即可 |
| 05 | Bellman 与表格学习 | 主线（最重要） | 4B 期望、条件概率 | CPU 即可 |
| 06 | DQN | 主线（最重要） | 6B CNN、帧堆叠 | Pong 训练在 CPU 上很慢；先用 CartPole 项目练流程 |
| 07 | 高层 RL 库 | 主线 | — | — |
| 08 | DQN 扩展（Rainbow） | 主线 | 4B 重要性采样（PER） | Atari 实验建议 GPU |
| 09 | RL 加速方法 | 主线 | 多进程基础 | GPU 收益明显 |
| 10 | 股票交易 | 选读 | ML 的 train/val/test 与数据泄漏 | 结果不代表真实投资收益 |
| 11 | 策略梯度 | 主线（最重要） | 4B log 概率、方差 | CartPole 部分 CPU 即可 |
| 12 | A2C/A3C | 主线 | 11 | 多环境并行 |
| 13 | TextWorld | 专题 | 6B 第 5 节、[注意力机制](https://zh.d2l.ai/chapter_attention-mechanisms/index.html) | TextWorld、transformers；`chatgpt_*.py` 需要 OpenAI API |
| 14 | Web 导航 | 专题 | 6B CNN + LSTM | MiniWoB 浏览器环境，配置复杂 |
| 15 | 连续动作空间 | 主线 | 4B 正态分布 | PyBullet、旧版 gym 0.25 |
| 16 | PPO/TRPO/SAC | 主线（最重要） | 4B 重要性采样、15 | MuJoCo |
| 17 | 黑盒优化（ES/GA） | 选读 | — | 多进程 |
| 18 | 高级探索 | 选读 | 7B UCB | Atari 部分建议 GPU |
| 19 | RLHF | 专题 | 12、奖励模型思想 | 标注 UI（nicegui） |
| 20 | AlphaGo Zero/MuZero | 专题 | 树搜索（MCTS） | 自对弈训练耗时长 |
| 21 | 离散优化（魔方） | 专题 | — | 训练耗时长 |
| 22 | 多智能体 RL | 专题 | 06、12 | MAgent2 |

“主线（最重要）”的 05、06、11、16 四章，建议在阶段考试通过后**第二遍精读**。

## 4. 每天、每周怎么学

**单次学习（约 90 分钟）：**20 分钟阅读 → 20 分钟手算 → 30 分钟运行与改参 → 10 分钟闭卷复述 → 10 分钟写学习日志。

**每周复盘（约 20 分钟），在学习日志里回答：**

1. 本周完成了哪些过关证据（练习通过、手算、项目、考试）？
2. 哪一个概念仍然说不清？下周第一件事就是解决它。
3. 有没有一次实验结果和预期不符？原因是什么？

**防止半途而废：**

- 卡住 30 分钟以上：先查[排错手册](DEBUGGING_PLAYBOOK.md)，再把问题缩到最小例子；
- 连续两周没进展：退回上一个已经过关的阶段做一个小项目，而不是继续往后读；
- 训练太慢：缩小环境或步数先跑通流程，正式实验再上 GPU（可选择云 GPU）。

## 5. 推荐的外部资源（按用途）

| 用途 | 资源 | 什么时候用 |
|---|---|---|
| 深度学习基础（中文） | [动手学深度学习](https://zh.d2l.ai/) | 预备篇 5、5B、6、6B 卡住时；Chapter13 前读注意力机制 |
| 神经网络直觉 | [3Blue1Brown 神经网络系列](https://www.3blue1brown.com/topics/neural-networks) | 开始 5B 之前 |
| RL 理论（经典教材） | [Sutton & Barto《Reinforcement Learning: An Introduction》](http://incompleteideas.net/book/the-book-2nd.html) | 第 2–6 章配合 7B 与 Chapter01–06；第 13 章配合 Chapter11 |
| RL 中文入门 | [Datawhale《蘑菇书 EasyRL》](https://datawhalechina.github.io/easy-rl/) | 想看中文推导时 |
| 策略方法速查 | [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/) | Chapter11–16，对照 VPG/PPO/SAC 伪代码 |
| 动手课程 | [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) | 想换一个角度、或做额外项目时 |
| 官方文档 | [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)、[Gymnasium Basic Usage](https://gymnasium.farama.org/introduction/basic_usage/) | 遇到 API 问题时 |
| 原书代码 | [Packt 官方仓库](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition) | 对照本仓库源码精读路线 |

原则：**本仓库是主线，外部资源只用来解决具体卡点**。同时跟三门课，往往哪门都学不完。

## 6. 学完以后能做什么

完成核心路线（约 6 个月）后，合理的期望是：

- 能独立实现表格 Q-learning、DQN、REINFORCE、A2C、PPO，并在 CartPole/LunarLander 等环境中训练成功；
- 能读懂原书大部分代码和主流 RL 论文的算法部分；
- 能用多个随机种子做公平对比，能诊断常见的训练失败。

它**不能**直接让你胜任机器人真机部署、大模型 RLHF 训练等工作——那需要在此基础上补仿真（如 MuJoCo/Isaac）、sim-to-real 或 LLM 工程等专门知识。
