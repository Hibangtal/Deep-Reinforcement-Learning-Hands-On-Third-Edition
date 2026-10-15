# 我的强化学习笔记

这是一套从零基础到能够独立实现、训练和诊断强化学习算法的学习系统，以 *Deep Reinforcement Learning Hands-On, Third Edition* 的 22 章为主线，并链接当前仓库的真实源码。

完整的逐章、逐主题核对结果见 [原书覆盖审计](05_reference/COVERAGE.md)。

## 使用方法

1. 第一次使用先打开[从这里开始](00_start_here/README.md)，不要直接跳进高级章节。
2. 在项目根目录按 `requirements.txt` 建立 Python 3.11 环境，启动 `jupyter lab my_learning`。
3. 完全零基础先完成预备篇；已有基础可直接做入学测试。
4. 每章按“手算 → 代码 → 改参 → 自测”学习，并在 Chapter05/06 后完成两个引导项目。
5. Atari、MuJoCo、TextWorld、浏览器环境、MAgent 等重型章节按各章 `requirements.txt` 配置。

> 这些是原创学习笔记和小型实现，不复制原书正文。公式符号尽量保持教材常用约定；若与具体脚本变量名不同，以语义为准。

## 开始前与学完后

| 材料 | 用途 |
|---|---|
| [从这里开始](00_start_here/README.md) | 环境、学习顺序和入门诊断 |
| [零基础预备篇](01_prerequisites/Prerequisite00_Index.ipynb) | Python、NumPy、绘图、数学、ML、PyTorch、Gymnasium 与入学测试 |
| [00 · 学习路线与入门诊断](00_start_here/00_Study_Roadmap.ipynb) | 摸底、选择路线、阶段过关标准 |
| [阶段考试](04_assessments/Milestone_Exams.ipynb) | 检查能否闭卷推导、实现和排错 |
| [可复现实验模板](03_guided_projects/Experiment_Template.ipynb) | 多随机种子、置信带、调参记录与报告规范 |
| [综合项目](03_guided_projects/Capstone_Projects.ipynb) | GridWorld、DQN、PPO/SAC 三档项目验收 |
| [完整 GridWorld 项目](03_guided_projects/Guided_Project01_GridWorld.ipynb) | 第一次端到端完成 Q-learning |
| [完整 DQN CartPole 项目](03_guided_projects/Guided_Project02_DQN_CartPole.ipynb) | 从 replay 到独立评估的深度 RL 项目 |
| [掌握清单](05_reference/LEARNING_CHECKLIST.md) | 用能力证据而不是阅读进度打卡 |
| [排错手册](05_reference/DEBUGGING_PLAYBOOK.md) | 从环境契约到评估协议的排查顺序 |
| [中英术语表](05_reference/GLOSSARY.md) | 快速查找常用概念及一句话解释 |

## 章节索引

| 章 | Notebook | 原书英文主题 |
|---:|---|---|
| 01 | [什么是强化学习](02_chapters/Chapter01.ipynb) | `What Is Reinforcement Learning?` |
| 02 | [环境接口](02_chapters/Chapter02.ipynb) | `OpenAI Gym API and Gymnasium` |
| 03 | [PyTorch 深度学习基础](02_chapters/Chapter03.ipynb) | `Deep Learning with PyTorch` |
| 04 | [交叉熵方法](02_chapters/Chapter04.ipynb) | `The Cross-Entropy Method` |
| 05 | [表格学习与 Bellman 方程](02_chapters/Chapter05.ipynb) | `Tabular Learning and the Bellman Equation` |
| 06 | [深度 Q 网络](02_chapters/Chapter06.ipynb) | `Deep Q-Networks` |
| 07 | [高层 RL 库](02_chapters/Chapter07.ipynb) | `Higher-Level RL Libraries` |
| 08 | [DQN 扩展](02_chapters/Chapter08.ipynb) | `DQN Extensions` |
| 09 | [RL 加速方法](02_chapters/Chapter09.ipynb) | `Ways to Speed Up RL` |
| 10 | [用 RL 做股票交易](02_chapters/Chapter10.ipynb) | `Stocks Trading Using RL` |
| 11 | [策略梯度](02_chapters/Chapter11.ipynb) | `Policy Gradients - an Alternative` |
| 12 | [Actor-Critic](02_chapters/Chapter12.ipynb) | `Actor-Critic Methods - A2C and A3C` |
| 13 | [文本世界](02_chapters/Chapter13.ipynb) | `The TextWorld Environment` |
| 14 | [Web 导航](02_chapters/Chapter14.ipynb) | `Web Navigation` |
| 15 | [连续动作空间](02_chapters/Chapter15.ipynb) | `Continuous Action Space` |
| 16 | [信赖域方法](02_chapters/Chapter16.ipynb) | `Trust Regions - PPO, TRPO, ACKTR, and SAC` |
| 17 | [黑盒优化](02_chapters/Chapter17.ipynb) | `Black-Box Optimizations in RL` |
| 18 | [高级探索](02_chapters/Chapter18.ipynb) | `Advanced Exploration` |
| 19 | [RLHF](02_chapters/Chapter19.ipynb) | `Reinforcement Learning with Human Feedback` |
| 20 | [AlphaGo Zero 与 MuZero](02_chapters/Chapter20.ipynb) | `AlphaGo Zero and MuZero` |
| 21 | [离散优化中的 RL](02_chapters/Chapter21.ipynb) | `RL in Discrete Optimization` |
| 22 | [多智能体强化学习](02_chapters/Chapter22.ipynb) | `Multi-Agent RL` |

## 学习路线建议

- **完全零基础：**环境与 Jupyter → Prerequisite00–08 → Chapter01–05 → GridWorld 项目 → Chapter06 → DQN 项目。
- **基础主线：**01 → 06，理解 MDP、Gym、PyTorch、Bellman、Q-learning 与 DQN。
- **价值方法：**07 → 10，学习工程抽象、DQN 扩展、性能与交易环境建模。
- **策略方法：**11 → 18，依次掌握 policy gradient、actor-critic、连续控制、PPO/SAC 与探索。
- **高级应用：**19 → 22，学习 RLHF、MuZero、离散优化与多智能体 RL。

## 维护说明

## 目录结构

```text
00_start_here       环境、使用方式与学习路线
01_prerequisites    零基础预备篇与入学测试
02_chapters         原书 22 章伴读
03_guided_projects  完整引导项目、实验模板、开放项目
04_assessments      阶段考试
05_reference        覆盖审计、排错、术语和掌握清单
_tools              生成与验证脚本
```

## 维护说明

`_tools` 中四个 build 脚本分别生成章节、预备篇、综合材料和引导项目。修改内容后运行生成器，再执行 `python my_learning/_tools/validate_notebooks.py --execute`。
