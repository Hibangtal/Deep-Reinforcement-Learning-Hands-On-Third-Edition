# 我的强化学习笔记

这组 notebook 对应 *Deep Reinforcement Learning Hands-On, Third Edition* 的 22 章，并链接到当前仓库的配套源码。每章从一个具体问题出发，依次经过原书覆盖地图、前置知识、符号表、分步推导、手算例子、公式与代码对照、两个原理实验、完整算法、真实源码精读，以及一段会真实产生训练曲线的诊断实验。每章还给出逐步调参方案和失败案例复盘，目标是让初学者不只“运行成功”，还知道公式从哪里来、曲线为何变化、失败后先查什么。

完整的逐章、逐主题核对结果见 [原书覆盖审计](COVERAGE.md)。

## 使用方法

1. 在项目根目录按 `requirements.txt` 建立 Python 3.11 环境。
2. 启动 Jupyter：`jupyter lab my_learning`。
3. 按章节顺序学习；代码运行前先预测结果，运行后完成“两分钟自测”，再进入下一章。
4. Atari、MuJoCo、TextWorld、浏览器环境、MAgent 等章节需要额外依赖；先读 notebook 主线，再按各章 `requirements.txt` 配置。

> 这些是原创学习笔记和小型实现，不复制原书正文。公式符号尽量保持教材常用约定；若与具体脚本变量名不同，以语义为准。

## 开始前与学完后

| 材料 | 用途 |
|---|---|
| [00 · 学习路线与入门诊断](00_Study_Roadmap.ipynb) | 摸底、选择路线、阶段过关标准 |
| [阶段考试](Milestone_Exams.ipynb) | 检查能否闭卷推导、实现和排错 |
| [可复现实验模板](Experiment_Template.ipynb) | 多随机种子、置信带、调参记录与报告规范 |
| [综合项目](Capstone_Projects.ipynb) | GridWorld、DQN、PPO/SAC 三档项目验收 |
| [掌握清单](LEARNING_CHECKLIST.md) | 用能力证据而不是阅读进度打卡 |
| [排错手册](DEBUGGING_PLAYBOOK.md) | 从环境契约到评估协议的排查顺序 |
| [中英术语表](GLOSSARY.md) | 快速查找常用概念及一句话解释 |

## 章节索引

| 章 | Notebook | 原书英文主题 |
|---:|---|---|
| 01 | [什么是强化学习](Chapter01.ipynb) | `What Is Reinforcement Learning?` |
| 02 | [环境接口](Chapter02.ipynb) | `OpenAI Gym API and Gymnasium` |
| 03 | [PyTorch 深度学习基础](Chapter03.ipynb) | `Deep Learning with PyTorch` |
| 04 | [交叉熵方法](Chapter04.ipynb) | `The Cross-Entropy Method` |
| 05 | [表格学习与 Bellman 方程](Chapter05.ipynb) | `Tabular Learning and the Bellman Equation` |
| 06 | [深度 Q 网络](Chapter06.ipynb) | `Deep Q-Networks` |
| 07 | [高层 RL 库](Chapter07.ipynb) | `Higher-Level RL Libraries` |
| 08 | [DQN 扩展](Chapter08.ipynb) | `DQN Extensions` |
| 09 | [RL 加速方法](Chapter09.ipynb) | `Ways to Speed Up RL` |
| 10 | [用 RL 做股票交易](Chapter10.ipynb) | `Stocks Trading Using RL` |
| 11 | [策略梯度](Chapter11.ipynb) | `Policy Gradients - an Alternative` |
| 12 | [Actor-Critic](Chapter12.ipynb) | `Actor-Critic Methods - A2C and A3C` |
| 13 | [文本世界](Chapter13.ipynb) | `The TextWorld Environment` |
| 14 | [Web 导航](Chapter14.ipynb) | `Web Navigation` |
| 15 | [连续动作空间](Chapter15.ipynb) | `Continuous Action Space` |
| 16 | [信赖域方法](Chapter16.ipynb) | `Trust Regions - PPO, TRPO, ACKTR, and SAC` |
| 17 | [黑盒优化](Chapter17.ipynb) | `Black-Box Optimizations in RL` |
| 18 | [高级探索](Chapter18.ipynb) | `Advanced Exploration` |
| 19 | [RLHF](Chapter19.ipynb) | `Reinforcement Learning with Human Feedback` |
| 20 | [AlphaGo Zero 与 MuZero](Chapter20.ipynb) | `AlphaGo Zero and MuZero` |
| 21 | [离散优化中的 RL](Chapter21.ipynb) | `RL in Discrete Optimization` |
| 22 | [多智能体强化学习](Chapter22.ipynb) | `Multi-Agent RL` |

## 学习路线建议

- **基础主线：**01 → 06，理解 MDP、Gym、PyTorch、Bellman、Q-learning 与 DQN。
- **价值方法：**07 → 10，学习工程抽象、DQN 扩展、性能与交易环境建模。
- **策略方法：**11 → 18，依次掌握 policy gradient、actor-critic、连续控制、PPO/SAC 与探索。
- **高级应用：**19 → 22，学习 RLHF、MuZero、离散优化与多智能体 RL。

## 维护说明

`build_notebooks.py` 生成章节，`build_mastery_materials.py` 生成跨章节材料。修改内容后运行两个生成器，再用 `python my_learning/validate_notebooks.py --execute` 检查结构、链接和代码。若要像当前交付版本一样缓存曲线输出，再运行 `jupyter nbconvert --execute --to notebook --inplace my_learning/*.ipynb`。
