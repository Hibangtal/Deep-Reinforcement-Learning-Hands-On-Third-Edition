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
