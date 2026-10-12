# 原书覆盖审计

本文件按原书目录的顶层主题逐项核对 22 个 notebook。它用于回答“哪些内容讲了、讲到什么深度、去哪里看”，避免仅凭文件数量宣称完成。

- **详细：**正文含定义、推导、手算或完整数据流。
- **完整/完整小例子：**含端到端算法流程和可执行验证。
- **源码路线：**解释真实仓库文件、类和函数的阅读顺序。
- **训练诊断：**每章都有真实运行的微型曲线、逐步调参顺序和失败案例；重型环境的正式成绩仍需运行原项目脚本。
- **结构理解/概览/地图：**作为非主线内容放在源码精读或扩展中。

## [01 · What Is Reinforcement Learning? / 什么是强化学习](Chapter01.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Supervised / unsupervised / reinforcement learning | 问题引入与心智模型 | 对比理解 |
| Complications in RL | 常见坑与扩展 | 延迟奖励、非平稳、探索 |
| RL formalisms: reward, agent, environment, action, observation | 符号表与交互递归 | 详细 |
| Markov process / MRP / MDP | 分步推导、矩阵求值实验 | 详细 |
| Policy | 价值定义与后续章节连接 | 主线 |

源码精读路线包含 **0** 个重点文件。

## [02 · OpenAI Gym API and Gymnasium / 环境接口](Chapter02.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| The anatomy of the agent | 交互契约与算法流程 | 详细 |
| Hardware and software requirements | 索引的环境说明 | 实用 |
| Gymnasium spaces / environment / creation / CartPole | 符号表、代码实验 | 详细 |
| The random CartPole agent | 第一段可运行实验 | 完整小例子 |
| Wrappers / rendering / extra API | 源码路线、常见坑、扩展 | 详细 |

源码精读路线包含 **4** 个重点文件。

## [03 · Deep Learning with PyTorch / PyTorch 深度学习基础](Chapter03.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Tensors / tensor creation / operations / GPU | 符号表、训练实验、源码精读 | 详细 |
| Gradients and autograd | 链式法则推导与第二实验 | 详细 |
| NN building blocks / custom layers | `OurModule` 源码路线 | 详细 |
| Loss functions and optimizers | 训练循环映射 | 详细 |
| TensorBoard | 源码路线与工程说明 | 实用 |
| GAN on Atari images | 折叠源码精读 | 结构理解 |
| PyTorch Ignite | 事件驱动训练说明 | 结构理解 |

源码精读路线包含 **4** 个重点文件。

## [04 · The Cross-Entropy Method / 交叉熵方法](Chapter04.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Taxonomy of RL methods | 心智模型与章节连接 | 概览 |
| CEM in practice | 最大似然推导 | 详细 |
| CEM on CartPole | 算法流程、源码路线 | 详细 |
| CEM on FrozenLake | 失败原因、改进与源码路线 | 详细 |
| Theoretical background | elite MLE / 交叉熵解释 | 主线 |

源码精读路线包含 **4** 个重点文件。

## [05 · Tabular Learning and the Bellman Equation / 表格学习与 Bellman 方程](Chapter05.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Value, state, and optimality | 价值定义与手算 | 详细 |
| Bellman equation of optimality | 分步推导 | 详细 |
| Value of the action | V/Q 对照 | 详细 |
| Value iteration method | 算法伪代码与收敛解释 | 详细 |
| Value iteration in practice | 第一个与第二实验 | 完整 |
| Q-iteration for FrozenLake | 源码路线 | 完整 |

源码精读路线包含 **2** 个重点文件。

## [06 · Deep Q-Networks / 深度 Q 网络](Chapter06.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Real-life value iteration | 从模型到采样的过渡 | 详细 |
| Tabular Q-learning | TD 更新与 chain 实验 | 详细 |
| Deep Q-learning | loss、replay、target network 推导 | 详细 |
| DQN on Pong: wrappers/model/training/performance/play | 完整流程与真实源码路线 | 详细 |
| Things to try | 练习与扩展 | 实践 |

源码精读路线包含 **4** 个重点文件。

## [07 · Higher-Level RL Libraries / 高层 RL 库](Chapter07.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Why RL libraries? | 组件化动机 | 详细 |
| PTAN action selectors / agents | 数据流与源码路线 | 详细 |
| Experience sources / first-last | n-step 推导与实验 | 详细 |
| Replay buffers / TargetNet / Ignite helpers | 组件职责表 | 详细 |
| PTAN CartPole solver | 端到端伪代码与源码 | 完整 |
| Other RL libraries | 扩展 | 概览 |

源码精读路线包含 **6** 个重点文件。

## [08 · DQN Extensions / DQN 扩展](Chapter08.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Basic and tuned DQN | baseline 与公平比较 | 详细 |
| N-step DQN | 第 7 章推导与源码 | 详细 |
| Double DQN | 选择/评价分离 | 详细 |
| Noisy networks | 参数噪声精读 | 详细 |
| Prioritized replay | 概率与重要性权重 | 详细 |
| Dueling DQN | 可辨识分解 | 详细 |
| Categorical DQN | 分布投影说明与实验 | 详细 |
| Combining everything / Rainbow | 组合表与源码路线 | 完整 |

源码精读路线包含 **7** 个重点文件。

## [09 · Ways to Speed Up RL / RL 加速方法](Chapter09.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Why speed matters / baseline | 时间预算与基准 | 详细 |
| PyTorch computation graph | 无用梯度与 detach | 详细 |
| Several environments | 批量环境 | 详细 |
| Playing and training in separate processes | 生产者/消费者架构 | 详细 |
| Tweaking wrappers | 预处理位置与复制成本 | 详细 |
| Benchmark results | 实验设计与解释 | 完整 |

源码精读路线包含 **6** 个重点文件。

## [10 · Stocks Trading Using RL / 用 RL 做股票交易](Chapter10.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Why trading / problem decisions | 状态、动作、奖励定义 | 详细 |
| Data | OHLC 读取、切分、特征 | 源码精读 |
| Trading environment | 记账与终止 | 详细 |
| Models | feed-forward / convolution | 结构对比 |
| Training code | 端到端流程 | 完整 |
| Results: feed-forward / convolution | 评估解读 | 详细 |
| Things to try | 练习与扩展 | 实践 |

源码精读路线包含 **6** 个重点文件。

## [11 · Policy Gradients - an Alternative / 策略梯度](Chapter11.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Values and policy | 价值法与策略法对照 | 详细 |
| REINFORCE | 完整 likelihood-ratio 推导 | 详细 |
| REINFORCE issues | 方差、baseline、credit assignment | 详细 |
| CartPole policy gradients | 渐进源码路线 | 完整 |
| Pong policy gradients | 像素与长轨迹工程 | 源码精读 |

源码精读路线包含 **6** 个重点文件。

## [12 · Actor-Critic Methods - A2C and A3C / Actor-Critic](Chapter12.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Variance reduction / CartPole variance | baseline 与实验直觉 | 详细 |
| Advantage Actor-Critic (A2C) | TD advantage、联合 loss | 详细 |
| A2C on Pong / tuning | 源码路线 | 完整 |
| Asynchronous A3C | 梯度并行与同步 | 详细 |

源码精读路线包含 **5** 个重点文件。

## [13 · The TextWorld Environment / 文本世界](Chapter13.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Interactive fiction / environment | POMDP 与 TextWorld | 详细 |
| Deep NLP basics | token、embedding、序列模型 | 详细 |
| Baseline DQN | 文本编码到动作价值 | 完整流程 |
| Tweaking observations | 预处理与状态信息 | 详细 |
| Transformers | 预训练编码器源码路线 | 结构理解 |
| ChatGPT | 交互/自动 agent 入口 | 应用与限制 |

源码精读路线包含 **7** 个重点文件。

## [14 · Web Navigation / Web 导航](Chapter14.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Evolution / browser automation / challenges | 问题建模与安全边界 | 详细 |
| MiniWoB / MiniWoB++ | 环境接口与源码路线 | 详细 |
| Simple clicking approach | 坐标离散与模型 | 完整 |
| Adding text description | 多模态输入 | 详细 |
| Human demonstrations | 行为克隆推导与数据 | 详细 |
| Things to try | 练习与扩展 | 实践 |

源码精读路线包含 **6** 个重点文件。

## [15 · Continuous Action Space / 连续动作空间](Chapter15.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Why continuous space? | 离散/连续动作差异 | 详细 |
| A2C method | Gaussian policy 与连续采样 | 详细 |
| Deep deterministic policy gradients | DDPG 推导与源码 | 详细 |
| Distributional policy gradients | D4PG 投影与源码 | 详细 |
| Things to try | 练习与扩展 | 实践 |

源码精读路线包含 **6** 个重点文件。

## [16 · Trust Regions - PPO, TRPO, ACKTR, and SAC / 信赖域方法](Chapter16.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Environments / A2C baseline | 公平基线与评估 | 详细 |
| PPO | ratio、clip、epoch 流程 | 详细 |
| TRPO | KL 约束、共轭梯度、线搜索 | 详细 |
| ACKTR | 自然梯度与 K-FAC | 结构理解 |
| SAC | 最大熵、双 Q、温度 | 详细 |
| Overall results | 同协议比较 | 详细 |

源码精读路线包含 **6** 个重点文件。

## [17 · Black-Box Optimizations in RL / 黑盒优化](Chapter17.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Black-box methods | 适用边界 | 详细 |
| Evolution strategies | 梯度估计推导与 CartPole/HalfCheetah | 详细 |
| Genetic algorithms | 选择、变异、deep GA、novelty | 详细 |
| GA on CartPole / HalfCheetah | 源码路线 | 完整 |

源码精读路线包含 **5** 个重点文件。

## [18 · Advanced Exploration / 高级探索](Chapter18.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Problems with epsilon-greedy / importance | 深度探索动机 | 详细 |
| Noisy / count / prediction-based exploration | 三类机制 | 详细 |
| MountainCar experiments | DQN/PPO 各变体比较 | 源码路线 |
| Atari experiments | 可扩展性与结果 | 源码路线 |

源码精读路线包含 **8** 个重点文件。

## [19 · Reinforcement Learning with Human Feedback / RLHF](Chapter19.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Reward functions in complex environments | 为什么需要偏好 | 详细 |
| Theoretical method / RLHF and LLMs | Bradley-Terry、KL | 详细 |
| Initial A2C training | 实验闭环起点 | 源码路线 |
| Labeling process | UI、pair 采样与偏差 | 详细 |
| Reward model training | loss、验证 | 详细 |
| A2C + reward model / multi-round fine-tuning | 完整闭环 | 详细 |
| Overall results | 独立评价 | 详细 |

源码精读路线包含 **6** 个重点文件。

## [20 · AlphaGo Zero and MuZero / AlphaGo Zero 与 MuZero](Chapter20.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Model-based vs model-free / board games | 概念对比 | 详细 |
| AlphaGo Zero: MCTS/self-play/training | PUCT 与闭环 | 详细 |
| Connect 4 AlphaZero implementation/results | 游戏、MCTS、模型、训练源码 | 完整 |
| MuZero high-level model/training | 三模型与 unroll loss | 详细 |
| Connect 4 MuZero | 节点、模型、搜索、数据 | 完整源码路线 |
| MuZero results / Atari | 结果与扩展 | 结构理解 |

源码精读路线包含 **7** 个重点文件。

## [21 · RL in Discrete Optimization / 离散优化中的 RL](Chapter21.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| Rubik's cube / God's number | 问题与最优性 | 详细 |
| Actions / states / solving approaches | 环境编码与合法变换 | 源码精读 |
| Training architecture/process | curriculum 与 value/policy targets | 详细 |
| Model application/results | 搜索与指标 | 详细 |
| Code outline: env/train/search | 真实源码路线 | 完整 |
| 2x2 / 3x3 experiments | 配置、结果与限制 | 结构理解 |
| Further improvements | 扩展 | 地图 |

源码精读路线包含 **7** 个重点文件。

## [22 · Multi-Agent RL / 多智能体强化学习](Chapter22.ipynb)

| 原书主题 | 本 notebook 对应位置 | 深度 |
|---|---|---|
| What is multi-agent RL? | Markov game 与非平稳性 | 详细 |
| MAgent environment/install/random setup | 环境接口与随机 baseline | 源码路线 |
| DQN for tigers | 共享网络与 group experience | 详细 |
| Tiger collaboration | 团队奖励与协作 | 详细 |
| Training tigers and deer | 双群体同时学习 | 完整流程 |
| Battle environment | 竞争/self-play | 源码路线 |

源码精读路线包含 **6** 个重点文件。
