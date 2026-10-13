# 强化学习中英术语表

| 英文 | 中文 | 一句话理解 |
|---|---|---|
| agent / environment | 智能体 / 环境 | 做决定的一方与产生下一状态、奖励的一方 |
| observation / state | 观测 / 状态 | 看到的信息；完整状态还应足以预测未来 |
| action / reward | 动作 / 奖励 | 决策与环境给出的即时标量反馈 |
| return | 回报 | 从当前时刻开始的折扣累计奖励 |
| policy | 策略 | 从状态到动作分布或动作的映射 |
| value / Q-value | 状态价值 / 动作价值 | 在策略下未来回报的条件期望 |
| Bellman equation | Bellman 方程 | 当前价值等于即时奖励加下一步价值 |
| bootstrap | 自举 | 用当前估计的一部分构造新的学习目标 |
| temporal-difference error | TD 误差 | target 与当前价值估计之差 |
| on-policy / off-policy | 同策略 / 异策略 | 生成数据的策略是否就是被更新的策略 |
| replay buffer | 经验回放池 | 保存并重新抽样 transition，打散相关性 |
| target network | 目标网络 | 延迟变化的网络，用于稳定 bootstrap target |
| advantage | 优势 | 某动作相对状态基线好多少 |
| entropy | 熵 | 策略随机性的度量，常用于维持探索 |
| policy gradient | 策略梯度 | 直接沿期望回报对策略参数求梯度 |
| actor / critic | 演员 / 评论家 | 产生动作的策略与估计价值的模型 |
| GAE | 广义优势估计 | 在偏差与方差之间折中地累计 TD 误差 |
| importance sampling | 重要性采样 | 用概率比校正不同数据分布的期望 |
| KL divergence | KL 散度 | 衡量新旧策略分布移动幅度的不对称距离 |
| intrinsic reward | 内在奖励 | 为新颖性、预测误差等探索信号提供的奖励 |
| self-play | 自我博弈 | 使用自己或历史版本作为对手生成训练数据 |
| reward model | 奖励模型 | 从偏好或示例学习出的奖励近似 |
| reward hacking | 奖励投机 | 优化代理奖励却偏离真实意图 |
| centralized training | 集中训练 | 训练时使用多个 agent 的联合信息 |
| decentralized execution | 分散执行 | 执行时每个 agent 只依赖自身可用信息 |

看到同一概念的不同译法时，以公式和数据语义为准，不要被名字绑住。

