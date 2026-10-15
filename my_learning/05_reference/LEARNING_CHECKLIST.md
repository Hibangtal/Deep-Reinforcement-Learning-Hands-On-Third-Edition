# 强化学习掌握清单

这不是阅读进度表，而是能力证据表。只有能够独立完成“证据”栏，才勾选掌握。

| 能力 | 掌握证据 | 状态 |
|---|---|---|
| 回报与折扣 | 手算有限轨迹的 $G_t$，解释不同 $\gamma$ 的影响 | ☐ |
| MDP 与策略 | 给新问题定义 state/action/reward/termination | ☐ |
| Bellman 方程 | 从回报递推推导 expectation/optimality equation | ☐ |
| 表格规划 | 从零实现 value iteration 并用残差停止 | ☐ |
| Q-learning | 正确处理 terminal mask，解释 on/off-policy | ☐ |
| PyTorch 训练循环 | 不看模板写出 forward/loss/backward/step | ☐ |
| DQN | 画出 replay、online/target、batch target 数据流 | ☐ |
| DQN 扩展 | 解释 Double、Dueling、PER、n-step 各解决什么 | ☐ |
| 策略梯度 | 从 likelihood ratio 解释 `log_prob * advantage` | ☐ |
| Actor-Critic | 区分 actor、critic、return、advantage 的梯度边界 | ☐ |
| PPO | 解释 ratio、clip、KL、entropy，并读懂异常曲线 | ☐ |
| 连续控制 | 正确处理 action bounds、探索和双 critic | ☐ |
| 探索 | 区分 external/intrinsic reward，并识别随机电视问题 | ☐ |
| RLHF | 解释偏好 loss、reward hacking 和 KL 约束 | ☐ |
| 搜索与规划 | 讲清 MCTS 四阶段与 backup 的玩家视角 | ☐ |
| 多智能体 | 解释非平稳、对手池与次优均衡 | ☐ |
| 实验设计 | 固定预算，运行多 seed，报告均值和不确定性 | ☐ |
| 故障诊断 | 对异常曲线提出可证伪假设并逐项排查 | ☐ |
| 独立项目 | 完成一个别人能复现的综合项目 | ☐ |

## 毕业标准

- [ ] [阶段考试](../04_assessments/Milestone_Exams.ipynb)的前三个关卡达到 70%。
- [ ] 完成[项目 A](../03_guided_projects/Capstone_Projects.ipynb)以及项目 B/C 中至少一个。
- [ ] 项目包含多个随机种子、独立评估和失败案例。
- [ ] 不看章节代码，能写出 Bellman target、DQN loss 或 PPO surrogate 中至少两项。
- [ ] 能向别人解释一次“loss 变好但策略变差”的真实案例。

