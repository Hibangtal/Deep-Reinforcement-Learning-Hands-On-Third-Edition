# 强化学习排错手册

RL 排错最有效的顺序不是“先调学习率”，而是从最确定、最便宜的环节向随机、昂贵的环节推进。

## 1. 环境契约

- observation/action 的 shape、dtype、range 是否与 space 一致？
- reward 是在哪个 transition 产生的？符号和尺度正确吗？
- `terminated` 与 `truncated` 是否被区分？真正终止是否停止 bootstrap？
- 固定动作脚本能否得到预期轨迹？
- wrapper 是否同步更新了 observation/action space？

## 2. 数据管线

- buffer 中抽样一条 transition，人工重算 target。
- batch 维、动作维、atom 维是否被错误广播？
- replay warm-up 是否足够？样本是否来自多个 episode？
- n-step 的实际步数、$\gamma^n$ 和 episode 边界是否一致？
- 多进程 worker 是否使用不同 seed？

## 3. 数学实现

- terminal mask 前后的 target 是否符合手算？
- `gather` 是否沿动作维取当前动作？
- Double DQN 是否真的由 online 选择、target 评价？
- actor 更新时 critic/advantage 中不该传播的路径是否 detach？
- PPO 的 old log-prob 是否在多个 epoch 中保持固定？

## 4. 优化与数值

- 参数是否真的变化？梯度是否为 `None`、0、NaN 或爆炸？
- 学习率先做数量级扫描，不要同时改 batch/network。
- observation、return、advantage 是否需要归一化？
- 动作分布是否过早饱和？entropy 是否快速归零？
- Q 值尺度是否远大于可能的折扣回报上界？

## 5. 评估协议

- evaluation 是否关闭训练探索噪声和 intrinsic reward？
- 训练与评估环境是否相互独立？
- checkpoint 是按验证指标选择，还是偷看测试集？
- 是否报告多个 seed，而非最好的一次？
- 横轴是 gradient step、环境帧数还是 wall-clock？比较时是否一致？

## 常见曲线 → 优先假设

| 现象 | 优先检查 | 不要立刻做什么 |
|---|---|---|
| loss 不变 | optimizer 参数、梯度图、数据是否变化 | 盲目加深网络 |
| loss 下降、return 不升 | target/评估/action/reward 语义 | 只继续训练更久 |
| Q 值爆炸 | terminal mask、reward scale、target sync | 只做 gradient clipping |
| return 突然坍缩 | KL、entropy、学习率、旧策略数据 | 用最好 checkpoint 掩盖问题 |
| seed 间差异巨大 | 探索、初始化、稀疏奖励、样本量 | 只报告最好 seed |
| 训练好、评估差 | evaluation 模式、探索噪声、过拟合/泄漏 | 在测试集继续调参 |

## 最小故障报告

记录：代码版本、完整配置、seed、首次异常步数、异常前后曲线、一个最小复现、已经排除的假设、下一次只改变的变量。没有这些信息，“它不收敛”还不是一个可调试的问题。

