# 看完视频，怎样一步一步读公式

[学习入口](../START_HERE.md) · [数学预备篇](../01_prerequisites/Prerequisite04_Math.ipynb)

固定顺序：**一句人话 → 标符号 → 代具体数字 → 一步变形 → 对照代码 → 换一个例子**。某一步解释不了就停在那一步，不要求靠背诵跨过去。

## 1. 求和与折扣：给每项奖励一个权重（P03/P04）

奖励序列为`[1,2,3]`，折扣gamma=0.5。

$$G_0=\sum_{k=0}^{2}\gamma^k r_k
=1\times1+0.5\times2+0.25\times3=2.75.$$

`Σ`表示相加；`k`是位置0、1、2；`gamma^k`是权重；`r_k`是这个位置的奖励。编程中的幂是`**`，不是`^`。

```python
rewards = [1, 2, 3]
gamma = 0.5
weight = 1.0
total = 0.0
for reward in rewards:
    total += weight * reward
    weight *= gamma
print(total)
```

先预测再运行，应得到2.75。把gamma改成0/1，结果应为1/6。如果不符，逐轮打印weight和total。

## 2. 期望：不是“取中间”，而是按概率加权（P08）

有75%机会得到0，25%机会得到4：

$$\mathbb E[R]=\sum_r p(r)r=0.75\times0+0.25\times4=1.$$

实际每次得到0或4，没有一次必须得到1。把概率改成各50%，先算期望2，再采样验证；样本平均会波动。Bellman里的转移概率加权就是同样思想。

## 3. 梯度下降：先算一维，再扩展网络（P07/P09）

设预测参数w想靠近3，损失`L(w)=(w−3)²`。令中间量u=w−3，则`dL/du=2u`，`du/dw=1`，链式法则给`dL/dw=2(w−3)`。

在w=1时，梯度是−4。学习率eta=0.1：

$$w_{new}=w-\eta\frac{dL}{dw}=1-0.1(-4)=1.4.$$

旧损失4，新损失2.56，方向合理。把学习率改成2看看损失是否反而变大；梯度方向正确不表示任意步长都好。对应PyTorch：`backward()`算梯度，`step()`按优化规则更新。

## 4. Bellman从哪里来（R04）

以下采用常用教材下标：从时刻t行动后得到的奖励记作`R_(t+1)`，与上面Python列表从0编号只差记法。

$$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\cdots$$
$$=R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+\cdots)
=R_{t+1}+\gamma G_{t+1}.$$

第一步只把首项拿出来，第二步给剩下项提出gamma。还没有概率或神经网络。

定义`V^π(s)=E[G_t|S_t=s]`，再按策略动作概率、环境转移概率展开条件期望：

$$V^\pi(s)=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V^\pi(s')].$$

最优Q则是：

$$Q^*(s,a)=\mathbb E[r+\gamma\max_{a'}Q^*(s',a')\mid s,a].$$

`π(a|s)`是在状态s选动作a的概率；`p(s',r|s,a)`是执行动作后到下一状态并获得奖励的概率；撇号表示“下一步”，不是求导。终止后未来价值设0。它来自固定策略取期望、最优策略选最大值的区别。

手算：确定转移、r=1、gamma=.9、next_value=2，得到2.8。再设一半机会到value=2、一半到value=0，同样即时奖励1，期望变为1.9。对照 `Chapter05/01_frozenlake_v_iteration.py` 的`count/total`、`reward + GAMMA * value`和`max`。

## 5. Q-learning到DQN：先有目标，再决定怎么拟合（R05/R06）

一次采样构造目标`y=r+gamma×next_max`。当前估计Q=2、目标y=4.6，差额delta=2.6；alpha=.5时，`Q_new=Q+alpha×delta=3.3`。它是朝目标挪一步，不是每次直接覆盖为目标。

DQN用网络输出Q，用平方误差`(Q−y)²`训练；目标网络在这次更新中不求梯度。若真正终止，则y=r。普通时间限制截断与任务终止要分开，不能见到停止就无条件把未来价值清零。

检查自己：若Q=1、r=2且terminal=True，不论next_max是多少，y都应为2。给自己的实现写一个这种反例。

## 6. 策略梯度的log为什么出现（R10）

先对有限离散轨迹讲，忽略折扣记号的细节。轨迹τ的回报R(τ)由环境给出：

$$J(\theta)=\sum_\tau p_\theta(\tau)R(\tau),\qquad
\nabla J=\sum_\tau \nabla p_\theta(\tau)R(\tau).$$

利用`∇log p=(∇p)/p`，有`∇p=p∇log p`，于是：

$$\nabla J=\mathbb E_{\tau\sim p_\theta}[R(\tau)\nabla\log p_\theta(\tau)].$$

在环境动力学不依赖策略参数的假设下，轨迹log概率中只有策略项需要对θ求导；乘积变成log的和。再用采样平均近似期望，就得到实现里的`log_prob×return`；优化器做下降，所以loss前放负号。

先用一条轨迹解释每个因子，再读第11章完整推导中的reward-to-go和baseline；不要跳过期望和采样这层含义。

## 7. PPO的min要真的算两遍（R13）

新策略给某动作概率0.7，旧策略0.5，比率rho=1.4。epsilon=.2，把rho裁到1.2。

| 优势A | rho×A | clip(rho)×A | min取什么 |
|---|---|---|---|
| 2 | 2.8 | 2.4 | 2.4 |
| −2 | −2.8 | −2.4 | −2.8 |

所以不能把PPO解释成“任何时候都直接使用裁剪比率”。代码中对min再取负均值作为loss。clip限制目标中的激励方式，不是保证策略KL永远小于某值。

变式：rho=.6，A分别取2与−2，epsilon仍为.2，两个min分别应为1.2、−1.6。解释清楚后才读多epoch复用和GAE。

每次交给助手的推导请保留中间式。只有最终数值时，无法判断你是理解了还是碰巧算对。
