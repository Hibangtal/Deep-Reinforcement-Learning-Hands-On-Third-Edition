# 视频不是另一条主线：按当前课卡观看

[学习入口](../START_HERE.md) · [逐课目录](../06_lessons/README.md)

核验日期：**2026-10-17**。下面优先使用作者官网、作者仓库或作者发布页。核验课程目录/标题与实际播放是两件事：本轮没有逐段播放视频，也没有验证你的网络、登录状态、字幕开关或地区限制。B站部分直接抓取返回412；已明确标出，不能据此承诺打开即播。没有编造精确时间戳、全套时长或未确认的单集编号。

## 只保留几条主资源

| 用途 | 资源与定位 | 对应课号 | 观看边界 |
|---|---|---|---|
| Python从零 | [莫烦Python基础](https://mofanpy.com/tutorials/python-basic/basic/) | P00–P05 | 课卡列出print、容器、循环、函数、类的顺序；暂不学GUI/并发 |
| 数组 | [莫烦NumPy作者视频](https://www.bilibili.com/video/av16378934/)；[配套目录](https://mofanpy.com/tutorials/data-manipulation/np-pd/) | P06 | 先看2.1–2.5；不要被后半Pandas分散注意力 |
| 看懂曲线 | [莫烦Matplotlib](https://mofanpy.com/tutorials/data-manipulation/plt/) | P06、R09 | 基本使用、坐标轴、图例；不需要掌握所有图形 |
| 导数直觉 | [3Blue1Brown官方双语：用几何来求导](https://www.bilibili.com/video/BV1zx411m7Le/) | P07 | 只补导数直觉；英文语音/双语，索引已核对，直接抓取412 |
| 条件概率直觉 | [3Blue1Brown官方双语：贝叶斯定理](https://www.bilibili.com/video/BV1R7411a76r/) | P08 | 先做本地期望练习，再看条件概率图示；不要求先学完整贝叶斯统计 |
| ML与神经网直觉 | [莫烦有趣的机器学习](https://mofanpy.com/tutorials/machine-learning/ML-intro/) | P09–P10、专题前置 | 看指定概念，不刷完整合集 |
| ML讲解与投影片 | [李宏毅ML2021官网](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php) | P09、R03 | Introduction→General Guidance；分类loss到CEM前再看；选择Chinese视频 |
| PyTorch入门 | [莫烦PyTorch](https://mofanpy.com/tutorials/machine-learning/torch/) | P11 | 观念可用，旧API不照搬；使用本地预备篇代码 |
| RL公式主线 | [赵世钰课程](https://www.bilibili.com/video/BV1sd4y167NS/)；[作者书籍/课件主页](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning) | R01、R04–R06、R10–R11 | 基本概念→Bellman→TD→DQN→PG→AC；每段课名在课卡中 |
| 算法对照 | [王树森DRL作者目录](https://github.com/wangshusen/DRL) | R07–R08、R12、专题 | 中文视频链接与slides并列；按主题定位，YouTube可达性未实测 |
| PPO与探索/模仿 | [李宏毅MLDS2018官网](https://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html) | R13、O14、O18 | 页面底部Deep Reinforcement Learning，PPO part 1/2分别看；配套PDF可替代播放 |

莫烦Python的[print页面](https://mofanpy.com/tutorials/python-basic/basic/print)可看到视频源切换（含B站）与图文；官网目录是定位入口，不等于每个播放器都已验证。数学视频若无法播放，先用[公式慢讲](../00_start_here/FORMULA_LAB.md)及预备篇完成同一目标，不必停学等待网络。

## 为什么不把22章都硬配一个视频

本书原作者的代码顺序不等同于这些课程的顺序。CEM课用分类损失视频补基础；PTAN课用n-step视频补概念；SAC、MuZero、文本/浏览器、RLHF等课明确标注视频是前置或相邻主题，真正对应的推导和实验在本地notebook。本轮未核验到适合该路线的中文SAC逐节专讲，不会拿普通Actor-Critic影片冒充。

PPO可以额外阅读[Hugging Face官方Unit 8](https://huggingface.co/learn/deep-rl-course/unit8/introduction)的直觉和裁剪可视化；它是英文图文备选，不标成视频，也不要求再跟它运行另一套依赖。

## 视频之后必须做什么

每段留下三样东西：一句自己的解释、一个手算结果、一个代码输出。每课写明停止点；达到目标就返回仓库。看不懂符号就退到P07/P08；能听懂却写不出代码就退到P03/P04；运行结果与视频不同先核对版本、随机种子与终止语义。

旧视频可能用Gym四项返回、Variable、旧TensorFlow等；本仓库主线使用Gymnasium与PyTorch，环境专章另有依赖。学习思想时跟视频，运行API时跟对应课卡和实际环境，不在同一环境盲目降级整套包。
