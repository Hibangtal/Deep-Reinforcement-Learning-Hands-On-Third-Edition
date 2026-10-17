# 第一周：先让Python听懂你的指令

[返回学习入口](../START_HERE.md) · [阶段一第01章：运行程序](../06_lessons/01_Python基础/01_P00_运行程序.md)

这是P00–P05按需引用的操作手册，不是另一条课程路线。跟当前学习章节只做相应课号，完成后返回该章的反馈与总结，不要求一周内全部做完。

“第一周”是开始方式，不是考试期限；下面每一小节可以占一天。做到哪里不懂就停在那里。此阶段不用NumPy、PyTorch、Jupyter。

## 第一次：找到运行代码的地方（P00）

打开编辑器的终端。终端一般显示目录路径和提示符；下面命令不包含提示符。在终端输入：

```bash
pwd
python --version
python -c 'import sys; print(sys.executable)'
```

`pwd`显示的位置应是本仓库。若不是，输入：

```bash
cd /home/hibant/workspace/jb_space/Deep-Reinforcement-Learning-Hands-On-Third-Edition
```

若系统只有`python3`，把本教程命令中的`python`统一换成`python3`。若提示`command not found`，把该命令与原始报错发给助手，不要先安装全部机器学习依赖。

在编辑器中，新建 `my_learning/scratch` 文件夹，再新建 `hello.py` 文件。`.py`表示Python代码文件。输入并保存：

```python
print(2 + 3)
print("今天我运行了第一个程序")
```

回终端输入：

```bash
python my_learning/scratch/hello.py
```

预期是两行：第一行`5`，第二行是那句中文。括号把要显示的内容交给`print`；带引号的内容是文字，不带引号的`2 + 3`会先计算。

现在把`3`改为`4`，保存、重新运行。第一行应变`6`。如果仍为5，检查是否保存、是否运行了同一个文件。完成后运行`python my_learning/coach.py quiz P00`，按学习入口的方法记录证据。

## 第二次：变量不是神秘符号（P01）

先看[莫烦print](https://mofanpy.com/tutorials/python-basic/basic/print)与课程目录的变量一节，再在自己的文件中逐行写：

```python
reward = 2
print(reward)
reward = reward + 3
print(reward)
```

运行前预测两行输出。结果是2、5。第二个赋值先读取原来的2，加3，再把5绑定给reward。

故意把最后一行改成`print(rewad)`。你会看到`NameError`：拼写错了，Python不认识这个名字。找报错末行的名字，再对照上面的定义修正。能解释这个错误，比背一串语法有用。

## 第三次：列表像按顺序摆放的小盒子（P02）

```python
rewards = [1, 0, 2]
print(rewards[0])
print(len(rewards))
rewards.append(3)
print(rewards)
```

先画四个小格子，编号0、1、2、3。前三条奖励对应前三格；append在末尾加入一格。输出依次是1、3、`[1, 0, 2, 3]`。

变式：把第一条奖励改成5；预测总和与平均值，再运行`sum(rewards)`和`sum(rewards)/len(rewards)`检查。不要直接复制参考输出当自己的证据。

## 第四次：循环是一项一项地做（P03）

```python
rewards = [1, 0, 2]
total = 0
for reward in rewards:
    total = total + reward
    print(total)
```

冒号后面缩进的两行会反复执行。每次reward拿到列表里的下一项。手算表是：

| 次数 | reward | 旧total | 新total |
|---|---|---|---|
| 1 | 1 | 0 | 1 |
| 2 | 0 | 1 | 1 |
| 3 | 2 | 1 | 3 |

变式：把`print(total)`移到循环外，与`for`左边对齐，输出变几行？先回答再运行。若看到`IndentationError`，检查缩进，统一用四个空格。

## 第五次：函数把一件事封装起来（P04）

```python
def total_reward(rewards):
    total = 0
    for reward in rewards:
        total = total + reward
    return total

print(total_reward([1, 0, 2]))
print(total_reward([]))
```

`def`定义功能；括号里的rewards是输入；`return`把结果交回。两行输出是3和0。空列表不会执行循环，所以返回初始的0。现在从空白处再写一遍，不看这段代码。

把上述求和与折扣回报区别开：总和让每一项权重相同；折扣回报让越晚的奖励权重越小。下一步照[公式慢讲](FORMULA_LAB.md)手算后，再写P04课卡的函数。

## 第六次与复盘（P05）

阅读对象/类的指定视频，再打开原仓库agent脚本，只认出以下链条：创建环境→创建agent→while反复step→输出总奖励。类一时不熟，可以分两次学，不必同一天跳到神经网络。

安装Jupyter后可做 [Python小步自动批改](02_Python_Baby_Steps.ipynb)。它把这些操作变成函数练习，未完成和错误会给中文提示；前五次不用等安装即可在.py文件练习。

本周复盘给助手的材料：你自己的hello文件、一个报错及修复、循环变量表、自己写的求和函数。助手应先给提示与变式题，而不是替你直接完成下一章。
