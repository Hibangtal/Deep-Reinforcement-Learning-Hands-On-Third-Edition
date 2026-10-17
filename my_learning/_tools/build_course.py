"""Build the reading guides and quiz manifest from one maintained source."""

import json
import os
from pathlib import Path

from course_content import LESSONS, RESOURCES
from course_structure import RECAPS, STAGES
from exercise_kit import Exercise, exercise_block, md


ROOT = Path(__file__).resolve().parent.parent


def build_python_practice():
    exercises = [
        Exercise("P01：一次奖励累计", "输入旧总奖励和新奖励，返回两者之和。先预测再运行。",
                 "def add_reward(total, reward):\n    raise NotImplementedError",
                 "def add_reward(total, reward):\n    return total + reward",
                 "def test_add_reward():\n    assert add_reward(2, 3) == 5, '先算右侧再return，不要只print'\n    assert add_reward(4, -2) == 2, '奖励也可能为负'",
                 "相加用+；return会把结果交给调用者。"),
        Exercise("P02：末次奖励", "返回列表最后一个奖励；空列表返回0，不修改原列表。",
                 "def last_reward(rewards):\n    raise NotImplementedError",
                 "def last_reward(rewards):\n    if not rewards:\n        return 0\n    return rewards[-1]",
                 "def test_last_reward():\n    values = [2, 5, 7]\n    assert last_reward(values) == 7, '负索引-1表示最后一个'\n    assert values == [2, 5, 7], '不能用pop破坏数据'\n    assert last_reward([]) == 0, '先检查空列表'\n    assert last_reward([4]) == 4, '一项列表也适用'",
                 "先用if处理空列表，再读取最后一项。"),
        Exercise("P03：自己写累计循环", "用for实现奖励求和，空列表结果为0。",
                 "def sum_rewards(rewards):\n    raise NotImplementedError",
                 "def sum_rewards(rewards):\n    total = 0\n    for reward in rewards:\n        total += reward\n    return total",
                 "def test_sum_rewards():\n    assert sum_rewards([1, 0, 2]) == 3, 'total初始化放在循环外'\n    assert sum_rewards([]) == 0, '空列表应返回初始总和'\n    assert sum_rewards([-2, 3, 4]) == 5, '不能只返回最后一项'",
                 "先在纸上填写每轮reward和total；测试不会判断是否用了for，请自行遵守练习要求。"),
        Exercise("P04：折扣回报", "按从左到右顺序，第一项权重1，后一项权重乘gamma。空列表返回0。",
                 "def discounted_return(rewards, gamma):\n    raise NotImplementedError",
                 "def discounted_return(rewards, gamma):\n    total = 0.0\n    weight = 1.0\n    for reward in rewards:\n        total += weight * reward\n        weight *= gamma\n    return total",
                 "def test_discounted_return():\n    assert abs(discounted_return([1, 2, 3], .5) - 2.75) < 1e-9, '逐项算1+1+0.75'\n    assert discounted_return([2, 4], 0) == 2, 'gamma=0只留第一项'\n    assert discounted_return([2, 4], 1) == 6, 'gamma=1是普通总和'\n    assert discounted_return([], .9) == 0, '不要除以列表长度'",
                 "维护weight=1，每处理一项后才乘gamma。"),
        Exercise("P04：return与默认参数", "写average_reward，空列表返回默认值default（默认0），非空返回平均值。",
                 "def average_reward(rewards, default=0):\n    raise NotImplementedError",
                 "def average_reward(rewards, default=0):\n    if not rewards:\n        return default\n    return sum(rewards) / len(rewards)",
                 "def test_average_reward():\n    assert average_reward([1, 2]) == 1.5, '平均用/，不是整除//'\n    assert average_reward([]) == 0, '先处理空列表'\n    assert average_reward([], default=-1) == -1, '调用参数覆盖默认值'",
                 "空列表不能做除法；先判断后返回。"),
        Exercise("P05：对象各有自己的数据", "创建Counter，初始steps=0；每次step()累加1并返回新的steps。",
                 "class Counter:\n    def __init__(self):\n        raise NotImplementedError\n\n    def step(self):\n        raise NotImplementedError",
                 "class Counter:\n    def __init__(self):\n        self.steps = 0\n\n    def step(self):\n        self.steps += 1\n        return self.steps",
                 "def test_counter():\n    first = Counter()\n    second = Counter()\n    assert first.steps == 0, '初始化实例属性self.steps'\n    assert first.step() == 1, '方法需要return新计数'\n    assert first.step() == 2, '调用后保留状态'\n    assert second.steps == 0, '两个实例的状态应独立'",
                 "self代表当前对象；不是一个所有对象共用的全局变量。"),
    ]
    cells = [md("# Python小步自动批改\n\n[第一周教程](FIRST_WEEK.md) · [课程入口](../START_HERE.md)\n\n每次只做当前课号的题。把raise NotImplementedError替换为自己的代码；先运行检查工具格再运行题目格。不要改自动检查。参考答案折叠；看过答案后应换例子独立重写。此处通过只证明列出的行为样例，不是完整掌握。")]
    cells.extend(exercise_block(exercises))
    for index, cell in enumerate(cells):
        cell["id"] = f"python-small-{index}"
    document = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.11"}}}
    (ROOT / "00_start_here/02_Python_Baby_Steps.ipynb").write_text(
        json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def relative_link(source, target):
    return Path(os.path.relpath(ROOT / target, (ROOT / source).parent)).as_posix()


def course_layout():
    layout = {}
    for stage in STAGES:
        for number, lesson_id in enumerate(stage["ids"], 1):
            if lesson_id in layout:
                raise ValueError(f"Duplicate lesson in stages: {lesson_id}")
            layout[lesson_id] = dict(stage=stage, number=number,
                                    guide=f"06_lessons/{stage['folder']}/{number:02d}_{lesson_id}_{RECAPS[lesson_id][0]}.md")
    if set(layout) != {item["id"] for item in LESSONS} or set(layout) != set(RECAPS):
        raise ValueError("Stages, recaps and lessons must contain exactly the same IDs")
    return layout


def build_stage_guides(layout, manifest):
    by_id = {item["id"]: item for item in manifest}
    index = ["# 分阶段学习目录", "", "[日常学习入口](../START_HERE.md)", "",
             "先选当前阶段，再按阶段中的章节顺序学。每章的导学、学习步骤、反馈和总结都在同一个文件里，不用在多个md之间猜顺序。", "",
             "## P、R、O到底是什么", "",
             "- **P = Prerequisites，前置基础**：P00–P11，分在阶段一、二。",
             "- **R = Reinforcement Learning，强化学习主线**：R01–R14，分在阶段三、四、五。",
             "- **O = Optional，选修专题**：放在阶段六，按先修和兴趣选择，不阻塞主线。", "",
             "它们是反馈工具的稳定课号，不是难度等级。**学习章节不等于原书章节**：例如R03对应原书第4章；O10等选修编号沿用原书章号。旧的测验和进度编号不变。", "",
             "## 六个阶段", "", "| 阶段入口 | 课号 | 结束时能做什么 |", "|---|---|---|"]
    for stage_index, stage in enumerate(STAGES):
        path = f"06_lessons/{stage['folder']}/README.md"
        link = lambda target: relative_link(path, target)
        rows = []
        for lesson_id in stage["ids"]:
            item = by_id[lesson_id]
            number = layout[lesson_id]["number"]
            prerequisites = "、".join(f"[{required}]({link(layout[required]['guide'])})" for required in item["requires"]) or "无"
            chapters = "、".join(map(str, item["chapters"])) or "前置基础"
            rows.append(f"| {number:02d} | [{item['title']}]({link(item['guide'])}) | {lesson_id} | {prerequisites} | {chapters} |")
        first = layout[stage["ids"][0]]["guide"]
        if stage_index < 4:
            next_stage = STAGES[stage_index + 1]
            exit_text = f"完成下面复盘并获得反馈后，进入[{next_stage['title']}]({link('06_lessons/' + next_stage['folder'] + '/README.md')})。"
        elif stage_index == 4:
            exit_text = f"主线到这里结束。先交毕业项目复核，再按兴趣去[阶段六选修]({link('06_lessons/' + STAGES[5]['folder'] + '/README.md')})；无需全部学完。"
        else:
            exit_text = "当前专题完成后回本页选择，不自动进入下一专题；有依赖关系的O14、O21需分别先完成O13、O20及课内列出的其他先修。"
        start_text = (f"从[本阶段第01章]({link(first)})开始，每次只开一章及它明确引用的材料。"
                      if stage_index < 5 else "从下表按兴趣选择一章，先核对该章先修；表格顺序不是必修顺序。")
        text = f"""# {stage['title']}

[全部阶段](../README.md) · [日常学习入口]({link('START_HERE.md')})

## 阶段开始：先看这里

**进入条件：** {stage['entry']}

**阶段目标：** {stage['goal']}

{start_text} 一章可拆成多次45–60分钟，学不完就停在当前步骤，不按文件数量赶进度。

## 本阶段章节

下面“章序”只在本阶段内编号；“原书章”用来查教材，不是另一条学习顺序。

| 章序 | 学习章节 | 稳定课号 | 先修 | 原书章 |
|---|---|---|---|---|
{chr(10).join(rows)}

<a id="stage-review"></a>

## 阶段结束：复盘与验收

**把章节串起来：** {stage['review']}

**要交付的证据：** {stage['evidence']}

- [ ] 完成本阶段主线章节；选修区只检查你所选专题及其先修。
- [ ] 不翻答案完成上面的综合复盘，记录真实卡点。
- [ ] 执行 `python my_learning/coach.py report`，连同代码、输出和手算交给助手点评。
- [ ] 根据反馈修正一个错误，隔一天换例子再做一次。

**未通过怎么退：** {stage['repair']}

工具只检查每课概念题和自报记录，**不会自动判定本阶段综合验收通过**。这里的勾选也是你的自查，不替代导师对证据的复核。

## 下一步

{exit_text}
"""
        (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / path).write_text(text, encoding="utf-8")
        ids = "、".join(stage["ids"]) if stage_index == 5 else f"{stage['ids'][0]}–{stage['ids'][-1]}"
        index.append(f"| [{stage['title']}]({stage['folder']}/README.md) | {ids} | {stage['goal']} |")
    index.extend(["", "## 每章都按同一条路径", "",
                  "本章开始 → 带问题看视频 → 数字到公式 → 对照材料并动手 → 反馈与补救 → 本章结束总结 → 下一步。", "",
                  "阶段末先回阶段目录完成综合复盘。读到本章引用时才打开notebook、源码或参考文档；它们是配套材料，不是同时要完成的平行课程。"])
    (ROOT / "06_lessons/README.md").write_text("\n".join(index) + "\n", encoding="utf-8")


def build():
    destination = ROOT / "06_lessons"
    destination.mkdir(exist_ok=True)
    manifest = []
    previous_core = None
    layout = course_layout()
    core_ids = [item["id"] for item in LESSONS if item["track"] == "core"]
    for item in LESSONS:
        requires = item["requires"]
        if requires is None:
            requires = [previous_core] if previous_core else []
        if item["track"] == "core":
            previous_core = item["id"]
        position = layout[item["id"]]
        stage = position["stage"]
        guide = position["guide"]
        link = lambda target: relative_link(guide, target)
        resource_name, url, access = RESOURCES[item["video"]]
        readings = "\n".join(f"- [{name}]({link(path)})" for name, path in item["reading"])
        prerequisites = "、".join(f"[{required}]({link(layout[required]['guide'])})" for required in requires) or "无，从本章开始"
        chapters = "、".join(map(str, item["chapters"])) or "无，这是前置基础"
        recall = "\n\n".join(f"{number}. {question['question']}" for number, question in enumerate(item["quiz"], 1))
        answers = "\n\n".join(f"{number}. {question['explanation']}" for number, question in enumerate(item["quiz"], 1))
        navigation = ["[本阶段目录](README.md)"]
        if item["track"] == "core":
            core_index = core_ids.index(item["id"])
            if core_index:
                previous = core_ids[core_index - 1]
                navigation.append(f"[上一章：{RECAPS[previous][0]}]({link(layout[previous]['guide'])})")
            if position["number"] == len(stage["ids"]):
                navigation.append("[下一步：先做阶段总结](README.md#stage-review)")
            else:
                following = core_ids[core_index + 1]
                navigation.append(f"[下一章：{RECAPS[following][0]}]({link(layout[following]['guide'])})")
        else:
            navigation.append("[下一步：专题复盘与自主选课](README.md#stage-review)")
        quiz = [dict(question, remedy=f"重做 {guide} 的公式与实践；核对：{item['expected']}") for question in item["quiz"]]
        item_text = f"""# 第{position['number']:02d}章 · {item['title']}

[全部阶段](../README.md) → [{stage['title']}](README.md) → 本章（{item['id']}）

原书对应章：{chapters}。这里的学习章序只在本阶段内计数；{item['id']}是测验/进度用的稳定课号，不是原书章号。

先修：{prerequisites}。定位：{'主线必修' if item['track'] == 'core' else '专题选修，不阻塞主线'}。

## 1. 本章开始：导学与目标

**本章要弄懂的核心：** {RECAPS[item['id']][1]}

**学完要交的证据：** {item['expected']}

**开始自查：** {'先确认能打开终端、保存文件；不懂这些词就跟第一周教程操作，不需要先自学其他章节。' if not requires else '闭卷解释先修课的一个概念，并重做其中一个小例子；解释不清先返回上面的先修链接。'}

**本章怎么分次学：** 第一次看指定视频并完成手算；第二次读对应材料、做实践；最后自测、总结并交反馈。每次45–60分钟只是参考，算法项目可以分多天。每次停止时记下“停在哪一步、下次第一件事”；不要同时打开多个新视频。

## 2. 带问题看视频

[打开：{resource_name}]({url})

{item['watch']}

{access} 所有资源的核验边界见[视频指南]({link('05_reference/VIDEO_GUIDE.md')})。

每看约10–15分钟暂停，写一句“它解决什么问题”，预测下面算例的结果，再继续。这里的10–15分钟是学习节奏，不是视频时间戳。

## 3. 从数字走到公式

{item['math']}

先圈出输入，再标出要计算的量，给每个符号写中文含义。手算之后换一组数字重算。回到下面材料只读对应主题的“分步推导/手算/符号表”，再向前后扩展；暂时不要求一次读完整章。

## 4. 对照材料并动手

{readings}

{item['task']}

使用notebook时：先预测输出，运行一格，说明为什么，再修改一个数。首次运行前用“另存为”保留个人副本；不要用维护生成器覆盖自己的作业。Python基础阶段可以一直使用.py文件。

## 5. 反馈、补救与过关

对照结果：{item['expected']}

终端在仓库根目录运行：

```bash
python my_learning/coach.py quiz {item['id']}
```

每道题会即时解释正确/错误及补课位置。全部答对只是概念检查；接着完成实践并如实记录自己的文件路径、实际输出或推导。下面参数是填写格式，需换成你的真实内容：

```bash
python my_learning/coach.py record {item['id']} --evidence '我的文件路径和实际输出/推导' --reflection '我原来理解错了什么，现在如何解释'
python my_learning/coach.py report
```

未通过：回到本课第3节，把例子拆成逐行/逐项结果；卡在语法退回[阶段一]({link('06_lessons/' + STAGES[0]['folder'] + '/README.md')})，卡在shape退回[P06]({link(layout['P06']['guide'])})，卡在求导退回[P07]({link(layout['P07']['guide'])})，卡在期望退回[P08]({link(layout['P08']['guide'])})。先看提示再看答案；看过答案后合上它，换数字重做并注明借助过答案。

通过后隔一天闭卷复述，再做一个变式。自动工具不会运行你的作业，也不会替导师判定开放题；把report、代码和推导[发给助手]({link('START_HERE.md')}#feedback)，获得下一轮针对性反馈。完成主线记录后仍需复核毕业项目。

## 6. 本章结束：总结与回顾

**把这一章压缩成一句话：** {RECAPS[item['id']][1]}

先不看选项，用自己的话回答；这是课内概念题的闭卷复述，不是新增考试：

{recall}

<details>
<summary>先自己答，再展开核对要点</summary>

{answers}

</details>

### 结束检查

- [ ] 能用自己的话解释本章核心，不只复读小结。
- [ ] 完成第3节手算/推演，换数字或情景再核对一次。
- [ ] 完成第4节实践，保存真实输出和一处错误修正。
- [ ] 做完概念检查并记录实践证据、困惑；不会的地方按第5节补救。

在自己的笔记中写四行：**我现在会了什么 / 证据在哪里 / 还不懂什么 / 下次从哪一步开始**。如果只是看完视频，不勾选实践完成。

## 7. 下一步

{' · '.join(navigation)}

未完成本章就停在这里继续；阶段最后一章先去阶段总结，完成综合复盘再前进。选修没有必须按顺序读完的要求。
"""
        (ROOT / guide).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / guide).write_text(item_text, encoding="utf-8")
        manifest.append(dict(id=item["id"], title=item["title"], track=item["track"], requires=requires,
                             guide=guide, evidence=item["expected"], quiz=quiz, chapters=item["chapters"],
                             stage=stage["title"], chapter_number=position["number"]))
    build_stage_guides(layout, manifest)
    (ROOT / "course.json").write_text(json.dumps({"version": 1, "lessons": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_python_practice()
    print(f"Built {len(manifest)} lesson guides and course.json")


if __name__ == "__main__":
    build()
