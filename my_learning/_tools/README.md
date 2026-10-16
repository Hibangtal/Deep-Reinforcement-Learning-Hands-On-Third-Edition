# 生成与验证工具

- `build_notebooks.py`：生成 22 章伴读 Notebook 与总目录。
- `build_prerequisites.py`：生成零基础预备篇（13 个 notebook）。
- `foundations_probability.py`、`foundations_neural_net.py`、`foundations_cnn.py`、`foundations_bandit.py`：4B/5B/6B/7B 补课 notebook 的内容。
- `foundations_extras.py`：追加到 NumPy、数学、机器学习、PyTorch 预备章的教学小节与练习。
- `exercise_kit.py`：自动批改练习工具。每道题由“题目 + 待填函数 + 检查函数 + 折叠答案”组成；参考答案保存在单元格 metadata 中，用于自检。
- `build_mastery_materials.py`：生成学习路线、考试、实验模板与综合项目说明。
- `build_guided_projects.py`：生成环境入门和两个完整引导项目。
- `training_content.py`：各章训练曲线、调参与失败案例内容。
- `validate_notebooks.py`：检查结构、链接、语法，可实际执行代码，并用参考答案自检全部练习。

从项目根目录运行：

```bash
python my_learning/_tools/build_notebooks.py
python my_learning/_tools/build_prerequisites.py
python my_learning/_tools/build_mastery_materials.py
python my_learning/_tools/build_guided_projects.py
python my_learning/_tools/validate_notebooks.py --execute
```

## 新增一道自动批改练习

在对应的 `foundations_*.py` 中添加一个 `Exercise(...)`：

- `stub`：给学习者的函数骨架，函数体为 `raise NotImplementedError`；
- `solution`：参考答案，定义同名函数；
- `test`：一个 `def test_xxx():`，用带中文提示的 `assert` 检查结果。

重新生成后运行验证脚本；参考答案未通过检查时验证会失败。
