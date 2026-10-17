# 环境按阶段准备

[学习入口](../START_HERE.md)

## P00–P05：先用已有Python

反馈工具与第一周示例只依赖标准库，Python 3.10及以上即可。仓库原书训练基线是3.11；本次检查默认解释器为3.13且没有NumPy/PyTorch/Jupyter，所以不把“机器上有Python”当成“原书环境已配好”。

```bash
python --version
python -c 'import sys; print(sys.executable)'
python my_learning/coach.py status
```

暂时不安装根目录requirements：它同时包含Atari、RL库等，第一周用不到。

## P06：建立独立的Python 3.11学习环境

先检查`python3.11 --version`。若不存在，把系统与报错交给助手配置3.11；不要用3.13/3.14强装整套书籍固定版本。本教程未自动安装解释器或下载大型依赖。

Linux终端在仓库根目录：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy<2' matplotlib jupyterlab ipykernel
python -m ipykernel install --prefix .venv --name rl-book --display-name 'RL Book Python 3.11'
python -m jupyterlab my_learning
```

若`.venv`已经存在，先核对其Python版本，不覆盖别人的环境。已有环境不合适时另建`.venv-rl`并相应替换命令路径。缺少venv组件、网络下载失败、权限错误时保留完整报错交给助手；不要反复升级全局pip。

浏览器打开Jupyter给出的本机地址。左侧是文件树，双击.ipynb，右上角选择刚建的kernel。在新代码格运行：

```python
import sys
import numpy as np
print(sys.executable)
print(np.array([1, 2, 3]).shape)
```

预期解释器路径在当前环境里、shape为`(3,)`。`Shift+Enter`执行当前格；`[*]`表示还在运行；菜单Kernel→Restart Kernel重置内存。先将作业另存到自己的scratch目录，再做修改；原教程的相对链接以原目录为基准。

## P11/R02：再加深度学习与经典控制

在同一个3.11环境运行（本次只提供与原书一致的版本方案，未声称下载/安装已成功）：

```bash
python -m pip install torch==2.5.0 'gymnasium[classic-control]==0.29.1'
python -m pip check
python -c 'import torch, gymnasium; print(torch.__version__, gymnasium.__version__)'
```

初学小网络使用CPU即可；torch默认安装的下载大小和设备支持取决于平台。若需CPU专用wheel或GPU配置，届时按本机系统选PyTorch官方安装方式，不先购买设备。

此后才打开原来的[环境与Jupyter notebook](01_Environment_and_Jupyter.ipynb)，它会同时导入数组、绘图、torch、gymnasium。

## 跑原书训练脚本之前

先读脚本顶部import和本章requirements，安装对应依赖。第7章的PTAN、TensorBoard等不包含在上述最小环境里；需要完整主线复现时，在单独3.11环境按根requirements配置。第13/14/15/16/19/21/22章还各有专门requirements，旧gym、浏览器、ROM与仿真环境不要随意混装。

先运行无渲染小例子，再开启可视化；先小预算冒烟，再完整训练。把`python -m pip freeze`、seed、配置、命令与日志存到自己的实验目录。Notebook结构检查通过并不代表原书长训练已复现。
