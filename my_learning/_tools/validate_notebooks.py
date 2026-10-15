"""Structural validation for the generated learning notebooks."""

from __future__ import annotations

import json
import re
import argparse
import contextlib
import io
import warnings
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent
SUPPLEMENTS = {
    "00_start_here/00_Study_Roadmap.ipynb": ["十分钟摸底", "四阶段路线", "什么时候算“掌握”"],
    "04_assessments/Milestone_Exams.ipynb": ["关卡 A", "关卡 B", "关卡 C", "关卡 D"],
    "03_guided_projects/Experiment_Template.ipynb": ["多个随机种子", "置信带", "最小报告结构"],
    "03_guided_projects/Capstone_Projects.ipynb": ["项目 A", "项目 B", "项目 C", "项目交付物"],
}
PREREQUISITES = {
    "01_prerequisites/Prerequisite00_Index.ipynb": ["推荐顺序", "什么暂时不用学", "时间建议"],
    "01_prerequisites/Prerequisite01_Python.ipynb": ["数据：一个 transition", "控制流与函数", "类：把状态和行为"],
    "01_prerequisites/Prerequisite02_NumPy.ipynb": ["数组、shape、dtype", "axis 与动作索引", "广播：方便，也危险"],
    "01_prerequisites/Prerequisite03_Matplotlib.ipynb": ["移动平均做了什么", "曲线阅读顺序"],
    "01_prerequisites/Prerequisite04_Math.ipynb": ["向量、矩阵与加权和", "导数与链式法则", "softmax 与交叉熵"],
    "01_prerequisites/Prerequisite05_Machine_Learning.ipynb": ["监督学习的最小闭环", "train / validation / test", "过拟合"],
    "01_prerequisites/Prerequisite06_PyTorch.ipynb": ["Tensor 与 NumPy", "Autograd", "完整训练循环", "detach"],
    "01_prerequisites/Prerequisite07_Gymnasium.ipynb": ["最小随机 agent", "Spaces 是数据契约", "停止 bootstrap"],
    "01_prerequisites/Prerequisite08_Readiness_Check.ipynb": ["概念题", "NumPy batch", "PyTorch 训练循环", "终止语义"],
}
GUIDED = {
    "00_start_here/01_Environment_and_Jupyter.ipynb": ["当前 Notebook 到底使用哪个 Python", "Cell 有状态", "怎样阅读 traceback"],
    "03_guided_projects/Guided_Project01_GridWorld.ipynb": ["随机基线", "训练 Q-learning", "独立评估", "故障注入"],
    "03_guided_projects/Guided_Project02_DQN_CartPole.ipynb": ["DQN 数据流", "replay buffer", "一次 batch 更新", "完整训练", "独立评估"],
}


def cell_source(cell: dict) -> str:
    """Normalize both legal nbformat source representations."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def validate_local_links(markdown: str, path: Path, failures: list[str]) -> None:
    for target in re.findall(r"\]\(([^)#]+)", markdown):
        if "://" in target:
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            failures.append(f"{path.relative_to(HERE)}: missing link {target}")


def execute_code_cells(doc: dict, owner: str, execute: bool) -> int:
    namespace: dict = {}
    code_count = 0
    for index, cell in enumerate(doc["cells"]):
        if cell["cell_type"] != "code":
            continue
        code_count += 1
        source = cell_source(cell)
        compiled = compile(source, f"{owner}:cell-{index}", "exec")
        if execute:
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
                exec(compiled, namespace)
    return code_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="also run every lightweight code cell")
    args = parser.parse_args()
    notebooks = sorted((HERE / "02_chapters").glob("Chapter[0-9][0-9].ipynb"))
    assert len(notebooks) == 22, f"expected 22 notebooks, got {len(notebooks)}"
    failures: list[str] = []
    for path in notebooks:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            assert doc["nbformat"] == 4
            assert len(doc["cells"]) >= 26
            all_markdown = "\n".join(
                cell_source(cell) for cell in doc["cells"] if cell["cell_type"] == "markdown"
            )
            assert '<a id="extension"></a>' in all_markdown
            assert f'<a id="chapter-{path.stem[-2:]}"></a>' in all_markdown
            required_sections = [
                "先想一个问题",
                "原书覆盖地图",
                "符号表",
                "分步推导",
                "手算一个最小例子",
                "公式 ↔ 代码逐项对照",
                "怎样读运行结果",
                "完整算法",
                "第二个验证实验",
                "项目源码精读路线",
                "训练曲线、调参与失败复盘",
                "曲线怎么读",
                "调参过程",
                "失败案例复盘",
                "实用技巧与常见坑",
                "动手练习",
                "练习参考答案",
                "脑中应留下什么",
                "两分钟自测",
                "与前后章节连接",
            ]
            for section in required_sections:
                assert section in all_markdown, f"missing teaching section: {section}"
            assert "\x07" not in all_markdown and "\t" not in all_markdown
            in_fence = False
            for line in all_markdown.splitlines():
                if line.lstrip().startswith("```"):
                    in_fence = not in_fence
                elif not in_fence:
                    assert not line.startswith("    "), f"unexpected Markdown code indentation: {line[:40]!r}"
            code_count = execute_code_cells(doc, path.name, args.execute)
            assert code_count >= 3, f"expected at least three executable experiments, got {code_count}"
            validate_local_links(all_markdown, path, failures)
        except Exception as exc:  # report every broken notebook at once
            failures.append(f"{path.name}: {exc}")

    for name, required_text in (SUPPLEMENTS | PREREQUISITES | GUIDED).items():
        path = HERE / name
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            assert doc["nbformat"] == 4
            markdown = "\n".join(
                cell_source(cell) for cell in doc["cells"] if cell["cell_type"] == "markdown"
            )
            for text in required_text:
                assert text in markdown, f"missing mastery section: {text}"
            assert execute_code_cells(doc, name, args.execute) >= 1
            validate_local_links(markdown, path, failures)
        except Exception as exc:
            failures.append(f"{name}: {exc}")

    for name in ["LEARNING_CHECKLIST.md", "DEBUGGING_PLAYBOOK.md", "GLOSSARY.md"]:
        if not (HERE / "05_reference" / name).exists():
            failures.append(f"missing support document: {name}")
    for markdown_path in HERE.rglob("*.md"):
        validate_local_links(markdown_path.read_text(encoding="utf-8"), markdown_path, failures)
    root_notebooks = list(HERE.glob("*.ipynb"))
    if root_notebooks:
        failures.append("notebooks must live in numbered learning-stage directories")
    if failures:
        raise SystemExit("\n".join(failures))
    coverage = HERE / "05_reference" / "COVERAGE.md"
    assert coverage.exists(), "missing COVERAGE.md"
    coverage_text = coverage.read_text(encoding="utf-8")
    assert coverage_text.count("## [") == 22, "coverage audit must contain 22 chapter sections"
    suffix = ", and code execution" if args.execute else ""
    total = len(notebooks) + len(SUPPLEMENTS) + len(PREREQUISITES) + len(GUIDED)
    print(f"Validated {total} notebooks: JSON, sections, anchors, local links{suffix} OK")


if __name__ == "__main__":
    main()
