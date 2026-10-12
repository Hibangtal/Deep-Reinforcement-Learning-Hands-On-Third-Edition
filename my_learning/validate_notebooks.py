"""Structural validation for the generated learning notebooks."""

from __future__ import annotations

import json
import re
import argparse
import contextlib
import io
import warnings
from pathlib import Path


HERE = Path(__file__).resolve().parent


def cell_source(cell: dict) -> str:
    """Normalize both legal nbformat source representations."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="also run every lightweight code cell")
    args = parser.parse_args()
    notebooks = sorted(HERE.glob("Chapter[0-9][0-9].ipynb"))
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
            namespace: dict = {}
            code_count = 0
            for index, cell in enumerate(doc["cells"]):
                if cell["cell_type"] != "code":
                    continue
                code_count += 1
                source = cell_source(cell)
                compiled = compile(source, f"{path.name}:cell-{index}", "exec")
                if args.execute:
                    captured = io.StringIO()
                    with contextlib.redirect_stdout(captured), warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
                        exec(compiled, namespace)
            assert code_count >= 3, f"expected at least three executable experiments, got {code_count}"
            for target in re.findall(r"\]\((\.\./[^)#]+)\)", all_markdown):
                resolved = (HERE / target).resolve()
                if not resolved.exists():
                    failures.append(f"{path.name}: missing link {target}")
        except Exception as exc:  # report every broken notebook at once
            failures.append(f"{path.name}: {exc}")
    if failures:
        raise SystemExit("\n".join(failures))
    coverage = HERE / "COVERAGE.md"
    assert coverage.exists(), "missing COVERAGE.md"
    coverage_text = coverage.read_text(encoding="utf-8")
    assert coverage_text.count("## [") == 22, "coverage audit must contain 22 chapter sections"
    suffix = ", and code execution" if args.execute else ""
    print(f"Validated {len(notebooks)} notebooks: JSON, sections, anchors, local links{suffix} OK")


if __name__ == "__main__":
    main()
