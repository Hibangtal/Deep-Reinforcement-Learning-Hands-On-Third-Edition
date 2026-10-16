"""Shared helpers for auto-checked exercises in the learning notebooks.

Each exercise is rendered as three cells:

1. Markdown: task description and an optional hint (collapsed).
2. Code: a stub that raises ``NotImplementedError`` plus a ``test_xxx`` function
   and a ``check(test_xxx)`` call.  Running the cell never crashes the notebook:
   unfinished or wrong answers are reported as text, so "Run All" still works.
3. Markdown: the reference solution, collapsed so the learner tries first.

``selftest_notebook`` replaces every stub with its reference solution and runs
the notebook top to bottom, asserting that every test passes.  This guarantees
that the reference solutions and the tests agree with each other.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import dataclass
from textwrap import dedent


CHECK_HELPER = r'''
# 自动检查工具：每道练习末尾都会调用 check(test_xxx)。
# 未完成或有错时只打印提示，不会中断 Run All。
def check(test):
    try:
        test()
    except NotImplementedError:
        print(f"[未完成] {test.__name__}：把 raise NotImplementedError 换成你的代码，再运行本格。")
    except AssertionError as error:
        print(f"[未通过] {test.__name__}：{error}")
    except Exception as error:  # 代码本身报错，也给出可读提示
        print(f"[代码报错] {test.__name__}：{type(error).__name__}: {error}")
    else:
        print(f"[通过] {test.__name__}")
'''


@dataclass
class Exercise:
    """One auto-checked exercise.

    ``stub`` and ``solution`` define the same function(s); ``test`` defines a
    single ``test_*`` function that uses plain ``assert`` statements with
    Chinese messages that tell the learner what is wrong.
    """

    title: str
    prompt: str
    stub: str
    solution: str
    test: str
    hint: str = ""

    @property
    def test_name(self) -> str:
        for line in dedent(self.test).splitlines():
            if line.startswith("def test_"):
                return line[4:line.index("(")]
        raise ValueError(f"exercise {self.title!r} has no test_ function")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text: str, *, role: str | None = None) -> dict:
    cell = {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": dedent(text).strip() + "\n",
    }
    if role:
        cell["metadata"]["learning_role"] = role
    return cell


def helper_cell() -> dict:
    return code(CHECK_HELPER, role="check-helper")


def exercise_cells(number: int, ex: Exercise) -> list[dict]:
    header = f"### 练习 {number}：{ex.title}\n\n{dedent(ex.prompt).strip()}"
    if ex.hint:
        header += f"\n\n<details><summary>提示（卡住 10 分钟再看）</summary>\n\n{dedent(ex.hint).strip()}\n\n</details>"
    body = (
        dedent(ex.stub).strip() + "\n\n\n# ---- 自动检查（不用修改） ----\n"
        + dedent(ex.test).strip() + f"\n\ncheck({ex.test_name})"
    )
    stub_cell = code(body, role="exercise")
    # Store the solution in metadata so the self-test can swap it in.
    stub_cell["metadata"]["solution"] = dedent(ex.solution).strip() + "\n"
    stub_cell["metadata"]["test"] = dedent(ex.test).strip() + "\n"
    stub_cell["metadata"]["test_name"] = ex.test_name
    answer = md(
        "<details><summary>参考答案（先独立完成，再对照）</summary>\n\n```python\n"
        + dedent(ex.solution).strip() + "\n```\n\n</details>"
    )
    return [md(header), stub_cell, answer]


def exercise_block(exercises: list[Exercise], *, title: str = "## 自动批改练习") -> list[dict]:
    cells = [md(f"""
    {title}

    每道题先读题、再写代码、最后运行同一格。看到 `[通过]` 才算完成；`[未通过]` 会说明哪里不对。
    参考答案默认折叠，建议独立尝试至少 10 分钟后再看。
    """), helper_cell()]
    for index, ex in enumerate(exercises, start=1):
        cells.extend(exercise_cells(index, ex))
    return cells


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def selftest_notebook(doc: dict, owner: str) -> int:
    """Run a notebook with reference solutions swapped in; return #exercises."""
    namespace: dict = {}
    checked = 0
    for index, cell in enumerate(doc["cells"]):
        if cell["cell_type"] != "code":
            continue
        meta = cell.get("metadata", {})
        if meta.get("learning_role") == "exercise":
            source = meta["solution"] + "\n" + meta["test"]
            compiled = compile(source, f"{owner}:solution-{index}", "exec")
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(compiled, namespace)
                try:
                    namespace[meta["test_name"]]()
                except AssertionError as error:
                    raise AssertionError(f"{owner} cell {index}: reference solution fails: {error}") from error
            checked += 1
        else:
            compiled = compile(_source(cell), f"{owner}:cell-{index}", "exec")
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(compiled, namespace)
    return checked
