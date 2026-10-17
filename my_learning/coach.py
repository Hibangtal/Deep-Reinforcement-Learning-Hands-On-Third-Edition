"""A local, standard-library study companion; never executes learner code."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent


def load_course(path: Path = ROOT / "course.json") -> list[dict]:
    lessons = json.loads(path.read_text(encoding="utf-8"))["lessons"]
    known = set()
    for lesson in lessons:
        if lesson["id"] in known or not set(lesson["requires"]) <= known:
            raise ValueError("课程编号重复或先修顺序有误")
        known.add(lesson["id"])
        if not lesson["quiz"]:
            raise ValueError("每课需要检查题")
        for question in lesson["quiz"]:
            if not 0 <= question["answer"] < len(question["choices"]):
                raise ValueError("检查题答案超出选项范围")
    return lessons


def load_progress(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "lessons": {}}
    progress = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(progress, dict) or progress.get("version") != 1:
        raise ValueError("进度文件版本不支持；原文件保留，请先备份检查")
    if not isinstance(progress.get("lessons"), dict):
        raise ValueError("进度文件格式有误；原文件保留，请先备份检查")
    for record in progress["lessons"].values():
        if not isinstance(record, dict) or not isinstance(record.get("attempts", []), list):
            raise ValueError("进度记录格式有误；原文件保留")
        for attempt in record.get("attempts", []):
            if not isinstance(attempt, dict) or not isinstance(attempt.get("passed"), bool):
                raise ValueError("测验记录格式有误；原文件保留")
    return progress


def save_progress(path: Path, progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(progress, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def completed(lessons: list[dict], progress: dict) -> set[str]:
    done = set()
    for lesson in lessons:
        record = progress["lessons"].get(lesson["id"], {})
        attempts = record.get("attempts", [])
        if (attempts and attempts[-1]["passed"] and record.get("evidence")
                and record.get("reflection") and set(lesson["requires"]) <= done):
            done.add(lesson["id"])
    return done


def next_lesson(lessons: list[dict], progress: dict) -> dict | None:
    done = completed(lessons, progress)
    return next((lesson for lesson in lessons if lesson["track"] == "core"
                 and lesson["id"] not in done and set(lesson["requires"]) <= done), None)


def show_status(lessons: list[dict], progress: dict) -> None:
    done = completed(lessons, progress)
    print("进度 = 概念检查通过 + 实践自报；不代表已由导师审核掌握。")
    print("P=前置基础；R=强化学习主线；O=选修。课号不是原书章号。")
    current_stage = None
    for lesson in lessons:
        stage = lesson.get("stage", "学习章节")
        if stage != current_stage:
            print(f"\n{stage}")
            current_stage = stage
        record = progress["lessons"].get(lesson["id"], {})
        attempts = record.get("attempts", [])
        if lesson["id"] in done:
            status = "已记录"
        elif not set(lesson["requires"]) <= done:
            status = "待先修"
        elif attempts and attempts[-1]["passed"]:
            status = "待实践"
        elif attempts:
            status = "需补课"
        else:
            status = "可开始"
        label = "选修" if lesson["track"] == "optional" else "主线"
        print(f"{lesson['id']:4s} [{status}] [{label}] {lesson['title']}")
    upcoming = next_lesson(lessons, progress)
    if upcoming:
        print(f"\n下一步：python my_learning/coach.py lesson {upcoming['id']}")
    else:
        print("\n主线学习记录齐全：请用毕业项目证据接受复核；也可选择专题。")


def quiz(lesson: dict, progress: dict, read=input, write=print) -> bool:
    answers = []
    for index, question in enumerate(lesson["quiz"], 1):
        write(f"\n{index}. {question['question']}")
        for number, choice in enumerate(question["choices"], 1):
            write(f"  {number}. {choice}")
        while True:
            raw = read("你的选择（输入数字，Ctrl+C 退出）：").strip()
            if raw.isascii() and raw.isdigit() and 1 <= int(raw) <= len(question["choices"]):
                answer = int(raw) - 1
                break
            write("请输入本题选项对应的数字。")
        correct = answer == question["answer"]
        answers.append({"choice": answer, "correct": correct})
        write(("[正确] " if correct else "[需补课] ") + question["explanation"])
        if not correct:
            write("补救：" + question["remedy"])
    passed = all(answer["correct"] for answer in answers)
    record = progress["lessons"].setdefault(lesson["id"], {})
    record.setdefault("attempts", []).append({"time": timestamp(), "passed": passed, "answers": answers})
    if not passed:
        record.pop("evidence", None)
        record.pop("reflection", None)
    write("概念检查通过。接着完成课卡实践；答案通过不代表实践已验证。" if passed
          else "本次未过关。按提示补课，换数字重做，再复测；旧实践标记已失效。")
    return passed


def record_evidence(lesson: dict, progress: dict, evidence: str, reflection: str) -> None:
    record = progress["lessons"].get(lesson["id"], {})
    attempts = record.get("attempts", [])
    if not attempts or not attempts[-1]["passed"]:
        raise ValueError("先完成本课概念检查；最近一次测验必须全部正确")
    if not evidence.strip() or not reflection.strip():
        raise ValueError("证据和复盘不能为空")
    record.update(evidence=evidence.strip(), reflection=reflection.strip(), recorded_at=timestamp())


def show_report(lessons: list[dict], progress: dict) -> None:
    print("请作为我的强化学习导师复核以下记录。先指出一个具体问题，再给一个小练习，等我回答后继续。")
    print("以下实践均为学习者自报，选择题由本地工具检查；请检查我的实际代码和推导。")
    by_id = {lesson["id"]: lesson for lesson in lessons}
    for lesson_id, record in progress["lessons"].items():
        if lesson_id not in by_id:
            continue
        attempts = record.get("attempts", [])
        print(f"\n{lesson_id} {by_id[lesson_id]['title']}")
        if attempts:
            print("最近测验：" + ("通过" if attempts[-1]["passed"] else "未通过"))
            for index, answer in enumerate(attempts[-1].get("answers", []), 1):
                if not answer["correct"]:
                    print(f"需补课：第 {index} 题；{by_id[lesson_id]['quiz'][index - 1]['remedy']}")
        print("证据：", record.get("evidence", "尚未提交"))
        print("复盘：", record.get("reflection", "尚未提交"))
    upcoming = next_lesson(lessons, progress)
    print("\n下一课：", upcoming["id"] if upcoming else "主线记录完成，复核毕业项目")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="零基础 RL 陪学：课卡、检查、补课与本地进度")
    parser.add_argument("--progress", type=Path, default=ROOT / "progress.json")
    parser.add_argument("command", choices=["status", "lesson", "quiz", "record", "report"], nargs="?", default="status")
    parser.add_argument("lesson_id", nargs="?")
    parser.add_argument("--evidence", default="")
    parser.add_argument("--reflection", default="")
    args = parser.parse_args(argv)
    try:
        lessons = load_course()
        progress = load_progress(args.progress)
        if args.command == "status":
            show_status(lessons, progress)
        elif args.command == "report":
            show_report(lessons, progress)
        else:
            lesson = next((item for item in lessons if item["id"] == (args.lesson_id or "").upper()), None)
            if lesson is None:
                raise ValueError("请输入有效课号，例如 lesson P00；status 可查看所有课号")
            if args.command == "lesson":
                if "stage" in lesson:
                    print(f"{lesson['stage']} / 第{lesson['chapter_number']:02d}章")
                print(f"{lesson['id']} · {lesson['title']}\n课卡：{ROOT / lesson['guide']}\n提交：{lesson['evidence']}")
                print("先修：", ", ".join(lesson["requires"]) or "无")
            else:
                missing = set(lesson["requires"]) - completed(lessons, progress)
                if missing:
                    raise ValueError("先完成先修课的测验与实践记录：" + ", ".join(sorted(missing)))
                if args.command == "quiz":
                    passed = quiz(lesson, progress)
                    save_progress(args.progress, progress)
                    return 0 if passed else 1
                record_evidence(lesson, progress, args.evidence, args.reflection)
                save_progress(args.progress, progress)
                print("实践自报已保存，尚未经导师审核。使用 report 生成复核材料。")
                upcoming = next_lesson(lessons, progress)
                if upcoming:
                    print("下一课：", upcoming["id"], upcoming["title"])
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\n本次操作已取消，已保存的进度不变。")
        return 130
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"无法继续：{error}。不会重置已有进度。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
