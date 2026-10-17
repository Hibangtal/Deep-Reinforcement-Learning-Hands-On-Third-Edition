"""Check the learner feedback lifecycle without optional ML dependencies."""

import contextlib
import copy
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import coach
import build_course
from exercise_kit import selftest_notebook
from course_structure import STAGES


class CoachTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "progress.json"
        self.progress = coach.load_progress(self.path)
        self.lessons = coach.load_course()
        self.first = self.lessons[0]

    def answer_all(self, lesson, correct=True):
        answers = iter(str((question["answer"] if correct else
                            (question["answer"] + 1) % len(question["choices"])) + 1)
                       for question in lesson["quiz"])
        return coach.quiz(lesson, self.progress, read=lambda _: next(answers), write=lambda _: None)

    def test_wrong_answers_explain_and_do_not_unlock(self):
        output = []
        answers = iter(str((question["answer"] + 1) % len(question["choices"]) + 1)
                       for question in self.first["quiz"])
        self.assertFalse(coach.quiz(self.first, self.progress, read=lambda _: next(answers), write=output.append))
        self.assertTrue(any("补救" in message for message in output))
        self.assertEqual(coach.next_lesson(self.lessons, self.progress)["id"], "P00")
        with self.assertRaises(ValueError):
            coach.record_evidence(self.first, self.progress, "输出5", "会运行文件")

    def test_quiz_alone_is_not_practice(self):
        self.answer_all(self.first)
        self.assertNotIn("P00", coach.completed(self.lessons, self.progress))
        with self.assertRaises(ValueError):
            coach.record_evidence(self.first, self.progress, " ", "复盘")
        coach.record_evidence(self.first, self.progress, "hello.py输出5和6", "修改后需要重跑")
        self.assertEqual(coach.next_lesson(self.lessons, self.progress)["id"], "P01")

    def test_round_trip_and_report(self):
        self.answer_all(self.first)
        coach.record_evidence(self.first, self.progress, "自己的代码输出5", "解释器读取文件")
        coach.save_progress(self.path, self.progress)
        self.assertEqual(coach.load_progress(self.path), self.progress)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            coach.show_report(self.lessons, coach.load_progress(self.path))
        self.assertIn("自己的代码输出5", output.getvalue())
        self.assertIn("自报", output.getvalue())
        self.assertIn("P01", output.getvalue())

    def test_failed_retest_invalidates_progress(self):
        for lesson in self.lessons[:2]:
            self.answer_all(lesson)
            coach.record_evidence(lesson, self.progress, "实践证据", "复盘")
        self.assertIn("P01", coach.completed(self.lessons, self.progress))
        self.answer_all(self.first, correct=False)
        self.assertNotIn("P00", coach.completed(self.lessons, self.progress))
        self.assertNotIn("P01", coach.completed(self.lessons, self.progress))
        self.assertNotIn("evidence", self.progress["lessons"]["P00"])

    def test_invalid_input_then_correct_answer(self):
        answers = iter(["bad", "0", "99", "١", *[str(question["answer"] + 1) for question in self.first["quiz"]]])
        self.assertTrue(coach.quiz(self.first, self.progress, read=lambda _: next(answers), write=lambda _: None))

    def test_interrupt_does_not_mutate_previous_progress(self):
        self.answer_all(self.first)
        before = copy.deepcopy(self.progress)
        responses = iter([str(self.first["quiz"][0]["answer"] + 1)])

        def interrupted_read(_):
            try:
                return next(responses)
            except StopIteration:
                raise EOFError

        with self.assertRaises(EOFError):
            coach.quiz(self.first, self.progress, read=interrupted_read, write=lambda _: None)
        self.assertEqual(self.progress, before)

    def test_corrupt_files_are_preserved(self):
        for content in ["{broken", '{"version":1,"lessons":[]}',
                        '{"version":1,"lessons":{"P00":{"attempts":[{"passed":"yes"}]}}}']:
            self.path.write_text(content, encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                result = coach.main(["--progress", str(self.path), "record", "P00",
                                     "--evidence", "some", "--reflection", "some"])
            self.assertEqual(result, 2)
            self.assertEqual(self.path.read_text(encoding="utf-8"), content)

    def test_failed_atomic_write_preserves_original(self):
        coach.save_progress(self.path, self.progress)
        before = self.path.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("simulated write failure")):
            with self.assertRaises(OSError):
                coach.save_progress(self.path, {"version": 1, "lessons": {"new": {}}})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_cli_blocks_skipping_prerequisites(self):
        with contextlib.redirect_stderr(io.StringIO()):
            result = coach.main(["--progress", str(self.path), "quiz", "R13"])
        self.assertEqual(result, 2)
        self.assertFalse(self.path.exists())

    def test_cli_piped_success_failure_and_eof(self):
        command = [sys.executable, str(coach.ROOT / "coach.py"), "--progress", str(self.path), "quiz", "P00"]
        good = "\n".join(str(question["answer"] + 1) for question in self.first["quiz"]) + "\n"
        result = subprocess.run(command, input=good, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        before = self.path.read_bytes()
        result = subprocess.run(command, input="", capture_output=True, text=True)
        self.assertEqual(result.returncode, 130)
        self.assertEqual(self.path.read_bytes(), before)
        bad = "\n".join(str((question["answer"] + 1) % len(question["choices"]) + 1)
                        for question in self.first["quiz"]) + "\n"
        result = subprocess.run(command, input=bad, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("补救", result.stdout)

    def test_complete_curriculum_and_optional_tracks(self):
        for lesson in self.lessons:
            self.assertTrue(set(lesson["requires"]) <= coach.completed(self.lessons, self.progress))
            self.assertTrue(self.answer_all(lesson))
            coach.record_evidence(lesson, self.progress, "测试夹具中的实践自报", "测试夹具中的复盘")
        self.assertIsNone(coach.next_lesson(self.lessons, self.progress))
        self.assertEqual(len(coach.completed(self.lessons, self.progress)), len(self.lessons))

    def test_paths_chapter_coverage_and_diagnostic_choices(self):
        coverage = set()
        for lesson in self.lessons:
            coverage.update(lesson["chapters"])
            self.assertTrue((coach.ROOT / lesson["guide"]).is_file())
            text = (coach.ROOT / lesson["guide"]).read_text(encoding="utf-8")
            for section in ["带问题看视频", "从数字走到公式", "反馈、补救与过关"]:
                self.assertIn(section, text)
            self.assertGreaterEqual(len(lesson["quiz"]), 2)
            for question in lesson["quiz"]:
                self.assertTrue(question["remedy"])
                self.assertTrue(question["explanation"])
                self.assertEqual(len(question["choices"]), len(set(question["choices"])))
        self.assertEqual(coverage, set(range(1, 23)))

    def test_beginner_notebook_reference_solutions(self):
        path = coach.ROOT / "00_start_here/02_Python_Baby_Steps.ipynb"
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(selftest_notebook(document, path.name), 6)
        namespace = {}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for cell in document["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile(cell["source"], str(path), "exec"), namespace)
        self.assertEqual(output.getvalue().count("[未完成]"), 6)

    def test_stage_layout_and_chapter_guidance(self):
        self.assertEqual(len(STAGES), 6)
        grouped_ids = [lesson_id for stage in STAGES for lesson_id in stage["ids"]]
        self.assertEqual(grouped_ids, [lesson["id"] for lesson in self.lessons])
        for stage in STAGES:
            directory = coach.ROOT / "06_lessons" / stage["folder"]
            stage_text = (directory / "README.md").read_text(encoding="utf-8")
            for heading in ["阶段开始", "本阶段章节", 'id="stage-review"', "阶段结束", "下一步"]:
                self.assertIn(heading, stage_text)
            for number, lesson_id in enumerate(stage["ids"], 1):
                lesson = next(item for item in self.lessons if item["id"] == lesson_id)
                guide = coach.ROOT / lesson["guide"]
                self.assertEqual(guide.parent, directory)
                self.assertEqual(lesson["chapter_number"], number)
                self.assertEqual(lesson["stage"], stage["title"])
                self.assertTrue(guide.name.startswith(f"{number:02d}_{lesson_id}_"))
                self.assertIn(guide.name, stage_text)
                text = guide.read_text(encoding="utf-8")
                for heading in ["本章开始", "本章结束", "结束检查", "下一步", "原书对应章"]:
                    self.assertIn(heading, text)
                if number == len(stage["ids"]) or lesson["track"] == "optional":
                    self.assertIn("README.md#stage-review", text)
        self.assertEqual([path.name for path in (coach.ROOT / "06_lessons").glob("*.md")], ["README.md"])

    def test_existing_progress_survives_guide_relocation(self):
        self.answer_all(self.first)
        coach.record_evidence(self.first, self.progress, "迁移前的实践", "迁移前的复盘")
        coach.save_progress(self.path, self.progress)
        migrated = coach.load_course()
        saved = coach.load_progress(self.path)
        self.assertIn("P00", coach.completed(migrated, saved))
        self.assertEqual(coach.next_lesson(migrated, saved)["id"], "P01")
        self.assertEqual(saved["lessons"]["P00"]["evidence"], "迁移前的实践")

    def test_navigation_links_and_stage_exits(self):
        for stage_index, stage in enumerate(STAGES):
            directory = coach.ROOT / "06_lessons" / stage["folder"]
            for guide in directory.glob("*.md"):
                text = guide.read_text(encoding="utf-8")
                for target in re.findall(r"\]\(([^)]+)\)", text):
                    if "://" in target:
                        continue
                    filename, separator, anchor = target.partition("#")
                    resolved = (guide.parent / filename).resolve()
                    self.assertTrue(resolved.exists(), (guide, target))
                    if separator:
                        self.assertIn(f'id="{anchor}"', resolved.read_text(encoding="utf-8"))
            if stage_index < 5:
                text = (directory / "README.md").read_text(encoding="utf-8")
                self.assertIn(f"../{STAGES[stage_index + 1]['folder']}/README.md", text)
            for index, lesson_id in enumerate(stage["ids"]):
                lesson = next(item for item in self.lessons if item["id"] == lesson_id)
                text = (coach.ROOT / lesson["guide"]).read_text(encoding="utf-8")
                if stage_index == 5:
                    self.assertNotIn("下一章：", text)
                elif index + 1 < len(stage["ids"]):
                    following = next(item for item in self.lessons if item["id"] == stage["ids"][index + 1])
                    self.assertIn(Path(following["guide"]).name, text)

    def test_generator_matches_files_and_preserves_progress(self):
        generated = Path(self.temporary.name) / "generated"
        (generated / "00_start_here").mkdir(parents=True)
        progress_path = generated / "progress.json"
        progress_path.write_text('learner-owned sentinel', encoding="utf-8")
        with patch.object(build_course, "ROOT", generated), contextlib.redirect_stdout(io.StringIO()):
            build_course.build()
            first = {path.relative_to(generated): path.read_bytes() for path in generated.rglob("*") if path.is_file()}
            build_course.build()
        second = {path.relative_to(generated): path.read_bytes() for path in generated.rglob("*") if path.is_file()}
        self.assertEqual(first, second)
        self.assertEqual(progress_path.read_text(encoding="utf-8"), 'learner-owned sentinel')
        for relative, content in second.items():
            if relative.name != "progress.json":
                self.assertEqual((coach.ROOT / relative).read_bytes(), content, relative)

    def test_cli_shows_stages_and_locates_new_chapter(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(coach.main(["--progress", str(self.path), "status"]), 0)
        for stage in STAGES:
            self.assertEqual(output.getvalue().count(stage["title"]), 1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(coach.main(["--progress", str(self.path), "lesson", "P00"]), 0)
        self.assertIn("阶段一 · Python基础 / 第01章", output.getvalue())
        self.assertIn(self.first["guide"], output.getvalue())


if __name__ == "__main__":
    unittest.main()
