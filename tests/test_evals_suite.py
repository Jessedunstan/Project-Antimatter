"""Unit tests verifying the integrity of the evals/ seeded-bug benchmark suite."""

from pathlib import Path
from evals.tasks import get_benchmark_tasks
from evals.run import _run_pytest_in_workspace


def test_all_10_seeded_bugs_fail_before_repair(tmp_path: Path):
    tasks = get_benchmark_tasks()
    assert len(tasks) == 10

    for task in tasks:
        task_dir = tmp_path / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / task.source_filename).write_text(task.buggy_code, encoding="utf-8")
        (task_dir / task.test_filename).write_text(task.test_code, encoding="utf-8")

        passed, output = _run_pytest_in_workspace(task_dir, task.test_filename)
        assert passed is False, f"Task {task.task_id} unexpectedly passed before repair:\n{output}"
