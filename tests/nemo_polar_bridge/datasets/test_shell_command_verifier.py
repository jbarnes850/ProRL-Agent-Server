"""Tests for the shell-harness task type (execution-based verifier)."""

from __future__ import annotations

import argparse

from nemo_polar_bridge.datasets.prepare_dataset import (
    SHELL_EXEC_EXECUTION_TYPES,
    _build_shell_exec_agent_command,
    build_task_template,
)
from nemo_polar_bridge.datasets.verifiers import (
    extract_shell_command,
    grade_shell_command_execution,
)


def _args(**overrides):
    base = dict(
        execution_type="shell_command_exec",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        model_max_tokens=256,
        model_temperature=1.0,
        model_top_p=1.0,
        model_request_timeout_seconds=120.0,
        test_timeout_seconds=30.0,
        task_timeout_seconds=240.0,
        runtime_image="polar-spark-shell:latest",
        answer_format="none",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_extract_shell_command_strips_fence_and_prose():
    assert extract_shell_command("Sure:\n```bash\necho hi\n```\nhope this helps") == "echo hi"
    assert extract_shell_command("```\nseq 1 3 | wc -l\n```") == "seq 1 3 | wc -l"
    # comment-only lines inside the fence are not treated as the command
    assert extract_shell_command("```bash\n# do the thing\necho ok\n```") == "echo ok"


def test_grade_rewards_correct_execution_output():
    result = grade_shell_command_execution("```bash\necho hello | tr a-z A-Z\n```", "HELLO")
    assert result["passed"] is True
    assert result["reward"] == 1.0
    assert result["reason"] == "shell_exec_match"


def test_grade_scores_by_execution_not_command_text():
    # command text mentions the answer but produces the wrong stdout -> reward 0
    result = grade_shell_command_execution("```bash\necho HELLO_is_not_this\n```", "HELLO")
    assert result["passed"] is False
    assert result["reward"] == 0.0
    assert result["reason"] == "shell_exec_stdout_mismatch"


def test_grade_flags_nonzero_exit_and_missing_command():
    assert grade_shell_command_execution("```bash\nexit 5\n```", "x")["reason"] == "shell_exec_nonzero_exit"
    assert grade_shell_command_execution("", "x")["reason"] == "shell_exec_no_command"


def test_build_task_template_routes_shell_branch():
    template = build_task_template(_args())
    assert template["agent"]["harness"] == "shell"
    assert template["metadata"]["execution_type"] == "shell_command_exec"
    assert template["evaluator"]["strategy"] == "verifier_result_file"
    command = template["agent"]["custom_shell"]["command"]
    assert "grade_shell(" in command
    assert "subprocess.run(" in command


def test_shell_heredoc_is_valid_python():
    import ast

    command = _build_shell_exec_agent_command(_args())
    body = command.split("<<'PY'\n", 1)[1].rsplit("PY", 1)[0]
    ast.parse(body)  # raises SyntaxError if the generated sandbox script is malformed


def test_shell_exec_execution_type_registered():
    assert "shell_command_exec" in SHELL_EXEC_EXECUTION_TYPES
