"""Every command must parse and reach its own handler."""

import pytest

from specdec.cli import build_parser


@pytest.mark.parametrize(
    ("arguments", "handler"),
    [
        (["generate", "hello"], "run_generate"),
        (["report", "--run", "runs/x"], "run_report"),
        (["smoke"], "run_smoke_command"),
        (["prepare-data"], "run_prepare_data"),
        (["arm", "--arm", "{}", "--workload", "w", "--out", "x"], "run_one_arm"),
        (["final", "--out", "x"], "run_frozen"),
    ],
)
def test_commands_reach_their_handlers(arguments, handler):
    assert build_parser().parse_args(arguments).handler.__name__ == handler
