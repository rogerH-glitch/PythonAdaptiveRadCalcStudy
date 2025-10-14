from src.cli_parser import parse_args, DEFAULT_EVAL_MODE


def test_eval_mode_defaults_to_grid_when_not_specified():
    args = parse_args([])  # simulate no flags given
    assert args.eval_mode == DEFAULT_EVAL_MODE == "grid"


def test_eval_mode_respects_explicit_choice_center():
    args = parse_args(["--eval-mode", "center"])
    assert args.eval_mode == "center"


def test_eval_mode_respects_explicit_choice_search():
    args = parse_args(["--eval-mode", "search"])
    assert args.eval_mode == "search"
