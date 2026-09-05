"""Tests for the command-line interface.

Only the commands that need no inference are exercised here. ``ask`` and
``chat`` are covered by the integration suite, which runs against a live
engine.
"""

from __future__ import annotations

import pytest

from minerva.cli import build_parser, main


class TestParser:
    def test_a_subcommand_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_ask_takes_a_prompt_and_a_thinking_level(self) -> None:
        args = build_parser().parse_args(["ask", "-t", "sol", "why is the sky blue?"])
        assert args.command == "ask"
        assert args.prompt == "why is the sky blue?"
        assert args.thinking == "sol"

    def test_hebrew_note_names_survive_argument_parsing(self) -> None:
        args = build_parser().parse_args(["ask", "-t", "סול", "hi"])
        assert args.thinking == "סול"

    def test_chat_accepts_the_generation_flags(self) -> None:
        args = build_parser().parse_args(["chat", "--show-thinking", "--no-tools"])
        assert args.show_thinking is True
        assert args.no_tools is True

    def test_pull_defaults_to_no_model(self) -> None:
        assert build_parser().parse_args(["pull"]).model is None


class TestInformationalCommands:
    def test_thinking_prints_the_whole_scale(self, capsys: pytest.CaptureFixture) -> None:
        assert main(["thinking", "--no-color"]) == 0
        out = capsys.readouterr().out
        for note in ("do", "re", "mi", "fa", "sol", "la", "si"):
            assert note in out
        for note in ("דו", "רה", "מי", "פה", "סול", "לה", "סי"):
            assert note in out

    def test_thinking_marks_the_extended_levels(self, capsys: pytest.CaptureFixture) -> None:
        main(["thinking", "--no-color"])
        lines = capsys.readouterr().out.splitlines()
        la_line = next(line for line in lines if " la " in line)
        do_line = next(line for line in lines if " do " in line)
        assert "yes" in la_line
        assert "no" in do_line

    def test_models_lists_swift(self, capsys: pytest.CaptureFixture) -> None:
        assert main(["models", "--no-color"]) == 0
        assert "swift" in capsys.readouterr().out

    def test_models_verbose_shows_the_full_spec(self, capsys: pytest.CaptureFixture) -> None:
        assert main(["models", "--no-color", "-v"]) == 0
        out = capsys.readouterr().out
        assert "Minerva Swift" in out
        assert "minerva:swift" in out, "Swift runs on Minerva's own engine"
        assert "9.9M" in out
        assert "do (דו)" in out, "a base model reports the silent thinking level"

    def test_tools_lists_the_builtins_with_their_parameters(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert main(["tools", "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "calculate" in out
        assert "expression" in out
        assert "current_time" in out

    def test_version_flag(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        assert "minerva" in capsys.readouterr().out


class TestErrorHandling:
    def test_an_unreachable_ollama_gives_a_nonzero_exit_and_advice(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Port 1 is reserved; nothing is listening there. This is a real
        # connection failure, not a simulated one.
        monkeypatch.setenv("MINERVA_ENGINE", "ollama")
        monkeypatch.setenv("MINERVA_OLLAMA_HOST", "http://127.0.0.1:1")
        assert main(["doctor", "--no-color"]) == 1
        out = capsys.readouterr().out
        assert "unreachable" in out
        assert "ollama serve" in out

    def test_missing_checkpoints_advise_training_not_downloading(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("MINERVA_ENGINE", "minerva")
        monkeypatch.setenv("MINERVA_CHECKPOINT_DIR", str(tmp_path / "empty"))
        assert main(["doctor", "--no-color"]) == 1
        out = capsys.readouterr().out
        assert "minerva train" in out
        assert "prepare-data" in out
        # The engine list legitimately names ollama; the *advice* must not.
        assert "ollama serve" not in out, "wrong advice for the native engine"

    def test_an_unknown_thinking_level_is_reported_not_traced(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert main(["ask", "-t", "octave", "hi"]) == 2
        assert "ThinkingLevelError" in capsys.readouterr().err

    def test_an_unknown_model_is_reported_not_traced(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert main(["ask", "-m", "hummingbird", "hi"]) == 2
        err = capsys.readouterr().err
        assert "ModelNotFoundError" in err
        assert "swift" in err


class TestTrainForwardsEveryTrainerFlag:
    """`minerva train` must not silently offer less than the module it wraps.

    CI failed with "unrecognized arguments: --seed 2027" because the CLI
    declared no --seed, even though the trainer had accepted one all along.
    The gap stays invisible until something actually uses the missing flag, so
    it is checked against the trainer's own source rather than a hand-written
    list that would drift the same way.
    """

    def test_every_trainer_flag_is_offered_by_the_cli(self) -> None:
        import argparse
        import inspect
        import re

        from minerva.cli import build_parser
        from minerva.training import trainer

        # The trainer builds its parser inside main(), so read the flags it
        # registers straight from that function's source.
        declared = set(
            re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', inspect.getsource(trainer.main))
        )
        assert "--seed" in declared, "sanity: the trainer really does take --seed"

        cli = build_parser()
        train = next(
            action.choices["train"]
            for action in cli._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        offered = {
            option
            for action in train._actions
            for option in action.option_strings
            if option.startswith("--")
        }
        missing = declared - offered
        assert not missing, f"minerva train does not offer trainer flags: {sorted(missing)}"
