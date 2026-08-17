"""Tests for configuration loading and precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from minerva.config import MinervaConfig, load_config
from minerva.errors import ConfigurationError
from minerva.thinking import ThinkingLevel


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip inherited MINERVA_* variables so tests start from a known state."""
    import os

    for key in list(os.environ):
        if key.startswith("MINERVA_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run in an empty directory so a stray ./minerva.toml cannot leak in."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    return tmp_path


class TestDefaults:
    def test_sensible_out_of_the_box_values(self) -> None:
        config = load_config()
        assert config.engine == "ollama"
        assert config.default_model == "swift"
        assert config.thinking_level is ThinkingLevel.FA
        assert config.ollama_host.startswith("http")
        assert config.source_file is None

    def test_invalid_values_are_rejected_at_construction(self) -> None:
        with pytest.raises(ConfigurationError):
            MinervaConfig(request_timeout=0)
        with pytest.raises(ConfigurationError):
            MinervaConfig(max_tool_iterations=0)

    def test_replace_produces_a_modified_copy(self) -> None:
        config = MinervaConfig()
        other = config.replace(default_model="swift", thinking_level="sol")
        assert other.thinking_level is ThinkingLevel.SOL
        assert config.thinking_level is ThinkingLevel.FA


class TestEnvironment:
    def test_env_vars_are_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINERVA_OLLAMA_HOST", "http://gpu-box:11434")
        monkeypatch.setenv("MINERVA_DEFAULT_MODEL", "swift")
        monkeypatch.setenv("MINERVA_THINKING_LEVEL", "sol")
        monkeypatch.setenv("MINERVA_ENABLE_TOOLS", "false")
        monkeypatch.setenv("MINERVA_REQUEST_TIMEOUT", "42.5")

        config = load_config()
        assert config.ollama_host == "http://gpu-box:11434"
        assert config.thinking_level is ThinkingLevel.SOL
        assert config.enable_tools is False
        assert config.request_timeout == 42.5

    def test_a_hebrew_note_works_in_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MINERVA_THINKING_LEVEL", "סי")
        assert load_config().thinking_level is ThinkingLevel.SI

    @pytest.mark.parametrize(
        ("text", "expected"), [("1", True), ("yes", True), ("off", False), ("FALSE", False)]
    )
    def test_boolean_spellings(
        self, monkeypatch: pytest.MonkeyPatch, text: str, expected: bool
    ) -> None:
        monkeypatch.setenv("MINERVA_SHOW_THINKING", text)
        assert load_config().show_thinking is expected

    def test_a_non_boolean_is_a_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINERVA_STREAM", "sometimes")
        with pytest.raises(ConfigurationError, match="expected a boolean"):
            load_config()

    def test_a_non_numeric_timeout_is_a_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MINERVA_REQUEST_TIMEOUT", "soon")
        with pytest.raises(ConfigurationError, match="expected a number"):
            load_config()


class TestConfigFile:
    def test_a_flat_toml_file_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "minerva.toml"
        path.write_text('thinking_level = "la"\nollama_host = "http://box:11434"\n')
        config = load_config()
        assert config.thinking_level is ThinkingLevel.LA
        assert config.ollama_host == "http://box:11434"
        assert config.source_file == Path("minerva.toml")

    def test_a_minerva_table_is_also_accepted(self, tmp_path: Path) -> None:
        (tmp_path / "minerva.toml").write_text('[minerva]\nthinking_level = "si"\n')
        assert load_config().thinking_level is ThinkingLevel.SI

    def test_the_local_file_wins_over_the_shared_one(self, tmp_path: Path) -> None:
        (tmp_path / "minerva.toml").write_text('thinking_level = "do"\n')
        (tmp_path / "minerva.local.toml").write_text('thinking_level = "si"\n')
        assert load_config().thinking_level is ThinkingLevel.SI

    def test_the_environment_beats_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "minerva.toml").write_text('thinking_level = "do"\n')
        monkeypatch.setenv("MINERVA_THINKING_LEVEL", "sol")
        assert load_config().thinking_level is ThinkingLevel.SOL

    def test_hyphenated_keys_are_accepted(self, tmp_path: Path) -> None:
        (tmp_path / "minerva.toml").write_text('default-model = "swift"\n')
        assert load_config().default_model == "swift"

    def test_unknown_keys_are_ignored(self, tmp_path: Path) -> None:
        # A config file may carry settings for another version of Minerva.
        (tmp_path / "minerva.toml").write_text('future_setting = 3\nthinking_level = "mi"\n')
        assert load_config().thinking_level is ThinkingLevel.MI

    def test_malformed_toml_is_reported_clearly(self, tmp_path: Path) -> None:
        (tmp_path / "minerva.toml").write_text("this is not = = toml\n")
        with pytest.raises(ConfigurationError, match="not valid TOML"):
            load_config()

    def test_an_explicit_missing_path_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_config(tmp_path / "nope.toml")

    def test_minerva_config_env_var_selects_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "elsewhere.toml"
        path.write_text('thinking_level = "re"\n')
        monkeypatch.setenv("MINERVA_CONFIG", str(path))
        assert load_config().thinking_level is ThinkingLevel.RE
