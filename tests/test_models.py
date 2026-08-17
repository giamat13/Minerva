"""Tests for the model catalogue and the spec -> request translation.

The catalogue tests are parametrised over every registered model, so a new
model added to the registry is automatically held to the same standard.
"""

from __future__ import annotations

import pytest

from minerva.engines.base import SamplingParams
from minerva.engines.ollama import OllamaEngine
from minerva.errors import CapabilityError, ModelNotFoundError
from minerva.messages import Role, system, user
from minerva.models.base import MinervaModel, ModelSpec
from minerva.models.registry import get_spec, list_specs, model_names, register_spec
from minerva.models.swift import SWIFT
from minerva.thinking import ThinkingLevel
from minerva.tools.registry import ToolRegistry, default_registry


@pytest.fixture
def offline_engine() -> OllamaEngine:
    """A real engine object. Constructing it opens no socket."""
    return OllamaEngine(host="http://127.0.0.1:11434")


@pytest.fixture
def model(offline_engine: OllamaEngine) -> MinervaModel:
    return MinervaModel(SWIFT, offline_engine, tools=default_registry())


class TestCatalogue:
    def test_swift_is_registered(self) -> None:
        assert "swift" in model_names()
        assert get_spec("swift") is SWIFT

    def test_aliases_resolve(self) -> None:
        assert get_spec("minerva-swift") is SWIFT
        assert get_spec("SWIFT") is SWIFT

    def test_unknown_models_list_what_exists(self) -> None:
        with pytest.raises(ModelNotFoundError) as exc_info:
            get_spec("hummingbird")
        assert "swift" in str(exc_info.value)

    def test_catalogue_is_ordered_smallest_first(self) -> None:
        tiers = [spec.tier for spec in list_specs()]
        order = {"small": 0, "medium": 1, "large": 2}
        assert tiers == sorted(tiers, key=lambda tier: order.get(tier, 99))

    @pytest.mark.parametrize("spec", list_specs(), ids=lambda spec: spec.name)
    def test_every_spec_is_complete(self, spec: ModelSpec) -> None:
        assert spec.name == spec.name.lower().strip()
        assert spec.display_name
        assert spec.version
        assert len(spec.description) > 20
        assert spec.tier in {"small", "medium", "large"}
        assert spec.engine
        assert spec.engine_model, "a spec must say which weights it runs"
        assert spec.context_length > 0

    @pytest.mark.parametrize("spec", list_specs(), ids=lambda spec: spec.name)
    def test_thinking_ceiling_is_at_least_the_default(self, spec: ModelSpec) -> None:
        assert spec.max_thinking >= spec.default_thinking

    @pytest.mark.parametrize("spec", list_specs(), ids=lambda spec: spec.name)
    def test_engine_model_candidates_are_unique(self, spec: ModelSpec) -> None:
        candidates = spec.candidate_engine_models()
        assert len(candidates) == len(set(candidates))

    def test_a_model_can_be_registered_at_runtime(self) -> None:
        experimental = SWIFT.with_overrides(name="swift-test", display_name="Swift (test)")
        register_spec(experimental, aliases=("st",))
        try:
            assert get_spec("swift-test") is experimental
            assert get_spec("st") is experimental
        finally:
            from minerva.models import registry as registry_module

            registry_module._SPECS.pop("swift-test", None)
            registry_module._ALIASES.pop("st", None)


class TestSwiftSpec:
    def test_it_is_the_small_tier_and_targets_a_small_model(self) -> None:
        assert SWIFT.tier == "small"
        assert SWIFT.engine_model == "qwen3:1.7b"
        assert SWIFT.engine_model_fallbacks

    def test_it_supports_tools_and_thinking(self) -> None:
        assert SWIFT.supports_tools
        assert SWIFT.supports_thinking

    def test_its_ceiling_is_sol(self) -> None:
        # Deliberate: a 1.7B model gains nothing from LA/SI.
        assert SWIFT.max_thinking is ThinkingLevel.SOL
        assert SWIFT.default_thinking is ThinkingLevel.FA

    def test_it_ships_a_system_prompt(self) -> None:
        assert SWIFT.system_prompt
        assert "Swift" in SWIFT.system_prompt

    def test_its_sampling_defaults_are_set_deliberately(self) -> None:
        assert SWIFT.sampling.temperature == 0.7
        assert SWIFT.sampling.context_length == SWIFT.context_length


class TestThinkingResolution:
    def test_none_falls_back_to_the_model_default(self) -> None:
        assert SWIFT.resolve_thinking(None) is ThinkingLevel.FA

    def test_requests_above_the_ceiling_are_clamped_not_rejected(self) -> None:
        assert SWIFT.resolve_thinking("si") is ThinkingLevel.SOL
        assert SWIFT.resolve_thinking("la") is ThinkingLevel.SOL

    def test_requests_below_the_ceiling_pass_through(self) -> None:
        assert SWIFT.resolve_thinking("mi") is ThinkingLevel.MI
        assert SWIFT.resolve_thinking("דו") is ThinkingLevel.DO

    def test_a_model_without_thinking_always_resolves_to_do(self) -> None:
        plain = SWIFT.with_overrides(name="plain", supports_thinking=False)
        assert plain.resolve_thinking("sol") is ThinkingLevel.DO


class TestRequestBuilding:
    def test_the_system_prompt_is_prepended(self, model: MinervaModel) -> None:
        request = model.build_request([user("hi")], verify_installed=False)
        assert request.messages[0].role is Role.SYSTEM
        assert request.messages[0].content == SWIFT.system_prompt
        assert request.messages[1].content == "hi"

    def test_a_caller_supplied_system_prompt_is_not_duplicated(
        self, model: MinervaModel
    ) -> None:
        request = model.build_request(
            [system("you are a pirate"), user("hi")], verify_installed=False
        )
        roles = [m.role for m in request.messages]
        assert roles.count(Role.SYSTEM) == 1
        assert request.messages[0].content == "you are a pirate"

    def test_the_system_prompt_can_be_overridden(self, model: MinervaModel) -> None:
        request = model.build_request(
            [user("hi")], system_prompt="be terse", verify_installed=False
        )
        assert request.messages[0].content == "be terse"

    def test_an_empty_override_suppresses_the_system_prompt(
        self, model: MinervaModel
    ) -> None:
        request = model.build_request([user("hi")], system_prompt="", verify_installed=False)
        assert all(m.role is not Role.SYSTEM for m in request.messages)

    def test_tools_are_attached_as_specs(self, model: MinervaModel) -> None:
        request = model.build_request([user("hi")], verify_installed=False)
        names = [spec["function"]["name"] for spec in request.tools]
        assert "calculate" in names

    def test_no_tools_are_sent_when_the_registry_is_empty(
        self, model: MinervaModel
    ) -> None:
        request = model.build_request(
            [user("hi")], tools=ToolRegistry(), verify_installed=False
        )
        assert request.tools == []

    def test_the_thinking_profile_travels_with_the_request(
        self, model: MinervaModel
    ) -> None:
        request = model.build_request([user("hi")], thinking="mi", verify_installed=False)
        assert request.thinking is not None
        assert request.thinking.level is ThinkingLevel.MI

    def test_the_ceiling_is_applied_at_request_time(self, model: MinervaModel) -> None:
        request = model.build_request([user("hi")], thinking="si", verify_installed=False)
        assert request.thinking is not None
        assert request.thinking.level is ThinkingLevel.SOL

    def test_model_sampling_defaults_are_used(self, model: MinervaModel) -> None:
        request = model.build_request([user("hi")], verify_installed=False)
        assert request.sampling.temperature == 0.7

    def test_caller_sampling_overrides_the_model_default(self, model: MinervaModel) -> None:
        request = model.build_request(
            [user("hi")], sampling=SamplingParams(temperature=0.1), verify_installed=False
        )
        assert request.sampling.temperature == 0.1
        assert request.sampling.top_p == SWIFT.sampling.top_p, "unset fields are inherited"

    def test_a_model_that_cannot_use_tools_refuses_them(
        self, offline_engine: OllamaEngine
    ) -> None:
        spec = SWIFT.with_overrides(name="no-tools", supports_tools=False)
        model = MinervaModel(spec, offline_engine, tools=default_registry())
        with pytest.raises(CapabilityError, match="does not support tool calling"):
            model.build_request([user("hi")], verify_installed=False)


class TestEngineModelResolution:
    def test_without_verification_the_preferred_tag_is_used(
        self, model: MinervaModel
    ) -> None:
        assert model.resolve_engine_model(verify=False) == "qwen3:1.7b"


class TestSamplingMerge:
    def test_unset_override_fields_do_not_clobber(self) -> None:
        base = SamplingParams(temperature=0.7, top_p=0.8)
        merged = base.merged_with(SamplingParams(temperature=0.2))
        assert merged.temperature == 0.2
        assert merged.top_p == 0.8

    def test_merging_with_none_returns_the_original(self) -> None:
        base = SamplingParams(temperature=0.7)
        assert base.merged_with(None) is base

    def test_as_dict_omits_unset_fields(self) -> None:
        assert SamplingParams(temperature=0.5).as_dict() == {"temperature": 0.5}
