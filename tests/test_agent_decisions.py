"""
Tests for the deterministic core of the investigation agent -- the
parts that must NEVER depend on an LLM being available, since these
are the money-decision logic the whole architecture is built around.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.investigation_agent import (
    node_decide_action, node_assess_severity, _rule_based_root_cause,
    ROOT_CAUSE_TAXONOMY, SEVERITY_BLOCK, SEVERITY_REVIEW,
)
from agent.llm_provider import resolve_provider, call_llm


def test_decision_thresholds_are_exactly_as_documented():
    # These specific numbers are printed in the README -- if they drift,
    # the README is wrong and needs updating alongside the code.
    assert SEVERITY_BLOCK == 0.75
    assert SEVERITY_REVIEW == 0.35


def test_high_severity_blocks():
    state = {"severity_score": 0.9}
    result = node_decide_action(state)
    assert result["decision"] == "BLOCK"


def test_mid_severity_reviews():
    state = {"severity_score": 0.5}
    result = node_decide_action(state)
    assert result["decision"] == "REVIEW"


def test_low_severity_allows():
    state = {"severity_score": 0.1}
    result = node_decide_action(state)
    assert result["decision"] == "ALLOW"


def test_boundary_values_exactly_at_threshold():
    # >= BLOCK_THRESHOLD should block, not review
    assert node_decide_action({"severity_score": 0.75})["decision"] == "BLOCK"
    # >= REVIEW_THRESHOLD but < BLOCK should review
    assert node_decide_action({"severity_score": 0.35})["decision"] == "REVIEW"
    # just under REVIEW should allow
    assert node_decide_action({"severity_score": 0.349})["decision"] == "ALLOW"


def test_severity_score_is_bounded_0_to_1():
    # feed extreme inputs and confirm the score never exceeds 1.0
    state = {
        "transactions": [{"proba": 1.0, "billing_shipping_mismatch": 1, "is_new_customer": 1}] * 50,
        "affected_merchants": ["m1"] * 50,
        "device_correlation": {"reuse_ratio": 1.0, "unique_devices": 1},
        "ip_correlation": {"reuse_ratio": 1.0, "unique_ips": 1},
    }
    result = node_assess_severity(state)
    assert 0.0 <= result["severity_score"] <= 1.0


def test_rule_based_root_cause_stays_within_taxonomy():
    # sweep a grid of plausible inputs and confirm every output is a
    # valid taxonomy member -- this is the guardrail the LLM path also
    # relies on, so the rule-based fallback must uphold it too
    for device_reuse in [0.1, 0.5, 0.9]:
        for ip_reuse in [0.1, 0.5, 0.9]:
            for mismatch in [0.05, 0.3, 0.6]:
                for new_cust in [0.2, 0.6, 0.9]:
                    for proba in [0.05, 0.2, 0.5]:
                        label = _rule_based_root_cause(device_reuse, ip_reuse, mismatch, new_cust, proba)
                        assert label in ROOT_CAUSE_TAXONOMY, f"'{label}' not in taxonomy"


def test_no_provider_configured_returns_none_not_exception(monkeypatch):
    # Clear both env vars so this stays hermetic on a machine that has a key set.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    text, meta = call_llm("test prompt", api_key=None, provider=None)
    assert text is None
    assert meta["ok"] is False
    assert meta["provider"] is None


def test_provider_resolution_prefers_explicit_over_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env-key")
    provider, key = resolve_provider(api_key="sk-or-explicit-key", provider="openrouter")
    assert provider == "openrouter"
    assert key == "sk-or-explicit-key"


def test_provider_resolution_falls_back_to_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env-key")
    provider, key = resolve_provider(api_key=None, provider=None)
    assert provider == "openai"
    assert key == "sk-openai-env-key"
