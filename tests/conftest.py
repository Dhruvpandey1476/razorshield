"""Shared fixtures.

Tests that run the agent must not touch the committed artifacts: the audit
trail is a shipped file, and an appended entry changes what evaluation/report.py
renders, which then fails the drift guards. Redirect it to a temp file, and
clear provider keys so a developer's environment cannot make tests hit a real
API or change their results.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def isolate_agent_side_effects(tmp_path, monkeypatch):
    from agent import investigation_agent
    monkeypatch.setattr(investigation_agent, "AUDIT_LOG_PATH",
                        str(tmp_path / "audit_trail.jsonl"))
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENROUTER_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
