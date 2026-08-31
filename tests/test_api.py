"""
API-level tests. These require the pipeline to have been run at least
once (artifacts/ populated) -- CI runs ./run_pipeline.sh before pytest,
see .github/workflows/ci.yml.
"""
import os
import sys
import json
import hmac
import hashlib

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.main import app

client = TestClient(app)

ART_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
pipeline_has_run = os.path.exists(os.path.join(ART_DIR, "metrics.json"))
requires_pipeline = pytest.mark.skipif(
    not pipeline_has_run, reason="artifacts/ not populated -- run ./run_pipeline.sh first"
)


def test_health_endpoint_always_works():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root_serves_frontend_or_fallback():
    r = client.get("/")
    assert r.status_code == 200


@requires_pipeline
def test_metrics_endpoint_shape():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "precision" in data
    assert "combined_system" in data
    assert 0.0 <= data["precision"] <= 1.0


@requires_pipeline
def test_spikes_endpoint_returns_four_events():
    r = client.get("/spikes")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 4


@requires_pipeline
def test_control_event_is_not_fraud():
    r = client.get("/control-event")
    assert r.status_code == 200
    # the control event's whole purpose is that it's genuinely not fraud
    assert "NOT fraud" in r.json()["description"] or "not fraud" in r.json()["description"].lower()


@requires_pipeline
def test_investigate_endpoint_returns_valid_decision():
    r = client.post("/spikes/SPK-001/investigate", json={})
    assert r.status_code == 200
    data = r.json()
    assert data["decision"] in ("ALLOW", "REVIEW", "BLOCK")
    assert 0.0 <= data["severity_score"] <= 1.0


@requires_pipeline
def test_investigate_control_event_allows():
    # this is the specificity check that matters most -- a genuine
    # legitimate spike must not get blocked
    r = client.post("/spikes/CTRL-001/investigate", json={})
    assert r.status_code == 200
    assert r.json()["decision"] == "ALLOW"


def test_webhook_rejects_missing_secret(monkeypatch):
    monkeypatch.delenv("RAZORPAY_WEBHOOK_SECRET", raising=False)
    r = client.post("/webhooks/razorpay", content=b'{"event":"payment.captured"}',
                     headers={"X-Razorpay-Signature": "anything"})
    assert r.status_code == 501


def test_webhook_rejects_invalid_signature(monkeypatch):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "ci_test_secret")
    r = client.post("/webhooks/razorpay", content=b'{"event":"payment.captured"}',
                     headers={"X-Razorpay-Signature": "not_a_real_signature"})
    assert r.status_code == 400


def test_webhook_accepts_correctly_signed_request(monkeypatch):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "ci_test_secret")
    payload = {
        "event": "payment.captured",
        "payload": {"payment": {"entity": {
            "id": "pay_ci_test", "amount": 100000, "method": "upi",
            "email": "ci_test@example.com", "created_at": 1700000000,
        }}}
    }
    raw_body = json.dumps(payload).encode()
    sig = hmac.new(b"ci_test_secret", raw_body, hashlib.sha256).hexdigest()
    r = client.post("/webhooks/razorpay", content=raw_body,
                     headers={"X-Razorpay-Signature": sig, "Content-Type": "application/json"})
    assert r.status_code == 200
    assert r.json()["received"] is True
