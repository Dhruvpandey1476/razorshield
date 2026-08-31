"""
Tests for backend/razorpay_webhook.py, specifically the signature
verification, since that's the securityvcritical claim: an unsigned
or tampered request must NEVER be scored, an incorrectly signed
request must be rejected and a correctly signed request must pass.
"""
import hashlib
import hmac
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend import razorpay_webhook as rzp


SECRET = "test_secret_for_ci_only"


def _sign(raw_body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()


def test_valid_signature_verifies():
    raw_body = json.dumps({"event": "payment.captured", "payload": {}}).encode()
    sig = _sign(raw_body)
    assert rzp.verify_signature(raw_body, sig, SECRET) is True


def test_tampered_body_rejected():
    raw_body = json.dumps({"event": "payment.captured", "amount": 100}).encode()
    sig = _sign(raw_body)
    tampered_body = json.dumps({"event": "payment.captured", "amount": 999999}).encode()
    assert rzp.verify_signature(tampered_body, sig, SECRET) is False


def test_wrong_secret_rejected():
    raw_body = json.dumps({"event": "payment.captured"}).encode()
    sig = _sign(raw_body, secret="wrong_secret")
    assert rzp.verify_signature(raw_body, sig, SECRET) is False


def test_missing_signature_rejected():
    raw_body = json.dumps({"event": "payment.captured"}).encode()
    assert rzp.verify_signature(raw_body, "", SECRET) is False
    assert rzp.verify_signature(raw_body, None, SECRET) is False


def test_missing_secret_rejected():
    raw_body = json.dumps({"event": "payment.captured"}).encode()
    sig = _sign(raw_body)
    assert rzp.verify_signature(raw_body, sig, "") is False
    assert rzp.verify_signature(raw_body, sig, None) is False


def test_parse_webhook_event_extracts_entity():
    payload = {
        "event": "payment.captured",
        "payload": {"payment": {"entity": {"id": "pay_test123", "amount": 50000}}},
    }
    event, entity = rzp.parse_webhook_event(payload)
    assert event == "payment.captured"
    assert entity["id"] == "pay_test123"
    assert entity["amount"] == 50000


def test_parse_webhook_event_handles_missing_entity():
    event, entity = rzp.parse_webhook_event({"event": "some.other.event"})
    assert event == "some.other.event"
    assert entity == {}


def test_map_payment_entity_converts_paise_to_inr():
    entity = {"id": "pay_x", "amount": 150000, "method": "upi", "email": "a@b.com", "created_at": 1700000000}
    features = rzp.map_payment_entity_to_features(entity)
    assert features["amount_inr"] == 1500.0


def test_map_payment_entity_flags_new_customer_once():
    entity = {"id": "pay_y", "amount": 1000, "method": "card", "email": "unique_test_customer@example.com", "created_at": 1700000001}
    f1 = rzp.map_payment_entity_to_features(entity)
    entity2 = {**entity, "id": "pay_z", "created_at": 1700000002}
    f2 = rzp.map_payment_entity_to_features(entity2)
    assert f1["is_new_customer"] == 1
    assert f2["is_new_customer"] == 0  # seen this email before, in the same process
