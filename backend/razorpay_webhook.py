"""Razorpay webhook receiver: signature verification, payload parsing, scoring."""
import os
import json
import hmac
import hashlib
import time
from collections import deque, defaultdict
from datetime import datetime, timezone

import joblib
import pandas as pd

ART_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
WINDOW_SECONDS = 24 * 3600
RISK_TIERS = {"low": 0, "medium": 1, "high": 2}

_recent_by_key = defaultdict(deque)   # email/contact -> deque of unix timestamps
_seen_customers = set()
_model = None
_feature_cols = None


def _load_model():
    global _model, _feature_cols
    if _model is not None:
        return _model, _feature_cols

    model_path = os.path.join(ART_DIR, "xgb_fraud_model.joblib")
    cols_path = os.path.join(ART_DIR, "feature_columns.json")
    if os.path.exists(model_path) and os.path.exists(cols_path):
        _model = joblib.load(model_path)
        with open(cols_path) as f:
            _feature_cols = json.load(f)
    return _model, _feature_cols


def verify_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    """HMAC-SHA256 over the raw body, constant-time compared to the header."""
    if not signature or not secret:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def parse_webhook_event(payload: dict):
    """-> (event_name, payment_entity) from {event, payload:{payment:{entity}}}."""
    event = payload.get("event", "unknown")
    entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
    return event, entity


def _velocity(key: str, now_ts: float) -> int:
    """Count of prior events for this key in the trailing window, then record."""
    dq = _recent_by_key[key]
    cutoff = now_ts - WINDOW_SECONDS
    while dq and dq[0] < cutoff:
        dq.popleft()
    count = len(dq)
    dq.append(now_ts)
    return count


def map_payment_entity_to_features(entity: dict, merchant_risk_tier: str = "medium") -> dict:
    """Map a Razorpay payment entity onto the model's feature set."""
    email = entity.get("email") or ""
    contact = entity.get("contact") or ""
    key = email or contact or entity.get("id", "unknown")
    now_ts = entity.get("created_at", time.time())

    is_new = key not in _seen_customers
    _seen_customers.add(key)
    recent_count = _velocity(key, now_ts)

    dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)

    return {
        "amount_inr": (entity.get("amount", 0) or 0) / 100.0,   # paise -> rupees
        "customer_txn_count_24h": recent_count,
        "is_new_customer": int(is_new),
        "billing_shipping_mismatch": 0,
        "hour_of_day": dt.hour,
        "is_night": int(dt.hour in (1, 2, 3, 4)),
        "device_txn_count_1h": 0,           # unavailable from webhook payload
        "ip_txn_count_1h": 0,               # unavailable from webhook payload
        "merchant_risk_score": RISK_TIERS.get(merchant_risk_tier, 1),
        "amount_zscore_for_merchant": 0.0,  # needs merchant history, not per-event
        "method": entity.get("method", "unknown"),
        "merchant_category": "unknown",
    }


def score_transaction(features: dict):
    """-> (proba, ok, note). ok=False when the model hasn't been trained yet."""
    model, feature_cols = _load_model()
    if model is None:
        return None, False, "Model not found -- run ./run_pipeline.sh first to train it."

    row = {c: 0 for c in feature_cols}
    for k, v in features.items():
        if k in row:
            row[k] = v
    for col in (f"method_{features.get('method')}",
                f"merchant_category_{features.get('merchant_category')}"):
        if col in row:
            row[col] = 1

    X = pd.DataFrame([row])[feature_cols]
    return float(model.predict_proba(X)[:, 1][0]), True, None
