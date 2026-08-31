"""FastAPI app: serves the frontend at / and the risk API alongside it."""
import os
import sys
import json
import time
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.investigation_agent import investigate_spike, investigate_merchant_bucket, AUDIT_LOG_PATH
from backend import razorpay_webhook as rzp

ART_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")

app = FastAPI(
    title="RazorShield API",
    description="Agentic AI risk manager -- fraud-spike detection and investigation",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Artifacts are served directly so the frontend can reference generated
# images (the SHAP plot) by URL without a dedicated endpoint.
if os.path.isdir(ART_DIR):
    app.mount("/artifacts", StaticFiles(directory=ART_DIR), name="artifacts")

STATIC_DIR = os.path.join(APP_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class InvestigateRequest(BaseModel):
    """Per-request LLM override so the frontend can pass a user-supplied key."""
    llm_api_key: Optional[str] = None
    llm_provider: Optional[str] = None   # "openrouter" | "openai"
    llm_model: Optional[str] = None


def _load_json(name):
    path = os.path.join(ART_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{name} not found -- run ./run_pipeline.sh first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifacts_present": os.path.exists(os.path.join(ART_DIR, "metrics.json")),
        "model_present": os.path.exists(os.path.join(ART_DIR, "xgb_fraud_model.joblib")),
    }


@app.get("/")
def root():
    index_path = os.path.join(APP_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "service": "RazorShield",
        "note": "app/index.html not found -- API is live, but no frontend to serve.",
        "endpoints": ["/metrics", "/spikes", "/spikes/{spike_id}/investigate",
                      "/audit-trail", "/transactions/flagged", "/dashboard-data", "/control-event"],
    }


@app.get("/metrics")
def get_metrics():
    return _load_json("metrics.json")


@app.get("/spikes")
def list_spikes():
    return _load_json("spike_event_detection_summary.json")


@app.get("/control-event")
def control_event():
    return _load_json("control_event_summary.json")


@app.post("/spikes/{spike_id}/investigate")
def investigate(spike_id: str, req: InvestigateRequest = InvestigateRequest()):
    """Run the investigation agent on a spike event and return its audit entry."""
    try:
        return investigate_spike(
            spike_id,
            llm_api_key=req.llm_api_key, llm_provider=req.llm_provider, llm_model=req.llm_model,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/merchants/{merchant_id}/investigate")
def investigate_merchant(merchant_id: str, bucket: str, req: InvestigateRequest = InvestigateRequest()):
    """Investigate an arbitrary merchant/time-bucket, for ad-hoc analyst queries."""
    try:
        return investigate_merchant_bucket(
            merchant_id, bucket,
            llm_api_key=req.llm_api_key, llm_provider=req.llm_provider, llm_model=req.llm_model,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/audit-trail")
def audit_trail(limit: int = 50):
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    with open(AUDIT_LOG_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    return entries[-limit:]


@app.get("/transactions/flagged")
def flagged_transactions(limit: int = 100):
    path = os.path.join(ART_DIR, "combined_predictions.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="run the pipeline first")
    df = pd.read_csv(path)
    flagged = df[df["combined_pred"] == 1].sort_values("proba", ascending=False)
    cols = ["payment_id", "merchant_id", "merchant_category", "timestamp",
            "amount_inr", "method", "proba", "is_spike_bucket", "spike_event_id", "is_fraud"]
    return json.loads(flagged[cols].head(limit).to_json(orient="records"))


@app.get("/timeline")
def timeline():
    """15-minute bucketed volume and anomaly score for the test window."""
    path = os.path.join(ART_DIR, "timeline_buckets.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="run the pipeline first")
    df = pd.read_csv(path)
    return {
        "labels": df["bucket"].tolist(),
        "txn_count": df["txn_count"].tolist(),
        "spike_score": df["max_spike_score"].round(2).tolist(),
    }


@app.get("/dashboard-data")
def dashboard_data():
    """Everything the frontend needs to hydrate, in one round trip."""
    control_path = os.path.join(ART_DIR, "control_event_summary.json")
    return {
        "metrics": _load_json("metrics.json"),
        "spikes": _load_json("spike_event_detection_summary.json"),
        "shap_top_features": _load_json("shap_global_importance.json")[:8],
        "control_event": _load_json("control_event_summary.json") if os.path.exists(control_path) else None,
    }


@app.get("/llm-status")
def llm_status():
    """Whether the SERVER has a default provider."""
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    return {
        "server_default_provider": "openrouter" if has_openrouter else ("openai" if has_openai else None),
        "server_has_openrouter_key": has_openrouter,
        "server_has_openai_key": has_openai,
    }


@app.post("/webhooks/razorpay")
async def razorpay_webhook_receiver(request: Request, x_razorpay_signature: str = Header(default=None)):
    """Receive, verify and score a Razorpay payment webhook."""
    raw_body = await request.body()
    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(status_code=501, detail=(
            "RAZORPAY_WEBHOOK_SECRET is not set. Set it to the same secret configured "
            "in your Razorpay Dashboard webhook settings, then retry."
        ))
    if not rzp.verify_signature(raw_body, x_razorpay_signature or "", secret):
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")

    event, entity = rzp.parse_webhook_event(json.loads(raw_body))
    if not entity:
        return {"received": True, "event": event, "note": "No payment entity in payload."}

    features = rzp.map_payment_entity_to_features(entity)
    proba, ok, note = rzp.score_transaction(features)

    result = {
        "received": True,
        "event": event,
        "payment_id": entity.get("id"),
        "amount_inr": features["amount_inr"],
        "method": features["method"],
        "scored": ok,
    }
    if ok:
        result["fraud_probability"] = round(proba, 4)
        result["note"] = ("Device/IP features were unavailable from this payload and "
                          "defaulted -- see backend/razorpay_webhook.py.")
    else:
        result["note"] = note
    return result


@app.post("/webhooks/razorpay/simulate")
def simulate_razorpay_webhook(amount_inr: float = 999.0, method: str = "upi",
                              email: str = "test@example.com"):
    """Build and correctly sign a payment.captured payload."""
    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "demo_secret_for_local_testing_only")
    now = int(time.time())
    payload = {
        "event": "payment.captured",
        "payload": {"payment": {"entity": {
            "id": f"pay_SIMULATED{now}",
            "amount": int(amount_inr * 100),
            "currency": "INR",
            "method": method,
            "email": email,
            "contact": "+919999999999",
            "order_id": f"order_SIMULATED{now}",
            "created_at": now,
            "notes": {},
        }}},
    }
    raw_body = json.dumps(payload).encode("utf-8")
    signature = rzp.hmac.new(secret.encode(), raw_body, rzp.hashlib.sha256).hexdigest()
    return {"raw_body": payload, "signature": signature}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
