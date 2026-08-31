"""LangGraph investigation agent."""
import os
import json
import uuid
from datetime import datetime, timezone
from typing import TypedDict, Optional, List, Dict, Any

import pandas as pd
from langgraph.graph import StateGraph, END

try:
    from agent.llm_provider import call_llm
except ImportError:  # running the file directly rather than as a package
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llm_provider import call_llm

ART_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
AUDIT_LOG_PATH = os.path.join(ART_DIR, "audit_trail.jsonl")

SEVERITY_BLOCK = 0.75
SEVERITY_REVIEW = 0.35

ROOT_CAUSE_TAXONOMY = [
    "Card testing / BIN attack",
    "Account takeover ring",
    "Promo or coupon abuse",
    "Synthetic identity fraud",
    "Data breach fallout (compromised card data)",
    "Legitimate demand spike (false-positive candidate)",
    "Insufficient evidence to classify",
]

# Signatures go into the prompt with the labels. Without them the classifier
# confused card testing with account takeover, which sit at opposite ends of
# the new-customer rate: card testing uses throwaway identities, takeover uses
# aged real ones.
ROOT_CAUSE_SIGNATURES = {
    "Card testing / BIN attack":
        "Very high device AND IP reuse (few endpoints, many attempts), HIGH new-customer "
        "rate (throwaway identities), often small ticket sizes. Automated probing for live cards.",
    "Account takeover ring":
        "LOW new-customer rate -- the accounts are real and AGED, which is the whole point. "
        "Devices/IPs are usually DISTRIBUTED, not shared. Often elevated ticket sizes as "
        "accounts are drained. If the new-customer rate is high, this is NOT account takeover.",
    "Promo or coupon abuse":
        "Many new accounts, low ticket sizes, low billing mismatch, clustered on one merchant "
        "or campaign. Financially motivated but not card fraud.",
    "Synthetic identity fraud":
        "HIGH new-customer rate AND high billing/shipping mismatch, but device/IP reuse is LOW "
        "-- each fabricated identity brings its own device, so there is no shared-endpoint signal.",
    "Data breach fallout (compromised card data)":
        "Sudden burst of unrelated cards across many merchants, moderate reuse, mismatch "
        "elevated, no single shared origin.",
    "Legitimate demand spike (false-positive candidate)":
        "Volume is anomalous but everything else is ordinary: low device/IP reuse, low mismatch, "
        "low model probability, normal new-customer mix. A busy merchant, not an attack.",
    "Insufficient evidence to classify":
        "Use this when the evidence does not clearly match any signature above. Preferred over "
        "guessing -- an uncertain label in an audit trail is worse than an honest abstention.",
}


class InvestigationState(TypedDict, total=False):
    spike_id: Optional[str]
    merchant_id: Optional[str]
    bucket: Optional[str]
    transactions: List[Dict[str, Any]]
    affected_merchants: List[str]
    device_correlation: Dict[str, Any]
    ip_correlation: Dict[str, Any]
    shap_reasons: Dict[str, List[Dict[str, Any]]]
    severity_score: float
    severity_reasoning: List[str]
    root_cause_hypothesis: Dict[str, Any]
    decision: str
    decision_reasoning: str
    analyst_narrative: Optional[str]
    audit_entry: Dict[str, Any]
    llm_api_key: Optional[str]
    llm_provider: Optional[str]
    llm_model: Optional[str]


def _llm_args(state):
    return {
        "api_key": state.get("llm_api_key"),
        "provider": state.get("llm_provider"),
        "model": state.get("llm_model"),
    }


def _ratios(state):
    """Evidence shared by the severity scorer and the root-cause classifier."""
    txns = state.get("transactions", [])
    n = len(txns) or 1
    return {
        "n": n,
        "avg_proba": sum(t.get("proba", 0) for t in txns) / n,
        "mismatch": sum(t.get("billing_shipping_mismatch", 0) for t in txns) / n,
        "new_cust": sum(t.get("is_new_customer", 0) for t in txns) / n,
        "device_reuse": state.get("device_correlation", {}).get("reuse_ratio", 0),
        "ip_reuse": state.get("ip_correlation", {}).get("reuse_ratio", 0),
    }


def node_ingest(state: InvestigationState) -> InvestigationState:
    df = pd.read_csv(os.path.join(ART_DIR, "combined_predictions.csv"))
    if state.get("spike_id"):
        subset = df[df["spike_event_id"] == state["spike_id"]]
    elif state.get("merchant_id") and state.get("bucket"):
        subset = df[(df["merchant_id"] == state["merchant_id"]) & (df["bucket"] == state["bucket"])]
    else:
        subset = df[df["combined_pred"] == 1].head(50)

    state["transactions"] = subset.to_dict(orient="records")
    return state


def node_correlate_merchants(state: InvestigationState) -> InvestigationState:
    state["affected_merchants"] = sorted({t["merchant_id"] for t in state.get("transactions", [])})
    return state


def node_correlate_device_ip(state: InvestigationState) -> InvestigationState:
    txns = state.get("transactions", [])
    n = len(txns) or 1
    for field, key in (("device_id", "device_correlation"), ("ip_address", "ip_correlation")):
        unique = len({t[field] for t in txns})
        state[key] = {
            "n_transactions": n,
            ("unique_devices" if field == "device_id" else "unique_ips"): unique,
            "reuse_ratio": round(1 - (unique / n), 3),
        }
    return state


def node_pull_shap_reasons(state: InvestigationState) -> InvestigationState:
    shap_path = os.path.join(ART_DIR, "per_txn_shap_explanations.json")
    per_txn = {}
    if os.path.exists(shap_path):
        with open(shap_path, encoding="utf-8") as f:
            per_txn = json.load(f)

    reasons = {t["payment_id"]: per_txn[t["payment_id"]]
               for t in state.get("transactions", []) if t["payment_id"] in per_txn}

    # Nothing in this cluster made the per transaction SHAP sample, fall back
    # to the global drivers so the audit entry still carries an explanation.
    global_path = os.path.join(ART_DIR, "shap_global_importance.json")
    if not reasons and os.path.exists(global_path):
        with open(global_path, encoding="utf-8") as f:
            reasons = {"_global_top_drivers": json.load(f)[:5]}

    state["shap_reasons"] = reasons
    return state


def node_assess_severity(state: InvestigationState) -> InvestigationState:
    r = _ratios(state)
    n_merchants = len(state.get("affected_merchants", []))

    score = (
        0.30 * min(r["avg_proba"] * 4, 1.0)
        + 0.20 * r["device_reuse"]
        + 0.20 * r["ip_reuse"]
        + 0.15 * r["mismatch"]
        + 0.10 * r["new_cust"]
        + 0.05 * min(n_merchants / 10, 1.0)
    )
    state["severity_score"] = round(min(score, 1.0), 3)
    state["severity_reasoning"] = [
        f"Average model fraud probability across {r['n']} transactions: {r['avg_proba']:.3f}",
        f"Device fingerprint reuse ratio: {r['device_reuse']:.1%} "
        f"({state['device_correlation']['unique_devices']} unique devices across {r['n']} txns)",
        f"IP address reuse ratio: {r['ip_reuse']:.1%} "
        f"({state['ip_correlation']['unique_ips']} unique IPs across {r['n']} txns)",
        f"Billing/shipping mismatch rate: {r['mismatch']:.1%}",
        f"New-customer rate: {r['new_cust']:.1%}",
        f"Spans {n_merchants} distinct merchant(s)",
    ]
    return state


def _rule_based_root_cause(device_reuse, ip_reuse, mismatch_ratio, new_cust_ratio, avg_proba):
    """Deterministic classifier used when no LLM provider is configured."""
    if device_reuse > 0.85 and ip_reuse > 0.85 and new_cust_ratio > 0.7:
        return "Card testing / BIN attack"
    if mismatch_ratio > 0.45 and new_cust_ratio > 0.6:
        return "Synthetic identity fraud"
    if device_reuse < 0.3 and ip_reuse < 0.3 and avg_proba < 0.15:
        return "Legitimate demand spike (false-positive candidate)"
    if new_cust_ratio > 0.5 and mismatch_ratio < 0.2 and avg_proba > 0.10:
        return "Account takeover ring"
    return "Insufficient evidence to classify"


def _parse_label(text):
    """Pull LABEL/JUSTIFICATION out of the model's reply."""
    def line_after(prefix):
        line = next((l for l in text.splitlines() if l.startswith(prefix)), "")
        return line.replace(prefix, "").strip()
    return line_after("LABEL:"), line_after("JUSTIFICATION:")


def node_classify_root_cause(state: InvestigationState) -> InvestigationState:
    """Label why the cluster looks fraudulent."""
    r = _ratios(state)
    fallback = _rule_based_root_cause(
        r["device_reuse"], r["ip_reuse"], r["mismatch"], r["new_cust"], r["avg_proba"])

    taxonomy = "\n".join(
        f"- {c}\n    signature: {ROOT_CAUSE_SIGNATURES[c]}" for c in ROOT_CAUSE_TAXONOMY)
    prompt = (
        "You are a fraud-pattern classifier. Pick exactly ONE label from the fixed "
        "list below and copy it VERBATIM. Each label is followed by the signature "
        "that defines it -- match the evidence against those signatures, not "
        f"against the label name alone:\n{taxonomy}\n\n"
        "If the evidence contradicts a signature on a key ratio, that label is wrong. "
        "If nothing matches cleanly, choose 'Insufficient evidence to classify' "
        "rather than guessing.\n\n"
        "Evidence:\n"
        f"- Average model fraud probability: {r['avg_proba']:.3f}\n"
        f"- Device fingerprint reuse ratio: {r['device_reuse']:.2f}\n"
        f"- IP address reuse ratio: {r['ip_reuse']:.2f}\n"
        f"- Billing/shipping mismatch rate: {r['mismatch']:.2f}\n"
        f"- New-customer rate: {r['new_cust']:.2f}\n"
        f"- Transactions examined: {r['n']}\n\n"
        "Respond in exactly this format:\n"
        "LABEL: <one label from the list>\n"
        "JUSTIFICATION: <one sentence citing only the numbers above>"
    )
    text, meta = call_llm(prompt, max_tokens=200, **_llm_args(state))

    if not meta["ok"]:
        state["root_cause_hypothesis"] = {
            "label": fallback,
            "justification": f"[Rule-based fallback -- {meta['error']}] "
                             f"device_reuse={r['device_reuse']:.2f}, ip_reuse={r['ip_reuse']:.2f}, "
                             f"mismatch_ratio={r['mismatch']:.2f}, new_cust_ratio={r['new_cust']:.2f}",
            "source": "rule_based_fallback",
            "llm_meta": meta,
        }
        return state

    label, justification = _parse_label(text)
    if label not in ROOT_CAUSE_TAXONOMY:
        state["root_cause_hypothesis"] = {
            "label": "Insufficient evidence to classify",
            "justification": f"[Guardrail: {meta['provider']}/{meta['model']} output "
                             f"'{label}' not in allowed taxonomy, rejected and "
                             "replaced with safe default]",
            "source": "llm_output_rejected",
            "llm_meta": meta,
        }
    else:
        state["root_cause_hypothesis"] = {
            "label": label,
            "justification": justification or "(no justification returned)",
            "source": f"llm_{meta['provider']}_{meta['model']}",
            "llm_meta": meta,
        }
    return state


def node_decide_action(state: InvestigationState) -> InvestigationState:
    score = state.get("severity_score", 0)
    if score >= SEVERITY_BLOCK:
        state["decision"] = "BLOCK"
        state["decision_reasoning"] = (
            f"Severity score {score:.3f} >= BLOCK threshold ({SEVERITY_BLOCK}). "
            "High device/IP reuse combined with elevated model confidence indicates "
            "a coordinated pattern. Recommend bounded temporary block on the "
            "involved device/IP cluster and mandatory human review before release."
        )
    elif score >= SEVERITY_REVIEW:
        state["decision"] = "REVIEW"
        state["decision_reasoning"] = (
            f"Severity score {score:.3f} is between REVIEW ({SEVERITY_REVIEW}) and "
            f"BLOCK ({SEVERITY_BLOCK}) thresholds. Signals are suspicious but not "
            "conclusive enough for an automatic block. Routed to human analyst "
            "queue with full evidence attached."
        )
    else:
        state["decision"] = "ALLOW"
        state["decision_reasoning"] = (
            f"Severity score {score:.3f} below REVIEW threshold ({SEVERITY_REVIEW}). "
            "No sufficiently strong correlated-risk signal found; transactions "
            "proceed normally."
        )
    return state


def node_narrate_for_analyst(state: InvestigationState) -> InvestigationState:
    """Turn the finished evidence into prose. Runs after the decision is final."""
    reasoning = state.get("severity_reasoning", [])
    decision = state.get("decision", "")
    score = state.get("severity_score", 0)
    spike_id = state.get("spike_id") or state.get("merchant_id", "unknown")
    root_cause = state.get("root_cause_hypothesis", {}).get("label", "not classified")

    fallback = (
        f"[Template narration -- no LLM provider configured] Investigation {spike_id}: "
        f"decision {decision} at severity {score}. Root-cause hypothesis: {root_cause}. "
        + " ".join(reasoning) + " "
        "See decision_reasoning field for the threshold logic applied."
    )

    prompt = (
        "You are writing a short analyst-facing summary for a fraud investigation "
        "record. Use ONLY the facts below -- do not invent any number, merchant "
        "name, or claim not present in this evidence. The decision is already "
        "final; do not second-guess it, just explain it clearly in 3-4 sentences "
        "for a human fraud analyst who needs to act on this fast.\n\n"
        f"Investigation ID: {spike_id}\n"
        f"Decision already made: {decision} (severity score {score})\n"
        f"Root-cause hypothesis (from a separate classifier, already fixed): {root_cause}\n"
        "Evidence:\n" + "\n".join(f"- {r}" for r in reasoning) + "\n\n"
        f"Full decision rationale: {state.get('decision_reasoning', '')}"
    )
    text, meta = call_llm(prompt, max_tokens=300, **_llm_args(state))

    if meta["ok"] and text:
        state["analyst_narrative"] = text
    elif meta["provider"] is None:
        state["analyst_narrative"] = fallback
    else:
        state["analyst_narrative"] = fallback.replace(
            "no LLM provider configured", f"{meta['provider']} call failed: {meta['error']}")
    return state


def node_log_audit(state: InvestigationState) -> InvestigationState:
    entry = {
        "investigation_id": f"inv_{uuid.uuid4().hex[:10]}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "spike_id": state.get("spike_id"),
        "merchant_id": state.get("merchant_id"),
        "n_transactions_examined": len(state.get("transactions", [])),
        "affected_merchants": state.get("affected_merchants", []),
        "device_correlation": state.get("device_correlation"),
        "ip_correlation": state.get("ip_correlation"),
        "shap_top_drivers": state.get("shap_reasons"),
        "severity_score": state.get("severity_score"),
        "severity_reasoning": state.get("severity_reasoning"),
        "decision": state.get("decision"),
        "decision_reasoning": state.get("decision_reasoning"),
        "root_cause_hypothesis": state.get("root_cause_hypothesis"),
        "analyst_narrative": state.get("analyst_narrative"),
        "thresholds_used": {"review": SEVERITY_REVIEW, "block": SEVERITY_BLOCK},
    }
    state["audit_entry"] = entry
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return state


PIPELINE = [
    ("ingest", node_ingest),
    ("correlate_merchants", node_correlate_merchants),
    ("correlate_device_ip", node_correlate_device_ip),
    ("pull_shap_reasons", node_pull_shap_reasons),
    ("classify_root_cause", node_classify_root_cause),
    ("assess_severity", node_assess_severity),
    ("decide_action", node_decide_action),
    ("narrate_for_analyst", node_narrate_for_analyst),
    ("log_audit", node_log_audit),
]

_graph = None


def build_graph():
    g = StateGraph(InvestigationState)
    for name, fn in PIPELINE:
        g.add_node(name, fn)
    g.set_entry_point(PIPELINE[0][0])
    for (name, _), (next_name, _) in zip(PIPELINE, PIPELINE[1:]):
        g.add_edge(name, next_name)
    g.add_edge(PIPELINE[-1][0], END)
    return g.compile()


def get_graph():
    """Compiled once and reused; compiling per request is pure overhead."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _run(initial_state):
    return get_graph().invoke(initial_state)["audit_entry"]


def investigate_spike(spike_id, llm_api_key=None, llm_provider=None, llm_model=None):
    return _run({"spike_id": spike_id, "llm_api_key": llm_api_key,
                 "llm_provider": llm_provider, "llm_model": llm_model})


def investigate_merchant_bucket(merchant_id, bucket, llm_api_key=None,
                                llm_provider=None, llm_model=None):
    return _run({"merchant_id": merchant_id, "bucket": bucket, "llm_api_key": llm_api_key,
                 "llm_provider": llm_provider, "llm_model": llm_model})


def _report(entry, header):
    print(f"\n=== {header} ===")
    print(f"Decision: {entry['decision']}")
    print(f"Severity: {entry['severity_score']}")
    print(f"Root cause: {entry['root_cause_hypothesis']['label']}")
    print(f"Reasoning: {entry['decision_reasoning']}")


def main():
    open(AUDIT_LOG_PATH, "w").close()

    with open(os.path.join(ART_DIR, "spike_ground_truth.json"), encoding="utf-8") as f:
        spikes = json.load(f)
    for spk in spikes:
        _report(investigate_spike(spk["id"]), f"Investigation: {spk['id']}")

    # The control is a legitimate demand spike, so ALLOW is the correct result.
    ctrl_path = os.path.join(ART_DIR, "control_event_ground_truth.json")
    if os.path.exists(ctrl_path):
        with open(ctrl_path, encoding="utf-8") as f:
            ctrl = json.load(f)
        entry = investigate_spike(ctrl["id"])
        _report(entry, f"Specificity control: {ctrl['id']} (legitimate spike, expected ALLOW)")


if __name__ == "__main__":
    main()
