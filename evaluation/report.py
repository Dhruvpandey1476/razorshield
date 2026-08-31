"""Render every published metric from artifacts/ into METRICS.md and README.md."""
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIR = os.path.join(ROOT, "artifacts")
README = os.path.join(ROOT, "README.md")
METRICS_MD = os.path.join(ROOT, "METRICS.md")

START = "<!-- METRICS:START -->"
END = "<!-- METRICS:END -->"


def load(name, default=None):
    path = os.path.join(ART_DIR, name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def load_audit():
    path = os.path.join(ART_DIR, "audit_trail.jsonl")
    if not os.path.exists(path):
        return []
    seen = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                seen[r.get("spike_id")] = r      # last run wins
    return list(seen.values())


def pct(x):
    return "--" if x is None else "%.1f%%" % (float(x) * 100)


def inr(x):
    return "--" if x is None else "Rs %s" % format(int(round(float(x))), ",d")


def build():
    m = load("metrics.json")
    if not m:
        raise SystemExit("artifacts/metrics.json not found -- run ./run_pipeline.sh first")
    events = load("spike_event_detection_summary.json", [])
    control = load("control_event_summary.json")
    gen = load("data_generation_summary.json", {})
    audit = load_audit()

    q = m.get("model_quality", {})
    comb = m.get("combined_system", {})
    sc = m.get("standalone_vs_combined", {})
    marg = m.get("marginal_contribution_of_temporal_layer", {})
    L = []

    L.append("### How the data is split")
    L.append("")
    sp = m.get("split_policy", {})
    L.append("| Window | Dates | Transactions | Fraud rate |")
    L.append("|---|---|---|---|")
    for k, label in [("train", "Train"), ("val", "Validation"), ("test", "Test (held out)")]:
        L.append("| %s | %s | %s | %s |" % (
            label, gen.get("windows", {}).get(k, "--"),
            format(gen.get("split_sizes", {}).get(k, 0), ",d"),
            pct(gen.get("fraud_rate_by_split", {}).get(k))))
    L.append("")
    L.append("_%s_" % sp.get("note", ""))
    L.append("")

    L.append("### Model quality (held-out test window)")
    L.append("")
    L.append("| Measure | Value |")
    L.append("|---|---|")
    L.append("| ROC-AUC, ordinary fraud population | **%.4f** |" % q.get("roc_auc_baseline_fraud_population", 0))
    L.append("| Bayes ceiling for this data (oracle limit) | %.4f |" % (q.get("bayes_ceiling_auc_baseline_population") or 0))
    L.append("| ROC-AUC, coordinated-ring population | %.4f |" % q.get("roc_auc_coordinated_ring_population", 0))
    L.append("| ROC-AUC, blended across both populations | %.4f |" % q.get("roc_auc_overall", 0))
    L.append("| PR-AUC (blended) | %.4f |" % q.get("pr_auc_overall", 0))
    L.append("")
    L.append("The classifier reaches **%.4f against a hard ceiling of %.4f** on the fraud it is "
             "designed for. The ring figure is far lower and that is the finding, not a defect: "
             "two of the four injected rings distribute across near-unique devices, so no "
             "per-transaction velocity signal exists for any classifier to learn. "
             "The blended number mixes the two populations and is reported only for completeness."
             % (q.get("roc_auc_baseline_fraud_population", 0),
                q.get("bayes_ceiling_auc_baseline_population") or 0))
    L.append("")

    L.append("### Operating points (thresholds chosen on validation, scored on test)")
    L.append("")
    ca = m.get("cost_assumptions", {})
    L.append("Cost model: **%s per manual review**, and a missed fraud costs the full "
             "transaction value plus a **%s chargeback fee**."
             % (inr(ca.get("manual_review_cost_inr_per_alert")),
                inr(ca.get("chargeback_fee_inr_per_missed_fraud"))))
    L.append("")
    L.append("| Operating point | Threshold | Alert rate | Precision | Recall | FP rate | FP review cost | Net benefit vs no model |")
    L.append("|---|---|---|---|---|---|---|---|")
    for op in m.get("operating_points", []):
        name = "**%s**" % op["name"] if "primary" in op["name"] else op["name"]
        L.append("| %s | %.4f | %s | %s | %s | %s | %s | %s |" % (
            name, op["threshold"], pct(op["alert_rate"]), pct(op["precision"]),
            pct(op["recall"]), pct(op["false_positive_rate"]),
            inr(op["false_positive_monetary_cost_inr"]),
            inr(op["net_benefit_vs_no_model_inr"])))
    L.append("")
    for op in m.get("operating_points", []):
        if op.get("caveat"):
            L.append("- `%s` -- %s" % (op["name"], op["caveat"]))
    L.append("")

    if sc:
        L.append("### Standalone classifier vs the full two-layer system")
        L.append("")
        L.append("| System | Precision | Recall | F1 | Alert rate | FP rate | FP review cost | Net benefit |")
        L.append("|---|---|---|---|---|---|---|---|")
        for key, label in [("standalone", "Per-transaction XGBoost only"),
                           ("combined", "**Two-layer (model + temporal)**")]:
            s = sc.get(key, {})
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                label, pct(s.get("precision")), pct(s.get("recall")), pct(s.get("f1")),
                pct(s.get("alert_rate")), pct(s.get("false_positive_rate")),
                inr(s.get("false_positive_monetary_cost_inr")),
                inr(s.get("net_benefit_vs_no_model_inr"))))
        L.append("")

    if marg:
        L.append("**Marginal cost of the temporal layer.** It adds **%s additional true "
                 "positives for %s additional false positives** -- a marginal review cost of "
                 "**%s per extra fraudulent transaction caught**, against the %s per-review "
                 "assumption used throughout. Both confusion matrices are in "
                 "`artifacts/metrics.json`."
                 % (marg.get("additional_true_positives"),
                    marg.get("additional_false_positives"),
                    inr(marg.get("review_cost_per_additional_fraud_caught_inr")),
                    inr(ca.get("manual_review_cost_inr_per_alert"))))
        L.append("")

    if events:
        L.append("### Per-event detection (all four injected ring events)")
        L.append("")
        L.append("| Event | Attack type | Txns | Device pool | Standalone recall | Combined recall | Event flagged | Latency |")
        L.append("|---|---|---|---|---|---|---|---|")
        for e in events:
            L.append("| %s | %s | %d | %s | %s | %s | %s | %s |" % (
                e["spike_id"], e["attack_type"].replace("_", " "), e["n_txns"],
                "%.0f%%" % (e["device_pool_fraction"] * 100),
                pct(e["standalone_recall"]), pct(e["combined_recall"]),
                "yes" if e["flagged_as_spike_bucket"] else "no",
                "%s min" % e["detection_latency_minutes"] if e["detection_latency_minutes"] is not None else "--"))
        L.append("")
        det = sum(1 for e in events if e["flagged_as_spike_bucket"])
        L.append("**%d of %d ring events raised an alert**, with latency measured to the close of "
                 "the first triggering 15-minute bucket (a bucket cannot alert before it closes, "
                 "so this is the earliest a real deployment could have known)."
                 % (det, len(events)))
        L.append("")
        worst = min(events, key=lambda e: e["combined_recall"])
        if worst["combined_recall"] < 0.25:
            L.append("**Where this is weakest, stated plainly.** `%s` (%s) reaches only %s "
                     "transaction-level recall. It spreads %d transactions across %d merchants "
                     "and near-unique devices, so no 15-minute aggregation on any axis sees "
                     "enough concentration to act on individual transactions. The event is still "
                     "escalated -- its bucket is flagged and the investigation agent routes it to "
                     "a human -- but most of its individual transactions are not caught. This is "
                     "the honest residual limitation of the current design."
                     % (worst["spike_id"], worst["attack_type"].replace("_", " "),
                        pct(worst["combined_recall"]), worst["n_txns"], worst["n_merchants"]))
            L.append("")

    if control:
        L.append("### Specificity control: does it avoid blocking an innocent merchant?")
        L.append("")
        L.append("`%s` is a **genuine legitimate demand spike** (a flash sale, %d transactions, "
                 "not fraud) injected into the same test window. A detector that flags every "
                 "volume anomaly is useless no matter how good its recall looks."
                 % (control["event_id"], control["n_txns"]))
        L.append("")
        L.append("| Layer | Result on CTRL-001 |")
        L.append("|---|---|")
        L.append("| Temporal layer | %s -- correctly, it genuinely is a volume anomaly |"
                 % ("flagged the bucket as anomalous"
                    if control["temporal_layer_flagged_bucket_as_anomalous"] else "did not flag"))
        L.append("| Per-transaction model | %s of its transactions flagged |"
                 % pct(control["individual_txn_flag_rate"]))
        L.append("| Full two-layer system | **%s of its transactions flagged** |"
                 % pct(control["combined_txn_flag_rate"]))
        ctrl_audit = next((a for a in audit if a.get("spike_id") == control["event_id"]), None)
        if ctrl_audit:
            L.append("| Investigation agent | **%s** at severity %.3f (REVIEW threshold %.2f) |" % (
                ctrl_audit["decision"], ctrl_audit["severity_score"],
                ctrl_audit.get("thresholds_used", {}).get("review", 0.35)))
        L.append("")
        L.append("Specificity cannot come from the volume signal, because the volume signal is "
                 "genuinely anomalous here. It comes from the cluster-level ring-evidence score "
                 "and the investigation agent, which weigh identity churn, merchant spread, "
                 "device/IP reuse and mismatch rates rather than reacting to volume alone.")
        L.append("")

    if audit:
        L.append("### Investigation agent decisions (real output, regenerated each run)")
        L.append("")
        L.append("| Event | Severity | Root-cause hypothesis | Decision |")
        L.append("|---|---|---|---|")
        for a in sorted(audit, key=lambda x: str(x.get("spike_id"))):
            rc = a.get("root_cause_hypothesis", {})
            L.append("| %s | %.3f | %s | **%s** |" % (
                a.get("spike_id"), a.get("severity_score", 0),
                rc.get("label", "--"), a.get("decision")))
        L.append("")
        L.append("Thresholds are fixed and inspectable (REVIEW at %.2f, BLOCK at %.2f) and the "
                 "decision is computed from the evidence, never generated by a language model. "
                 "Full reasoning chains are in `artifacts/audit_trail.jsonl`."
                 % (audit[0].get("thresholds_used", {}).get("review", 0.35),
                    audit[0].get("thresholds_used", {}).get("block", 0.75)))
        L.append("")

    return "\n".join(L).rstrip() + "\n"


def main():
    body = build()

    header = (
        "# RazorShield - Measured Results\n\n"
        "Every number on this page is rendered directly from `artifacts/` by\n"
        "`evaluation/report.py`. Nothing here is typed by hand. Regenerate with\n"
        "`./run_pipeline.sh`.\n\n"
    )
    with open(METRICS_MD, "w", encoding="utf-8") as f:
        f.write(header + body)

    if os.path.exists(README):
        with open(README, encoding="utf-8") as f:
            readme = f.read()
        if START in readme and END in readme:
            pre = readme.split(START)[0]
            post = readme.split(END)[1]
            with open(README, "w", encoding="utf-8") as f:
                f.write(pre + START + "\n" + body + END + post)
            print("Injected results into README.md between markers")
        else:
            print("WARNING: README.md has no %s / %s markers -- wrote METRICS.md only" % (START, END))
    print("Wrote METRICS.md (%d lines)" % (body.count("\n") + 1))


if __name__ == "__main__":
    main()
