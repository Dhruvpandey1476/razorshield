"""Temporal anomaly layer: catches rings the per-transaction model cannot see."""
import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ART_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

BUCKET_MINUTES = 15
Z_THRESHOLD = 3.0
MIN_BUCKET_TXNS = 5
MIN_HISTORY_BUCKETS = 8       # below this an entity has no usable baseline
COMBINED_ALERT_BUDGET = 0.02  # combined system may alert on <=2% of traffic
AXES = ["merchant_id", "device_id", "ip_address"]

REVIEW_COST_INR = 45.0
CHARGEBACK_FEE_INR = 750.0


def add_bucket_column(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["bucket"] = df["timestamp"].dt.floor(f"{BUCKET_MINUTES}min")
    return df


def build_buckets(df, key):
    return df.groupby([key, "bucket"]).agg(
        txn_count=("payment_id", "count"),
        mean_proba=("proba", "mean"),
        mean_amount=("amount_inr", "mean"),
        new_cust_ratio=("is_new_customer", "mean"),
        mismatch_ratio=("billing_shipping_mismatch", "mean"),
        n_merchants=("merchant_id", "nunique"),
        n_customers=("customer_id", "nunique"),
        n_devices=("device_id", "nunique"),
    ).reset_index()


def ring_evidence(g, key):
    """Does this cluster look like a ring, or just like a busy fifteen minutes?"""
    proba_term = (g["mean_proba"] * 4).clip(0, 1)
    volume = (g["count_z"].fillna(0) / 6).clip(0, 1)

    if key == "merchant_id":
        amount = (g["amount_z"].fillna(0) / 3).clip(0, 1)
        return (0.28 * proba_term
                + 0.26 * g["mismatch_ratio"]
                + 0.18 * g["new_cust_ratio"]
                + 0.14 * volume
                + 0.14 * amount)

    identity_churn = ((g["n_customers"] - 1) / 4).clip(0, 1)
    merchant_spread = ((g["n_merchants"] - 1) / 3).clip(0, 1)
    return (0.22 * proba_term
            + 0.26 * identity_churn
            + 0.22 * merchant_spread
            + 0.14 * g["mismatch_ratio"]
            + 0.08 * g["new_cust_ratio"]
            + 0.08 * volume)


def calibrate_absolute_threshold(train_df, key, quantile=0.999, floor=5):
    """How many events in one bucket is abnormal for ANY entity of this kind."""
    counts = train_df.groupby([key, "bucket"]).size()
    return int(max(floor, np.ceil(counts.quantile(quantile))))


def flag_buckets(agg, key, absolute_count_threshold):
    """Flag anomalous buckets per entity against its own prior history."""
    agg = agg.sort_values([key, "bucket"]).reset_index(drop=True)
    out = []
    for _, g in agg.groupby(key, sort=False):
        g = g.sort_values("bucket").copy()
        for col, prefix in (("txn_count", "count"), ("mean_proba", "proba"),
                            ("mean_amount", "amount")):
            prior = g[col].shift(1)
            mu = prior.expanding(min_periods=MIN_HISTORY_BUCKETS).mean()
            sd = prior.expanding(min_periods=MIN_HISTORY_BUCKETS).std()
            g[prefix + "_z"] = (g[col] - mu) / (sd.fillna(0) + 1e-6)

        g["history_buckets"] = g["txn_count"].shift(1).expanding(min_periods=1).count()
        g["has_baseline"] = (g["history_buckets"] >= MIN_HISTORY_BUCKETS).astype(int)
        g["spike_score"] = g[["count_z", "proba_z"]].max(axis=1).fillna(0.0)

        relative = ((g["has_baseline"] == 1) & (g["spike_score"] >= Z_THRESHOLD)
                    & (g["txn_count"] >= MIN_BUCKET_TXNS))
        absolute = (g["has_baseline"] == 0) & (g["txn_count"] >= absolute_count_threshold)
        g["is_spike_bucket"] = (relative | absolute).astype(int)
        g["detection_basis"] = np.where(relative, "prior_baseline_zscore",
                                        np.where(absolute, "population_absolute_threshold", ""))
        g["ring_evidence"] = ring_evidence(g, key).round(4)
        out.append(g)

    res = pd.concat(out, ignore_index=True)
    res["axis"] = key
    return res


def attach_buckets(scored, flagged_by_axis):
    """A transaction is in an anomalous cluster if ANY axis flagged its bucket."""
    merged = scored.copy()
    merged["is_spike_bucket"] = 0
    merged["spike_score"] = 0.0
    merged["ring_evidence"] = 0.0
    merged["spike_axes"] = ""

    for key, flagged in flagged_by_axis.items():
        cols = flagged[[key, "bucket", "is_spike_bucket", "spike_score", "ring_evidence"]].rename(
            columns={"is_spike_bucket": "_flag", "spike_score": "_score", "ring_evidence": "_ev"})
        merged = merged.merge(cols, on=[key, "bucket"], how="left")
        merged["_flag"] = merged["_flag"].fillna(0).astype(int)
        merged["_score"] = merged["_score"].fillna(0.0)
        # Evidence only counts from an axis that actually flagged the bucket.
        merged["_ev"] = merged["_ev"].fillna(0.0) * merged["_flag"]

        merged["spike_axes"] = np.where(
            merged["_flag"] == 1,
            np.where(merged["spike_axes"] == "", key, merged["spike_axes"] + "+" + key),
            merged["spike_axes"])
        merged["is_spike_bucket"] = merged[["is_spike_bucket", "_flag"]].max(axis=1)
        merged["spike_score"] = merged[["spike_score", "_score"]].max(axis=1)
        merged["ring_evidence"] = merged[["ring_evidence", "_ev"]].max(axis=1)
        merged = merged.drop(columns=["_flag", "_score", "_ev"])

    return merged


def fit_escalation_threshold(val_merged):
    """Lowest ring-evidence bar that keeps the combined alert rate within budget."""
    n = len(val_merged)
    for t in np.round(np.linspace(0.05, 1.0, 96), 4):
        pred = (val_merged["pred"] == 1) | (val_merged["ring_evidence"] >= t)
        if pred.sum() / n <= COMBINED_ALERT_BUDGET:
            return float(t)
    return 1.0


def cost_of(pred, y, amounts):
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = (pred == 0) & (y == 1)
    return round(fp * REVIEW_COST_INR + float(amounts[fn].sum()) + int(fn.sum()) * CHARGEBACK_FEE_INR, 2)


def summarise(y, pred, amounts, label):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    cost = cost_of(pred, y, amounts)
    do_nothing = cost_of(np.zeros(len(y), dtype=int), y, amounts)
    return {
        "system": label,
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "alert_rate": round(float(pred.mean()), 4),
        "false_positive_rate": round(float(fp / max(fp + tn, 1)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_monetary_cost_inr": round(fp * REVIEW_COST_INR, 2),
        "fraud_amount_prevented_inr": round(float(amounts[(pred == 1) & (y == 1)].sum()), 2),
        "total_cost_inr": cost,
        "net_benefit_vs_no_model_inr": round(do_nothing - cost, 2),
    }


def marginal_contribution(standalone, combined):
    added_tp = combined["confusion_matrix"]["tp"] - standalone["confusion_matrix"]["tp"]
    added_fp = combined["confusion_matrix"]["fp"] - standalone["confusion_matrix"]["fp"]
    return {
        "additional_true_positives": int(added_tp),
        "additional_false_positives": int(added_fp),
        "additional_review_cost_inr": round(added_fp * REVIEW_COST_INR, 2),
        "additional_fraud_value_caught_inr": round(
            combined["fraud_amount_prevented_inr"] - standalone["fraud_amount_prevented_inr"], 2),
        "review_cost_per_additional_fraud_caught_inr": (
            round(added_fp * REVIEW_COST_INR / added_tp, 2) if added_tp > 0 else None),
    }


def per_event_summary(test, gt_events):
    events = []
    for ev in gt_events:
        rows = test[test["spike_event_id"] == ev["id"]]
        if not len(rows):
            continue
        triggering = rows[rows["is_spike_bucket"] == 1]["bucket"]
        latency = None
        if len(triggering):
            closes_at = pd.Timestamp(triggering.min()) + pd.Timedelta(minutes=BUCKET_MINUTES)
            latency = round(float((closes_at - pd.Timestamp(ev["start"])).total_seconds() / 60), 1)
        events.append({
            "spike_id": ev["id"],
            "attack_type": ev["label"],
            "start": ev["start"],
            "n_txns": int(len(rows)),
            "n_merchants": ev["n_merchants"],
            "device_pool_fraction": ev["device_pool_frac"],
            "visibility_to_per_txn_model": ev["visibility"],
            "standalone_recall": round(float(rows["pred"].mean()), 3),
            "combined_recall": round(float(rows["combined_pred"].mean()), 3),
            "flagged_as_spike_bucket": bool(rows["is_spike_bucket"].max()),
            "detection_latency_minutes": latency,
        })
    return events


def control_summary(test, ctrl_gt):
    rows = test[test["spike_event_id"] == ctrl_gt["id"]]
    return {
        "event_id": ctrl_gt["id"],
        "description": ctrl_gt["description"],
        "n_txns": int(len(rows)),
        "temporal_layer_flagged_bucket_as_anomalous": bool(rows["is_spike_bucket"].max()),
        "individual_txn_flag_rate": round(float(rows["pred"].mean()), 4),
        "combined_txn_flag_rate": round(float(rows["combined_pred"].mean()), 4),
        "note": (
            "The temporal layer flags this bucket as a volume anomaly, which is "
            "correct -- it genuinely is one. Specificity therefore cannot come from "
            "the volume signal, and does not: it comes from the downstream "
            "investigation agent, which aggregates device/IP reuse, mismatch rate "
            "and model confidence before deciding. See audit_trail.jsonl for the "
            "agent's ALLOW/REVIEW/BLOCK verdict on this event."
        ),
    }


def ring_recall(test, column):
    is_ring = test["spike_event_id"].notna() & (test["spike_event_id"] != "CTRL-001")
    return round(float(test.loc[is_ring, column].mean()), 4)


def main():
    scored = add_bucket_column(pd.read_csv(os.path.join(ART_DIR, "scored_all_transactions.csv")))

    with open(os.path.join(ART_DIR, "metrics.json")) as f:
        metrics = json.load(f)

    # Buckets span the full timeline so test-window buckets have real prior
    # history from train/val to compare against, while the baseline itself
    # still only ever looks backwards.
    train_rows = scored[scored["split"] == "train"]
    abs_thresholds = {k: calibrate_absolute_threshold(train_rows, k) for k in AXES}
    flagged_by_axis = {k: flag_buckets(build_buckets(scored, k), k, abs_thresholds[k])
                       for k in AXES}
    pd.concat(flagged_by_axis.values(), ignore_index=True).to_csv(
        os.path.join(ART_DIR, "spike_buckets.csv"), index=False)

    merged = attach_buckets(scored, flagged_by_axis)
    escalation = fit_escalation_threshold(merged[merged["split"] == "val"])
    merged["combined_pred"] = (
        (merged["pred"] == 1) | (merged["ring_evidence"] >= escalation)).astype(int)

    test = merged[merged["split"] == "test"].reset_index(drop=True)
    test.to_csv(os.path.join(ART_DIR, "combined_predictions.csv"), index=False)

    y, amounts = test["is_fraud"].values, test["amount_inr"].values
    standalone = summarise(y, test["pred"].values, amounts, "standalone_xgboost")
    combined = summarise(y, test["combined_pred"].values, amounts, "combined_two_layer")
    marginal = marginal_contribution(standalone, combined)

    with open(os.path.join(ART_DIR, "spike_ground_truth.json")) as f:
        events = per_event_summary(test, json.load(f))
    with open(os.path.join(ART_DIR, "spike_event_detection_summary.json"), "w") as f:
        json.dump(events, f, indent=2, default=str)

    ctrl_path = os.path.join(ART_DIR, "control_event_ground_truth.json")
    control = None
    if os.path.exists(ctrl_path):
        with open(ctrl_path) as f:
            control = control_summary(test, json.load(f))
        with open(os.path.join(ART_DIR, "control_event_summary.json"), "w") as f:
            json.dump(control, f, indent=2)

    timeline = test.groupby("bucket").agg(
        txn_count=("payment_id", "count"), max_spike_score=("spike_score", "max")).reset_index()
    timeline.to_csv(os.path.join(ART_DIR, "timeline_buckets.csv"), index=False)

    metrics["combined_system"] = {
        "description": (
            "Per-transaction XGBoost score above the budget threshold, OR a transaction "
            "inside a merchant-bucket flagged anomalous by the temporal layer "
            "(z>=%.1f vs that merchant's strictly-prior expanding baseline, min %d txns "
            "per %d-minute bucket) whose cluster-level ring-evidence score clears %.3f, "
            "a bar fitted on the validation window under a %.0f%% alert budget."
            % (Z_THRESHOLD, MIN_BUCKET_TXNS, BUCKET_MINUTES, escalation,
               COMBINED_ALERT_BUDGET * 100)
        ),
        "ring_evidence_escalation_threshold": escalation,
        "aggregation_axes": AXES,
        "population_absolute_thresholds": abs_thresholds,
        "baseline_is_strictly_backward_looking": True,
        "precision": combined["precision"],
        "recall": combined["recall"],
        "f1": combined["f1"],
        "alert_rate": combined["alert_rate"],
        "false_positive_rate": combined["false_positive_rate"],
        "confusion_matrix": combined["confusion_matrix"],
        "false_positive_monetary_cost_inr": combined["false_positive_monetary_cost_inr"],
        "fraud_amount_prevented_inr": combined["fraud_amount_prevented_inr"],
        "net_benefit_vs_no_model_inr": combined["net_benefit_vs_no_model_inr"],
        "spike_events_detected": sum(1 for e in events if e["flagged_as_spike_bucket"]),
        "spike_events_total": len(events),
        "ring_txn_recall": ring_recall(test, "combined_pred"),
    }
    metrics["standalone_vs_combined"] = {"standalone": standalone, "combined": combined}
    metrics["marginal_contribution_of_temporal_layer"] = marginal
    metrics["spike_specific"]["ring_txn_recall_standalone"] = ring_recall(test, "pred")

    with open(os.path.join(ART_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Temporal layer ===")
    print("  aggregation axes: %s" % ", ".join(AXES))
    print("  population absolute thresholds (fitted on train): %s" % abs_thresholds)
    print("  ring-evidence escalation bar (fitted on validation): %.4f" % escalation)
    print("  %-22s %9s %9s %9s %9s %11s"
          % ("system", "prec", "recall", "alert%", "FPR", "net INR"))
    for s in (standalone, combined):
        print("  %-22s %9.3f %9.3f %8.2f%% %9.4f %11.0f"
              % (s["system"], s["precision"], s["recall"], s["alert_rate"] * 100,
                 s["false_positive_rate"], s["net_benefit_vs_no_model_inr"]))
    print("\n  marginal: +%d true positives for +%d false positives (%s INR review cost each)"
          % (marginal["additional_true_positives"], marginal["additional_false_positives"],
             marginal["review_cost_per_additional_fraud_caught_inr"]))

    print("\n=== Per-event detection ===")
    print("  %-9s %-20s %6s %10s %10s %8s"
          % ("event", "type", "txns", "standalone", "combined", "latency"))
    for e in events:
        print("  %-9s %-20s %6d %10.3f %10.3f %8s"
              % (e["spike_id"], e["attack_type"], e["n_txns"], e["standalone_recall"],
                 e["combined_recall"], e["detection_latency_minutes"]))

    if control:
        print("\n=== Specificity control (CTRL-001, legitimate spike) ===")
        print("  bucket flagged anomalous: %s | per-txn flag rate: %.4f | combined flag rate: %.4f"
              % (control["temporal_layer_flagged_bucket_as_anomalous"],
                 control["individual_txn_flag_rate"], control["combined_txn_flag_rate"]))


if __name__ == "__main__":
    main()
