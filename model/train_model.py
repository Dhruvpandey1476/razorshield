"""Train the per transaction fraud classifier and evaluate it on held out data."""
import os
import json

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve,
)
import joblib

ART_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

REVIEW_COST_INR = 45.0        # ops cost of one manual review
CHARGEBACK_FEE_INR = 750.0    # on top of losing the transaction value
ALERT_BUDGET_FRACTION = 0.01  # a team that can review 1% of transactions

FEATURES_NUM = [
    "amount_inr", "customer_txn_count_24h", "is_new_customer",
    "billing_shipping_mismatch", "hour_of_day", "is_night",
    "device_txn_count_1h", "ip_txn_count_1h", "merchant_risk_score",
    "amount_zscore_for_merchant",
]
CAT_FEATURES = ["method", "merchant_category"]
VELOCITY_COLS = ["device_txn_count_1h", "ip_txn_count_1h", "customer_txn_count_24h"]

SCORED_COLS = [
    "payment_id", "customer_id", "merchant_id", "merchant_category", "merchant_risk_tier",
    "timestamp", "amount_inr", "method", "device_id", "ip_address",
    "is_new_customer", "billing_shipping_mismatch", "customer_txn_count_24h",
    "device_txn_count_1h", "ip_txn_count_1h",
    "is_fraud", "spike_event_id", "split", "proba", "pred",
]

CAVEATS = {
    "cost_optimal_unconstrained":
        "Lowest expected rupee cost, but it alerts on {rate:.1f}% of all traffic. "
        "Reported for completeness -- no risk-ops team can staff this, which is "
        "precisely why the deployed operating point is budget-constrained.",
    "f1_optimal":
        "What maximising F1 would pick. Shown to make the difference visible: F1 "
        "implicitly prices one missed fraud equal to one false alarm, which is "
        "wrong by roughly two orders of magnitude in payments.",
}


def fit_merchant_stats(train_df):
    g = train_df.groupby("merchant_id")["amount_inr"]
    return {
        "mean": g.mean().to_dict(),
        "std": g.std().fillna(0.0).to_dict(),
        "global_mean": float(train_df["amount_inr"].mean()),
        "global_std": float(train_df["amount_inr"].std()),
    }


def engineer_features(df, merchant_stats):
    """merchant_stats must come from fit_merchant_stats() on the train window."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour_of_day"].isin([1, 2, 3, 4]).astype(int)
    df["merchant_risk_score"] = df["merchant_risk_tier"].map({"low": 0, "medium": 1, "high": 2})

    # Velocity columns come from the generator, which builds them with strictly
    # backward looking logic. Carried through unchanged so training and
    # webhook-time serving share one definition.
    missing = [c for c in VELOCITY_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"missing velocity columns {missing} -- re-run data/generate_data.py")

    mean_s = df["merchant_id"].map(merchant_stats["mean"]).fillna(merchant_stats["global_mean"])
    std_s = (df["merchant_id"].map(merchant_stats["std"])
             .fillna(merchant_stats["global_std"])
             .replace(0, np.nan).fillna(merchant_stats["global_std"]))
    df["amount_zscore_for_merchant"] = ((df["amount_inr"] - mean_s) / (std_s + 1e-6)).fillna(0.0)

    for c in CAT_FEATURES:
        df[c + "_raw"] = df[c]
    return pd.get_dummies(df, columns=CAT_FEATURES, prefix=CAT_FEATURES)


def feature_columns(df):
    dummies = [c for c in df.columns
               if any(c.startswith(p + "_") for p in CAT_FEATURES) and not c.endswith("_raw")]
    return FEATURES_NUM + sorted(dummies)


def total_cost(y_true, pred, amounts):
    """A false positive wastes a review; a false negative loses the value plus a fee."""
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn_mask = (pred == 0) & (y_true == 1)
    fn_amount = float(amounts[fn_mask].sum())
    fn_count = int(fn_mask.sum())
    return {
        "review_cost_inr": round(fp * REVIEW_COST_INR, 2),
        "missed_fraud_value_inr": round(fn_amount, 2),
        "chargeback_fees_inr": round(fn_count * CHARGEBACK_FEE_INR, 2),
        "total_cost_inr": round(fp * REVIEW_COST_INR + fn_amount + fn_count * CHARGEBACK_FEE_INR, 2),
    }


def evaluate_at(threshold, y_true, proba, amounts, name):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    cost = total_cost(y_true, pred, amounts)
    do_nothing = total_cost(y_true, np.zeros(len(y_true), dtype=int), amounts)
    return {
        "name": name,
        "threshold": round(float(threshold), 4),
        "alert_rate": round(float(pred.mean()), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "false_positive_rate": round(float(fp / max(fp + tn, 1)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_monetary_cost_inr": round(fp * REVIEW_COST_INR, 2),
        "fraud_amount_prevented_inr": round(float(amounts[(pred == 1) & (y_true == 1)].sum()), 2),
        "fraud_amount_missed_inr": round(float(amounts[(pred == 0) & (y_true == 1)].sum()), 2),
        "cost_breakdown": cost,
        "net_benefit_vs_no_model_inr": round(do_nothing["total_cost_inr"] - cost["total_cost_inr"], 2),
    }


def pick_thresholds(y_val, proba_val, amounts_val):
    """Choose candidate thresholds on validation only."""
    grid = np.unique(np.quantile(proba_val, np.linspace(0.0, 0.99999, 600)))
    costs = np.array([
        total_cost(y_val, (proba_val >= t).astype(int), amounts_val)["total_cost_inr"] for t in grid
    ])
    alert_rates = np.array([float((proba_val >= t).mean()) for t in grid])

    def cheapest_within(cap):
        ok = np.where(alert_rates <= cap)[0]
        return float(grid[ok[int(np.argmin(costs[ok]))]]) if len(ok) else float(grid[-1])

    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)

    thresholds = {
        "budget_1pct": cheapest_within(ALERT_BUDGET_FRACTION),
        "budget_2pct": cheapest_within(ALERT_BUDGET_FRACTION * 2),
        "budget_5pct": cheapest_within(ALERT_BUDGET_FRACTION * 5),
        "cost_optimal_unconstrained": float(grid[int(np.argmin(costs))]),
        "f1_optimal": float(thr[int(np.argmax(f1s[:-1]))]) if len(thr) else 0.5,
    }
    curve = [{"threshold": round(float(t), 5), "total_cost_inr": round(float(c), 2),
              "alert_rate": round(float(a), 5)}
             for t, c, a in zip(grid, costs, alert_rates)]
    return thresholds, curve


def train(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.5,
        scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        eval_metric="aucpr",
        random_state=42,
        n_jobs=1,             # single-threaded and exact splits: multi-threaded
        tree_method="exact",  # histogram building is not bit-reproducible
    )
    model.fit(X_train, y_train)
    return model


def auc_by_population(test, proba, y_test):
    """A blended AUC over two structurally different fraud populations is not."""
    is_ring = test["spike_event_id"].notna() & (test["spike_event_id"] != "CTRL-001")
    is_baseline = test["spike_event_id"].isna()
    ring_vs_legit = (is_ring | (is_baseline & (test["is_fraud"] == 0))).values
    return {
        "overall": float(roc_auc_score(y_test, proba)),
        "baseline": float(roc_auc_score(test.loc[is_baseline, "is_fraud"], proba[is_baseline.values])),
        "ring": float(roc_auc_score(test.loc[ring_vs_legit, "is_fraud"], proba[ring_vs_legit])),
    }


def save_scored_splits(model, splits, feat_cols, threshold):
    """The temporal layer needs train/val history to build a prior baseline."""
    frames = []
    for name, frame in splits.items():
        f = frame.copy()
        f["proba"] = model.predict_proba(f[feat_cols])[:, 1]
        f["pred"] = (f["proba"] >= threshold).astype(int)
        f["split"] = name
        frames.append(f)

    scored = pd.concat(frames, ignore_index=True).rename(
        columns={"merchant_category_raw": "merchant_category", "method_raw": "method"})
    scored[SCORED_COLS].to_csv(os.path.join(ART_DIR, "scored_all_transactions.csv"), index=False)
    scored[scored["split"] == "test"][SCORED_COLS].to_csv(
        os.path.join(ART_DIR, "scored_test_transactions.csv"), index=False)


def write_shap(model, X_test, test, preds):
    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(min(1500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(os.path.join(ART_DIR, "shap_summary.png"), dpi=140)
    plt.close()

    importance = sorted(zip(sample.columns.tolist(), np.abs(shap_values).mean(axis=0).tolist()),
                        key=lambda x: -x[1])
    with open(os.path.join(ART_DIR, "shap_global_importance.json"), "w") as f:
        json.dump([{"feature": k, "mean_abs_shap": round(v, 5)} for k, v in importance], f, indent=2)

    # Top-3 drivers per flagged transaction, consumed by the agent's audit trail.
    flagged = set(test[preds == 1].index)
    picked = [i for i in sample.index if i in flagged][:250]
    explanations = {}
    if picked:
        sub_X = X_test.loc[picked]
        sub_shap = explainer.shap_values(sub_X)
        for pos, idx in enumerate(picked):
            contribs = sorted(zip(sub_X.columns, sub_shap[pos]), key=lambda x: -abs(x[1]))
            explanations[str(test.loc[idx, "payment_id"])] = [
                {"feature": k, "shap_value": round(float(v), 4)} for k, v in contribs[:3]
            ]
    with open(os.path.join(ART_DIR, "per_txn_shap_explanations.json"), "w") as f:
        json.dump(explanations, f, indent=2)


def build_metrics(test, y_test, proba_test, preds, auc, operating_points, cost_curve, gen):
    primary = operating_points[0]

    p_curve, r_curve, _ = precision_recall_curve(y_test, proba_test)
    step = max(1, len(p_curve) // 300)
    pr_curve = [{"precision": round(float(p), 4), "recall": round(float(r), 4)}
                for p, r in zip(p_curve[::step], r_curve[::step])]

    ring_rows = test[test["spike_event_id"].notna()].copy()
    ring_rows["pred"] = preds[test["spike_event_id"].notna().values]

    return {
        "evaluation_window": "held-out temporal test window (2026-08-10 to 2026-08-14)",
        "split_policy": {
            "train": gen.get("windows", {}).get("train"),
            "validation": gen.get("windows", {}).get("val"),
            "test": gen.get("windows", {}).get("test"),
            "note": "Every threshold and operating point is selected on the validation "
                    "window. The test window is scored once, after all choices are frozen.",
        },
        "cost_assumptions": {
            "manual_review_cost_inr_per_alert": REVIEW_COST_INR,
            "chargeback_fee_inr_per_missed_fraud": CHARGEBACK_FEE_INR,
            "missed_fraud_also_costs": "the full transaction value",
            "alert_budget_fraction": ALERT_BUDGET_FRACTION,
        },
        "n_test_transactions": int(len(test)),
        "n_test_fraud": int(y_test.sum()),
        "model_quality": {
            "roc_auc_overall": round(auc["overall"], 4),
            "roc_auc_baseline_fraud_population": round(auc["baseline"], 4),
            "roc_auc_coordinated_ring_population": round(auc["ring"], 4),
            "pr_auc_overall": round(float(average_precision_score(y_test, proba_test)), 4),
            "bayes_ceiling_auc_baseline_population": gen.get("bayes_ceiling_auc_baseline_population"),
            "interpretation": (
                "Judge the classifier on the baseline-population AUC, read against the "
                "Bayes ceiling rather than against 1.0. The coordinated-ring AUC is much "
                "lower because two of the four injected rings distribute across "
                "near-unique devices, leaving no per-transaction velocity signal to "
                "learn. That gap is the measured case for the temporal layer. The "
                "blended figure mixes both populations and is reported for completeness."
            ),
        },
        "primary_operating_point": primary["name"],
        "operating_points": operating_points,
        "validation_cost_curve": cost_curve,
        "pr_curve_test": pr_curve,

        # Flat aliases for the primary point, so the API, dashboard and tests
        # have a stable shape to read.
        "threshold_used": primary["threshold"],
        "precision": primary["precision"],
        "recall": primary["recall"],
        "f1": primary["f1"],
        "roc_auc": round(auc["overall"], 4),
        "false_positive_rate": primary["false_positive_rate"],
        "confusion_matrix": primary["confusion_matrix"],
        "false_positive_monetary_cost_inr": primary["false_positive_monetary_cost_inr"],
        "fraud_amount_prevented_inr": primary["fraud_amount_prevented_inr"],
        "fraud_amount_missed_inr": primary["fraud_amount_missed_inr"],
        "net_benefit_vs_no_model_inr": primary["net_benefit_vs_no_model_inr"],

        "spike_specific": {
            "n_ring_labelled_txns": int((~ring_rows["spike_event_id"].eq("CTRL-001")).sum()),
            "spike_recall": round(float(recall_score(
                ring_rows["is_fraud"], ring_rows["pred"], zero_division=0)), 4),
            "note": "Standalone per-transaction model only. The combined system's "
                    "figure is added by model/spike_detector.py.",
        },
        "methodology_note": (
            "Synthetic data with a known generating process (see "
            "artifacts/data_generation_summary.json). Temporal three-way split; "
            "thresholds fitted on validation; per-merchant amount statistics fitted "
            "on train only to avoid leakage into the test window."
        ),
    }


def print_summary(auc, gen, y_test, proba_test, operating_points):
    print("=== RazorShield model evaluation (held-out test window) ===")
    print("  Bayes ceiling (baseline population) : %s"
          % gen.get("bayes_ceiling_auc_baseline_population"))
    print("  ROC-AUC baseline fraud population   : %.4f" % auc["baseline"])
    print("  ROC-AUC coordinated-ring population : %.4f   <- the blind spot, measured" % auc["ring"])
    print("  ROC-AUC blended (both populations)  : %.4f" % auc["overall"])
    print("  PR-AUC                              : %.4f"
          % average_precision_score(y_test, proba_test))
    print()
    print("  %-28s %8s %8s %8s %8s %8s %11s"
          % ("operating point", "thresh", "alert%", "prec", "recall", "FPR", "net INR"))
    for op in operating_points:
        print("  %-28s %8.4f %7.2f%% %8.3f %8.3f %8.4f %11.0f"
              % (op["name"], op["threshold"], op["alert_rate"] * 100, op["precision"],
                 op["recall"], op["false_positive_rate"], op["net_benefit_vs_no_model_inr"]))


def main():
    raw = pd.read_csv(os.path.join(ART_DIR, "transactions.csv"))
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    merchant_stats = fit_merchant_stats(raw[raw["split"] == "train"])
    df = engineer_features(raw, merchant_stats)
    splits = {name: df[df["split"] == name].reset_index(drop=True)
              for name in ("train", "val", "test")}
    train_df, val, test = splits["train"], splits["val"], splits["test"]

    feat_cols = feature_columns(df)
    y_train, y_val, y_test = (s["is_fraud"].values for s in (train_df, val, test))
    amounts_val, amounts_test = val["amount_inr"].values, test["amount_inr"].values

    model = train(train_df[feat_cols], y_train)
    proba_val = model.predict_proba(val[feat_cols])[:, 1]
    proba_test = model.predict_proba(test[feat_cols])[:, 1]

    thresholds, cost_curve = pick_thresholds(y_val, proba_val, amounts_val)
    named = [("budget_1pct", "budget_1pct (primary)"), ("budget_2pct", "budget_2pct"),
             ("budget_5pct", "budget_5pct"),
             ("cost_optimal_unconstrained", "cost_optimal_unconstrained"),
             ("f1_optimal", "f1_optimal")]
    operating_points = [
        evaluate_at(thresholds[key], y_test, proba_test, amounts_test, label)
        for key, label in named
    ]
    for op in operating_points:
        for key, template in CAVEATS.items():
            if op["name"].startswith(key):
                op["caveat"] = template.format(rate=op["alert_rate"] * 100)

    preds = (proba_test >= thresholds["budget_1pct"]).astype(int)
    auc = auc_by_population(test, proba_test, y_test)

    gen_path = os.path.join(ART_DIR, "data_generation_summary.json")
    gen = json.load(open(gen_path)) if os.path.exists(gen_path) else {}

    metrics = build_metrics(test, y_test, proba_test, preds, auc,
                            operating_points, cost_curve, gen)
    with open(os.path.join(ART_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(model, os.path.join(ART_DIR, "xgb_fraud_model.joblib"))
    with open(os.path.join(ART_DIR, "feature_columns.json"), "w") as f:
        json.dump(feat_cols, f)
    with open(os.path.join(ART_DIR, "merchant_amount_stats.json"), "w") as f:
        json.dump(merchant_stats, f)

    save_scored_splits(model, splits, feat_cols, thresholds["budget_1pct"])
    write_shap(model, test[feat_cols], test, preds)
    print_summary(auc, gen, y_test, proba_test, operating_points)


if __name__ == "__main__":
    main()
