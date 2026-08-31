"""Generate the synthetic Razorpay-shaped transaction dataset."""
import os
import json
import uuid
from collections import deque

import numpy as np
import pandas as pd
from faker import Faker

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)
np.random.seed(RNG_SEED)
fake = Faker()
Faker.seed(RNG_SEED)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

N_MERCHANTS = 250
N_CUSTOMERS = 9000
N_NORMAL_TXNS = 34000        # ordinary one-off customer activity
N_BURST_CLUSTERS = 1900      # short bursts from one device (legit retries,
                             # power users, subscription runs -- AND some abuse)

SIM_START = pd.Timestamp("2026-08-01 00:00:00")
SIM_END = pd.Timestamp("2026-08-14 23:59:59")
VAL_WINDOW_START = pd.Timestamp("2026-08-08 00:00:00")   # threshold tuning
TEST_WINDOW_START = pd.Timestamp("2026-08-10 00:00:00")  # never touched until scoring

PAYMENT_METHODS = ["upi", "card", "netbanking", "wallet", "emi"]
METHOD_WEIGHTS = [0.46, 0.28, 0.14, 0.08, 0.04]
CATEGORIES = ["ecommerce", "food_delivery", "travel", "saas", "gaming", "utilities", "education"]

# Ground truth for the label generating process, the model has to recover
# these from the observed columns.
COEF = {
    "merchant_risk": 1.05,
    "amount_z": 0.95,
    "is_night": 1.30,
    "mismatch": 1.55,
    "new_customer": 0.80,
    "device_velocity": 1.15,    # applied to log1p(device_txn_count_1h)
    "ip_velocity": 0.65,        # applied to log1p(ip_txn_count_1h)
    "customer_velocity": 0.45,  # applied to log1p(customer_txn_count_24h)
}
# Irreducible label noise, so the task is not trivially separable. Tuned so the
# Bayes ceiling lands near what supervised fraud models reach on real payment
# data rather than an implausible 0.99+.
NOISE_SD = 0.32
TARGET_FRAUD_RATE = 0.015

TARGET_HOUR_WEIGHTS = np.array(
    [1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 7, 6, 4, 2, 1], dtype=float
)
TARGET_HOUR_WEIGHTS = TARGET_HOUR_WEIGHTS / TARGET_HOUR_WEIGHTS.sum()


def build_merchants():
    return pd.DataFrame([
        {
            "merchant_id": "mch_%04d" % i,
            "category": CATEGORIES[int(rng.integers(0, len(CATEGORIES)))],
            "risk_tier": str(rng.choice(["low", "medium", "high"], p=[0.70, 0.24, 0.06])),
            "avg_ticket": float(rng.lognormal(mean=6.5, sigma=0.8)),
        }
        for i in range(N_MERCHANTS)
    ])


def build_customers():
    """A minority of customers share a device (households, kiosks, device farms), which."""
    device_pool = ["dev_" + uuid.uuid4().hex[:10] for _ in range(int(N_CUSTOMERS * 0.88))]
    ip_pool = [fake.ipv4() for _ in range(int(N_CUSTOMERS * 0.80))]
    return pd.DataFrame({
        "customer_id": ["cust_%06d" % i for i in range(N_CUSTOMERS)],
        "device_id": rng.choice(device_pool, N_CUSTOMERS),
        "ip_address": rng.choice(ip_pool, N_CUSTOMERS),
        "is_new": rng.binomial(1, 0.22, N_CUSTOMERS),
    })


MERCHANTS = build_merchants()
CUSTOMERS = build_customers()


def _sample_hours(n):
    return rng.choice(np.arange(24), size=n, p=TARGET_HOUR_WEIGHTS)


def _random_times(n, start, end):
    span = (end - start).total_seconds()
    offs = rng.uniform(0, span, n)
    return pd.to_datetime(start) + pd.to_timedelta(offs, unit="s")


def emit_normal_transactions(n):
    """One-off activity: independent customers spread across the period."""
    cust_idx = rng.integers(0, N_CUSTOMERS, n)
    mer_idx = rng.integers(0, N_MERCHANTS, n)
    days = _random_times(n, SIM_START, SIM_END).floor("D")
    ts = days + pd.to_timedelta(_sample_hours(n), unit="h") \
              + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")
    return _assemble(cust_idx, mer_idx, ts)


def emit_burst_clusters(n_clusters):
    """Bursts of 2-9 transactions from one customer/device within 30 minutes."""
    sizes = rng.integers(2, 10, n_clusters)
    starts = _random_times(n_clusters, SIM_START, SIM_END - pd.Timedelta(minutes=30)).floor("D")
    starts = starts + pd.to_timedelta(_sample_hours(n_clusters), unit="h")
    cust_idx, mer_idx, times = [], [], []
    for c in range(n_clusters):
        k = int(sizes[c])
        cust = int(rng.integers(0, N_CUSTOMERS))
        mer = int(rng.integers(0, N_MERCHANTS))
        for o in np.sort(rng.uniform(0, 30 * 60, k)):
            cust_idx.append(cust)
            mer_idx.append(mer)
            times.append(starts[c] + pd.Timedelta(seconds=float(o)))
    return _assemble(np.array(cust_idx), np.array(mer_idx), pd.to_datetime(times))


def _assemble(cust_idx, mer_idx, ts):
    n = len(cust_idx)
    cust = CUSTOMERS.iloc[cust_idx].reset_index(drop=True)
    mer = MERCHANTS.iloc[mer_idx].reset_index(drop=True)
    amounts = np.round(rng.lognormal(mean=np.log(mer["avg_ticket"].values + 1), sigma=0.6), 2)
    return pd.DataFrame({
        "payment_id": ["pay_" + uuid.uuid4().hex[:14] for _ in range(n)],
        "customer_id": cust["customer_id"].values,
        "merchant_id": mer["merchant_id"].values,
        "merchant_category": mer["category"].values,
        "merchant_risk_tier": mer["risk_tier"].values,
        "timestamp": pd.to_datetime(ts),
        "amount_inr": amounts,
        "method": rng.choice(PAYMENT_METHODS, size=n, p=METHOD_WEIGHTS),
        "device_id": cust["device_id"].values,
        "ip_address": cust["ip_address"].values,
        "is_new_customer": cust["is_new"].values,
        "billing_shipping_mismatch": rng.binomial(1, 0.07, n),
    })


def trailing_count(df, key_col, window_minutes):
    """Count of PRIOR transactions sharing key_col in the trailing window."""
    counts = np.zeros(len(df), dtype=int)
    times = df["timestamp"].values
    keys = df[key_col].values
    buckets = {}
    win = np.timedelta64(window_minutes, "m")
    for i in range(len(df)):
        k = keys[i]
        t = times[i]
        dq = buckets.setdefault(k, deque())
        cutoff = t - win
        while dq and dq[0] < cutoff:
            dq.popleft()
        counts[i] = len(dq)
        dq.append(t)
    return counts


def add_velocity_features(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["device_txn_count_1h"] = trailing_count(df, "device_id", 60)
    df["ip_txn_count_1h"] = trailing_count(df, "ip_address", 60)
    df["customer_txn_count_24h"] = trailing_count(df, "customer_id", 60 * 24) + 1
    return df


def compute_signal(df, amount_mu, amount_sd):
    amt_z = (np.log1p(df["amount_inr"].values) - amount_mu) / (amount_sd + 1e-9)
    is_night = np.isin(pd.to_datetime(df["timestamp"]).dt.hour.values, [1, 2, 3, 4]).astype(float)
    risk = pd.Series(df["merchant_risk_tier"].values).map(
        {"low": 0.0, "medium": 1.0, "high": 2.0}).values
    return (
        COEF["merchant_risk"] * risk
        + COEF["amount_z"] * amt_z
        + COEF["is_night"] * is_night
        + COEF["mismatch"] * df["billing_shipping_mismatch"].values
        + COEF["new_customer"] * df["is_new_customer"].values
        + COEF["device_velocity"] * np.log1p(df["device_txn_count_1h"].values)
        + COEF["ip_velocity"] * np.log1p(df["ip_txn_count_1h"].values)
        + COEF["customer_velocity"] * np.log1p(df["customer_txn_count_24h"].values)
    )


def calibrate_intercept(latent, target_rate):
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(latent - mid)))).mean() > target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def label_baseline(df):
    log_amt = np.log1p(df["amount_inr"].values)
    amount_mu, amount_sd = float(log_amt.mean()), float(log_amt.std())
    signal = compute_signal(df, amount_mu, amount_sd)
    latent = signal + rng.normal(0, NOISE_SD, len(df))
    intercept = calibrate_intercept(latent, TARGET_FRAUD_RATE)
    prob = 1 / (1 + np.exp(-(latent - intercept)))
    out = df.copy()
    out["is_fraud"] = (rng.uniform(0, 1, len(df)) < prob).astype(int)
    out["spike_event_id"] = None
    return out, signal


SPIKE_DEFS = [
    {
        "id": "SPK-001", "label": "card_testing", "start": "2026-08-10 09:00",
        "duration_min": 45, "n_txns": 140, "n_merchants": 9,
        "device_pool_frac": 0.05, "ip_pool_frac": 0.04,
        "new_cust_p": 0.85, "mismatch_p": 0.55, "amount_log_shift": -0.45,
        "visibility": "dense device pool -- individually VISIBLE to a per-txn model",
        "note": "Automated BIN/card testing: many tiny authorisations from a "
                "handful of devices, probing for live card numbers.",
    },
    {
        "id": "SPK-002", "label": "synthetic_identity", "start": "2026-08-11 22:15",
        "duration_min": 30, "n_txns": 95, "n_merchants": 6,
        "device_pool_frac": 0.95, "ip_pool_frac": 0.90,
        "new_cust_p": 0.95, "mismatch_p": 0.68, "amount_log_shift": 0.10,
        "visibility": "distributed device pool -- individually INVISIBLE to a per-txn model",
        "note": "Synthetic identities: each fabricated customer brings its own "
                "device and IP, so no per-transaction velocity signal exists.",
    },
    {
        "id": "SPK-003", "label": "account_takeover", "start": "2026-08-12 14:40",
        "duration_min": 60, "n_txns": 180, "n_merchants": 12,
        "device_pool_frac": 0.85, "ip_pool_frac": 0.80,
        "new_cust_p": 0.20, "mismatch_p": 0.12, "amount_log_shift": 0.60,
        "visibility": "distributed pool, established accounts -- individually INVISIBLE",
        "note": "Account-takeover ring: real, aged accounts drained from many "
                "different devices. One transaction at a time it looks like "
                "ordinary loyal-customer traffic.",
    },
    {
        "id": "SPK-004", "label": "card_testing_small", "start": "2026-08-13 03:10",
        "duration_min": 25, "n_txns": 70, "n_merchants": 5,
        "device_pool_frac": 0.07, "ip_pool_frac": 0.03,
        "new_cust_p": 0.80, "mismatch_p": 0.51, "amount_log_shift": -0.40,
        "visibility": "dense device pool -- individually VISIBLE to a per-txn model",
        "note": "Smaller overnight repeat of the card-testing pattern.",
    },
]

CONTROL_DEF = {
    "id": "CTRL-001", "start": "2026-08-12 18:00", "duration_min": 40,
    "n_txns": 110, "n_merchants": 1,
    "device_pool_frac": 0.95, "ip_pool_frac": 0.92,
    "new_cust_p": 0.25, "mismatch_p": 0.05, "amount_log_shift": 0.0,
    "description": "Genuine legitimate demand spike (flash sale) -- NOT fraud. "
                   "Specificity control: tests whether the system avoids blocking "
                   "an innocent merchant having a good day.",
}


def _pool(n, frac, prefix=None):
    size = max(2, int(round(n * frac)))
    items = [prefix + uuid.uuid4().hex[:10] for _ in range(size)] if prefix \
        else [fake.ipv4() for _ in range(size)]
    return list(rng.choice(items, n))


def _event_chunk(spec, merchant_rows, is_fraud, event_id):
    n = spec["n_txns"]
    start = pd.Timestamp(spec["start"])
    ts = _random_times(n, start, start + pd.Timedelta(minutes=spec["duration_min"]))
    amounts = np.round(
        rng.lognormal(mean=np.log(merchant_rows["avg_ticket"].values + 1) + spec["amount_log_shift"],
                      sigma=0.35), 2)
    return pd.DataFrame({
        "payment_id": ["pay_" + uuid.uuid4().hex[:14] for _ in range(n)],
        "customer_id": ["cust_evt_" + uuid.uuid4().hex[:8] for _ in range(n)],
        "merchant_id": merchant_rows["merchant_id"].values,
        "merchant_category": merchant_rows["category"].values,
        "merchant_risk_tier": merchant_rows["risk_tier"].values,
        "timestamp": pd.to_datetime(ts),
        "amount_inr": amounts,
        "method": rng.choice(["upi", "card"], size=n, p=[0.65, 0.35]),
        "device_id": _pool(n, spec["device_pool_frac"], prefix="dev_"),
        "ip_address": _pool(n, spec["ip_pool_frac"]),
        "is_new_customer": rng.binomial(1, spec["new_cust_p"], n),
        "billing_shipping_mismatch": rng.binomial(1, spec["mismatch_p"], n),
        "is_fraud": int(is_fraud),
        "spike_event_id": event_id,
    })


def inject_events(baseline):
    chunks = []
    for spec in SPIKE_DEFS:
        targets = MERCHANTS.sample(spec["n_merchants"], replace=False, random_state=RNG_SEED)
        rows = targets.sample(spec["n_txns"], replace=True, random_state=RNG_SEED).reset_index(drop=True)
        chunks.append(_event_chunk(spec, rows, is_fraud=True, event_id=spec["id"]))

    ctrl_merchant = MERCHANTS[MERCHANTS["category"] == "ecommerce"].sample(
        1, random_state=RNG_SEED).iloc[0]
    ctrl_rows = pd.DataFrame([ctrl_merchant.to_dict()] * CONTROL_DEF["n_txns"]).reset_index(drop=True)
    chunks.append(_event_chunk(CONTROL_DEF, ctrl_rows, is_fraud=False, event_id=CONTROL_DEF["id"]))

    full = pd.concat([baseline] + chunks, ignore_index=True)
    return full.sort_values("timestamp").reset_index(drop=True)


def bayes_ceiling_auc(signal, labels):
    """AUC an oracle would reach knowing the true signal but not the noise."""
    from sklearn.metrics import roc_auc_score
    if labels.sum() in (0, len(labels)):
        return None
    return float(roc_auc_score(labels, signal))


def main():
    stream = pd.concat(
        [emit_normal_transactions(N_NORMAL_TXNS), emit_burst_clusters(N_BURST_CLUSTERS)],
        ignore_index=True,
    )
    stream = add_velocity_features(stream)
    baseline, signal = label_baseline(stream)
    ceiling = bayes_ceiling_auc(signal, baseline["is_fraud"].values)

    full = inject_events(baseline)
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    # Recomputed across the full stream so injected rings carry the velocity
    # their device pools imply. Baseline labels were fixed before injection.
    full = add_velocity_features(full)

    full["split"] = np.where(
        full["timestamp"] >= TEST_WINDOW_START, "test",
        np.where(full["timestamp"] >= VAL_WINDOW_START, "val", "train"))
    full["is_test_window"] = (full["split"] == "test").astype(int)

    out_path = os.path.join(OUT_DIR, "transactions.csv")
    full.to_csv(out_path, index=False)

    with open(os.path.join(OUT_DIR, "spike_ground_truth.json"), "w") as f:
        json.dump(SPIKE_DEFS, f, indent=2, default=str)
    ctrl_meta = dict(CONTROL_DEF)
    ctrl_meta["merchant_id"] = str(full[full.spike_event_id == "CTRL-001"]["merchant_id"].iloc[0])
    with open(os.path.join(OUT_DIR, "control_event_ground_truth.json"), "w") as f:
        json.dump(ctrl_meta, f, indent=2, default=str)

    summary = {
        "n_transactions": int(len(full)),
        "n_merchants": N_MERCHANTS,
        "n_customers": N_CUSTOMERS,
        "windows": {
            "train": "%s to %s" % (SIM_START.date(), (VAL_WINDOW_START - pd.Timedelta(days=1)).date()),
            "val": "%s to %s" % (VAL_WINDOW_START.date(), (TEST_WINDOW_START - pd.Timedelta(days=1)).date()),
            "test": "%s to %s" % (TEST_WINDOW_START.date(), SIM_END.date()),
        },
        "split_sizes": {str(k): int(v) for k, v in full["split"].value_counts().items()},
        "fraud_rate_by_split": {
            k: round(float(full[full.split == k].is_fraud.mean()), 4) for k in ["train", "val", "test"]
        },
        "generating_coefficients": COEF,
        "label_noise_sd": NOISE_SD,
        "bayes_ceiling_auc_baseline_population": round(ceiling, 4) if ceiling else None,
        "bayes_ceiling_note": (
            "AUC achievable by an oracle that knows the exact generating signal but "
            "not the irreducible noise term. No model trained on the observed columns "
            "can exceed this on the baseline population. Injected ring events are "
            "excluded from this figure because they are labelled by construction "
            "rather than drawn from the logistic process."
        ),
        "injected_events": [
            {"id": s["id"], "label": s["label"], "n_txns": s["n_txns"],
             "device_pool_frac": s["device_pool_frac"], "visibility": s["visibility"]}
            for s in SPIKE_DEFS
        ] + [{"id": CONTROL_DEF["id"], "label": "legitimate_demand_spike",
              "n_txns": CONTROL_DEF["n_txns"], "device_pool_frac": CONTROL_DEF["device_pool_frac"],
              "visibility": "negative control -- must NOT be blocked"}],
    }
    with open(os.path.join(OUT_DIR, "data_generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Generated %d transactions across %d merchants / %d customers"
          % (len(full), N_MERCHANTS, N_CUSTOMERS))
    for k in ["train", "val", "test"]:
        sub = full[full.split == k]
        print("  %-5s: %6d txns | fraud rate %5.2f%%" % (k, len(sub), sub.is_fraud.mean() * 100))
    print("  Injected ring txns: %d (incl. %d legitimate-control txns)"
          % (full.spike_event_id.notna().sum(), CONTROL_DEF["n_txns"]))
    print("  Bayes-ceiling AUC on baseline population: %.4f" % ceiling)
    print("  Saved to %s" % out_path)


if __name__ == "__main__":
    main()
