# RazorShield - Measured Results

Every number on this page is rendered directly from `artifacts/` by
`evaluation/report.py`. Nothing here is typed by hand. Regenerate with
`./run_pipeline.sh`.

### How the data is split

| Window | Dates | Transactions | Fraud rate |
|---|---|---|---|
| Train | 2026-08-01 to 2026-08-07 | 22,213 | 1.6% |
| Validation | 2026-08-08 to 2026-08-09 | 6,544 | 1.7% |
| Test (held out) | 2026-08-10 to 2026-08-14 | 16,345 | 4.4% |

_Every threshold and operating point is selected on the validation window. The test window is scored once, after all choices are frozen._

### Model quality (held-out test window)

| Measure | Value |
|---|---|
| ROC-AUC, ordinary fraud population | **0.9018** |
| Bayes ceiling for this data (oracle limit) | 0.9306 |
| ROC-AUC, coordinated-ring population | 0.7350 |
| ROC-AUC, blended across both populations | 0.7889 |
| PR-AUC (blended) | 0.1898 |

The classifier reaches **0.9018 against a hard ceiling of 0.9306** on the fraud it is designed for. The ring figure is far lower and that is the finding, not a defect: two of the four injected rings distribute across near-unique devices, so no per-transaction velocity signal exists for any classifier to learn. The blended number mixes the two populations and is reported only for completeness.

### Operating points (thresholds chosen on validation, scored on test)

Cost model: **Rs 45 per manual review**, and a missed fraud costs the full transaction value plus a **Rs 750 chargeback fee**.

| Operating point | Threshold | Alert rate | Precision | Recall | FP rate | FP review cost | Net benefit vs no model |
|---|---|---|---|---|---|---|---|
| **budget_1pct (primary)** | 0.9071 | 0.7% | 41.5% | 6.9% | 0.4% | Rs 3,105 | Rs 156,616 |
| budget_2pct | 0.7986 | 2.0% | 35.4% | 16.1% | 1.3% | Rs 9,450 | Rs 338,241 |
| budget_5pct | 0.5816 | 4.6% | 25.3% | 26.7% | 3.6% | Rs 25,290 | Rs 559,555 |
| cost_optimal_unconstrained | 0.1820 | 12.7% | 15.1% | 43.9% | 11.3% | Rs 79,290 | Rs 840,038 |
| f1_optimal | 0.8069 | 1.9% | 35.9% | 15.6% | 1.3% | Rs 8,910 | Rs 327,044 |

- `cost_optimal_unconstrained` -- Lowest expected rupee cost, but it alerts on 12.7% of all traffic. Reported for completeness -- no risk-ops team can staff this, which is precisely why the deployed operating point is budget-constrained.
- `f1_optimal` -- What maximising F1 would pick. Shown to make the difference visible: F1 implicitly prices one missed fraud equal to one false alarm, which is wrong by roughly two orders of magnitude in payments.

### Standalone classifier vs the full two-layer system

| System | Precision | Recall | F1 | Alert rate | FP rate | FP review cost | Net benefit |
|---|---|---|---|---|---|---|---|
| Per-transaction XGBoost only | 41.5% | 6.9% | 11.8% | 0.7% | 0.4% | Rs 3,105 | Rs 156,616 |
| **Two-layer (model + temporal)** | 60.3% | 46.1% | 52.3% | 3.3% | 1.4% | Rs 9,765 | Rs 598,825 |

**Marginal cost of the temporal layer.** It adds **280 additional true positives for 148 additional false positives** -- a marginal review cost of **Rs 24 per extra fraudulent transaction caught**, against the Rs 45 per-review assumption used throughout. Both confusion matrices are in `artifacts/metrics.json`.

### Per-event detection (all four injected ring events)

| Event | Attack type | Txns | Device pool | Standalone recall | Combined recall | Event flagged | Latency |
|---|---|---|---|---|---|---|---|
| SPK-001 | card testing | 140 | 5% | 6.4% | 97.1% | yes | 15.0 min |
| SPK-002 | synthetic identity | 95 | 95% | 1.1% | 75.8% | yes | 15.0 min |
| SPK-003 | account takeover | 180 | 85% | 0.6% | 7.2% | yes | 5.0 min |
| SPK-004 | card testing small | 70 | 7% | 10.0% | 94.3% | yes | 5.0 min |

**4 of 4 ring events raised an alert**, with latency measured to the close of the first triggering 15-minute bucket (a bucket cannot alert before it closes, so this is the earliest a real deployment could have known).

**Where this is weakest, stated plainly.** `SPK-003` (account takeover) reaches only 7.2% transaction-level recall. It spreads 180 transactions across 12 merchants and near-unique devices, so no 15-minute aggregation on any axis sees enough concentration to act on individual transactions. The event is still escalated -- its bucket is flagged and the investigation agent routes it to a human -- but most of its individual transactions are not caught. This is the honest residual limitation of the current design.

### Specificity control: does it avoid blocking an innocent merchant?

`CTRL-001` is a **genuine legitimate demand spike** (a flash sale, 110 transactions, not fraud) injected into the same test window. A detector that flags every volume anomaly is useless no matter how good its recall looks.

| Layer | Result on CTRL-001 |
|---|---|
| Temporal layer | flagged the bucket as anomalous -- correctly, it genuinely is a volume anomaly |
| Per-transaction model | 0.0% of its transactions flagged |
| Full two-layer system | **0.0% of its transactions flagged** |
| Investigation agent | **ALLOW** at severity 0.197 (REVIEW threshold 0.35) |

Specificity cannot come from the volume signal, because the volume signal is genuinely anomalous here. It comes from the cluster-level ring-evidence score and the investigation agent, which weigh identity churn, merchant spread, device/IP reuse and mismatch rates rather than reacting to volume alone.

### Investigation agent decisions (real output, regenerated each run)

| Event | Severity | Root-cause hypothesis | Decision |
|---|---|---|---|
| CTRL-001 | 0.197 | Insufficient evidence to classify | **ALLOW** |
| SPK-001 | 0.889 | Card testing / BIN attack | **BLOCK** |
| SPK-002 | 0.530 | Synthetic identity fraud | **REVIEW** |
| SPK-003 | 0.383 | Insufficient evidence to classify | **REVIEW** |
| SPK-004 | 0.847 | Card testing / BIN attack | **BLOCK** |

Thresholds are fixed and inspectable (REVIEW at 0.35, BLOCK at 0.75) and the decision is computed from the evidence, never generated by a language model. Full reasoning chains are in `artifacts/audit_trail.jsonl`.
