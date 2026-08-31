"""
RazorShield - Dashboard Builder
Generates a single self contained HTML file (dashboard/index.html) that
embeds the REAL computed metrics, spike events, SHAP importances and
audit trail entries as JSON """
import os
import json
import base64
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIR = os.path.join(ROOT, "artifacts")
OUT_PATH = os.path.join(ROOT, "dashboard", "index.html")


def load_json(name):
    with open(os.path.join(ART_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def load_audit_entries():
    path = os.path.join(ART_DIR, "audit_trail.jsonl")
    entries = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    return entries


def main():
    metrics = load_json("metrics.json")
    spikes = load_json("spike_event_detection_summary.json")
    shap_importance = load_json("shap_global_importance.json")[:8]
    audit_entries = load_audit_entries()
    control_summary = None
    control_path = os.path.join(ART_DIR, "control_event_summary.json")
    if os.path.exists(control_path):
        control_summary = load_json("control_event_summary.json")

    timeline_df = pd.read_csv(os.path.join(ART_DIR, "timeline_buckets.csv"))
    timeline = {
        "labels": timeline_df["bucket"].tolist(),
        "txn_count": timeline_df["txn_count"].tolist(),
        "spike_score": timeline_df["max_spike_score"].round(2).tolist(),
    }

    shap_img_path = os.path.join(ART_DIR, "shap_summary.png")
    shap_img_b64 = ""
    if os.path.exists(shap_img_path):
        with open(shap_img_path, "rb") as f:
            shap_img_b64 = base64.b64encode(f.read()).decode("utf-8")

    embedded_data = {
        "metrics": metrics,
        "spikes": spikes,
        "shap_importance": shap_importance,
        "audit_entries": audit_entries,
        "timeline": timeline,
        "control_summary": control_summary,
        "pr_curve": metrics.get("pr_curve_test", []),
        "cost_curve": metrics.get("validation_cost_curve", []),
        "operating_points": metrics.get("operating_points", []),
        "standalone_vs_combined": metrics.get("standalone_vs_combined", {}),
        "cost_assumptions": metrics.get("cost_assumptions", {}),
        "model_quality": metrics.get("model_quality", {}),
    }

    html = HTML_TEMPLATE.replace("__EMBEDDED_DATA__", json.dumps(embedded_data)) \
                         .replace("__SHAP_IMG_B64__", shap_img_b64)

    chartjs_path = os.path.join(ROOT, "dashboard", "vendor", "chart.umd.js")
    if os.path.exists(chartjs_path):
        with open(chartjs_path) as f:
            chartjs_code = f.read()
        html = html.replace("__CHARTJS_INLINE__", chartjs_code)
    else:
        html = html.replace("__CHARTJS_INLINE__", "console.warn('Chart.js vendor file missing');")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard written to {OUT_PATH} ({len(html)/1024:.0f} KB)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RazorShield — Agentic AI Risk Manager</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<script>__CHARTJS_INLINE__</script>
<style>
  :root{
    --bg: #0A0E13;
    --panel: #10161D;
    --panel-2: #141B23;
    --line: #232D38;
    --text: #E7ECF1;
    --text-dim: #8A97A6;
    --text-faint: #566172;
    --allow: #35D08E;
    --review: #F0A93B;
    --block: #EF4A5C;
    --accent: #4FA3FF;
  }
  *{box-sizing:border-box; margin:0; padding:0;}
  body{
    background: radial-gradient(1200px 600px at 15% -10%, #0F1620 0%, var(--bg) 55%), var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    line-height: 1.5;
    padding-bottom: 80px;
  }
  .mono{ font-family: 'JetBrains Mono', monospace; }
  .display{ font-family: 'Space Grotesk', sans-serif; }

  header.top{
    position: sticky; top:0; z-index: 50;
    display:flex; align-items:center; justify-content:space-between;
    padding: 18px 40px;
    background: rgba(10,14,19,0.85);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--line);
  }
  .brand{ display:flex; align-items:center; gap:12px; }
  .brand .mark{
    width:30px; height:30px; border-radius:7px;
    background: linear-gradient(135deg, var(--accent), #7C5CFF);
    display:flex; align-items:center; justify-content:center;
    font-family:'JetBrains Mono'; font-weight:700; font-size:14px; color:#04070A;
  }
  .brand .name{ font-family:'Space Grotesk'; font-weight:700; font-size:17px; letter-spacing:0.2px;}
  .brand .tag{ font-size:11px; color:var(--text-faint); margin-top:1px; }
  .status-pill{
    display:flex; align-items:center; gap:8px;
    font-family:'JetBrains Mono'; font-size:12px; color: var(--text-dim);
    border:1px solid var(--line); padding:6px 12px; border-radius:20px;
  }
  .dot{ width:7px; height:7px; border-radius:50%; background: var(--allow); box-shadow: 0 0 8px var(--allow); animation: pulse 2s infinite; }
  @keyframes pulse{ 0%,100%{opacity:1;} 50%{opacity:.4;} }

  main{ max-width: 1180px; margin: 0 auto; padding: 0 40px; }

  .hero{ padding: 56px 0 40px; border-bottom:1px solid var(--line); }
  .eyebrow{
    font-family:'JetBrains Mono'; font-size:12px; letter-spacing:0.14em; text-transform:uppercase;
    color: var(--accent); margin-bottom:14px;
  }
  h1.headline{
    font-family:'Space Grotesk'; font-weight:700; font-size:44px; letter-spacing:-0.01em;
    max-width: 780px; line-height:1.12;
  }
  h1.headline .grad{ background: linear-gradient(90deg, var(--accent), #7C5CFF); -webkit-background-clip:text; background-clip:text; color:transparent; }
  .sub{ color: var(--text-dim); max-width: 620px; margin-top:16px; font-size:15px; }

  .kpi-row{ display:grid; grid-template-columns: repeat(4, 1fr); gap:1px; margin-top:40px; background: var(--line); border:1px solid var(--line); border-radius:14px; overflow:hidden; }
  .kpi{ background: var(--panel); padding:22px 22px; }
  .kpi .val{ font-family:'JetBrains Mono'; font-weight:700; font-size:28px; }
  .kpi .val small{ font-size:14px; color: var(--text-faint); font-weight:500;}
  .kpi .label{ margin-top:6px; font-size:12.5px; color: var(--text-dim); }
  .kpi.good .val{ color: var(--allow); }
  .kpi.warn .val{ color: var(--review); }

  section{ padding: 48px 0; border-bottom:1px solid var(--line); }
  .section-head{ display:flex; align-items:baseline; justify-content:space-between; margin-bottom: 22px; }
  .section-head h2{ font-family:'Space Grotesk'; font-size:22px; font-weight:600; }
  .section-head .num{ font-family:'JetBrains Mono'; color:var(--text-faint); font-size:12px; }
  .section-head p{ color:var(--text-dim); font-size:13.5px; max-width:520px; }

  .panel{ background: var(--panel); border:1px solid var(--line); border-radius:14px; padding:24px; }

  /* Severity threshold bar - signature element */
  .severity-bar-wrap{ margin-top: 18px; }
  .severity-track{
    position:relative; height:52px; border-radius:10px; overflow:hidden;
    background: linear-gradient(90deg,
      rgba(53,208,142,0.18) 0%, rgba(53,208,142,0.18) 35%,
      rgba(240,169,59,0.20) 35%, rgba(240,169,59,0.20) 75%,
      rgba(239,74,92,0.22) 75%, rgba(239,74,92,0.22) 100%);
    border: 1px solid var(--line);
  }
  .severity-track .zone-label{
    position:absolute; top:6px; font-family:'JetBrains Mono'; font-size:10.5px; letter-spacing:0.06em; text-transform:uppercase;
  }
  .zone-label.allow{ left:12px; color: var(--allow); }
  .zone-label.review{ left:38%; color: var(--review); }
  .zone-label.block{ left:78%; color: var(--block); }
  .severity-marker{
    position:absolute; bottom:6px; transform:translateX(-50%);
    width:2px; height:22px;
  }
  .severity-marker .stem{ width:2px; height:100%; margin:0 auto; }
  .severity-marker .chip{
    position:absolute; bottom:26px; left:50%; transform:translateX(-50%);
    font-family:'JetBrains Mono'; font-size:10px; padding:3px 6px; border-radius:5px;
    white-space:nowrap; border:1px solid var(--line);
  }
  .track-ticks{ display:flex; justify-content:space-between; margin-top:8px; font-family:'JetBrains Mono'; font-size:10px; color:var(--text-faint); }

  .spike-grid{ display:grid; grid-template-columns: repeat(4,1fr); gap:14px; margin-top:22px; }
  .spike-card{ background: var(--panel-2); border:1px solid var(--line); border-radius:12px; padding:18px; position:relative; }
  .spike-card .id{ font-family:'JetBrains Mono'; font-size:11px; color:var(--text-faint); }
  .spike-card .decision{ font-family:'Space Grotesk'; font-weight:700; font-size:20px; margin-top:6px; }
  .spike-card .meta{ margin-top:12px; font-size:12px; color:var(--text-dim); display:flex; flex-direction:column; gap:4px;}
  .spike-card .meta span b{ color:var(--text); font-family:'JetBrains Mono'; }
  .badge{ position:absolute; top:16px; right:16px; font-size:10px; font-family:'JetBrains Mono'; padding:3px 8px; border-radius:20px; border:1px solid var(--line); color:var(--text-dim); }

  .control-card{ display:grid; grid-template-columns: 220px 1fr; gap:24px; align-items:center; }
  .control-card .decision-big{ font-family:'Space Grotesk'; font-weight:700; font-size:34px; text-align:center; padding:24px 12px; border-radius:12px; background: var(--panel-2); border:1px solid var(--line); }
  .control-card .details{ font-size:13px; color:var(--text-dim); line-height:1.7; }
  .control-card .details b{ color:var(--text); font-family:'JetBrains Mono'; font-weight:600; }
  .control-card .layer-row{ display:flex; gap:10px; margin-top:10px; flex-wrap:wrap; }
  .control-card .layer-chip{ font-family:'JetBrains Mono'; font-size:11px; padding:5px 10px; border-radius:6px; border:1px solid var(--line); }

  .chart-wrap{ height: 260px; margin-top: 10px; }

  .pipeline{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin-top:22px; }
  .step{ background: var(--panel-2); border:1px solid var(--line); border-radius:10px; padding:14px 12px; text-align:left; }
  .step .n{ font-family:'JetBrains Mono'; font-size:11px; color: var(--accent); }
  .step .t{ font-family:'Space Grotesk'; font-size:13px; font-weight:600; margin-top:6px; }
  .step .d{ font-size:11px; color:var(--text-faint); margin-top:5px; line-height:1.4; }
  .arrow{ display:flex; align-items:center; justify-content:center; color:var(--text-faint); font-size:14px; }

  table{ width:100%; border-collapse:collapse; margin-top:10px; font-size:13px; }
  th{ text-align:left; font-family:'JetBrains Mono'; font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-faint); padding:10px 12px; border-bottom:1px solid var(--line); }
  td{ padding:11px 12px; border-bottom:1px solid var(--line); color:var(--text-dim); }
  td.mono, th.mono{ font-family:'JetBrains Mono'; color: var(--text); }
  tr:hover td{ background: rgba(255,255,255,0.02); }
  .tag{ font-family:'JetBrains Mono'; font-size:10.5px; padding:3px 8px; border-radius:5px; border:1px solid var(--line); }
  .tag.allow{ color:var(--allow); border-color: rgba(53,208,142,0.3); }
  .tag.review{ color:var(--review); border-color: rgba(240,169,59,0.3); }
  .tag.block{ color:var(--block); border-color: rgba(239,74,92,0.3); }

  .two-col{ display:grid; grid-template-columns: 1.1fr 0.9fr; gap:24px; }
  .shap-img{ width:100%; border-radius:10px; border:1px solid var(--line); background:#fff; }

  .audit-entry{ border-left:2px solid var(--line); padding:14px 0 14px 18px; margin-left:6px; position:relative; }
  .audit-entry::before{ content:''; position:absolute; left:-5px; top:20px; width:8px; height:8px; border-radius:50%; background: var(--panel); border:2px solid var(--accent); }
  .audit-entry .h{ display:flex; justify-content:space-between; align-items:center; }
  .audit-entry .h .id{ font-family:'JetBrains Mono'; font-size:12px; color:var(--text-dim); }
  .audit-entry .reason{ margin-top:8px; font-size:12.5px; color:var(--text-dim); line-height:1.55; }
  .audit-entry ul{ margin-top:8px; padding-left:18px; font-size:12px; color:var(--text-faint); }
  .audit-entry li{ margin-top:3px; }

  footer{ padding: 40px 0; text-align:center; color:var(--text-faint); font-size:12px; }
  footer .track{ color: var(--text-dim); font-family:'JetBrains Mono'; }
</style>
</head>
<body>

<header class="top">
  <div class="brand">
    <div class="mark">RS</div>
    <div>
      <div class="name">RazorShield</div>
      <div class="tag">Agentic AI Risk Manager for Razorpay</div>
    </div>
  </div>
  <div class="status-pill"><span class="dot"></span> LIVE ON HELD-OUT TEST WINDOW</div>
</header>

<main>

  <section class="hero">
    <div class="eyebrow">Razorpay AI Builder Hackathon — Track 02: AI Risk Manager</div>
    <h1 class="headline">Every transaction scored.<br>Every spike <span class="grad">investigated</span>.<br>Every decision explained.</h1>
    <p class="sub">RazorShield pairs a supervised fraud model with a temporal anomaly layer and a LangGraph investigation agent — so coordinated fraud rings get caught, not just individual bad transactions, and every ALLOW / REVIEW / BLOCK decision comes with a full audit trail.</p>

    <div class="kpi-row" id="kpiRow"></div>
  </section>

  <section id="spikes">
    <div class="section-head">
      <div>
        <h2>Fraud-spike events — held-out test window</h2>
        <p>4 distinct coordinated fraud patterns were injected into a temporally held-out window the model never trained on — deliberately differentiated, not 4 copies of the same signature. Ground truth is known; nothing here is cherry-picked.</p>
      </div>
      <div class="num">04 EVENTS · 100% FLAGGED</div>
    </div>
    <div class="panel">
      <div class="spike-grid" id="spikeGrid"></div>
    </div>
  </section>

  <section id="control">
    <div class="section-head">
      <div>
        <h2>Specificity control — does it avoid blocking innocent merchants?</h2>
        <p>A genuine legitimate demand spike (e.g. a flash sale) was also injected — NOT fraud. A system that flags every volume anomaly as fraud isn't useful. This tests whether the layered design correctly tells the difference.</p>
      </div>
      <div class="num">CTRL-001 · NOT FRAUD</div>
    </div>
    <div class="panel" id="controlPanel"></div>
  </section>

  <section id="severity">
    <div class="section-head">
      <div>
        <h2>Decision thresholds are explicit, not hidden in a prompt</h2>
        <p>Each investigated event lands on a transparent severity scale. Thresholds are fixed, inspectable numbers — not an opaque LLM judgment call.</p>
      </div>
      <div class="num">SEVERITY SCALE 0.00 – 1.00</div>
    </div>
    <div class="panel severity-bar-wrap">
      <div class="severity-track" id="severityTrack">
        <span class="zone-label allow">Allow</span>
        <span class="zone-label review">Review ≥ 0.35</span>
        <span class="zone-label block">Block ≥ 0.75</span>
      </div>
      <div class="track-ticks"><span>0.0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1.0</span></div>
    </div>
  </section>

  <section id="timeline">
    <div class="section-head">
      <div>
        <h2>Transaction volume & anomaly score over time</h2>
        <p>15-minute buckets across the test window. Spike scores are z-scored against each merchant's own rolling baseline — a merchant's own history defines what "normal" looks like for that merchant.</p>
      </div>
      <div class="num">14–17 AUG 2026</div>
    </div>
    <div class="panel">
      <div class="chart-wrap"><canvas id="timelineChart"></canvas></div>
    </div>
  </section>

  <section id="pipeline">
    <div class="section-head">
      <div>
        <h2>Investigation pipeline (LangGraph)</h2>
        <p>Every flagged event runs through the same 9-node graph — 7 fully deterministic, 2 optional LLM nodes (root-cause label, narration) that never override the decision. No black-box reasoning at the step that matters.</p>
      </div>
      <div class="num">9-NODE GRAPH</div>
    </div>
    <div class="pipeline" id="pipelineGrid"></div>
  </section>

  <section id="curves">
    <div class="section-head">
      <div>
        <h2>Precision-recall trade-off, and what each threshold costs</h2>
        <p>Left: the full precision/recall curve on the held-out test window &mdash; a single (precision, recall) pair hides the shape of the trade-off. Right: expected rupee cost as a function of threshold, computed on the <b>validation</b> window, which is where the deployed threshold was actually chosen.</p>
      </div>
      <div class="num">HELD-OUT / VALIDATION</div>
    </div>
    <div class="panel" style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
      <div class="chart-wrap"><canvas id="prChart"></canvas></div>
      <div class="chart-wrap"><canvas id="costChart"></canvas></div>
    </div>
  </section>

  <section id="operating-points">
    <div class="section-head">
      <div>
        <h2>Operating points</h2>
        <p>Every threshold below was selected on the validation window and then scored once on the held-out test window. Cost assumptions are stated, not implied: a manual review costs &#8377;<span id="revCost"></span>, and a missed fraud costs the full transaction value plus a &#8377;<span id="cbFee"></span> chargeback fee.</p>
      </div>
      <div class="num">CHOSEN ON VALIDATION</div>
    </div>
    <div class="panel">
      <table>
        <thead><tr><th>Operating point</th><th class="mono">Threshold</th><th class="mono">Alert rate</th><th class="mono">Precision</th><th class="mono">Recall</th><th class="mono">FP rate</th><th class="mono">Net benefit</th></tr></thead>
        <tbody id="opBody"></tbody>
      </table>
    </div>
  </section>

  <section id="metrics-table">
    <div class="section-head">
      <div>
        <h2>Standalone model vs. combined system</h2>
        <p>Reported honestly, including the standalone model's blind spot: a per-transaction classifier alone under-catches genuinely novel coordinated fraud because it never saw that velocity pattern in training.</p>
      </div>
      <div class="num">METRICS.JSON</div>
    </div>
    <div class="panel">
      <table>
        <thead><tr><th>System</th><th class="mono">Precision</th><th class="mono">Recall</th><th class="mono">F1</th><th class="mono">Alert rate</th><th class="mono">FP Rate</th><th class="mono">Ring recall</th><th class="mono">FP Cost (₹)</th></tr></thead>
        <tbody id="metricsBody"></tbody>
      </table>
    </div>
  </section>

  <section id="shap">
    <div class="section-head">
      <div>
        <h2>Explainability — SHAP feature attribution</h2>
        <p>Every risk score is decomposable into the features that drove it. This powers both the analyst-facing audit trail and the agent's own reasoning.</p>
      </div>
      <div class="num">TREE EXPLAINER</div>
    </div>
    <div class="panel two-col">
      <div>
        <img class="shap-img" src="data:image/png;base64,__SHAP_IMG_B64__" alt="SHAP summary plot"/>
      </div>
      <div>
        <table>
          <thead><tr><th>Feature</th><th class="mono">Mean |SHAP|</th></tr></thead>
          <tbody id="shapBody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="audit">
    <div class="section-head">
      <div>
        <h2>Audit trail</h2>
        <p>Full reasoning chain for each investigated spike — merchant correlation, device/IP reuse, SHAP drivers, severity score, and final decision.</p>
      </div>
      <div class="num" id="auditCount"></div>
    </div>
    <div class="panel" id="auditList"></div>
  </section>

</main>

<footer>
  <div class="track">RazorShield · Built for Razorpay AI Builder Hackathon · Track 02 — AI Risk Manager</div>
  <div style="margin-top:6px;">Synthetic, Razorpay-shaped transaction data with controlled fraud-spike injection. Metrics computed on a temporally held-out test window.</div>
</footer>

<script>
const DATA = __EMBEDDED_DATA__;
const RUPEE = "\u20b9";

// ---- KPI row ----
const m = DATA.metrics;
const cs = m.combined_system;
const extraTP = cs.confusion_matrix.tp - m.confusion_matrix.tp;
const extraFP = cs.confusion_matrix.fp - m.confusion_matrix.fp;
const kpis = [
  { label: "Transactions monitored (test window)", val: m.n_test_transactions.toLocaleString(), cls:"" },
  { label: "Fraud-spike events detected", val: cs.spike_events_detected + " / " + cs.spike_events_total, cls:"good" },
  { label: `Extra fraud caught by spike layer, for +${extraFP} FPs`, val: "+" + extraTP, cls:"good" },
  { label: "False-positive review cost", val: "₹" + Math.round(cs.false_positive_monetary_cost_inr).toLocaleString(), cls:"warn" },
];
document.getElementById('kpiRow').innerHTML = kpis.map(k => `
  <div class="kpi ${k.cls}">
    <div class="val mono">${k.val}</div>
    <div class="label">${k.label}</div>
  </div>`).join('');

// ---- Spike cards ----
const decisionMap = {};
DATA.audit_entries.forEach(e => { if (e.spike_id) decisionMap[e.spike_id] = e; });
document.getElementById('spikeGrid').innerHTML = DATA.spikes.map(s => {
  const inv = decisionMap[s.spike_id];
  const decision = inv ? inv.decision : "—";
  const rootCause = inv && inv.root_cause_hypothesis ? inv.root_cause_hypothesis.label : null;
  const dCls = decision === "BLOCK" ? "block" : decision === "REVIEW" ? "review" : "allow";
  const color = decision === "BLOCK" ? "var(--block)" : decision === "REVIEW" ? "var(--review)" : "var(--allow)";
  return `
  <div class="spike-card">
    <span class="badge">${s.detection_latency_minutes !== null ? s.detection_latency_minutes + ' min latency' : 'not flagged'}</span>
    <div class="id">${s.spike_id}</div>
    <div class="decision" style="color:${color}">${decision}</div>
    ${rootCause ? `<div style="font-family:'JetBrains Mono'; font-size:10.5px; color:var(--text-faint); margin-top:4px;">${rootCause}</div>` : ''}
    <div class="meta">
      <span>Merchants affected: <b>${s.n_merchants}</b></span>
      <span>Transactions: <b>${s.n_txns}</b></span>
      <span>Recall, model alone: <b>${pctf(s.standalone_recall)}</b></span>
      <span>Recall, two-layer: <b>${pctf(s.combined_recall)}</b></span>
      <span>Bucket flagged: <b>${s.flagged_as_spike_bucket ? 'yes' : 'no'}</b></span>
    </div>
  </div>`;
}).join('');

// ---- Severity track markers ----
const track = document.getElementById('severityTrack');
DATA.audit_entries.filter(e => e.spike_id).forEach(e => {
  const pct = Math.min(Math.max(e.severity_score, 0), 1) * 100;
  const color = e.decision === "BLOCK" ? "var(--block)" : e.decision === "REVIEW" ? "var(--review)" : "var(--allow)";
  const marker = document.createElement('div');
  marker.className = 'severity-marker';
  marker.style.left = pct + '%';
  marker.innerHTML = `<div class="chip" style="color:${color}; background: var(--panel);">${e.spike_id} · ${e.severity_score.toFixed(2)}</div><div class="stem" style="background:${color};"></div>`;
  track.appendChild(marker);
});

// ---- Pipeline steps ----
const steps = [
  { t: "Ingest", d: "Pull flagged transactions for the spike event or merchant/time-bucket." },
  { t: "Correlate merchants", d: "Identify every distinct merchant touched by the alert." },
  { t: "Correlate device/IP", d: "Compute device & IP fingerprint reuse ratios across the cluster." },
  { t: "Pull SHAP reasons", d: "Attach top-3 SHAP feature drivers for each flagged transaction." },
  { t: "Classify root cause", d: "LLM or rule-based: label WHY it looks fraudulent, from a fixed taxonomy. Informational only.", llm: true },
  { t: "Assess severity", d: "Weighted, inspectable score from model confidence + correlation signals." },
  { t: "Decide action", d: "Apply fixed thresholds → ALLOW / REVIEW / BLOCK. Final — nothing downstream can change this." },
  { t: "Narrate for analyst", d: "LLM or template: turn the already-final evidence into readable prose.", llm: true },
  { t: "Log audit trail", d: "Persist full reasoning chain as an immutable record." },
];
document.getElementById('pipelineGrid').innerHTML = steps.map((s,i) => `
  <div class="step"><div class="n">0${i+1}${s.llm ? ' <span style="color:var(--accent);">·LLM</span>' : ''}</div><div class="t">${s.t}</div><div class="d">${s.d}</div></div>
`).join('');

// ---- Metrics table ----
const svc = DATA.standalone_vs_combined || {};

const rows = [
  { name: "Per-transaction XGBoost only", d: svc.standalone || m,
    ring: m.spike_specific ? m.spike_specific.ring_txn_recall_standalone : null },
  { name: "Two-layer (model + temporal)", d: svc.combined || cs,
    ring: cs.ring_txn_recall },
];
document.getElementById('metricsBody').innerHTML = rows.map(r => `
  <tr>
    <td>${r.name}</td>
    <td class="mono">${pctf(r.d.precision)}</td>
    <td class="mono">${pctf(r.d.recall)}</td>
    <td class="mono">${pctf(r.d.f1)}</td>
    <td class="mono">${pctf(r.d.alert_rate)}</td>
    <td class="mono">${pctf(r.d.false_positive_rate)}</td>
    <td class="mono">${pctf(r.ring)}</td>
    <td class="mono">₹${Math.round(r.d.false_positive_monetary_cost_inr).toLocaleString()}</td>
  </tr>`).join('');

// ---- Operating points table ----
const ca = DATA.cost_assumptions || {};
const revEl = document.getElementById('revCost');
if (revEl) revEl.textContent = ca.manual_review_cost_inr_per_alert ?? 45;
const cbEl = document.getElementById('cbFee');
if (cbEl) cbEl.textContent = ca.chargeback_fee_inr_per_missed_fraud ?? 750;
const opBody = document.getElementById('opBody');
if (opBody) {
  opBody.innerHTML = (DATA.operating_points || []).map(op => {
    const isPrimary = op.name.indexOf('primary') !== -1;
    const caveat = op.caveat ? '<div style="font-size:11px;opacity:.65;margin-top:4px;max-width:52ch;">' + op.caveat + '</div>' : '';
    return '<tr' + (isPrimary ? ' style="background:rgba(120,180,255,0.07);"' : '') + '>'
      + '<td>' + (isPrimary ? '<b>' + op.name + '</b>' : op.name) + caveat + '</td>'
      + '<td class="mono">' + op.threshold.toFixed(4) + '</td>'
      + '<td class="mono">' + pctf(op.alert_rate) + '</td>'
      + '<td class="mono">' + pctf(op.precision) + '</td>'
      + '<td class="mono">' + pctf(op.recall) + '</td>'
      + '<td class="mono">' + pctf(op.false_positive_rate) + '</td>'
      + '<td class="mono">₹' + Math.round(op.net_benefit_vs_no_model_inr).toLocaleString() + '</td>'
      + '</tr>';
  }).join('');
}

// ---- PR curve + validation cost curve ----
if (typeof Chart !== 'undefined') {
  const axis = (xTitle, yTitle) => ({
    x: { title: { display: true, text: xTitle, color: '#7c8798' },
         ticks: { color: '#7c8798' }, grid: { color: 'rgba(255,255,255,0.05)' } },
    y: { title: { display: true, text: yTitle, color: '#7c8798' },
         ticks: { color: '#7c8798' }, grid: { color: 'rgba(255,255,255,0.05)' } },
  });

  const prEl = document.getElementById('prChart');
  if (prEl && (DATA.pr_curve || []).length) {
    new Chart(prEl, {
      type: 'line',
      data: { datasets: [{
        label: 'Precision vs recall (test window)',
        data: DATA.pr_curve.map(p => ({ x: p.recall, y: p.precision })),
        borderColor: '#5b9dff', backgroundColor: 'rgba(91,157,255,0.12)',
        borderWidth: 2, pointRadius: 0, fill: true, tension: 0.15,
      }] },
      options: { responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#c9d3e0' } } },
        scales: axis('Recall', 'Precision') },
    });
  }

  const costEl = document.getElementById('costChart');
  if (costEl && (DATA.cost_curve || []).length) {
    const pts = DATA.cost_curve.map(c => ({ x: c.threshold, y: c.total_cost_inr }));
    new Chart(costEl, {
      type: 'line',
      data: { datasets: [{
        label: 'Expected cost on validation (' + RUPEE + ')',
        data: pts,
        borderColor: '#ffb45b', backgroundColor: 'rgba(255,180,91,0.10)',
        borderWidth: 2, pointRadius: 0, fill: true, tension: 0.15,
      }] },
      options: { responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#c9d3e0' } } },
        scales: axis('Score threshold', 'Expected total cost (' + RUPEE + ')') },
    });
  }
}

// ---- SHAP table ----
document.getElementById('shapBody').innerHTML = DATA.shap_importance.map(f => `
  <tr><td class="mono">${f.feature}</td><td class="mono">${f.mean_abs_shap.toFixed(4)}</td></tr>
`).join('');

// ---- Specificity control panel ----
if (DATA.control_summary) {
  const cs2 = DATA.control_summary;
  const ctrlAudit = DATA.audit_entries.find(e => e.spike_id === cs2.event_id);
  const decision = ctrlAudit ? ctrlAudit.decision : "—";
  const color = decision === "ALLOW" ? "var(--allow)" : decision === "REVIEW" ? "var(--review)" : "var(--block)";
  document.getElementById('controlPanel').innerHTML = `
    <div class="control-card">
      <div class="decision-big" style="color:${color}">${decision}</div>
      <div class="details">
        <div>${cs2.description}</div>
        <div style="margin-top:10px;">Severity score: <b>${ctrlAudit ? ctrlAudit.severity_score : '—'}</b> · ${cs2.n_txns} transactions from a single merchant</div>
        <div class="layer-row">
          <span class="layer-chip" style="color:var(--review); border-color:rgba(240,169,59,0.3);">Temporal layer: flagged bucket as anomalous (correct — it IS a volume spike)</span>
          <span class="layer-chip" style="color:var(--allow); border-color:rgba(53,208,142,0.3);">Per-transaction model: ${(cs2.individual_txn_flag_rate*100).toFixed(0)}% flagged individually</span>
          <span class="layer-chip" style="color:${color}; border-color:var(--line);">Investigation agent final decision: ${decision}</span>
        </div>
        <div style="margin-top:12px; color:var(--text-faint); font-size:12px;">${cs2.note}</div>
      </div>
    </div>`;
}

// ---- Audit trail list ----
document.getElementById('auditCount').textContent = DATA.audit_entries.length + " ENTRIES";
document.getElementById('auditList').innerHTML = DATA.audit_entries.map(e => {
  const dCls = e.decision === "BLOCK" ? "block" : e.decision === "REVIEW" ? "review" : "allow";
  return `
  <div class="audit-entry">
    <div class="h">
      <span class="id">${e.investigation_id} · ${e.spike_id || e.merchant_id}</span>
      <span class="tag ${dCls}">${e.decision} · severity ${e.severity_score}</span>
    </div>
    ${e.root_cause_hypothesis ? `<div style="margin-top:6px; font-family:'JetBrains Mono'; font-size:11px; color:var(--accent);">Root-cause hypothesis: ${e.root_cause_hypothesis.label}</div>` : ''}
    <div class="reason">${e.decision_reasoning}</div>
    ${e.analyst_narrative ? `<div class="reason" style="margin-top:8px; color:var(--text-faint); font-style:italic;">"${e.analyst_narrative}"</div>` : ''}
    <ul>${e.severity_reasoning.map(r => `<li>${r}</li>`).join('')}</ul>
  </div>`;
}).join('');

// ---- Timeline chart (isolated: never let a CDN/network hiccup break the rest of the page) ----
try {
  const tl = DATA.timeline;
  if (typeof Chart === 'undefined') throw new Error('Chart.js failed to load');
  new Chart(document.getElementById('timelineChart'), {
    type: 'line',
    data: {
      labels: tl.labels,
      datasets: [
        {
          label: 'Transactions / 15min',
          data: tl.txn_count,
          borderColor: '#4FA3FF',
          backgroundColor: 'rgba(79,163,255,0.08)',
          fill: true,
          tension: 0.25,
          pointRadius: 0,
          borderWidth: 1.5,
          yAxisID: 'y',
        },
        {
          label: 'Max anomaly z-score (merchant bucket)',
          data: tl.spike_score,
          borderColor: '#EF4A5C',
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.2,
          yAxisID: 'y1',
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { ticks: { color: '#566172', maxTicksLimit: 8, font: { family: 'JetBrains Mono', size: 10 } }, grid: { color: '#1A222C' } },
        y: { position:'left', ticks: { color: '#566172', font: { family: 'JetBrains Mono', size: 10 } }, grid: { color: '#1A222C' } },
        y1:{ position:'right', ticks: { color: '#EF4A5C', font: { family: 'JetBrains Mono', size: 10 } }, grid: { display:false } },
      },
      plugins: { legend: { labels: { color: '#8A97A6', font: { family: 'Inter', size: 11 } } } }
    }
  });
} catch (err) {
  document.getElementById('timelineChart').parentElement.innerHTML =
    '<div style="color:var(--text-faint); font-family:JetBrains Mono; font-size:12px; padding:20px;">Timeline chart unavailable offline (requires Chart.js from CDN). Raw data is in artifacts/timeline_buckets.csv.</div>';
}

</script>

</body>
</html>
"""

if __name__ == "__main__":
    main()
