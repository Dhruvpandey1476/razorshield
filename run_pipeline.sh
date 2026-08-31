#!/usr/bin/env bash
# RazorShield - full pipeline runner
# Regenerates data, retrains the model, runs the spike detector, and
# re-investigates all ground-truth spike events, then rebuilds the
# static demo dashboard.
set -e
cd "$(dirname "$0")"

echo "[1/6] Generating synthetic Razorpay-shaped transaction data..."
python data/generate_data.py

echo "[2/6] Training XGBoost model + SHAP explainability..."
python model/train_model.py

echo "[3/6] Running temporal fraud-spike detector..."
python model/spike_detector.py

echo "[4/6] Running LangGraph investigation agent on all spike events..."
rm -f artifacts/audit_trail.jsonl
python agent/investigation_agent.py

echo "[5/6] Regenerating METRICS.md + README results section from artifacts..."
python evaluation/report.py

echo "[6/6] Building demo dashboard..."
python dashboard/build_dashboard.py

echo ""
echo "Done."
echo ""
echo "TWO ways to view results:"
echo "  1. LIVE APP (recommended) -- a real running application:"
echo "       uvicorn backend.main:app --reload --port 8000"
echo "     then open http://localhost:8000/ -- click 'Investigate Live' on"
echo "     any card to run the actual agent on demand. Add an OpenRouter or"
echo "     OpenRouter key in the settings panel (gear icon) to see the real"
echo "     LLM nodes run instead of the deterministic fallback."
echo ""
echo "  2. STATIC REPORT -- no server needed, just open the file:"
echo "       dashboard/index.html"
echo "     A frozen snapshot of one full pipeline run, for offline viewing"
echo "     or screenshots. Does not update or accept clicks."
