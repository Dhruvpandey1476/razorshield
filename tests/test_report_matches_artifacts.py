"""Drift guard: fail the build if the committed docs disagree with artifacts/.

The README results section and METRICS.md are generated, so any mismatch means
someone re-ran the pipeline without regenerating the docs.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")
ART = os.path.join(ROOT, "artifacts")

pipeline_has_run = os.path.exists(os.path.join(ART, "metrics.json"))
requires_pipeline = pytest.mark.skipif(
    not pipeline_has_run, reason="artifacts/ not populated -- run ./run_pipeline.sh first")


@requires_pipeline
def test_readme_results_section_matches_artifacts():
    from evaluation.report import build, START, END
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    assert START in readme and END in readme, "README lost its METRICS markers"
    committed = readme.split(START)[1].split(END)[0].strip()
    assert committed == build().strip(), (
        "README results section is stale. Run: python evaluation/report.py")


@requires_pipeline
def test_metrics_md_matches_artifacts():
    from evaluation.report import build
    path = os.path.join(ROOT, "METRICS.md")
    assert os.path.exists(path), "METRICS.md missing -- run python evaluation/report.py"
    with open(path, encoding="utf-8") as f:
        assert build().strip() in f.read(), "METRICS.md is stale. Run: python evaluation/report.py"


@requires_pipeline
def test_no_hand_typed_percentages_outside_generated_block():
    """Prose outside the generated block must not assert its own metrics.

    Any percentage or rupee figure there is hand-typed and can drift. The
    few that are legitimately static (thresholds, cost assumptions, split
    dates) are allowlisted explicitly.
    """
    from evaluation.report import START, END
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    outside = readme.split(START)[0] + readme.split(END)[1]
    # Strip URLs first: shields.io badge paths percent-encode spaces as %20,
    # which is not a metric and must not trip this guard.
    outside = re.sub(r"\]\([^)]*\)", "]()", outside)
    outside = re.sub(r"https?://\S+", "", outside)
    allowed = {
        "0.35", "0.75",      # agent decision thresholds (asserted by another test)
        "1%", "2%",          # alert-budget policy, not a result
        "100%",              # rhetorical
    }
    found = set(re.findall(r"\b\d+\.\d+%|\b\d+%", outside))
    leaked = found - allowed
    assert not leaked, (
        "Hand-typed metrics found in README prose outside the generated block: %s. "
        "Move them inside the METRICS markers so they regenerate." % sorted(leaked))
