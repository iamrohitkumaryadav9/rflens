"""Tests of the in-container analyses, run inside the image (make test-analytics).

The expected numbers are worked out by hand or by a closed form, not by calling the code under
test a second time.
"""

import csv
import json
import math
import os
import subprocess
import sys

import analyses as an
import numpy as np
import pytest

RUNNER = os.path.join(os.path.dirname(os.path.abspath(an.__file__)), "runner.py")


def make_data(n_pass=40, n_fail=30, seed=1):
    """Feature 1: separates the groups; 2: noise; 3: constant; 4: mostly missing; 5: 1..n."""
    rng = np.random.default_rng(seed)
    n = n_pass + n_fail
    labels = np.array(["pass"] * n_pass + ["fail"] * n_fail, dtype=object)
    f1 = np.concatenate([rng.normal(0, 1, n_pass), rng.normal(2, 1, n_fail)])
    f2 = rng.normal(0, 1, n)
    f3 = np.full(n, 7.0)
    f4 = np.full(n, np.nan)
    f4[:5] = [1, 2, 3, 4, 5]
    f5 = np.arange(1, n + 1, dtype=float)
    ts = np.array([f"2008-07-{1 + i % 28:02d} 10:00:00" for i in range(n)], dtype=object)
    return an.Data(np.arange(1, n + 1), labels, ts, {1: f1, 2: f2, 3: f3, 4: f4, 5: f5})


def test_describe_matches_closed_form():
    d = make_data()
    r = an.describe(d, {"feature": 5})
    assert r["n"] == 70 and r["n_missing"] == 0
    assert r["mean"] == pytest.approx(35.5)
    assert r["median"] == pytest.approx(35.5)
    assert r["min"] == 1 and r["max"] == 70
    assert r["q1"] == pytest.approx(18.25) and r["q3"] == pytest.approx(52.75)
    assert r["sd"] == pytest.approx(math.sqrt(70 * 71 / 12))  # variance of 1..n with ddof=1


def test_describe_reports_missing_and_never_imputes():
    r = an.describe(make_data(), {"feature": 4})
    assert r["n"] == 5 and r["n_missing"] == 65
    assert r["missing_share"] == pytest.approx(65 / 70)
    assert r["mean"] == pytest.approx(3.0)


def test_distribution_counts_are_exact():
    d = make_data()
    r = an.distribution(d, {"feature": 5, "bins": 7, "by_label": True})
    assert sum(r["counts"]) == 70 and len(r["bin_edges"]) == 8
    assert r["counts"] == [10] * 7  # 1..70 in 7 equal-width bins of 9.857 holds 10 each
    assert sum(r["counts_by_label"]["pass"]["counts"]) == 40
    assert sum(r["counts_by_label"]["fail"]["counts"]) == 30
    assert r["skewness"] == pytest.approx(0.0, abs=1e-12)


def test_group_compare_mann_whitney_direction_and_effect():
    r = an.group_compare(make_data(), {"feature": 1, "test": "mann_whitney", "min_n": 20})
    assert r["groups"]["pass"]["n"] == 40 and r["groups"]["fail"]["n"] == 30
    assert r["higher_in"] == "fail" and r["rank_biserial"] > 0.7
    assert r["p_value"] < 1e-6 and r["significant_at_alpha"] is True
    assert r["auc_fail_greater_than_pass"] == pytest.approx((r["rank_biserial"] + 1) / 2)


def test_group_compare_welch_matches_formula():
    d = make_data()
    r = an.group_compare(d, {"feature": 1, "test": "welch_t", "min_n": 20})
    f, p = d.features[1][d.labels == "fail"], d.features[1][d.labels == "pass"]
    pooled = ((len(f) - 1) * f.var(ddof=1) + (len(p) - 1) * p.var(ddof=1)) / (len(f) + len(p) - 2)
    assert r["effect_size"]["value"] == pytest.approx((f.mean() - p.mean()) / math.sqrt(pooled))
    se = math.sqrt(f.var(ddof=1) / len(f) + p.var(ddof=1) / len(p))
    assert r["statistic_t_fail_minus_pass"] == pytest.approx((f.mean() - p.mean()) / se)


def test_group_compare_refuses_too_few_values():
    with pytest.raises(an.AnalysisError) as e:
        an.group_compare(make_data(), {"feature": 4, "test": "mann_whitney", "min_n": 20})
    assert e.value.code == "too_few_values" and "needs at least 20" in e.value.message


def test_feature_ranking_orders_excludes_and_adjusts():
    r = an.feature_ranking(make_data(), {"top_k": 5, "min_n": 20})
    # feature 5 (1..70, passes first) separates the groups perfectly; feature 1 does it noisily
    assert [t["feature"] for t in r["top"]][:2] == [5, 1]
    assert r["top"][0]["rank_biserial"] == pytest.approx(
        1.0
    )  # every fail value is above every pass
    assert r["top"][0]["higher_in"] == "fail"
    reasons = {e["feature"]: e["reason"] for e in r["excluded"]}
    assert reasons == {3: "constant", 4: "too_few_values"}
    assert r["n_tested"] == 3 and r["n_excluded"] == 2
    ps = [t["p_value"] for t in r["top"]]
    assert ps == sorted(ps)
    # BH with m = 3 tests: q_(i) = min_{j>=i} p_(j) * m / j
    p = ps
    expected_q = [min(p[j] * 3 / (j + 1) for j in range(i, 3)) for i in range(3)]
    assert [t["q_value"] for t in r["top"]] == pytest.approx([min(q, 1.0) for q in expected_q])
    assert all(t["q_value"] >= t["p_value"] for t in r["top"])


def test_feature_ranking_is_deterministic():
    d = make_data()
    a = an.feature_ranking(d, {"top_k": 5, "min_n": 20})
    b = an.feature_ranking(d, {"top_k": 5, "min_n": 20})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_missing_summary_orders_by_share_then_index():
    r = an.missing_summary(make_data(), {"top_k": 3})
    assert [t["feature"] for t in r["top"]] == [4, 1, 2]  # 4 has 65 missing; the rest tie at 0
    assert r["top"][0]["n_missing"] == 65 and r["n_features_with_missing"] == 1


def test_correlation_pearson_and_spearman_of_a_line():
    d = make_data()
    d.features[6] = 2 * d.features[5] + 1
    r = an.correlation(d, {"feature_a": 5, "feature_b": 6, "method": "spearman", "min_n": 20})
    assert r["coefficient"] == pytest.approx(1.0) and r["n_pairs"] == 70
    r = an.correlation(d, {"feature_a": 5, "feature_b": 6, "method": "pearson", "min_n": 20})
    assert r["coefficient"] == pytest.approx(1.0)


def test_correlation_constant_and_too_few_are_refused():
    d = make_data()
    with pytest.raises(an.AnalysisError) as e:
        an.correlation(d, {"feature_a": 5, "feature_b": 3, "method": "spearman", "min_n": 20})
    assert e.value.code == "constant_input"
    with pytest.raises(an.AnalysisError) as e:
        an.correlation(d, {"feature_a": 5, "feature_b": 4, "method": "spearman", "min_n": 20})
    assert e.value.code == "too_few_values"


def test_correlation_top_ties_by_feature_index():
    d = make_data()
    d.features[6] = d.features[5].copy()  # perfectly correlated with 5, like 5 itself
    d.features[7] = d.features[5].copy()
    r = an.correlation_top(d, {"feature": 5, "method": "spearman", "top_k": 2, "min_n": 20})
    assert [t["feature"] for t in r["top"]] == [6, 7]
    assert r["n_excluded"] == 2  # features 3 (constant) and 4 (too few pairs)


def test_outliers_iqr_matches_hand_computation():
    d = make_data()
    d.features[8] = np.concatenate([np.arange(1.0, 70.0), [1000.0]])
    r = an.outliers(d, {"feature": 8, "method": "iqr", "k": 1.5, "z": 3.0})
    q1, q3 = np.percentile(d.features[8], [25, 75])
    assert r["upper_threshold"] == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert r["n_outliers"] == 1
    assert r["outliers"][0] == {"run_id": 70, "value": 1000.0, "label": "fail"}


def test_filters_labels_dates_and_conditions():
    d = make_data()
    only_fail = an.apply_filters(d, {"labels": ["fail"]})
    assert only_fail.n_rows == 30
    some = an.apply_filters(d, {"conditions": [{"feature": 5, "op": "<=", "value": 10.0}]})
    assert some.n_rows == 10
    missing = an.apply_filters(d, {"conditions": [{"feature": 4, "op": "is_null"}]})
    assert missing.n_rows == 65
    with pytest.raises(an.AnalysisError) as e:
        an.apply_filters(d, {"conditions": [{"feature": 5, "op": ">", "value": 1e9}]})
    assert e.value.code == "no_rows"
    early = an.apply_filters(d, {"run_date_to": "2008-07-02"})
    assert early.n_rows == 6  # days 1 and 2 are i = 0, 1, 28, 29, 56, 57


@pytest.mark.parametrize(
    "params",
    [
        {"kind": "histogram", "feature": 5, "bins": 10, "by_label": True},
        {"kind": "box_by_label", "feature": 1},
        {"kind": "scatter", "feature_a": 1, "feature_b": 2},
        {"kind": "ranking_bar", "top_k": 3},
    ],
)
def test_plot_writes_a_png_of_the_stated_size(tmp_path, params):
    out = tmp_path / "plot.png"
    r = an.plot(make_data(), params, str(out))
    raw = out.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    assert (width, height) == (800, 600) == (r["image"]["width"], r["image"]["height"])
    assert r["data"]


def run_runner(tmp_path, data: an.Data, job: dict):
    indir, outdir = tmp_path / "in", tmp_path / "out"
    indir.mkdir()
    outdir.mkdir()
    with open(indir / "data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "label", "run_ts", *[f"f{k}" for k in sorted(data.features)]])
        for i in range(data.n_rows):
            row = [data.run_ids[i], data.labels[i], data.run_ts[i]]
            for k in sorted(data.features):
                v = data.features[k][i]
                row.append("" if np.isnan(v) else repr(float(v)))
            w.writerow(row)
    (indir / "job.json").write_text(json.dumps(job))
    env = {**os.environ, "RFLENS_IN_DIR": str(indir), "RFLENS_OUT_DIR": str(outdir)}
    proc = subprocess.run(
        [sys.executable, "-I", RUNNER], env=env, capture_output=True, text=True, check=False
    )
    return proc, json.loads((outdir / "result.json").read_text()), outdir


def test_runner_end_to_end_and_deterministic(tmp_path):
    job = {"analysis": "feature_ranking", "params": {"top_k": 3, "min_n": 20}}
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    proc1, first, _ = run_runner(tmp_path / "a", make_data(), job)
    proc2, second, _ = run_runner(tmp_path / "b", make_data(), job)
    assert proc1.returncode == 0 and proc2.returncode == 0
    assert first["ok"] is True and first["n_runs_used"] == 70
    assert first == second
    assert set(first["versions"]) == {"python", "numpy", "scipy"}


def test_runner_reports_analysis_errors_as_json(tmp_path):
    job = {
        "analysis": "group_compare",
        "params": {"feature": 4, "test": "mann_whitney", "min_n": 20},
    }
    proc, result, _ = run_runner(tmp_path, make_data(), job)
    assert proc.returncode == 0
    assert result["ok"] is False and result["error"]["code"] == "too_few_values"


def test_runner_refuses_an_unknown_analysis(tmp_path):
    proc, result, _ = run_runner(tmp_path, make_data(), {"analysis": "exec", "params": {}})
    assert result["ok"] is False and result["error"]["code"] == "unknown_analysis"
