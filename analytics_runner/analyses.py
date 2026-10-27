"""The fixed menu of analyses that run inside the sandbox container (numpy and scipy only).

Nothing here reads a database, a network or a path other than the files the runner opens: each
function takes the in-memory `Data` (one wide table of runs x features) and a parameter dict that
the host has already validated (rflens.analytics.menu), and returns a JSON-able dict.

Conventions (stated in the results so the writer can quote them):
- label values are `pass` and `fail`; the positive group of every comparison is `fail`;
- missing values are NaN and are dropped pairwise, per feature (inside each group for comparisons);
  nothing is imputed, and the number of values each statistic rests on is always returned;
- Mann-Whitney U: two-sided, scipy `method="asymptotic"` (tie-corrected normal approximation with
  continuity correction); statistic = U of the `fail` sample; effect size = rank-biserial
  correlation 2*U/(n_fail*n_pass) - 1 (positive: higher in `fail`);
- multiple testing: Benjamini-Hochberg q-values over every tested feature;
- ranking order: p ascending, ties by feature index ascending.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

ALPHA = 0.05
POSITIVE = "fail"
NEGATIVE = "pass"
MAX_EXCLUDED_LISTED = 100
MAX_SCATTER_POINTS = 5000
MAX_OUTLIERS_LISTED = 20
MISSING_POLICY = (
    "missing values are dropped pairwise per feature (inside each group for comparisons); "
    "nothing is imputed; every statistic reports the number of values it rests on"
)


class AnalysisError(Exception):
    """A request the data cannot answer (too few values, constant feature, ...)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Data:
    run_ids: np.ndarray  # int
    labels: np.ndarray  # str
    run_ts: np.ndarray  # str, 'YYYY-MM-DD HH:MM:SS'
    features: dict[int, np.ndarray] = field(default_factory=dict)  # idx -> float with NaN

    @property
    def n_rows(self) -> int:
        return len(self.run_ids)

    def select(self, mask: np.ndarray) -> "Data":
        return Data(
            self.run_ids[mask],
            self.labels[mask],
            self.run_ts[mask],
            {k: v[mask] for k, v in self.features.items()},
        )


# --- small helpers -----------------------------------------------------------------------------


def _f(x) -> float | None:
    """A JSON-safe float: NaN and infinities become None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _col(data: Data, feature: int) -> np.ndarray:
    if feature not in data.features:
        raise AnalysisError("unknown_feature", f"feature {feature} is not in the extract")
    return data.features[feature]


def _present(x: np.ndarray) -> np.ndarray:
    return x[~np.isnan(x)]


def _quantiles(v: np.ndarray) -> dict:
    q1, med, q3 = np.percentile(v, [25, 50, 75])  # linear interpolation
    return {"min": _f(v.min()), "q1": _f(q1), "median": _f(med), "q3": _f(q3), "max": _f(v.max())}


def apply_filters(data: Data, filters: dict) -> Data:
    """Rows kept: label in the list, run date in range (inclusive), and every feature condition."""
    mask = np.ones(data.n_rows, dtype=bool)
    labels = filters.get("labels")
    if labels:
        mask &= np.isin(data.labels, labels)
    day = np.array([t[:10] for t in data.run_ts])
    if filters.get("run_date_from"):
        mask &= day >= filters["run_date_from"]
    if filters.get("run_date_to"):
        mask &= day <= filters["run_date_to"]
    for cond in filters.get("conditions") or []:
        col = _col(data, cond["feature"])
        op, value = cond["op"], cond.get("value")
        with np.errstate(invalid="ignore"):  # comparisons with NaN are False, which is intended
            if op == "is_null":
                mask &= np.isnan(col)
            elif op == "not_null":
                mask &= ~np.isnan(col)
            elif op == "<":
                mask &= col < value
            elif op == "<=":
                mask &= col <= value
            elif op == ">":
                mask &= col > value
            elif op == ">=":
                mask &= col >= value
            elif op == "==":
                mask &= col == value
            else:  # the host validates this; refuse anything else
                raise AnalysisError("bad_filter", f"unknown filter operator {op!r}")
    out = data.select(mask)
    if out.n_rows == 0:
        raise AnalysisError("no_rows", "the filters leave no runs")
    return out


def _groups(data: Data, feature: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(fail values, pass values, fail rows, pass rows); missing dropped inside each group."""
    col = _col(data, feature)
    is_fail = data.labels == POSITIVE
    is_pass = data.labels == NEGATIVE
    return (
        _present(col[is_fail]),
        _present(col[is_pass]),
        int(is_fail.sum()),
        int(is_pass.sum()),
    )


def _mwu(fail: np.ndarray, pas: np.ndarray) -> dict:
    res = stats.mannwhitneyu(fail, pas, alternative="two-sided", method="asymptotic")
    u = float(res.statistic)
    rb = 2.0 * u / (len(fail) * len(pas)) - 1.0
    higher = "fail" if rb > 0 else "pass" if rb < 0 else "neither"
    return {
        "statistic_u_fail": _f(u),
        "p_value": _f(res.pvalue),
        "rank_biserial": _f(rb),
        "auc_fail_greater_than_pass": _f((rb + 1.0) / 2.0),
        "higher_in": higher,
    }


# --- the menu ----------------------------------------------------------------------------------


def describe(data: Data, p: dict) -> dict:
    col = _col(data, p["feature"])
    v = _present(col)
    if len(v) == 0:
        raise AnalysisError("no_values", f"feature {p['feature']} has no values in these runs")
    return {
        "feature": p["feature"],
        "n": int(len(v)),
        "n_missing": int(data.n_rows - len(v)),
        "missing_share": _f((data.n_rows - len(v)) / data.n_rows),
        "mean": _f(v.mean()),
        "sd": _f(v.std(ddof=1)) if len(v) > 1 else None,
        **_quantiles(v),
    }


def missing_summary(data: Data, p: dict) -> dict:
    rows = []
    for idx in sorted(data.features):
        n_missing = int(np.isnan(data.features[idx]).sum())
        rows.append((-(n_missing / data.n_rows), idx, n_missing))
    rows.sort()
    top = rows[: p["top_k"]]
    return {
        "n_runs": data.n_rows,
        "n_features": len(rows),
        "n_features_with_missing": sum(1 for r in rows if r[2] > 0),
        "n_features_all_missing": sum(1 for r in rows if r[2] == data.n_rows),
        "top": [
            {"feature": idx, "n_missing": n, "missing_share": _f(n / data.n_rows)}
            for _, idx, n in top
        ],
    }


def distribution(data: Data, p: dict) -> dict:
    col = _col(data, p["feature"])
    v = _present(col)
    if len(v) == 0:
        raise AnalysisError("no_values", f"feature {p['feature']} has no values in these runs")
    counts, edges = np.histogram(v, bins=p["bins"], range=(v.min(), v.max()))
    out = {
        "feature": p["feature"],
        "n": int(len(v)),
        "n_missing": int(data.n_rows - len(v)),
        "bins": p["bins"],
        "bin_edges": [_f(e) for e in edges],
        "counts": [int(c) for c in counts],
        "mean": _f(v.mean()),
        "skewness": _f(stats.skew(v, bias=False)) if len(v) >= 3 and v.min() < v.max() else None,
        **_quantiles(v),
    }
    if p.get("by_label"):
        out["counts_by_label"] = {}
        for label in (NEGATIVE, POSITIVE):
            sub = _present(col[data.labels == label])
            out["counts_by_label"][label] = {
                "n": int(len(sub)),
                "counts": [int(c) for c in np.histogram(sub, bins=edges)[0]],
            }
    return out


def group_compare(data: Data, p: dict) -> dict:
    fail, pas, n_fail_rows, n_pass_rows = _groups(data, p["feature"])
    min_n = p["min_n"]
    if len(fail) < min_n or len(pas) < min_n:
        raise AnalysisError(
            "too_few_values",
            f"feature {p['feature']} has {len(pas)} pass and {len(fail)} fail values; "
            f"each group needs at least {min_n}",
        )
    out = {
        "feature": p["feature"],
        "groups": {
            name: {
                "n": int(len(v)),
                "n_missing": rows - int(len(v)),
                "median": _f(np.median(v)),
                "mean": _f(v.mean()),
            }
            for name, v, rows in ((NEGATIVE, pas, n_pass_rows), (POSITIVE, fail, n_fail_rows))
        },
        "alpha": ALPHA,
    }
    if p["test"] == "mann_whitney":
        out["test"] = "mann_whitney_u (two-sided, asymptotic)"
        out.update(_mwu(fail, pas))
        out["effect_size"] = {"name": "rank_biserial", "value": out["rank_biserial"]}
    else:
        n1, n2 = len(fail), len(pas)
        s1, s2 = fail.var(ddof=1), pas.var(ddof=1)
        pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
        if pooled == 0:
            raise AnalysisError("constant_input", f"feature {p['feature']} is constant")
        res = stats.ttest_ind(fail, pas, equal_var=False)
        d = (fail.mean() - pas.mean()) / math.sqrt(pooled)
        out["test"] = "welch_t (two-sided)"
        out["statistic_t_fail_minus_pass"] = _f(res.statistic)
        out["p_value"] = _f(res.pvalue)
        out["effect_size"] = {"name": "cohen_d", "value": _f(d)}
        out["higher_in"] = "fail" if d > 0 else "pass" if d < 0 else "neither"
    out["significant_at_alpha"] = bool(out["p_value"] is not None and out["p_value"] < ALPHA)
    return out


def _pair(data: Data, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = _col(data, a), _col(data, b)
    ok = ~(np.isnan(x) | np.isnan(y))
    return x[ok], y[ok]


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    res = stats.spearmanr(x, y) if method == "spearman" else stats.pearsonr(x, y)
    return float(res[0]), float(res[1])


def correlation(data: Data, p: dict) -> dict:
    x, y = _pair(data, p["feature_a"], p["feature_b"])
    if len(x) < p["min_n"]:
        raise AnalysisError(
            "too_few_values",
            f"only {len(x)} runs have both features; at least {p['min_n']} are needed",
        )
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise AnalysisError("constant_input", "one of the two features is constant")
    r, pv = _corr(x, y, p["method"])
    return {
        "feature_a": p["feature_a"],
        "feature_b": p["feature_b"],
        "method": p["method"],
        "n_pairs": int(len(x)),
        "coefficient": _f(r),
        "p_value": _f(pv),
        "alpha": ALPHA,
    }


def correlation_top(data: Data, p: dict) -> dict:
    rows, excluded = [], 0
    for other in sorted(data.features):
        if other == p["feature"]:
            continue
        x, y = _pair(data, p["feature"], other)
        if len(x) < p["min_n"] or np.ptp(x) == 0 or np.ptp(y) == 0:
            excluded += 1
            continue
        r, pv = _corr(x, y, p["method"])
        if math.isfinite(r):
            rows.append((-abs(r), other, r, pv, len(x)))
        else:
            excluded += 1
    rows.sort()
    return {
        "feature": p["feature"],
        "method": p["method"],
        "n_compared": len(rows),
        "n_excluded": excluded,
        "top": [
            {"feature": o, "coefficient": _f(r), "p_value": _f(pv), "n_pairs": n}
            for _, o, r, pv, n in rows[: p["top_k"]]
        ],
    }


def feature_ranking(data: Data, p: dict) -> dict:
    tested, excluded = [], []
    for idx in sorted(data.features):
        fail, pas, _, _ = _groups(data, idx)
        if len(fail) < p["min_n"] or len(pas) < p["min_n"]:
            excluded.append({"feature": idx, "reason": "too_few_values"})
            continue
        if min(fail.min(), pas.min()) == max(fail.max(), pas.max()):
            excluded.append({"feature": idx, "reason": "constant"})
            continue
        m = _mwu(fail, pas)
        tested.append((m["p_value"], idx, len(fail), len(pas), m))
    if not tested:
        raise AnalysisError("no_features", "no feature has enough values in both groups")
    pvals = np.array([t[0] for t in tested])
    qvals = stats.false_discovery_control(pvals, method="bh")
    ranked = sorted(zip(tested, qvals, strict=True), key=lambda tq: (tq[0][0], tq[0][1]))
    reasons: dict[str, int] = {}
    for e in excluded:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    top = []
    for rank, ((pv, idx, n_fail, n_pass, m), q) in enumerate(ranked[: p["top_k"]], start=1):
        top.append(
            {
                "rank": rank,
                "feature": idx,
                "n_pass": n_pass,
                "n_fail": n_fail,
                "missing_share": _f(float(np.isnan(data.features[idx]).mean())),
                "statistic_u_fail": m["statistic_u_fail"],
                "p_value": _f(pv),
                "q_value": _f(q),
                "rank_biserial": m["rank_biserial"],
                "higher_in": m["higher_in"],
            }
        )
    return {
        "method": (
            "mann_whitney_u per feature (two-sided, asymptotic), rank-biserial effect size, "
            "Benjamini-Hochberg q-values over all tested features; ranked by p ascending, "
            "ties by feature index"
        ),
        "positive_group": POSITIVE,
        "alpha": ALPHA,
        "n_tested": len(tested),
        "n_excluded": len(excluded),
        "excluded_by_reason": reasons,
        "excluded": excluded[:MAX_EXCLUDED_LISTED],
        "n_significant_q_below_alpha": int((qvals < ALPHA).sum()),
        "top": top,
    }


def outliers(data: Data, p: dict) -> dict:
    col = _col(data, p["feature"])
    ok = ~np.isnan(col)
    v = col[ok]
    if len(v) < 3:
        raise AnalysisError("too_few_values", f"feature {p['feature']} has {len(v)} values")
    if p["method"] == "iqr":
        q1, q3 = np.percentile(v, [25, 75])
        spread = q3 - q1
        low, high = q1 - p["k"] * spread, q3 + p["k"] * spread
        rule = {"method": "iqr", "k": p["k"], "q1": _f(q1), "q3": _f(q3)}
    else:
        sd = v.std(ddof=1)
        if sd == 0:
            raise AnalysisError("constant_input", f"feature {p['feature']} is constant")
        low, high = v.mean() - p["z"] * sd, v.mean() + p["z"] * sd
        rule = {"method": "zscore", "z": p["z"], "mean": _f(v.mean()), "sd": _f(sd)}
    hit = ok & ((col < low) | (col > high))
    idx = np.flatnonzero(hit)
    beyond = np.where(col[idx] < low, low - col[idx], col[idx] - high)
    order = sorted(range(len(idx)), key=lambda i: (-beyond[i], data.run_ids[idx[i]]))
    return {
        "feature": p["feature"],
        "rule": rule,
        "lower_threshold": _f(low),
        "upper_threshold": _f(high),
        "n": int(len(v)),
        "n_outliers": int(len(idx)),
        "outliers": [
            {
                "run_id": int(data.run_ids[idx[i]]),
                "value": _f(col[idx[i]]),
                "label": str(data.labels[idx[i]]),
            }
            for i in order[:MAX_OUTLIERS_LISTED]
        ],
    }


# --- plots -------------------------------------------------------------------------------------


def plot(data: Data, p: dict, out_png: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kind = p["kind"]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    colours = {NEGATIVE: "#2a6f97", POSITIVE: "#c1121f"}
    if kind == "histogram":
        res = distribution(data, {"feature": p["feature"], "bins": p["bins"], "by_label": True})
        edges = np.array(res["bin_edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        width = (edges[1] - edges[0]) if len(edges) > 1 else 1.0
        if p.get("by_label"):
            for label in (NEGATIVE, POSITIVE):
                ax.bar(
                    centers,
                    res["counts_by_label"][label]["counts"],
                    width=width,
                    alpha=0.6,
                    label=f"{label} (n={res['counts_by_label'][label]['n']})",
                    color=colours[label],
                )
            ax.legend()
        else:
            ax.bar(centers, res["counts"], width=width, color="#2a6f97")
        ax.set_xlabel(f"feature {p['feature']} (unit unknown)")
        ax.set_ylabel("runs")
        ax.set_title(
            f"Distribution of feature {p['feature']} (n={res['n']}, {res['n_missing']} missing)"
        )
        drawn = res
    elif kind == "box_by_label":
        col = _col(data, p["feature"])
        series, drawn = [], {"feature": p["feature"], "groups": {}}
        for label in (NEGATIVE, POSITIVE):
            v = _present(col[data.labels == label])
            if len(v) == 0:
                raise AnalysisError("no_values", f"no {label} values for feature {p['feature']}")
            series.append(v)
            drawn["groups"][label] = {"n": int(len(v)), **_quantiles(v)}
        ax.boxplot(series, tick_labels=[NEGATIVE, POSITIVE])
        ax.set_ylabel(f"feature {p['feature']} (unit unknown)")
        ax.set_title(f"Feature {p['feature']} by label")
    elif kind == "scatter":
        x, y = _pair(data, p["feature_a"], p["feature_b"])
        lab = data.labels[
            ~(np.isnan(_col(data, p["feature_a"])) | np.isnan(_col(data, p["feature_b"])))
        ]
        if len(x) == 0:
            raise AnalysisError("no_values", "no run has both features")
        keep = np.arange(len(x))
        sampled = len(x) > MAX_SCATTER_POINTS
        if sampled:
            keep = np.sort(
                np.random.default_rng(0).choice(len(x), MAX_SCATTER_POINTS, replace=False)
            )
        for label in (NEGATIVE, POSITIVE):
            m = lab[keep] == label
            ax.scatter(x[keep][m], y[keep][m], s=8, alpha=0.6, c=colours[label], label=label)
        ax.legend()
        ax.set_xlabel(f"feature {p['feature_a']}")
        ax.set_ylabel(f"feature {p['feature_b']}")
        ax.set_title(f"Feature {p['feature_a']} vs {p['feature_b']} (n={len(x)})")
        drawn = {
            "feature_a": p["feature_a"],
            "feature_b": p["feature_b"],
            "n_pairs": int(len(x)),
            "n_drawn": int(len(keep)),
            "sampled_with_seed_0": bool(sampled),
        }
    else:  # ranking_bar
        res = feature_ranking(data, {"top_k": p["top_k"], "min_n": 20})
        top = res["top"]
        ax.barh(
            [f"feature {t['feature']}" for t in reversed(top)],
            [-math.log10(max(t["p_value"], 1e-300)) for t in reversed(top)],
            color="#2a6f97",
        )
        ax.axvline(-math.log10(ALPHA), color="grey", linestyle="--", label="p = 0.05")
        ax.legend()
        ax.set_xlabel("-log10(p), Mann-Whitney pass vs fail")
        ax.set_title(f"Top {len(top)} features associated with fail ({res['n_tested']} tested)")
        drawn = res
    fig.tight_layout()
    fig.savefig(out_png, format="png", metadata={"Software": "rflens analytics"})
    plt.close(fig)
    return {"kind": kind, "image": {"width": 800, "height": 600, "format": "png"}, "data": drawn}


ANALYSES = {
    "describe": describe,
    "missing_summary": missing_summary,
    "distribution": distribution,
    "group_compare": group_compare,
    "correlation": correlation,
    "correlation_top": correlation_top,
    "feature_ranking": feature_ranking,
    "outliers": outliers,
}
