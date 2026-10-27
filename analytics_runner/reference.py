"""Independent reference values for the golden stats questions (make golden-stats-refresh).

This file does NOT import analyses.py or runner.py and shares no code with them: it parses the
extract itself, calls scipy.stats directly with explicit arguments, and also recomputes every
Mann-Whitney p-value from the textbook formula (average ranks, tie-corrected variance, continuity
correction) with plain numpy and math. The two computations must agree (relative 1e-9) or the
script stops: that is the check on the analytics code's definitions.

    python /app/reference.py /in/data.csv   -> JSON on stdout
"""

import csv
import json
import math
import sys

import numpy as np
import scipy
from scipy import stats

TOP = 10


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    label = np.array([r[1] for r in body])
    cols = {}
    for k, name in enumerate(header[3:]):
        cols[int(name[1:])] = np.array(
            [float(r[3 + k]) if r[3 + k] != "" else np.nan for r in body]
        )
    return label, cols


def mwu_scipy(fail, pas):
    r = stats.mannwhitneyu(
        fail, pas, alternative="two-sided", method="asymptotic", use_continuity=True
    )
    return float(r.statistic), float(r.pvalue)


def mwu_textbook(fail, pas):
    n1, n2 = len(fail), len(pas)
    allv = np.concatenate([fail, pas])
    order = np.argsort(allv, kind="mergesort")
    sorted_v = allv[order]
    ranks = np.empty(len(allv))
    i = 0
    tie_term = 0.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1  # average rank, 1-based
        t = j - i + 1
        tie_term += t**3 - t
        i = j + 1
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    big_n = n1 + n2
    sigma = math.sqrt(n1 * n2 / 12 * ((big_n + 1) - tie_term / (big_n * (big_n - 1))))
    z = (max(u1, u2) - n1 * n2 / 2 - 0.5) / sigma
    return u1, min(1.0, math.erfc(z / math.sqrt(2)))


def split(label, col):
    return col[(label == "fail") & ~np.isnan(col)], col[(label == "pass") & ~np.isnan(col)]


def main(path):
    label, cols = load(path)
    tested = []
    for idx in sorted(cols):
        fail, pas = split(label, cols[idx])
        if (
            len(fail) < 20
            or len(pas) < 20
            or max(fail.max(), pas.max()) == min(fail.min(), pas.min())
        ):
            continue
        u, p = mwu_scipy(fail, pas)
        u2, p2 = mwu_textbook(fail, pas)
        if abs(u - u2) > 1e-6 or abs(p - p2) > 1e-9 * max(p, 1e-300) + 1e-300:
            raise SystemExit(f"feature {idx}: scipy ({u}, {p}) and textbook ({u2}, {p2}) disagree")
        tested.append((p, idx, len(fail), len(pas), u))
    tested.sort()
    ps = np.array([t[0] for t in tested])
    q = stats.false_discovery_control(ps, method="bh")
    ranking = [
        {
            "rank": i + 1,
            "feature": idx,
            "n_pass": npas,
            "n_fail": nfail,
            "p_value": p,
            "q_value": float(q[i]),
            "rank_biserial": 2 * u / (nfail * npas) - 1,
        }
        for i, (p, idx, nfail, npas, u) in enumerate(tested[:TOP])
    ]
    fail, pas = split(label, cols[60])
    u, p = mwu_scipy(fail, pas)
    compare60 = {
        "feature": 60,
        "n_pass": len(pas),
        "n_fail": len(fail),
        "median_pass": float(np.median(pas)),
        "median_fail": float(np.median(fail)),
        "mean_pass": float(pas.mean()),
        "mean_fail": float(fail.mean()),
        "p_value": p,
        "rank_biserial": 2 * u / (len(fail) * len(pas)) - 1,
    }
    v = cols[104][~np.isnan(cols[104])]
    counts, edges = np.histogram(v, bins=20, range=(v.min(), v.max()))
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    dist104 = {
        "feature": 104,
        "n": int(len(v)),
        "n_missing": int(len(cols[104]) - len(v)),
        "min": float(v.min()),
        "q1": float(q1),
        "median": float(med),
        "q3": float(q3),
        "max": float(v.max()),
        "counts": [int(c) for c in counts],
    }
    print(
        json.dumps(
            {
                "scipy": scipy.__version__,
                "n_tested": len(tested),
                "ranking": ranking,
                "compare60": compare60,
                "dist104": dist104,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
