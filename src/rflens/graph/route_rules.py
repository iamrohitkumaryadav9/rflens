"""The sql / stats boundary of the planner, as deterministic rules (like spec_rules.py for specs).

The planner (an LLM call) sends a SECOM question either to `sql` (simple aggregates: counts, sums,
averages, minimum / maximum, top-N by a count or a value, missing-value counts) or to `stats`
(statistical tests, effect sizes, association rankings, correlations, distributions with bins or
quartiles, histograms, outliers, plots). The model got this wrong live (run eval-e2e-aecd54a7:
"Which 8 SECOM sensor features have the most missing values?" went to stats), so the rules settle
it when they are decisive, and only then:

- the question must be about SECOM data (secom / sensor / feature N / passing or failing runs): an
  FCC question about a "distribution of licences" is never touched;
- a statistics word (test, significant, p-value, effect size, correlation, association, histogram,
  distribution, quartile, bins, outlier, plot, "differ" for pass/fail) means `stats`, whatever
  else the question says;
- otherwise an aggregate word (how many, count, sum, total, average, mean, minimum, maximum,
  highest, lowest, top N, most / fewest missing) means `sql`;
- anything else is left to the planner.

`boundary_route` returns the route or None; the planner applies it only to a `sql` or `stats` route.
"""

import re

SECOM_RE = re.compile(
    r"\bsecom\b|\bsensors?\b|\bfeatures?\s+\d+|\bsensor\s+features?\b|"
    r"\b(?:passing|failing|pass|fail)\b[\w\s]{0,20}\bruns?\b|\bproduction\s+runs?\b",
    re.I,
)
STATS_RE = re.compile(
    r"\bsignifican\w*|\bp-?values?\b|\bmann-?whitney\b|\bwelch\b|\bt-?test\b|\beffect\s+sizes?\b|"
    r"\bcorrelat\w*|\bassociat\w*|\bhistograms?\b|\bdistribut\w*|\bquartiles?\b|\bpercentiles?\b|"
    r"\bbins?\b|\boutliers?\b|\b(?:plot|chart|graph|figure)\b|\bstatistical\w*|\bhypothesis\b",
    re.I,
)
DIFFER_RE = re.compile(r"\bdiffer\w*|\bhow\s+large\s+is\s+the\s+difference\b", re.I)
PASS_FAIL_RE = re.compile(r"\b(?:pass|passing|fail|failing|failed)\b", re.I)
AGGREGATE_RE = re.compile(
    r"\bhow\s+many\b|\bcount\w*|\bsum\b|\btotal\b|\baverage\b|\bmean\b|\bminimum\b|\bmaximum\b|"
    r"\bmin\b|\bmax\b|\bhighest\b|\blowest\b|\blargest\b|\bsmallest\b|\btop\s+\d+|"
    r"\b(?:most|fewest|least|more|fewer)\b[\w\s]{0,20}\bmissing\b|\bmissing\s+(?:values?|readings?)",
    re.I,
)


def boundary_route(question: str) -> str | None:
    """'sql', 'stats' or None (not decisive: the planner's route stands)."""
    if not SECOM_RE.search(question):
        return None
    if STATS_RE.search(question) or (DIFFER_RE.search(question) and PASS_FAIL_RE.search(question)):
        return "stats"
    if AGGREGATE_RE.search(question):
        return "sql"
    return None
