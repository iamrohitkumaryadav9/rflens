"""The fixed menu of analyses: parameter models, limits and the dataset registry.

The model that picks an analysis returns a name and JSON parameters. They are validated here
(`extra="forbid"`, ranges, enums); there is no field that can hold code, SQL or a path. The code
that runs is analytics_runner/analyses.py, baked into the sandbox image. See
docs/design/analytics.md.
"""

import datetime
import hashlib
import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

N_FEATURES = 590  # SECOM: feature_idx 1..590
MAX_RUNS = 5000
MAX_RESULT_BYTES = 64 * 1024

Feature = Annotated[int, Field(ge=1, le=N_FEATURES, description="SECOM feature index, 1 to 590")]
TopK = Annotated[int, Field(ge=1, le=50)]
MinN = Annotated[int, Field(ge=10, le=1000)]


@dataclass(frozen=True)
class Dataset:
    name: str
    description: str
    n_features: int
    labels: tuple[str, ...]
    positive_label: str
    max_runs: int
    feature_unit: str


DATASETS = {
    "secom": Dataset(
        name="secom",
        description=(
            "UCI SECOM semiconductor manufacturing data: 1,567 production runs (2008), each with "
            "590 sensor features (numbered 1 to 590; the source gives no names or units) and a "
            "label pass (1,463 runs) or fail (104). Missing sensor values are common."
        ),
        n_features=N_FEATURES,
        labels=("pass", "fail"),
        positive_label="fail",
        max_runs=MAX_RUNS,
        feature_unit="UNKNOWN (the source publishes no units or names)",
    )
}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Condition(Strict):
    feature: Feature
    op: Literal["<", "<=", ">", ">=", "==", "is_null", "not_null"]
    value: float | None = None

    @model_validator(mode="after")
    def _value_matches_op(self) -> "Condition":
        if self.op in ("is_null", "not_null"):
            if self.value is not None:
                raise ValueError(f"operator {self.op} takes no value")
        elif self.value is None:
            raise ValueError(f"operator {self.op} needs a value")
        return self


class Filters(Strict):
    labels: list[Literal["pass", "fail"]] | None = Field(
        default=None, description="keep only runs with these labels"
    )
    run_date_from: str | None = Field(default=None, description="YYYY-MM-DD, inclusive")
    run_date_to: str | None = Field(default=None, description="YYYY-MM-DD, inclusive")
    conditions: list[Condition] = Field(default_factory=list, max_length=3)

    @field_validator("run_date_from", "run_date_to")
    @classmethod
    def _iso_date(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            return datetime.date.fromisoformat(v).isoformat()
        except ValueError as e:
            raise ValueError("must be a date like 2008-08-15") from e


class _WithFilters(Strict):
    filters: Filters = Field(default_factory=Filters)


class DescribeParams(_WithFilters):
    feature: Feature


class MissingSummaryParams(_WithFilters):
    top_k: TopK = 10


class DistributionParams(_WithFilters):
    feature: Feature
    bins: Annotated[int, Field(ge=5, le=50)] = 20
    by_label: bool = False


class GroupCompareParams(_WithFilters):
    feature: Feature
    test: Literal["mann_whitney", "welch_t"] = "mann_whitney"
    min_n: MinN = 20


class CorrelationParams(_WithFilters):
    feature_a: Feature
    feature_b: Feature
    method: Literal["spearman", "pearson"] = "spearman"
    min_n: MinN = 20

    @model_validator(mode="after")
    def _distinct(self) -> "CorrelationParams":
        if self.feature_a == self.feature_b:
            raise ValueError("feature_a and feature_b must differ")
        return self


class CorrelationTopParams(_WithFilters):
    feature: Feature
    method: Literal["spearman", "pearson"] = "spearman"
    top_k: TopK = 10
    min_n: MinN = 20


class FeatureRankingParams(_WithFilters):
    target: Literal["label"] = "label"
    positive: Literal["fail"] = "fail"
    method: Literal["mann_whitney"] = "mann_whitney"
    top_k: TopK = 10
    min_n: MinN = 20


class OutliersParams(_WithFilters):
    feature: Feature
    method: Literal["iqr", "zscore"] = "iqr"
    k: Annotated[float, Field(ge=0.5, le=10)] = 1.5
    z: Annotated[float, Field(ge=1, le=10)] = 3.0


class PlotParams(_WithFilters):
    kind: Literal["histogram", "box_by_label", "scatter", "ranking_bar"]
    feature: Feature | None = None
    feature_a: Feature | None = None
    feature_b: Feature | None = None
    bins: Annotated[int, Field(ge=5, le=50)] = 20
    by_label: bool = False
    top_k: Annotated[int, Field(ge=1, le=30)] = 10

    @model_validator(mode="after")
    def _fields_for_kind(self) -> "PlotParams":
        need = {
            "histogram": {"feature"},
            "box_by_label": {"feature"},
            "scatter": {"feature_a", "feature_b"},
            "ranking_bar": set(),
        }[self.kind]
        given = {k for k in ("feature", "feature_a", "feature_b") if getattr(self, k) is not None}
        if given != need:
            raise ValueError(
                f"plot kind {self.kind} needs exactly {sorted(need) or 'no feature'}, "
                f"got {sorted(given) or 'none'}"
            )
        if self.kind == "scatter" and self.feature_a == self.feature_b:
            raise ValueError("feature_a and feature_b must differ")
        return self


@dataclass(frozen=True)
class Analysis:
    name: str
    model: type[BaseModel]
    description: str
    all_features: bool = False  # the analysis reads every feature, not just the named ones


ANALYSES: dict[str, Analysis] = {
    a.name: a
    for a in (
        Analysis(
            "describe",
            DescribeParams,
            "Summary of one feature: n, missing count and share, mean, sd, min, quartiles, max.",
        ),
        Analysis(
            "missing_summary",
            MissingSummaryParams,
            "The features with the highest share of missing values, and counts of features with "
            "any or all values missing.",
            all_features=True,
        ),
        Analysis(
            "distribution",
            DistributionParams,
            "Histogram of one feature (equal-width bins between its min and max), plus n, "
            "missing count, mean, skewness, min, quartiles and max; optionally per label.",
        ),
        Analysis(
            "group_compare",
            GroupCompareParams,
            "Compare one feature between passing and failing runs: n, median and mean per "
            "group, the test (Mann-Whitney U or Welch t), p-value, effect size (rank-biserial "
            "or Cohen's d) and which group is higher. Each group needs min_n values.",
        ),
        Analysis(
            "correlation",
            CorrelationParams,
            "Correlation (Spearman or Pearson) between two features over the runs that have "
            "both, with the number of pairs and the p-value.",
        ),
        Analysis(
            "correlation_top",
            CorrelationTopParams,
            "The features most correlated (by absolute coefficient) with one feature.",
            all_features=True,
        ),
        Analysis(
            "feature_ranking",
            FeatureRankingParams,
            "Rank all features by association with a failing run: Mann-Whitney U per feature, "
            "rank-biserial effect size, Benjamini-Hochberg q-values, missing values dropped per "
            "feature; features with too few values or constant are excluded and listed.",
            all_features=True,
        ),
        Analysis(
            "outliers",
            OutliersParams,
            "Outliers of one feature by the IQR rule (k times the interquartile range) or the "
            "z-score rule; counts them and lists up to 20.",
        ),
        Analysis(
            "plot",
            PlotParams,
            "One PNG (800x600): histogram, box_by_label, scatter or ranking_bar of the "
            "feature ranking. Returns an artifact id and the numbers drawn.",
        ),
    )
}


def features_needed(analysis: str, params: BaseModel) -> list[int] | None:
    """The feature columns the extract must hold, or None for all of them."""
    spec = ANALYSES[analysis]
    if spec.all_features or (analysis == "plot" and params.kind == "ranking_bar"):  # type: ignore[attr-defined]
        return None
    wanted: set[int] = set()
    for name in ("feature", "feature_a", "feature_b"):
        value = getattr(params, name, None)
        if value is not None:
            wanted.add(value)
    wanted.update(c.feature for c in params.filters.conditions)  # type: ignore[attr-defined]
    return sorted(wanted)


def validate(analysis: str, raw: Any) -> BaseModel:
    """Parse the parameters of `analysis`; raises ValueError (unknown analysis) or
    pydantic.ValidationError."""
    if analysis not in ANALYSES:
        raise ValueError(f"unknown analysis {analysis!r}; available: {', '.join(ANALYSES)}")
    if not isinstance(raw, dict):
        raise ValueError("params must be a JSON object")
    return ANALYSES[analysis].model.model_validate(raw)


def normalised(params: BaseModel) -> dict[str, Any]:
    """Every field with its default filled in, JSON types only."""
    return params.model_dump(mode="json")


def analysis_id(analysis: str, params: dict[str, Any], extract_sha256: str) -> str:
    """Same analysis, parameters and data give the same id (64 hex characters)."""
    blob = json.dumps(
        {"analysis": analysis, "params": params, "extract": extract_sha256},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def describe_menu() -> dict[str, Any]:
    """What list_analyses returns: the dataset, the analyses and a JSON Schema for each."""
    ds = DATASETS["secom"]
    return {
        "dataset": {
            "name": ds.name,
            "description": ds.description,
            "n_features": ds.n_features,
            "labels": list(ds.labels),
            "positive_label": ds.positive_label,
            "feature_unit": ds.feature_unit,
            "max_runs_per_call": ds.max_runs,
        },
        "analyses": [
            {
                "name": a.name,
                "description": a.description,
                "params_schema": a.model.model_json_schema(),
            }
            for a in ANALYSES.values()
        ],
        "limits": {
            "result_bytes": MAX_RESULT_BYTES,
            "filter_conditions": 3,
            "top_k": 50,
            "bins": "5 to 50",
        },
    }
