from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
EVENT_PATH = OUTPUT_DIR / "s6_event_features.csv"
SUMMARY_PATH = OUTPUT_DIR / "s6_regime_liquidity_summary.json"
SCORED_PATH = OUTPUT_DIR / "s6_regime_liquidity_scored_events.csv"
DEVELOPMENT_END = "20260430"

WALK_FORWARD_FOLDS = (
    ("20251231", "20260101", "20260228"),
    ("20260228", "20260301", "20260430"),
    ("20260430", "20260501", "20260630"),
    ("20260630", "20260701", "20260831"),
    ("20260831", "20260901", "20260930"),
)

ENTRY_FEATURES = (
    "option_realized_volatility_20",
    "option_realized_volatility_60",
    "option_slope_acceleration_5",
    "option_slope_5",
    "option_slope_10",
    "option_rsi_5",
    "option_rsi_14",
    "option_volume_ratio_20",
    "option_recent20_traded_ratio",
    "option_recent20_round_010_volume_share",
    "option_recent20_max_no_trade_run",
    "log_signal_bar_volume",
    "entry_wait_minutes",
    "fill_exact_010",
    "intraday_futures_realized_volatility_5",
    "intraday_futures_realized_volatility_20",
    "intraday_spot_realized_volatility_5",
    "intraday_spot_realized_volatility_20",
    "intraday_basis_slope_10",
    "intraday_spot_futures_corr_20",
    "vix_slope_5",
    "vix_slope_20",
    "vix_slope_120",
    "vix_ma5_distance",
    "vix_ma20_distance",
    "vix_return_1",
    "entry_volatility_expansion_score",
    "entry_liquidity_score",
    "is_put",
    "band_low",
    "put_x_entry_expansion",
    "entry_expansion_x_liquidity",
)

LANDMARK_FEATURES = (
    "at2x_minutes_since_entry",
    "at2x_minute_from_open",
    "at2x_close_multiple",
    "at2x_high_multiple",
    "at2x_path_efficiency",
    "at2x_path_min_multiple",
    "at2x_path_traded_bars",
    "at2x_option_realized_volatility_5",
    "at2x_option_realized_volatility_20",
    "at2x_option_realized_volatility_60",
    "at2x_option_slope_acceleration_5",
    "at2x_option_slope_5",
    "at2x_option_slope_10",
    "at2x_option_rsi_5",
    "at2x_option_rsi_14",
    "at2x_option_volume_ratio_5",
    "at2x_option_volume_ratio_20",
    "at2x_option_recent20_traded_ratio",
    "at2x_option_recent20_round_010_volume_share",
    "at2x_option_recent20_max_no_trade_run",
    "at2x_option_recent20_mean_range_pct",
    "log_at2x_bar_volume",
    "at2x_intraday_futures_realized_volatility_5",
    "at2x_intraday_futures_realized_volatility_20",
    "at2x_intraday_futures_realized_volatility_60",
    "at2x_intraday_spot_realized_volatility_5",
    "at2x_intraday_spot_realized_volatility_20",
    "at2x_intraday_spot_realized_volatility_60",
    "at2x_intraday_basis_slope_10",
    "at2x_intraday_spot_futures_corr_20",
    "vix_slope_5",
    "vix_slope_20",
    "vix_slope_120",
    "vix_ma5_distance",
    "vix_ma20_distance",
    "vix_return_1",
    "at2x_volatility_expansion_score",
    "at2x_liquidity_score",
    "is_put",
    "band_low",
    "put_x_at2x_expansion",
    "at2x_expansion_x_liquidity",
)

MINIMAL_LANDMARK_FEATURES = (
    "at2x_option_realized_volatility_60",
    "at2x_option_realized_volatility_20",
    "at2x_option_slope_10",
    "vix_slope_120",
    "is_put",
    "band_low",
    "at2x_minutes_since_entry",
)


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def rounded(value: object, digits: int = 4) -> float | None:
    number = finite(value)
    return round(number, digits) if number is not None else None


def percentage(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100.0, 4) if denominator else 0.0


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def greater(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    left_values = numeric(frame, left)
    right_values = numeric(frame, right)
    return (left_values > right_values).fillna(False)


def add_regime_and_liquidity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["is_put"] = 1 - numeric(result, "is_call").fillna(0).astype(int)
    result["log_signal_bar_volume"] = np.log1p(
        numeric(result, "signal_bar_volume").clip(lower=0)
    )
    result["log_at2x_bar_volume"] = np.log1p(
        numeric(result, "at2x_bar_volume").clip(lower=0)
    )
    result["at2x_minute_from_open"] = (
        numeric(result, "at2x_minute").floordiv(100) * 60
        + numeric(result, "at2x_minute").mod(100)
        - 9 * 60
    )

    daily_points = (
        (numeric(result, "vix_ma5_distance") > 0).astype(int)
        + (numeric(result, "vix_slope_5") > 0).astype(int)
        + (numeric(result, "vix_return_1") > 0).astype(int)
    )
    entry_intraday_points = (
        greater(
            result,
            "intraday_futures_realized_volatility_5",
            "intraday_futures_realized_volatility_20",
        ).astype(int)
        + greater(
            result,
            "intraday_spot_realized_volatility_5",
            "intraday_spot_realized_volatility_20",
        ).astype(int)
        + greater(
            result,
            "intraday_futures_realized_volatility_20",
            "intraday_futures_realized_volatility_60",
        ).astype(int)
        + greater(
            result,
            "intraday_spot_realized_volatility_20",
            "intraday_spot_realized_volatility_60",
        ).astype(int)
    )
    at2x_intraday_points = (
        greater(
            result,
            "at2x_intraday_futures_realized_volatility_5",
            "at2x_intraday_futures_realized_volatility_20",
        ).astype(int)
        + greater(
            result,
            "at2x_intraday_spot_realized_volatility_5",
            "at2x_intraday_spot_realized_volatility_20",
        ).astype(int)
        + greater(
            result,
            "at2x_intraday_futures_realized_volatility_20",
            "at2x_intraday_futures_realized_volatility_60",
        ).astype(int)
        + greater(
            result,
            "at2x_intraday_spot_realized_volatility_20",
            "at2x_intraday_spot_realized_volatility_60",
        ).astype(int)
    )
    result["entry_vix_trend_points"] = daily_points
    result["entry_intraday_expansion_points"] = entry_intraday_points
    result["entry_volatility_expansion_score"] = (
        daily_points + entry_intraday_points
    )
    result["at2x_intraday_expansion_points"] = at2x_intraday_points
    result["at2x_volatility_expansion_score"] = (
        daily_points + at2x_intraday_points
    )

    contracting = (
        (numeric(result, "vix_slope_5") < 0)
        & (numeric(result, "vix_return_1") < 0)
        & (entry_intraday_points == 0)
    )
    expanding = (daily_points >= 2) & (entry_intraday_points >= 1)
    high_stable = (
        (numeric(result, "vix_ma20_distance") > 0)
        & (numeric(result, "vix_slope_20") > 0)
    )
    result["entry_volatility_regime"] = np.select(
        [expanding, contracting, high_stable],
        ["EXPANDING", "CONTRACTING", "HIGH_STABLE"],
        default="NORMAL",
    )

    at2x_contracting = (
        (numeric(result, "vix_slope_5") < 0)
        & (numeric(result, "vix_return_1") < 0)
        & (at2x_intraday_points == 0)
    )
    at2x_expanding = (daily_points >= 2) & (at2x_intraday_points >= 1)
    result["at2x_volatility_regime"] = np.select(
        [at2x_expanding, at2x_contracting, high_stable],
        ["EXPANDING", "CONTRACTING", "HIGH_STABLE"],
        default="NORMAL",
    )

    result["entry_liquidity_score"] = (
        (numeric(result, "fill_exact_010") > 0).astype(int)
        + (numeric(result, "entry_wait_minutes") <= 1).astype(int)
        + (numeric(result, "fill_capacity_10pct_contracts") >= 1).astype(int)
        + (
            numeric(result, "option_recent20_traded_ratio") >= 0.50
        ).astype(int)
        + (
            numeric(
                result,
                "option_recent20_round_010_volume_share",
            )
            >= 0.50
        ).astype(int)
    )
    result["at2x_liquidity_score"] = (
        (numeric(result, "at2x_option_recent20_traded_ratio") >= 0.50)
        .astype(int)
        + (
            numeric(
                result,
                "at2x_option_recent20_round_010_volume_share",
            )
            >= 0.50
        ).astype(int)
        + (numeric(result, "at2x_option_volume_ratio_20") >= 1.0).astype(int)
        + (numeric(result, "at2x_bar_volume") >= 10).astype(int)
        + (
            numeric(result, "at2x_option_recent20_max_no_trade_run") <= 2
        ).astype(int)
    )
    result["put_x_entry_expansion"] = (
        result["is_put"] * result["entry_volatility_expansion_score"]
    )
    result["entry_expansion_x_liquidity"] = (
        result["entry_volatility_expansion_score"]
        * result["entry_liquidity_score"]
    )
    result["put_x_at2x_expansion"] = (
        result["is_put"] * result["at2x_volatility_expansion_score"]
    )
    result["at2x_expansion_x_liquidity"] = (
        result["at2x_volatility_expansion_score"]
        * result["at2x_liquidity_score"]
    )
    return result


def period_name(date_value: str) -> str:
    if date_value < "20251001":
        return "PRE_2025Q4"
    if date_value <= "20251231":
        return "2025Q4"
    if date_value <= DEVELOPMENT_END:
        return "2026_JAN_APR"
    return "2026_MAY_SEP"


def group_outcome_summary(
    rows: pd.DataFrame,
    group_column: str,
) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    for key, group in rows.groupby(
        group_column,
        dropna=False,
        observed=True,
    ):
        reached_2x = int(numeric(group, "reached_2x").sum())
        reached_10x = int(numeric(group, "reached_10x").sum())
        documents.append(
            {
                group_column: str(key),
                "events": len(group),
                "expiry_days": int(group["expiry_date"].nunique()),
                "reached_2x_pct": percentage(reached_2x, len(group)),
                "reached_10x_pct": percentage(reached_10x, len(group)),
                "ten_given_two_pct": percentage(reached_10x, reached_2x),
                "median_fill_volume": rounded(
                    numeric(group, "fill_bar_volume").median()
                ),
                "median_recent20_traded_ratio": rounded(
                    numeric(
                        group,
                        "option_recent20_traded_ratio",
                    ).median()
                ),
            }
        )
    return documents


def liquidity_profile(frame: pd.DataFrame) -> dict[str, object]:
    rows = frame.copy()
    rows["period"] = rows["expiry_date"].map(period_name)
    periods: list[dict[str, object]] = []
    for period, group in rows.groupby("period", sort=False):
        exact = group[numeric(group, "fill_exact_010") > 0]
        non_exact = group[numeric(group, "fill_exact_010") <= 0]
        periods.append(
            {
                "period": period,
                "events": len(group),
                "expiry_days": int(group["expiry_date"].nunique()),
                "fill_exact_010_pct": percentage(len(exact), len(group)),
                "immediate_next_minute_pct": percentage(
                    int(
                        (
                            numeric(group, "entry_immediate_next_minute") > 0
                        ).sum()
                    ),
                    len(group),
                ),
                "median_fill_bar_volume": rounded(
                    numeric(group, "fill_bar_volume").median()
                ),
                "p75_fill_bar_volume": rounded(
                    numeric(group, "fill_bar_volume").quantile(0.75)
                ),
                "median_fill_capacity_10pct": rounded(
                    numeric(
                        group,
                        "fill_capacity_10pct_contracts",
                    ).median()
                ),
                "median_recent20_round_volume_share": rounded(
                    numeric(
                        group,
                        "option_recent20_round_010_volume_share",
                    ).median()
                ),
                "exact_010_median_fill_volume": rounded(
                    numeric(exact, "fill_bar_volume").median()
                ),
                "non_exact_median_fill_volume": rounded(
                    numeric(non_exact, "fill_bar_volume").median()
                ),
                "exact_010_reached_10x_pct": percentage(
                    int(numeric(exact, "reached_10x").sum()),
                    len(exact),
                ),
                "non_exact_reached_10x_pct": percentage(
                    int(numeric(non_exact, "reached_10x").sum()),
                    len(non_exact),
                ),
            }
        )
    earlier = rows[rows["expiry_date"] < "20251001"]
    later = rows[rows["expiry_date"] >= "20251001"]
    return {
        "data_limit": (
            "옵션 원자료는 1분 OHLCV 체결자료이며 매수·매도 호가잔량과 "
            "스프레드는 없다. 아래 수치는 호가원장 깊이가 아니라 "
            "실제 체결 유동성 대용치다."
        ),
        "periods": periods,
        "late_vs_earlier": {
            "earlier_events": len(earlier),
            "late_events": len(later),
            "earlier_exact_010_pct": percentage(
                int((numeric(earlier, "fill_exact_010") > 0).sum()),
                len(earlier),
            ),
            "late_exact_010_pct": percentage(
                int((numeric(later, "fill_exact_010") > 0).sum()),
                len(later),
            ),
            "earlier_median_fill_volume": rounded(
                numeric(earlier, "fill_bar_volume").median()
            ),
            "late_median_fill_volume": rounded(
                numeric(later, "fill_bar_volume").median()
            ),
            "earlier_median_capacity_10pct": rounded(
                numeric(
                    earlier,
                    "fill_capacity_10pct_contracts",
                ).median()
            ),
            "late_median_capacity_10pct": rounded(
                numeric(
                    later,
                    "fill_capacity_10pct_contracts",
                ).median()
            ),
        },
    }


def model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.10,
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=42,
                ),
            ),
        ]
    )


def score_metrics(
    target: np.ndarray,
    scores: np.ndarray,
) -> dict[str, object]:
    if len(target) == 0 or len(np.unique(target)) < 2:
        return {
            "rows": len(target),
            "positives": int(target.sum()) if len(target) else 0,
            "auc": None,
            "average_precision": None,
            "top_20pct_lift": None,
            "brier_score": None,
        }
    count = max(1, math.ceil(len(target) * 0.20))
    top = np.argsort(scores)[::-1][:count]
    prevalence = float(target.mean())
    return {
        "rows": len(target),
        "positives": int(target.sum()),
        "prevalence_pct": round(prevalence * 100.0, 4),
        "auc": round(float(roc_auc_score(target, scores)), 4),
        "average_precision": round(
            float(average_precision_score(target, scores)),
            4,
        ),
        "top_20pct_lift": round(
            float(target[top].mean() / prevalence),
            4,
        ),
        "brier_score": round(float(brier_score_loss(target, scores)), 4),
    }


def cluster_bootstrap_auc(
    rows: pd.DataFrame,
    scores: np.ndarray,
    iterations: int = 500,
) -> dict[str, float | None]:
    data = rows[["expiry_date", "target"]].copy()
    data["score"] = scores
    dates = data["expiry_date"].unique()
    generator = np.random.default_rng(42)
    values: list[float] = []
    for _ in range(iterations):
        sampled = generator.choice(dates, size=len(dates), replace=True)
        boot = pd.concat(
            [data[data["expiry_date"] == item] for item in sampled],
            ignore_index=True,
        )
        if boot["target"].nunique() < 2:
            continue
        values.append(float(roc_auc_score(boot["target"], boot["score"])))
    return {
        "low_95": (
            round(float(np.quantile(values, 0.025)), 4)
            if values
            else None
        ),
        "median": (
            round(float(np.median(values)), 4) if values else None
        ),
        "high_95": (
            round(float(np.quantile(values, 0.975)), 4)
            if values
            else None
        ),
    }


def available_features(
    frame: pd.DataFrame,
    requested: tuple[str, ...],
) -> list[str]:
    return [
        column
        for column in requested
        if column in frame.columns and numeric(frame, column).notna().any()
    ]


def fit_model(
    frame: pd.DataFrame,
    requested_features: tuple[str, ...],
    name: str,
) -> tuple[dict[str, object], pd.Series]:
    features = available_features(frame, requested_features)
    development = frame[frame["expiry_date"] <= DEVELOPMENT_END].copy()
    validation = frame[frame["expiry_date"] > DEVELOPMENT_END].copy()
    model = model_pipeline()
    model.fit(development[features], development["target"])
    development_scores = model.predict_proba(development[features])[:, 1]
    validation_scores = model.predict_proba(validation[features])[:, 1]
    validation_cluster = validation[["expiry_date", "target"]].copy()

    coefficients = model.named_steps["model"].coef_[0]
    coefficient_rows = sorted(
        (
            {
                "feature": feature,
                "standardized_coefficient": round(float(coefficient), 5),
            }
            for feature, coefficient in zip(features, coefficients)
        ),
        key=lambda item: abs(float(item["standardized_coefficient"])),
        reverse=True,
    )

    walk_documents: list[dict[str, object]] = []
    out_of_fold: list[pd.DataFrame] = []
    for train_end, test_start, test_end in WALK_FORWARD_FOLDS:
        train = frame[frame["expiry_date"] <= train_end]
        test = frame[
            (frame["expiry_date"] >= test_start)
            & (frame["expiry_date"] <= test_end)
        ]
        if (
            len(train) < 100
            or len(test) < 20
            or train["target"].nunique() < 2
            or test["target"].nunique() < 2
        ):
            walk_documents.append(
                {
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "skipped": True,
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
            )
            continue
        fold_model = model_pipeline()
        fold_model.fit(train[features], train["target"])
        scores = fold_model.predict_proba(test[features])[:, 1]
        walk_documents.append(
            {
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                **score_metrics(test["target"].to_numpy(), scores),
            }
        )
        part = test[["expiry_date", "target"]].copy()
        part["score"] = scores
        out_of_fold.append(part)

    walk_aggregate: dict[str, object] = {}
    if out_of_fold:
        combined = pd.concat(out_of_fold, ignore_index=True)
        walk_aggregate = score_metrics(
            combined["target"].to_numpy(),
            combined["score"].to_numpy(),
        )
        walk_aggregate["expiry_cluster_auc_ci"] = cluster_bootstrap_auc(
            combined,
            combined["score"].to_numpy(),
        )

    all_scores = pd.Series(np.nan, index=frame.index, dtype=float)
    all_scores.loc[development.index] = development_scores
    all_scores.loc[validation.index] = validation_scores
    return (
        {
            "name": name,
            "features": features,
            "development": score_metrics(
                development["target"].to_numpy(),
                development_scores,
            ),
            "validation": score_metrics(
                validation["target"].to_numpy(),
                validation_scores,
            ),
            "validation_expiry_cluster_auc_ci": cluster_bootstrap_auc(
                validation_cluster.assign(
                    target=validation["target"].to_numpy()
                ),
                validation_scores,
            ),
            "walk_forward": {
                "folds": walk_documents,
                "aggregate": walk_aggregate,
            },
            "largest_standardized_coefficients": coefficient_rows[:12],
        },
        all_scores,
    )


def univariate_stability(
    frame: pd.DataFrame,
    features: list[str],
) -> list[dict[str, object]]:
    development = frame[frame["expiry_date"] <= DEVELOPMENT_END]
    validation = frame[frame["expiry_date"] > DEVELOPMENT_END]
    documents: list[dict[str, object]] = []
    for feature in features:
        dev = development[[feature, "target"]].dropna()
        val = validation[[feature, "target"]].dropna()
        if (
            len(dev) < 40
            or len(val) < 40
            or dev["target"].nunique() < 2
            or val["target"].nunique() < 2
        ):
            continue
        positive_median = dev.loc[dev["target"] == 1, feature].median()
        negative_median = dev.loc[dev["target"] == 0, feature].median()
        direction = 1 if positive_median >= negative_median else -1
        dev_auc = roc_auc_score(dev["target"], dev[feature] * direction)
        val_auc = roc_auc_score(val["target"], val[feature] * direction)
        documents.append(
            {
                "feature": feature,
                "development_direction": (
                    "higher_for_10x" if direction > 0 else "lower_for_10x"
                ),
                "development_oriented_auc": round(float(dev_auc), 4),
                "validation_oriented_auc": round(float(val_auc), 4),
                "development_positive_median": rounded(positive_median, 6),
                "development_negative_median": rounded(negative_median, 6),
                "validation_positive_median": rounded(
                    val.loc[val["target"] == 1, feature].median(),
                    6,
                ),
                "validation_negative_median": rounded(
                    val.loc[val["target"] == 0, feature].median(),
                    6,
                ),
            }
        )
    return sorted(
        documents,
        key=lambda item: float(item["validation_oriented_auc"]),
        reverse=True,
    )


def landmark_group_summaries(frame: pd.DataFrame) -> dict[str, object]:
    rows = frame.copy()
    rows["time_to_2x_bucket"] = pd.cut(
        numeric(rows, "at2x_minutes_since_entry"),
        bins=[-1, 15, 30, 60, 120, math.inf],
        labels=["00_15", "16_30", "31_60", "61_120", "121_PLUS"],
    )
    rows["at2x_score_bucket"] = pd.cut(
        numeric(rows, "at2x_volatility_expansion_score"),
        bins=[-1, 1, 3, 5, math.inf],
        labels=["0_1", "2_3", "4_5", "6_7"],
    )
    rows["sample_period"] = np.where(
        rows["expiry_date"] <= DEVELOPMENT_END,
        "DEVELOPMENT",
        "VALIDATION",
    )
    rows["period_regime"] = (
        rows["sample_period"].astype(str)
        + "__"
        + rows["at2x_volatility_regime"].astype(str)
    )
    rows["period_score"] = (
        rows["sample_period"].astype(str)
        + "__"
        + rows["at2x_score_bucket"].astype(str)
    )
    return {
        "by_volatility_regime": group_outcome_summary(
            rows,
            "at2x_volatility_regime",
        ),
        "by_time_to_2x": group_outcome_summary(
            rows,
            "time_to_2x_bucket",
        ),
        "by_expansion_score": group_outcome_summary(
            rows,
            "at2x_score_bucket",
        ),
        "by_period_and_regime": group_outcome_summary(
            rows,
            "period_regime",
        ),
        "by_period_and_expansion_score": group_outcome_summary(
            rows,
            "period_score",
        ),
        "by_call_put": group_outcome_summary(rows, "call_put"),
        "by_premium_band": group_outcome_summary(rows, "premium_band"),
        "by_fill_exact_010": group_outcome_summary(rows, "fill_exact_010"),
    }


def selection_metrics(
    rows: pd.DataFrame,
    selected: pd.Series,
) -> dict[str, object]:
    mask = selected.reindex(rows.index).fillna(False).astype(bool)
    chosen = rows[mask]
    baseline = float(rows["target"].mean()) if len(rows) else 0.0
    chosen_rate = float(chosen["target"].mean()) if len(chosen) else 0.0
    return {
        "rows": len(rows),
        "selected": len(chosen),
        "coverage_pct": percentage(len(chosen), len(rows)),
        "selected_10x": int(chosen["target"].sum()) if len(chosen) else 0,
        "selected_10x_rate_pct": round(chosen_rate * 100.0, 4),
        "baseline_10x_rate_pct": round(baseline * 100.0, 4),
        "lift": round(chosen_rate / baseline, 4) if baseline > 0 else None,
    }


def development_derived_simple_rules(
    frame: pd.DataFrame,
) -> dict[str, object]:
    development = frame[frame["expiry_date"] <= DEVELOPMENT_END]
    validation = frame[frame["expiry_date"] > DEVELOPMENT_END]
    thresholds: dict[str, float] = {}
    for feature in (
        "at2x_option_realized_volatility_60",
        "at2x_option_realized_volatility_20",
        "at2x_option_slope_10",
        "vix_slope_120",
    ):
        positive = numeric(
            development[development["target"] == 1],
            feature,
        ).median()
        negative = numeric(
            development[development["target"] == 0],
            feature,
        ).median()
        thresholds[feature] = float((positive + negative) / 2.0)

    rules = {
        "RV60_HIGH": (
            numeric(frame, "at2x_option_realized_volatility_60")
            >= thresholds["at2x_option_realized_volatility_60"]
        ),
        "RV60_AND_SLOPE10": (
            (
                numeric(frame, "at2x_option_realized_volatility_60")
                >= thresholds["at2x_option_realized_volatility_60"]
            )
            & (
                numeric(frame, "at2x_option_slope_10")
                >= thresholds["at2x_option_slope_10"]
            )
        ),
        "RV60_AND_VIX120": (
            (
                numeric(frame, "at2x_option_realized_volatility_60")
                >= thresholds["at2x_option_realized_volatility_60"]
            )
            & (
                numeric(frame, "vix_slope_120")
                >= thresholds["vix_slope_120"]
            )
        ),
        "RV60_AND_PUT": (
            (
                numeric(frame, "at2x_option_realized_volatility_60")
                >= thresholds["at2x_option_realized_volatility_60"]
            )
            & (numeric(frame, "is_put") > 0)
        ),
    }
    return {
        "warning": (
            "RV60 등 후보가 후반 단변량 결과를 확인한 뒤 선택됐으므로 "
            "독립 채택 근거가 아니라 다음 신규 만기 잠금검증 후보이다."
        ),
        "development_midpoint_thresholds": {
            key: round(value, 8) for key, value in thresholds.items()
        },
        "rules": {
            name: {
                "development": selection_metrics(
                    development,
                    mask,
                ),
                "validation": selection_metrics(
                    validation,
                    mask,
                ),
            }
            for name, mask in rules.items()
        },
    }


def main() -> None:
    frame = pd.read_csv(EVENT_PATH, encoding="utf-8-sig")
    frame["expiry_date"] = frame["expiry_date"].astype(str).str.zfill(8)
    frame = add_regime_and_liquidity(frame)

    reached_two = frame[numeric(frame, "reached_2x") > 0].copy()
    same_bar = numeric(reached_two, "at2x_same_bar_10x") > 0
    actionable = reached_two[~same_bar].copy()
    strict = actionable[
        (numeric(actionable, "at2x_future_max_multiple") < 5)
        | (numeric(actionable, "at2x_future_max_multiple") >= 10)
    ].copy()
    strict["target"] = (
        numeric(strict, "at2x_future_max_multiple") >= 10
    ).astype(int)

    entry_model, entry_scores = fit_model(
        strict,
        ENTRY_FEATURES,
        "ENTRY_INFORMATION_ONLY",
    )
    landmark_model, landmark_scores = fit_model(
        strict,
        LANDMARK_FEATURES,
        "AT_2X_CONFIRMED_BAR",
    )
    minimal_model, minimal_scores = fit_model(
        strict,
        MINIMAL_LANDMARK_FEATURES,
        "AT_2X_MINIMAL_STABLE_FEATURES",
    )
    strict["entry_model_score"] = entry_scores
    strict["at2x_model_score"] = landmark_scores
    strict["at2x_minimal_model_score"] = minimal_scores

    validation = landmark_model["validation"]
    validation_ci = landmark_model["validation_expiry_cluster_auc_ci"]
    walk = landmark_model["walk_forward"]["aggregate"]
    passed = bool(
        (validation.get("auc") or 0) >= 0.60
        and (validation.get("top_20pct_lift") or 0) >= 1.50
        and (validation_ci.get("low_95") or 0) > 0.50
        and (walk.get("auc") or 0) >= 0.58
    )
    minimal_validation = minimal_model["validation"]
    minimal_ci = minimal_model["validation_expiry_cluster_auc_ci"]
    minimal_walk = minimal_model["walk_forward"]["aggregate"]
    minimal_metrics_passed = bool(
        (minimal_validation.get("auc") or 0) >= 0.60
        and (minimal_validation.get("top_20pct_lift") or 0) >= 1.50
        and (minimal_ci.get("low_95") or 0) > 0.50
        and (minimal_walk.get("auc") or 0) >= 0.58
    )

    available_landmark = available_features(strict, LANDMARK_FEATURES)
    result = {
        "status": "temporary_regime_liquidity_research",
        "data": {
            "event_rows": len(frame),
            "expiry_days": int(frame["expiry_date"].nunique()),
            "reached_2x_rows": len(reached_two),
            "same_bar_2x_10x_ambiguous_rows": int(same_bar.sum()),
            "actionable_after_2x_rows": len(actionable),
            "strict_2_to_below5_vs_10x_rows": len(strict),
            "strict_10x_rows": int(strict["target"].sum()),
            "development_end": DEVELOPMENT_END,
        },
        "definitions": {
            "volatility_expanding": (
                "직전 확정 VIX의 종가>SMA5·5일기울기>0·전일상승 중 "
                "2개 이상이고, 판단시점까지 현물·선물 RV5>RV20 또는 "
                "RV20>RV60 조건 중 1개 이상"
            ),
            "liquidity_score": (
                "0.10 실제 체결, 다음 1분 체결, 거래량10% 한 계약 이상, "
                "최근20분 체결밀도 50% 이상, 0.10 체결량 비중 50% 이상"
            ),
            "direct_comparison": (
                "2배 확인봉 이후 미래 최고배수가 5배 미만인 사건과 "
                "10배 이상인 사건만 비교. 2배와 10배가 같은 분봉인 "
                "순서불명 사건은 제외"
            ),
            "feature_cutoff": (
                "진입모델은 신호 완료봉까지, 2배모델은 첫 2배 확인봉 "
                "완료시점까지의 정보만 사용"
            ),
        },
        "liquidity_profile": liquidity_profile(frame),
        "entry_regime_outcomes": group_outcome_summary(
            frame,
            "entry_volatility_regime",
        ),
        "at2x_group_outcomes": landmark_group_summaries(strict),
        "models": {
            "entry_information_only": entry_model,
            "at_2x_confirmed_bar": landmark_model,
            "at_2x_minimal_stable_features": minimal_model,
        },
        "stable_univariate_candidates": univariate_stability(
            strict,
            available_landmark,
        )[:20],
        "exploratory_simple_rules": development_derived_simple_rules(
            strict
        ),
        "promotion_gate": {
            "requirements": {
                "validation_auc_min": 0.60,
                "validation_top20_lift_min": 1.50,
                "validation_cluster_auc_low95_above": 0.50,
                "walk_forward_auc_min": 0.58,
            },
            "predeclared_full_model_metrics_passed": passed,
            "post_selection_minimal_model_metrics_passed": (
                minimal_metrics_passed
            ),
            "independent_validation_passed": False,
            "passed": False,
            "decision": (
                "최소모델 수치는 기준을 넘었지만 후반 결과를 본 뒤 "
                "변수를 축소했으므로 정식 승격하지 않고 2026-09-28 "
                "이후 신규 만기 잠금검증 후보로 고정"
                if minimal_metrics_passed
                else (
                    "2배 확인시점 정보로도 분리력이 충분하지 않아 "
                    "동적 10배 비중 전략 승격 보류"
                )
            ),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    strict.to_csv(SCORED_PATH, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "data": result["data"],
                "liquidity_profile": result["liquidity_profile"],
                "entry_model_validation": entry_model["validation"],
                "landmark_model_validation": landmark_model["validation"],
                "landmark_walk_forward": landmark_model[
                    "walk_forward"
                ]["aggregate"],
                "promotion_gate": result["promotion_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
