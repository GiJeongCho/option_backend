from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
EVENT_PATH = OUTPUT_DIR / "s6_event_features.csv"
SUMMARY_PATH = OUTPUT_DIR / "s6_hypothesis_followup.json"
DEVELOPMENT_END = "20260430"

FOCUSED_FEATURES = (
    "option_realized_volatility_20",
    "option_realized_volatility_60",
    "option_slope_acceleration_5",
    "option_slope_5",
    "option_slope_10",
    "option_rsi_5",
    "option_rsi_14",
    "option_volume_ratio_20",
    "signed_intraday_futures_slope_5",
    "signed_intraday_spot_slope_5",
    "signed_intraday_futures_return_5",
    "signed_intraday_spot_return_5",
    "intraday_spot_futures_corr_20",
    "intraday_basis_slope_10",
    "vix_slope_120",
    "vix_ma20_distance",
    "vix_rsi_14",
    "weekly_futures_slope_20",
    "daily_futures_slope_20",
    "quarterly_futures_dte",
    "is_call",
    "band_low",
)

WALK_FORWARD_FOLDS = (
    ("20251231", "20260101", "20260228"),
    ("20260228", "20260301", "20260430"),
    ("20260430", "20260501", "20260630"),
    ("20260630", "20260701", "20260831"),
    ("20260831", "20260901", "20260930"),
)


def percentage(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 4) if denominator else 0.0


def model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.15,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )


def metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float | int | None]:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "rows": len(y),
            "positives": int(y.sum()) if len(y) else 0,
            "auc": None,
            "average_precision": None,
            "top_20pct_lift": None,
        }
    count = max(1, math.ceil(len(y) * 0.20))
    top = np.argsort(scores)[::-1][:count]
    prevalence = float(y.mean())
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "prevalence_pct": round(prevalence * 100, 4),
        "auc": round(float(roc_auc_score(y, scores)), 4),
        "average_precision": round(
            float(average_precision_score(y, scores)), 4
        ),
        "top_20pct_lift": round(float(y[top].mean() / prevalence), 4),
    }


def cluster_bootstrap_auc(
    rows: pd.DataFrame,
    scores: np.ndarray,
    iterations: int = 400,
) -> dict[str, float | None]:
    data = rows[["expiry_date", "target"]].copy()
    data["score"] = scores
    dates = data["expiry_date"].unique()
    generator = np.random.default_rng(42)
    values: list[float] = []
    for _ in range(iterations):
        sampled = generator.choice(dates, size=len(dates), replace=True)
        parts = [data[data["expiry_date"] == item] for item in sampled]
        boot = pd.concat(parts, ignore_index=True)
        if boot["target"].nunique() < 2:
            continue
        values.append(float(roc_auc_score(boot["target"], boot["score"])))
    if not values:
        return {"low_95": None, "median": None, "high_95": None}
    return {
        "low_95": round(float(np.quantile(values, 0.025)), 4),
        "median": round(float(np.median(values)), 4),
        "high_95": round(float(np.quantile(values, 0.975)), 4),
    }


def fixed_split(
    frame: pd.DataFrame,
    features: list[str],
    target_column: str,
) -> tuple[dict[str, object], Pipeline]:
    development = frame[frame["expiry_date"] <= DEVELOPMENT_END].copy()
    validation = frame[frame["expiry_date"] > DEVELOPMENT_END].copy()
    model = model_pipeline()
    model.fit(development[features], development[target_column])
    train_scores = model.predict_proba(development[features])[:, 1]
    validation_scores = model.predict_proba(validation[features])[:, 1]
    validation_with_target = validation.copy()
    validation_with_target["target"] = validation[target_column].astype(int)
    return (
        {
            "development": metrics(
                development[target_column].to_numpy(), train_scores
            ),
            "validation": metrics(
                validation[target_column].to_numpy(), validation_scores
            ),
            "validation_expiry_cluster_auc_ci": cluster_bootstrap_auc(
                validation_with_target,
                validation_scores,
            ),
        },
        model,
    )


def walk_forward(
    frame: pd.DataFrame,
    features: list[str],
    target_column: str,
) -> dict[str, object]:
    documents: list[dict[str, object]] = []
    oof_parts: list[pd.DataFrame] = []
    for train_end, test_start, test_end in WALK_FORWARD_FOLDS:
        train = frame[frame["expiry_date"] <= train_end]
        test = frame[
            (frame["expiry_date"] >= test_start)
            & (frame["expiry_date"] <= test_end)
        ]
        if (
            len(train) < 100
            or len(test) < 20
            or train[target_column].nunique() < 2
            or test[target_column].nunique() < 2
        ):
            documents.append(
                {
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "skipped": True,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "test_positives": int(test[target_column].sum()),
                }
            )
            continue
        model = model_pipeline()
        model.fit(train[features], train[target_column])
        scores = model.predict_proba(test[features])[:, 1]
        document = {
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            **metrics(test[target_column].to_numpy(), scores),
        }
        documents.append(document)
        part = test[["expiry_date", target_column]].copy()
        part["score"] = scores
        part = part.rename(columns={target_column: "target"})
        oof_parts.append(part)
    aggregate: dict[str, object] = {}
    if oof_parts:
        oof = pd.concat(oof_parts, ignore_index=True)
        aggregate = metrics(
            oof["target"].to_numpy(), oof["score"].to_numpy()
        )
        aggregate["expiry_cluster_auc_ci"] = cluster_bootstrap_auc(
            oof, oof["score"].to_numpy()
        )
    return {"folds": documents, "aggregate": aggregate}


def focused_tree(
    frame: pd.DataFrame,
    features: list[str],
    target_column: str,
) -> dict[str, object]:
    development = frame[frame["expiry_date"] <= DEVELOPMENT_END].copy()
    validation = frame[frame["expiry_date"] > DEVELOPMENT_END].copy()
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(development[features])
    validation_values = imputer.transform(validation[features])
    tree = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=max(20, int(len(development) * 0.05)),
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(train_values, development[target_column])
    train_scores = tree.predict_proba(train_values)[:, 1]
    validation_scores = tree.predict_proba(validation_values)[:, 1]
    return {
        "rules": export_text(tree, feature_names=features),
        "development": metrics(
            development[target_column].to_numpy(), train_scores
        ),
        "validation": metrics(
            validation[target_column].to_numpy(), validation_scores
        ),
    }


def first_passing_diagnostic(
    frame: pd.DataFrame,
    features: list[str],
    model: Pipeline,
    target_column: str,
) -> dict[str, object]:
    development = frame[frame["expiry_date"] <= DEVELOPMENT_END].copy()
    validation = frame[frame["expiry_date"] > DEVELOPMENT_END].copy()
    train_scores = model.predict_proba(development[features])[:, 1]
    threshold = float(np.quantile(train_scores, 0.80))
    validation["score"] = model.predict_proba(validation[features])[:, 1]
    ordered = validation.sort_values(
        ["expiry_date", "signal_minute", "score", "option_code"],
        ascending=[True, True, False, True],
    )
    baseline = ordered.groupby("expiry_date", as_index=False).first()
    passing = ordered[ordered["score"] >= threshold]
    selected = passing.groupby("expiry_date", as_index=False).first()

    def selection_summary(rows: pd.DataFrame) -> dict[str, object]:
        return {
            "selected_expiry_days": len(rows),
            "10x_days": int(rows["reached_10x"].sum()),
            "10x_rate_pct": percentage(
                int(rows["reached_10x"].sum()), len(rows)
            ),
            "5x_days": int(rows["reached_5x"].sum()),
            "5x_rate_pct": percentage(
                int(rows["reached_5x"].sum()), len(rows)
            ),
            "2x_days": int(rows["reached_2x"].sum()),
            "2x_rate_pct": percentage(
                int(rows["reached_2x"].sum()), len(rows)
            ),
            "low_band_days": int(rows["band_low"].sum()),
            "put_days": int((rows["is_call"] == 0).sum()),
        }

    return {
        "warning": (
            "후반 구간을 처음 확인한 뒤 작성한 진단이므로 독립 검증이 "
            "아니며 S6 채택 근거로 단독 사용하지 않음"
        ),
        "development_score_80pct_threshold": threshold,
        "baseline_first_event_each_expiry": selection_summary(baseline),
        "first_event_passing_threshold": selection_summary(selected),
        "target_column": target_column,
    }


def band_paired_comparison(frame: pd.DataFrame) -> dict[str, object]:
    per_day = (
        frame.groupby(["expiry_date", "premium_band"])
        .agg(
            events=("option_code", "count"),
            rate_2x=("reached_2x", "mean"),
            rate_5x=("reached_5x", "mean"),
            rate_10x=("reached_10x", "mean"),
            max_multiple=("max_multiple", "max"),
        )
        .reset_index()
    )
    result: dict[str, object] = {}
    for metric in ("rate_2x", "rate_5x", "rate_10x", "max_multiple"):
        pivot = per_day.pivot(
            index="expiry_date",
            columns="premium_band",
            values=metric,
        ).dropna()
        low = pivot["LOW_040_050"]
        high = pivot["HIGH_100_250"]
        difference = low - high
        statistic = wilcoxon(
            difference,
            alternative="two-sided",
            zero_method="wilcox",
        )
        result[metric] = {
            "matched_expiry_days": len(pivot),
            "low_mean": round(float(low.mean()), 6),
            "high_mean": round(float(high.mean()), 6),
            "low_minus_high_mean": round(float(difference.mean()), 6),
            "low_minus_high_median": round(float(difference.median()), 6),
            "wilcoxon_p": float(statistic.pvalue),
        }
    return result


def main() -> None:
    frame = pd.read_csv(EVENT_PATH, encoding="utf-8-sig")
    frame["expiry_date"] = frame["expiry_date"].astype(str).str.zfill(8)
    features = [item for item in FOCUSED_FEATURES if item in frame.columns]
    tasks = {
        "TEN_PLUS_VS_ALL": "reached_10x",
        "FIVE_PLUS_VS_ALL": "reached_5x",
    }
    task_documents: dict[str, object] = {}
    for task_name, target_column in tasks.items():
        fixed, model = fixed_split(frame, features, target_column)
        task_documents[task_name] = {
            "features": features,
            "fixed_split": fixed,
            "walk_forward": walk_forward(frame, features, target_column),
            "tree": focused_tree(frame, features, target_column),
            "first_passing_diagnostic": first_passing_diagnostic(
                frame,
                features,
                model,
                target_column,
            ),
        }
    ten_validation = task_documents["TEN_PLUS_VS_ALL"]["fixed_split"][
        "validation"
    ]
    ten_walk = task_documents["TEN_PLUS_VS_ALL"]["walk_forward"]["aggregate"]
    five_validation = task_documents["FIVE_PLUS_VS_ALL"]["fixed_split"][
        "validation"
    ]
    meaningful = bool(
        (ten_validation.get("auc") or 0) >= 0.60
        and (ten_walk.get("auc") or 0) >= 0.58
        and (
            task_documents["TEN_PLUS_VS_ALL"]["fixed_split"][
                "validation_expiry_cluster_auc_ci"
            ].get("low_95")
            or 0
        )
        > 0.50
    )
    result = {
        "status": "temporary_hypothesis_followup",
        "features_are_predeclared_from_recording": True,
        "feature_policy": (
            "절대 지수 수준·연도·행사가격을 제외하고 옵션 기울기/변동성, "
            "MA·RSI, VIX 장기기울기, 현선물 단기방향·상관·basis, "
            "분기선물 만기거리만 사용"
        ),
        "band_paired_comparison": band_paired_comparison(frame),
        "tasks": task_documents,
        "s6_gate": {
            "scope": "복합 분류 모델을 실행 신호로 직접 쓰는 경우",
            "ten_validation": ten_validation,
            "ten_walk_forward": ten_walk,
            "five_validation": five_validation,
            "passed": meaningful,
            "decision": (
                "S6 정식 백테스트 후보"
                if meaningful
                else (
                    "복합 모델 직접 승격은 보류. 별도로 발견한 "
                    "VIX상승+PUT+RV20 단순 규칙은 비용 포함 백테스트와 "
                    "독립 원자료 감사로 판정"
                )
            ),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "band_paired_comparison": result["band_paired_comparison"],
                "s6_gate": result["s6_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
