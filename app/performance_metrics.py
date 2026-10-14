from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence


TRADING_DAYS_PER_YEAR = 252


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _population_variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    average = _mean(values)
    return sum((value - average) ** 2 for value in values) / len(values)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return None
    return numerator / denominator


def _annual_to_period_rate(
    annual_rate: float,
    periods_per_year: int,
) -> float:
    if annual_rate <= -1.0:
        raise ValueError("연 수익률 입력은 -100%보다 커야 합니다.")
    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


def calculate_performance_statistics(
    returns: Sequence[float],
    equities: Sequence[float],
    market_returns: Sequence[float | None],
    initial_equity: float,
    ending_equity: float,
    *,
    risk_free_annual_rate: float = 0.0,
    target_annual_rate: float = 0.0,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
    holding_years: float | None = None,
) -> dict[str, float | int | None]:
    if not returns:
        raise ValueError("성과지표를 계산할 수익률이 없습니다.")
    if len(equities) != len(returns) + 1:
        raise ValueError("계좌곡선 길이가 수익률보다 정확히 하나 커야 합니다.")
    if len(market_returns) != len(returns):
        raise ValueError("시장 수익률과 전략 수익률 길이가 다릅니다.")
    if initial_equity <= 0:
        raise ValueError("초기 자산은 0보다 커야 합니다.")

    average_return = _mean(returns)
    daily_variance = _population_variance(returns)
    daily_deviation = math.sqrt(daily_variance)
    annualized_volatility = daily_deviation * math.sqrt(
        annualization_factor
    )
    years = (
        holding_years
        if holding_years is not None
        else len(returns) / annualization_factor
    )
    if years <= 0:
        raise ValueError("투자 보유 연수는 0보다 커야 합니다.")
    annualized_return = (
        (ending_equity / initial_equity) ** (1.0 / years) - 1.0
        if ending_equity > 0 and years > 0
        else None
    )
    risk_free_daily = _annual_to_period_rate(
        risk_free_annual_rate,
        annualization_factor,
    )
    target_daily = _annual_to_period_rate(
        target_annual_rate,
        annualization_factor,
    )

    shortfalls = [
        max(0.0, target_daily - value) for value in returns
    ]
    gains = [max(0.0, value - target_daily) for value in returns]
    lpm1 = _mean(shortfalls)
    lpm2 = _mean([value * value for value in shortfalls])
    omega_ratio = _safe_ratio(sum(gains), sum(shortfalls))
    downside_deviation = math.sqrt(lpm2)
    annualized_downside_deviation = (
        downside_deviation * math.sqrt(annualization_factor)
    )
    excess_annual_return = (
        annualized_return - risk_free_annual_rate
        if annualized_return is not None
        else None
    )
    sharpe_ratio = (
        _safe_ratio(excess_annual_return, annualized_volatility)
        if excess_annual_return is not None
        else None
    )
    sortino_ratio = (
        _safe_ratio(
            excess_annual_return,
            annualized_downside_deviation,
        )
        if excess_annual_return is not None
        else None
    )

    paired = [
        (strategy_return, market_return)
        for strategy_return, market_return in zip(
            returns,
            market_returns,
            strict=True,
        )
        if market_return is not None
    ]
    beta: float | None = None
    benchmark_observations = len(paired)
    if paired:
        strategy_values = [item[0] for item in paired]
        market_values = [float(item[1]) for item in paired]
        strategy_average = _mean(strategy_values)
        market_average = _mean(market_values)
        covariance = _mean(
            [
                (strategy_value - strategy_average)
                * (market_value - market_average)
                for strategy_value, market_value in paired
            ]
        )
        beta = _safe_ratio(
            covariance,
            _population_variance(market_values),
        )

    peak = equities[0]
    maximum_drawdown = 0.0
    for equity in equities:
        peak = max(peak, equity)
        if peak > 0:
            maximum_drawdown = min(
                maximum_drawdown,
                (equity - peak) / peak,
            )

    return {
        "observations": len(returns),
        "benchmark_observations": benchmark_observations,
        "average_daily_return": average_return,
        "annualized_return": annualized_return,
        "daily_return_deviation": daily_deviation,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "lpm1": lpm1,
        "lpm2": lpm2,
        "omega_ratio": omega_ratio,
        "sortino_ratio": sortino_ratio,
        "beta": beta,
        "maximum_drawdown": maximum_drawdown,
        "downside_deviation": downside_deviation,
        "annualized_downside_deviation": annualized_downside_deviation,
        "risk_free_daily": risk_free_daily,
        "target_daily": target_daily,
    }


def _parse_strategy_date(value: str) -> str:
    compact = value.strip().replace("-", "")
    parsed = datetime.strptime(compact, "%Y%m%d")
    return parsed.strftime("%Y%m%d")


def read_benchmark_returns(
    path: Path,
) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    result: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            date_value = _parse_strategy_date(row["날짜"].replace(" ", ""))
            pct_text = row["변동 %"].strip().replace("%", "")
            result[date_value] = float(pct_text) / 100.0
    return result


def build_performance_report(
    strategy_version: str,
    strategy_label: str,
    daily_rows: Sequence[dict[str, str]],
    benchmark_path: Path,
    *,
    risk_free_annual_pct: float = 0.0,
    target_annual_pct: float = 0.0,
) -> dict[str, object]:
    if not daily_rows:
        raise ValueError("전략 일별 결과가 없습니다.")
    rows = sorted(daily_rows, key=lambda row: _parse_strategy_date(row["date"]))
    start_date = _parse_strategy_date(rows[0]["date"])
    end_date = _parse_strategy_date(rows[-1]["date"])
    initial_equity = float(rows[0]["start_cash"])
    reported_ending_equity = float(rows[-1]["end_cash"])
    pnl_by_date = {
        _parse_strategy_date(row["date"]): float(row["pnl"])
        for row in rows
    }
    benchmark = read_benchmark_returns(benchmark_path)
    start_datetime = datetime.strptime(start_date, "%Y%m%d")
    end_datetime = datetime.strptime(end_date, "%Y%m%d")
    calendar: list[str] = []
    current = start_datetime
    while current <= end_datetime:
        if current.weekday() < 5:
            calendar.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    missing_strategy_dates = sorted(set(pnl_by_date) - set(calendar))
    if missing_strategy_dates:
        calendar = sorted(set(calendar) | set(missing_strategy_dates))
    missing_benchmark_strategy_dates = sorted(
        set(pnl_by_date) - set(benchmark)
    )
    holding_years = (end_datetime - start_datetime).days / 365.2425
    if holding_years <= 0:
        holding_years = len(calendar) / TRADING_DAYS_PER_YEAR

    equity = initial_equity
    equities = [equity]
    returns: list[float] = []
    market_returns: list[float | None] = []
    for date_value in calendar:
        pnl = pnl_by_date.get(date_value, 0.0)
        period_return = pnl / equity if equity > 0 else 0.0
        returns.append(period_return)
        equity += pnl
        equities.append(equity)
        market_returns.append(benchmark.get(date_value))

    if not math.isclose(
        equity,
        reported_ending_equity,
        abs_tol=0.5,
    ):
        raise ValueError(
            "일별 손익으로 재구성한 종료자산이 결과 파일과 다릅니다."
        )

    statistics = calculate_performance_statistics(
        returns,
        equities,
        market_returns,
        initial_equity,
        equity,
        risk_free_annual_rate=risk_free_annual_pct / 100.0,
        target_annual_rate=target_annual_pct / 100.0,
        holding_years=holding_years,
    )
    return {
        "strategyVersion": strategy_version,
        "strategyLabel": strategy_label,
        "period": {
            "start": start_date,
            "end": end_date,
            "strategyEventDays": len(rows),
            "weekdayObservations": len(calendar),
            "holdingYears": holding_years,
        },
        "assumptions": {
            "tradingDaysPerYear": TRADING_DAYS_PER_YEAR,
            "riskFreeAnnualPct": risk_free_annual_pct,
            "riskFreeSource": (
                "사용자 입력값"
                if risk_free_annual_pct
                else "KOFR 시계열 미제공으로 0% 적용"
            ),
            "targetAnnualPct": target_annual_pct,
            "targetSource": (
                "사용자 입력값"
                if target_annual_pct
                else "목표수익률 미지정으로 0% 적용"
            ),
            "benchmark": "코스피200 선물 일별 수익률 대용치",
            "returnCalendar": (
                "평일 기준, 옵션 만기일 외 전략수익률 0%"
            ),
            "annualReturnBasis": "첫 관측일부터 마지막 관측일까지 CAGR",
        },
        "metrics": {
            "annualizedReturnPct": (
                float(statistics["annualized_return"]) * 100.0
                if statistics["annualized_return"] is not None
                else None
            ),
            "dailyReturnDeviationPct": (
                float(statistics["daily_return_deviation"]) * 100.0
            ),
            "annualizedVolatilityPct": (
                float(statistics["annualized_volatility"]) * 100.0
            ),
            "sharpeRatio": statistics["sharpe_ratio"],
            "lpm1": statistics["lpm1"],
            "lpm2": statistics["lpm2"],
            "omegaRatio": statistics["omega_ratio"],
            "sortinoRatio": statistics["sortino_ratio"],
            "beta": statistics["beta"],
            "maximumDrawdownPct": (
                float(statistics["maximum_drawdown"]) * 100.0
            ),
        },
        "coverage": {
            "observations": statistics["observations"],
            "benchmarkObservations": statistics[
                "benchmark_observations"
            ],
            "missingBenchmarkDates": missing_benchmark_strategy_dates,
        },
        "warnings": [
            (
                "논문은 KOFR와 KOSPI 현물지수를 사용하지만 현재 "
                "저장소에는 해당 시계열이 없어 각각 0%와 코스피200 "
                "선물로 대체했습니다."
            ),
            (
                "전략은 옵션 만기일에만 손익이 발생하므로 주말을 제외한 "
                "평일 중 비만기일 수익률은 0%로 채웠습니다. 별도 KRX "
                "휴장일 달력이 없어 휴장 평일도 0% 관측치에 포함됩니다."
            ),
            (
                "논문 표의 수치와 직접 비교하기보다 동일 전략 간 "
                "상대 비교용으로 사용해야 합니다."
            ),
        ],
    }
