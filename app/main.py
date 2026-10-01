from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .backtest_service import BacktestResultService
from .data_service import DataNotFoundError, MarketDataService
from .strategy_analysis_service import StrategyAnalysisService


app = FastAPI(
    title="코스피200 위클리옵션 로컬 차트 API",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

service = MarketDataService()
backtest_service = BacktestResultService()
strategy_analysis_service = StrategyAnalysisService()


@app.get("/api/health")
def health() -> dict[str, object]:
    dates = service.list_option_dates()
    return {
        "status": "ok",
        "rowdataDirectory": str(service.rowdata_dir),
        "optionFileCount": len(dates),
        "optionMinDate": dates[0] if dates else None,
        "optionMaxDate": dates[-1] if dates else None,
        "volatilityFile": str(service.volatility_path),
        "volatilityFileFound": service.volatility_path.exists(),
    }


@app.get("/api/options/dates")
def option_dates() -> dict[str, object]:
    dates = service.list_option_dates()
    return {
        "dates": dates,
        "count": len(dates),
        "minDate": dates[0] if dates else None,
        "maxDate": dates[-1] if dates else None,
    }


@app.get("/api/options/contracts")
def option_contracts(
    date: str = Query(pattern=r"^\d{8}$"),
    session: Literal["day", "night"] = "day",
) -> dict[str, object]:
    try:
        contracts = service.list_contracts(date, session)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {
        "date": date,
        "session": session,
        "count": len(contracts),
        "contracts": contracts,
    }


@app.get("/api/options/bars")
def option_bars(
    date: str = Query(pattern=r"^\d{8}$"),
    code: str = Query(min_length=1, max_length=32),
    session: Literal["day", "night"] = "day",
    minute_grid: bool = False,
) -> dict[str, object]:
    try:
        bars = (
            service.option_bar_grid(date, code)
            if minute_grid and session == "day"
            else service.option_bars(date, code, session)
        )
        contracts = service.list_contracts(date, session)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    contract = next((item for item in contracts if item["code"] == code), None)
    return {
        "date": date,
        "session": session,
        "code": code,
        "contract": contract,
        "bars": bars,
    }


@app.get("/api/volatility")
def volatility(
    start: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
) -> dict[str, object]:
    try:
        bars = service.volatility_bars()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    if start is not None:
        bars = [item for item in bars if str(item["date"]) >= start]
    if end is not None:
        bars = [item for item in bars if str(item["date"]) <= end]

    return {
        "file": str(service.volatility_path),
        "count": len(bars),
        "bars": bars,
    }


@app.get("/api/backtest/options")
def backtest_options() -> dict[str, object]:
    try:
        return backtest_service.options()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/backtest/result")
def backtest_result(
    premium_range: str = Query(
        alias="premiumRange", min_length=4, max_length=32
    ),
    exit_mode: Literal["time_only", "two_x_all", "runner"] = Query(
        alias="exitMode"
    ),
    liquidity_mode: Literal["next_bar", "volume_10pct"] = Query(
        alias="liquidityMode"
    ),
    start_date: str | None = Query(
        default=None, alias="startDate", pattern=r"^\d{8}$"
    ),
    end_date: str | None = Query(
        default=None, alias="endDate", pattern=r"^\d{8}$"
    ),
) -> dict[str, object]:
    try:
        return backtest_service.result(
            premium_range,
            exit_mode,
            liquidity_mode,
            start_date,
            end_date,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-1")
def strategy_one_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_one()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-2")
def strategy_two_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_two()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-2-v02")
def strategy_two_v02_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_two_v02()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-2-v03")
def strategy_two_v03_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_two_v03()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-3")
def strategy_three_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_three()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-3-v02")
def strategy_three_v02_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_three_v02()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-3-v03")
def strategy_three_v03_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_three_v03()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-3-v04")
def strategy_three_v04_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_three_v04()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-4")
def strategy_four_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_four()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-5")
def strategy_five_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_five()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-5-v03")
def strategy_five_v03_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_five_v03()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-5-v04")
def strategy_five_v04_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_five_v04()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-5-v05")
def strategy_five_v05_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_five_v05()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/strategy-5-v06")
def strategy_five_v06_analysis() -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_five_v06()
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/strategy-analysis/trade-records")
def strategy_trade_records(
    strategy: str = Query(min_length=1, max_length=32),
    losses_only: bool = True,
) -> dict[str, object]:
    try:
        return strategy_analysis_service.strategy_trade_records(
            strategy, losses_only
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except DataNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
