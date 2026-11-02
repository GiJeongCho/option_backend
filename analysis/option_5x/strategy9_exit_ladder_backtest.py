# -*- coding: utf-8 -*-
"""
S9 - 같은 진입, '청산 배분'만 바꿔 비교 (2x 매도 vs 10x 홀드 배분)

하나의 매매기법(진입은 동일)에서, 포지션을 2배에서 얼마나 팔고 10배까지 얼마나
끌고 갈지를 다르게 가져가는 효과를 측정한다.
("여유있게 2배 먹고 끝낼 것인가, 10배 도전인가"의 정량 비교)

진입(공통):
  - 밴드: 안정(1.0~3.0) 또는 복권(0.4~0.8)
  - 게이트: 전일 VKOSPI ≥ 5일선, 풋 우선, 13시 이전(v01 엔진)
  - 사이징: 당일 시작자금 5%·최대 500만, 진입봉 거래량 10%

청산 배분(이번에 비교하는 변수):
  - all_2x            : 전량 2배 익절
  - all_10x           : 전량 10배 러너
  - half_2x_half_5x   : 절반 2배 + 나머지 5배
  - half_2x_half_10x  : 절반 2배 + 나머지 10배 (기준)
  - ladder_2x_5x_10x  : 1/3 2배 + 1/3 5배 + 나머지 10배

청산 실행은 검증된 strategy6_v01.execute_exits 재사용(트랜치 구성만 교체).
기간: 만기일 ≥ 2025-03-01.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import minimal_baseline_backtest as base
import strategy6_v01_vix_rv_put_backtest as v01
import strategy6_v02_vix_ma5_rising_backtest as v02

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STRATEGY_VERSION = "S9-v0.1"
PERIOD_START = "20250301"

BANDS = {
    "core_100_300": base.PremiumBand("core_100_300", 1.00, 3.00),
    "lottery_040_080": base.PremiumBand("lottery_040_080", 0.40, 0.80),
}
EXIT_CONFIGS = (
    "all_2x",
    "all_10x",
    "half_2x_half_5x",
    "half_2x_half_10x",
    "ladder_2x_5x_10x",
)

INITIAL_CASH = v01.INITIAL_CASH
RISK_FRACTION = 0.05
RISK_MAX = 5_000_000
MULT = base.MULTIPLIER
COMM = base.COMMISSION_PER_CONTRACT
T = v01.Tranche


def is_put(contract):
    return "PUT" in contract.call_put.upper() or "풋" in contract.call_put


def entry_gate(snap):
    if snap is None or not math.isfinite(snap.close):
        return False
    return (not math.isfinite(snap.ma5)) or snap.close >= snap.ma5


def select_band(contracts, band, slope120):
    cands = []
    for contract in contracts.values():
        ev = v01.first_band_event(contract, band, slope120)
        if ev is not None:
            cands.append((0 if is_put(contract) else 1, ev))
    if not cands:
        return None
    cands.sort(key=lambda t: (t[0], t[1].signal_minute, t[1].code))
    return cands[0][1]


def build_tranches(qty, config):
    if config == "all_2x":
        return [T(stage=1, target_multiple=2, quantity=qty, remaining=qty)]
    if config == "all_10x":
        return [T(stage=1, target_multiple=10, quantity=qty, remaining=qty)]
    if config == "half_2x_half_5x":
        a = math.ceil(qty / 2)
        out = [T(1, 2, a, a)]
        if qty - a > 0:
            out.append(T(2, 5, qty - a, qty - a))
        return out
    if config == "half_2x_half_10x":
        a = math.ceil(qty / 2)
        out = [T(1, 2, a, a)]
        if qty - a > 0:
            out.append(T(2, 10, qty - a, qty - a))
        return out
    if config == "ladder_2x_5x_10x":
        a = math.ceil(qty / 3)
        rem = qty - a
        b = math.ceil(rem / 2)
        c = rem - b
        out = [T(1, 2, a, a)]
        if b > 0:
            out.append(T(2, 5, b, b))
        if c > 0:
            out.append(T(3, 10, c, c))
        return out
    raise ValueError(config)


def simulate_day(date_value, start_cash, contracts, snap, band, config):
    if not entry_gate(snap) or not contracts:
        return _empty(date_value, start_cash, config), []
    event = select_band(contracts, band, snap.slope120 if snap else math.nan)
    if event is None:
        return _empty(date_value, start_cash, config), []

    contract = contracts[event.code]
    fill_bar = contract.bars[event.fill_index]
    budget = min(RISK_MAX, math.floor(start_cash * RISK_FRACTION))
    qty = min(
        math.floor(budget / (event.fill_price * MULT)),
        math.floor(start_cash / (event.fill_price * MULT + 2 * COMM)),
        math.floor(fill_bar.volume * base.ENTRY_PARTICIPATION),
    )
    if qty <= 0:
        return _empty(date_value, start_cash, config, signal=True), []

    buy_principal = round(qty * event.fill_price * MULT)
    buy_fee = qty * COMM
    trades = [{
        "date": date_value, "config": config, "code": event.code,
        "call_put": contract.call_put, "side": "BUY", "minute": event.fill_minute,
        "qty": qty, "price": round(event.fill_price, 4),
        "cash_flow": -(buy_principal + buy_fee),
    }]
    positions = build_tranches(qty, config)
    gross, sell_fees, sell_trades, reached = v01.execute_exits(
        date_value, contract, event, positions
    )
    for t in sell_trades:
        t["config"] = config
    trades.extend(sell_trades)

    pnl = gross - sell_fees - buy_principal - buy_fee
    end_cash = start_cash + pnl
    daily = {
        "strategy": f"{STRATEGY_VERSION}-{config}",
        "config": config,
        "date": date_value,
        "start_cash": start_cash,
        "end_cash": end_cash,
        "pnl": pnl,
        "trade_day": True,
        "code": event.code,
        "call_put": contract.call_put,
        "entry_price": round(event.fill_price, 4),
        "qty": qty,
        "reached_2x": bool(reached.get(2, False)),
        "reached_5x": bool(reached.get(5, False)),
        "reached_10x": bool(reached.get(10, False)),
        "vix_prev_close": snap.close if snap else None,
    }
    return daily, trades


def _empty(date_value, start_cash, config, signal=False):
    return {
        "strategy": f"{STRATEGY_VERSION}-{config}",
        "config": config, "date": date_value,
        "start_cash": start_cash, "end_cash": start_cash, "pnl": 0,
        "trade_day": False, "code": "", "call_put": "", "entry_price": 0.0, "qty": 0,
        "reached_2x": False, "reached_5x": False, "reached_10x": False,
        "vix_prev_close": None,
    }


def summarize(daily, band_name, config):
    traded = [r for r in daily if r["trade_day"]]
    pnls = [r["pnl"] for r in daily]
    equity = [INITIAL_CASH] + [r["end_cash"] for r in daily]
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak:
            mdd = min(mdd, (v - peak) / peak * 100)
    profits = sorted((p for p in pnls if p > 0), reverse=True)
    gp = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    return {
        "band": band_name,
        "exit_config": config,
        "trade_days": len(traded),
        "win_rate_pct": round(sum(p > 0 for p in pnls) / len(traded) * 100, 2) if traded else 0.0,
        "reached_10x_days": sum(bool(r["reached_10x"]) for r in daily),
        "total_pnl": sum(pnls),
        "total_return_pct": round(sum(pnls) / INITIAL_CASH * 100, 2),
        "profit_factor": round(gp / gl, 3) if gl else None,
        "max_drawdown_pct": round(mdd, 2),
        "pnl_excl_best_3": sum(pnls) - sum(profits[:3]),
    }


def write_csv_union(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        w = csv.DictWriter(target, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def main():
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [g for g in base.derive_expiry_groups(profile)
              if g["expiry_date"] >= PERIOD_START]
    files = base.target_file_map()
    vix_dates, snapshots = v02.vix_snapshots()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combos = [(bn, band, cfg) for bn, band in BANDS.items() for cfg in EXIT_CONFIGS]
    cash = {key: INITIAL_CASH for key in combos}
    daily_by_combo = {key: [] for key in combos}

    # 하루 파일은 한 번만 로드해 모든 조합에 재사용
    for group in groups:
        date_value = group["expiry_date"]
        contracts = (base.load_expiry_day(files[date_value], group)
                     if date_value in files else {})
        snap = v02.previous_snapshot(date_value, vix_dates, snapshots)
        for key in combos:
            _bn, band, config = key
            daily, _trades = simulate_day(
                date_value, cash[key], contracts, snap, band, config
            )
            cash[key] = daily["end_cash"]
            daily_by_combo[key].append(daily)

    comparison = []
    for key in combos:
        band_name, _band, config = key
        daily_rows = daily_by_combo[key]
        comparison.append(summarize(daily_rows, band_name, config))
        write_csv_union(
            OUTPUT_DIR / f"strategy9_{band_name}_{config}_daily.csv", daily_rows
        )

    result = {
        "strategy_version": STRATEGY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period_start": PERIOD_START,
        "expiry_days": len(groups),
        "initial_cash": INITIAL_CASH,
        "note": "같은 진입, 청산 배분만 변경 비교",
        "comparison": comparison,
    }
    (OUTPUT_DIR / "strategy9_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    write_csv_union(OUTPUT_DIR / "strategy9_comparison.csv", comparison)

    import pandas as pd  # noqa: WPS433
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"구간 {PERIOD_START}~  만기일 {len(groups)}일  초기자금 {INITIAL_CASH:,}원\n")
    df = pd.DataFrame(comparison)
    for band_name in BANDS:
        print(f"=== 밴드: {band_name} ===")
        print(df[df["band"] == band_name].to_string(index=False))
        print()
    print("저장: output/strategy9_comparison.json / _comparison.csv / strategy9_<band>_<config>_daily.csv")


if __name__ == "__main__":
    main()
