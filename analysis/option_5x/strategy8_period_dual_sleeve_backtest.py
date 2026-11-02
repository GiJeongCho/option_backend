# -*- coding: utf-8 -*-
"""
S8 - 2025.3~ 구간 듀얼 슬리브(dual-sleeve) 전략 백테스트

0.4(복권형)과 1.0~3.0(안정형)은 성격이 정반대이므로 '다르게' 진입/청산한다.
  - 안정 슬리브(1.0~3.0) : 메인 자금, 절반 2배 익절 + 나머지 10배 러너
  - 복권 슬리브(0.4~0.8) : 소액 고정, 2배 익절 없이 전량 10배 러너(꼬리만 노림)
  - 공통 게이트          : 전일 VKOSPI ≥ 5일선, 풋 우선, 13시 이전 진입(v01 엔진)

대조군:
  - core    : 안정 슬리브만
  - lottery : 복권 슬리브만
  - dual    : 둘 다 (슬리브별 다른 규칙)

청산은 검증된 strategy6_v01.execute_exits 를 재사용(슬리브별 트랜치만 교체).
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
STRATEGY_VERSION = "S8-v0.1"
PERIOD_START = "20250301"

CORE = base.PremiumBand("CORE_100_300", 1.00, 3.00)
LOTTERY = base.PremiumBand("LOTTERY_040_080", 0.40, 0.80)

INITIAL_CASH = v01.INITIAL_CASH            # 2,000만원
CORE_FRACTION = 0.05
CORE_MAX = 5_000_000
LOTTERY_FRACTION = 0.02
LOTTERY_MAX = 1_000_000
MULT = base.MULTIPLIER
COMM = base.COMMISSION_PER_CONTRACT

MODES = ("core", "lottery", "dual")


def write_csv_union(path: Path, rows: list[dict]) -> None:
    """행마다 키가 다를 수 있어 합집합을 필드명으로 사용."""
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def is_put(contract: base.ContractGrid) -> bool:
    return "PUT" in contract.call_put.upper() or "풋" in contract.call_put


def entry_gate(snap) -> bool:
    if snap is None or not math.isfinite(snap.close):
        return False
    if math.isfinite(snap.ma5):
        return snap.close >= snap.ma5
    return True


def select_band(contracts, band, slope120):
    """해당 밴드 첫 적격 사건 중 풋 우선 → 이른 신호 → 코드 순."""
    cands = []
    for contract in contracts.values():
        ev = v01.first_band_event(contract, band, slope120)
        if ev is None:
            continue
        cands.append((0 if is_put(contract) else 1, ev))
    if not cands:
        return None
    cands.sort(key=lambda t: (t[0], t[1].signal_minute, t[1].code))
    return cands[0][1]


def size_quantity(event, fill_bar, budget, cash):
    qty = min(
        math.floor(budget / (event.fill_price * MULT)),
        math.floor(cash / (event.fill_price * MULT + 2 * COMM)),
        math.floor(fill_bar.volume * base.ENTRY_PARTICIPATION),
    )
    return max(0, qty)


def simulate_day(date_value, start_cash, contracts, snap, mode):
    cash = start_cash
    total_buy = 0
    total_sell = 0
    trades = []
    info = {
        "core_code": "", "core_qty": 0, "core_2x": False, "core_10x": False,
        "lottery_code": "", "lottery_qty": 0, "lottery_10x": False,
    }

    sleeves = []
    if mode in ("core", "dual"):
        sleeves.append(("core", CORE, CORE_FRACTION, CORE_MAX))
    if mode in ("lottery", "dual"):
        sleeves.append(("lottery", LOTTERY, LOTTERY_FRACTION, LOTTERY_MAX))

    gate_ok = entry_gate(snap)
    for name, band, frac, cap in sleeves:
        if not gate_ok or not contracts:
            continue
        event = select_band(contracts, band, snap.slope120 if snap else math.nan)
        if event is None:
            continue
        contract = contracts[event.code]
        fill_bar = contract.bars[event.fill_index]
        budget = min(cap, math.floor(start_cash * frac))
        qty = size_quantity(event, fill_bar, budget, cash)
        if qty <= 0:
            continue
        buy_principal = round(qty * event.fill_price * MULT)
        buy_fee = qty * COMM
        cash -= buy_principal + buy_fee
        total_buy += buy_principal + buy_fee
        trades.append({
            "date": date_value, "mode": mode, "sleeve": name, "code": event.code,
            "call_put": contract.call_put, "side": "BUY", "minute": event.fill_minute,
            "qty": qty, "price": round(event.fill_price, 4),
            "cash_flow": -(buy_principal + buy_fee),
        })

        if name == "core":
            positions = v01.tranches(qty)              # 절반 2배 + 나머지 10배
        else:
            positions = [v01.Tranche(stage=1, target_multiple=10,
                                     quantity=qty, remaining=qty)]  # 전량 10배 러너

        gross, sell_fees, sell_trades, reached = v01.execute_exits(
            date_value, contract, event, positions
        )
        cash += gross - sell_fees
        total_sell += gross - sell_fees
        for t in sell_trades:
            t["mode"] = mode
            t["sleeve"] = name
        trades.extend(sell_trades)

        if name == "core":
            info["core_code"] = event.code
            info["core_qty"] = qty
            info["core_2x"] = reached[2]
            info["core_10x"] = reached[10]
        else:
            info["lottery_code"] = event.code
            info["lottery_qty"] = qty
            info["lottery_10x"] = reached[10]

    end_cash = start_cash - total_buy + total_sell
    pnl = end_cash - start_cash
    daily = {
        "strategy": f"{STRATEGY_VERSION}-{mode}",
        "mode": mode,
        "date": date_value,
        "start_cash": start_cash,
        "end_cash": end_cash,
        "pnl": pnl,
        "trade_day": total_buy > 0,
        "vix_prev_close": snap.close if snap else None,
        **info,
    }
    return daily, trades


def summarize(daily):
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
        "trade_days": len(traded),
        "win_rate_pct": round(sum(p > 0 for p in pnls) / len(traded) * 100, 2) if traded else 0.0,
        "core_2x_days": sum(bool(r["core_2x"]) for r in daily),
        "core_10x_days": sum(bool(r["core_10x"]) for r in daily),
        "lottery_10x_days": sum(bool(r["lottery_10x"]) for r in daily),
        "total_pnl": sum(pnls),
        "total_return_pct": round(sum(pnls) / INITIAL_CASH * 100, 2),
        "profit_factor": round(gp / gl, 3) if gl else None,
        "max_drawdown_pct": round(mdd, 2),
        "pnl_excl_best_3": sum(pnls) - sum(profits[:3]),
    }


def main():
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [g for g in base.derive_expiry_groups(profile)
              if g["expiry_date"] >= PERIOD_START]
    files = base.target_file_map()
    vix_dates, snapshots = v02.vix_snapshots()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = []
    for mode in MODES:
        cash = INITIAL_CASH
        daily_rows = []
        trade_rows = []
        for group in groups:
            date_value = group["expiry_date"]
            contracts = (base.load_expiry_day(files[date_value], group)
                         if date_value in files else {})
            snap = v02.previous_snapshot(date_value, vix_dates, snapshots)
            daily, trades = simulate_day(date_value, cash, contracts, snap, mode)
            cash = daily["end_cash"]
            daily_rows.append(daily)
            trade_rows.extend(trades)
        s = summarize(daily_rows)
        s["mode"] = mode
        comparison.append(s)
        write_csv_union(OUTPUT_DIR / f"strategy8_{mode}_daily.csv", daily_rows)
        write_csv_union(OUTPUT_DIR / f"strategy8_{mode}_trades.csv", trade_rows)

    cols = ["mode", "trade_days", "win_rate_pct", "core_2x_days", "core_10x_days",
            "lottery_10x_days", "total_pnl", "total_return_pct", "profit_factor",
            "max_drawdown_pct", "pnl_excl_best_3"]
    comparison = [{k: c.get(k) for k in cols} for c in comparison]

    result = {
        "strategy_version": STRATEGY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period_start": PERIOD_START,
        "expiry_days": len(groups),
        "initial_cash": INITIAL_CASH,
        "rules": {
            "core_band": [CORE.lower, CORE.upper],
            "core_exit": "절반 2배 익절 + 나머지 10배 러너",
            "core_budget": [CORE_FRACTION, CORE_MAX],
            "lottery_band": [LOTTERY.lower, LOTTERY.upper],
            "lottery_exit": "전량 10배 러너(2배 익절 없음)",
            "lottery_budget": [LOTTERY_FRACTION, LOTTERY_MAX],
            "gate": "전일 VKOSPI ≥ 5일선, 풋 우선, 13시 이전",
        },
        "comparison": comparison,
    }
    (OUTPUT_DIR / "strategy8_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    base.write_csv(OUTPUT_DIR / "strategy8_comparison.csv", comparison)

    import pandas as pd  # noqa: WPS433
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"구간 {PERIOD_START}~  만기일 {len(groups)}일  초기자금 {INITIAL_CASH:,}원\n")
    print(pd.DataFrame(comparison).to_string(index=False))
    print("\n저장: output/strategy8_comparison.json / _comparison.csv / strategy8_<mode>_daily.csv")


if __name__ == "__main__":
    main()
