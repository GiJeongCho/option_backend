# -*- coding: utf-8 -*-
"""
S10 - 조건부(동적) 청산 + 최종 듀얼 조합 백테스트 (2025.3~)

핵심: 1/3 고정분할이 아니라, '그날의 국면'으로 2배에 끝낼지 10배까지 갈지를 판정.

진입(공통): 풋 우선, 전일 VKOSPI ≥ 5일선, 13시 이전(v01 엔진), 당일 5%·최대 500만.

모드:
  - dual_final        : 안정(1~3)=사다리 2·5·10  +  복권(0.4~0.8)=전량 10배 (두 슬리브)
  - cond_core         : 안정밴드, '그날 판정'으로 전량 2배 or 전량 10배
  - cond_ladder_core  : 안정밴드, 공격일=전량 10배 / 보수일=사다리 2·5·10
  - static_ladder_core: 안정밴드, 항상 사다리 (기준)
  - all2x_core        : 안정밴드, 항상 2배 (기준)
  - all10x_core       : 안정밴드, 항상 10배 (기준)

그날 판정(공격 = 10배 겨냥):
  전일 VKOSPI가 5일선 위로 이격 ≥ 0.3  OR  전일 변동성 급등(일간수익률) ≥ 0.03
  → 둘 중 하나라도 참이면 공격(10배), 아니면 보수(2배)

청산 실행은 strategy6_v01.execute_exits 재사용.
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
STRATEGY_VERSION = "S10-v0.1"
PERIOD_START = "20250301"

CORE = base.PremiumBand("core_100_300", 1.00, 3.00)
LOTTERY = base.PremiumBand("lottery_040_080", 0.40, 0.80)

INITIAL_CASH = v01.INITIAL_CASH
MULT = base.MULTIPLIER
COMM = base.COMMISSION_PER_CONTRACT
T = v01.Tranche

# 그날 판정 임계값
DEV_THRESHOLD = 0.3     # 전일 VKOSPI - 5일선 이격
RET_THRESHOLD = 0.03    # 전일 VKOSPI 일간 변화율

MODES = (
    "dual_final",
    "cond_core",
    "cond_ladder_core",
    "static_ladder_core",
    "all2x_core",
    "all10x_core",
)


def is_put(contract):
    return "PUT" in contract.call_put.upper() or "풋" in contract.call_put


def entry_gate(snap):
    if snap is None or not math.isfinite(snap.close):
        return False
    return (not math.isfinite(snap.ma5)) or snap.close >= snap.ma5


def is_aggressive(snap) -> bool:
    """공격일(10배 겨냥) 판정."""
    if snap is None:
        return False
    dev = (snap.close - snap.ma5) if math.isfinite(snap.ma5) else float("nan")
    dr = snap.daily_return
    dev_hit = math.isfinite(dev) and dev >= DEV_THRESHOLD
    ret_hit = math.isfinite(dr) and dr >= RET_THRESHOLD
    return bool(dev_hit or ret_hit)


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
    if config == "all2x":
        return [T(1, 2, qty, qty)]
    if config == "all10x":
        return [T(1, 10, qty, qty)]
    if config == "ladder":  # 1/3 2배 + 1/3 5배 + 나머지 10배
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


def sleeves_for(mode, snap):
    """반환: [(sleeve_name, band, config, risk_fraction, risk_cap)]"""
    if mode == "dual_final":
        return [
            ("core", CORE, "ladder", 0.05, 5_000_000),
            ("lottery", LOTTERY, "all10x", 0.02, 1_000_000),
        ]
    agg = is_aggressive(snap)
    if mode == "cond_core":
        return [("core", CORE, "all10x" if agg else "all2x", 0.05, 5_000_000)]
    if mode == "cond_ladder_core":
        return [("core", CORE, "all10x" if agg else "ladder", 0.05, 5_000_000)]
    if mode == "static_ladder_core":
        return [("core", CORE, "ladder", 0.05, 5_000_000)]
    if mode == "all2x_core":
        return [("core", CORE, "all2x", 0.05, 5_000_000)]
    if mode == "all10x_core":
        return [("core", CORE, "all10x", 0.05, 5_000_000)]
    raise ValueError(mode)


def simulate_day(date_value, start_cash, contracts, snap, mode):
    cash = start_cash
    total_buy = 0
    total_sell = 0
    reached_10x = False
    decision = ""
    sleeves = sleeves_for(mode, snap)
    if mode in ("cond_core", "cond_ladder_core"):
        decision = "AGG" if is_aggressive(snap) else "CON"

    gate_ok = entry_gate(snap)
    for name, band, config, frac, cap in sleeves:
        if not gate_ok or not contracts:
            continue
        event = select_band(contracts, band, snap.slope120 if snap else math.nan)
        if event is None:
            continue
        contract = contracts[event.code]
        fill_bar = contract.bars[event.fill_index]
        budget = min(cap, math.floor(start_cash * frac))
        qty = min(
            math.floor(budget / (event.fill_price * MULT)),
            math.floor(cash / (event.fill_price * MULT + 2 * COMM)),
            math.floor(fill_bar.volume * base.ENTRY_PARTICIPATION),
        )
        if qty <= 0:
            continue
        buy_principal = round(qty * event.fill_price * MULT)
        buy_fee = qty * COMM
        cash -= buy_principal + buy_fee
        total_buy += buy_principal + buy_fee
        positions = build_tranches(qty, config)
        gross, sell_fees, _sell_trades, reached = v01.execute_exits(
            date_value, contract, event, positions
        )
        cash += gross - sell_fees
        total_sell += gross - sell_fees
        if reached.get(10, False):
            reached_10x = True

    end_cash = start_cash - total_buy + total_sell
    pnl = end_cash - start_cash
    return {
        "strategy": f"{STRATEGY_VERSION}-{mode}",
        "mode": mode,
        "date": date_value,
        "start_cash": start_cash,
        "end_cash": end_cash,
        "pnl": pnl,
        "trade_day": total_buy > 0,
        "decision": decision,
        "reached_10x": reached_10x,
        "vix_prev_close": snap.close if snap else None,
    }


def summarize(daily, mode):
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
    agg_days = sum(r["decision"] == "AGG" for r in daily)
    con_days = sum(r["decision"] == "CON" for r in daily)
    return {
        "mode": mode,
        "trade_days": len(traded),
        "agg_days": agg_days,
        "con_days": con_days,
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

    cash = {m: INITIAL_CASH for m in MODES}
    daily_by_mode = {m: [] for m in MODES}
    for group in groups:
        date_value = group["expiry_date"]
        contracts = (base.load_expiry_day(files[date_value], group)
                     if date_value in files else {})
        snap = v02.previous_snapshot(date_value, vix_dates, snapshots)
        for m in MODES:
            daily = simulate_day(date_value, cash[m], contracts, snap, m)
            cash[m] = daily["end_cash"]
            daily_by_mode[m].append(daily)

    comparison = [summarize(daily_by_mode[m], m) for m in MODES]
    for m in MODES:
        write_csv_union(OUTPUT_DIR / f"strategy10_{m}_daily.csv", daily_by_mode[m])

    result = {
        "strategy_version": STRATEGY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period_start": PERIOD_START,
        "expiry_days": len(groups),
        "initial_cash": INITIAL_CASH,
        "aggressive_rule": f"이격 ≥ {DEV_THRESHOLD} 또는 전일변화율 ≥ {RET_THRESHOLD}",
        "comparison": comparison,
    }
    (OUTPUT_DIR / "strategy10_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    write_csv_union(OUTPUT_DIR / "strategy10_comparison.csv", comparison)

    import pandas as pd  # noqa: WPS433
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(f"구간 {PERIOD_START}~  만기일 {len(groups)}일  초기자금 {INITIAL_CASH:,}원")
    print(f"공격일 판정: 이격≥{DEV_THRESHOLD} 또는 전일변화율≥{RET_THRESHOLD}\n")
    print(pd.DataFrame(comparison).to_string(index=False))
    print("\n저장: output/strategy10_comparison.json / _comparison.csv / strategy10_<mode>_daily.csv")


if __name__ == "__main__":
    main()
