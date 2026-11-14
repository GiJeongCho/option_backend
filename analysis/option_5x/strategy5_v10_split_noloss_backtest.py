# -*- coding: utf-8 -*-
"""
S5-v1.0 - 분할 바스켓 · 무손절 · 2배 전량 익절 (초기형, S5 v0.7 고도화)

요구사항2(맹박사)·녹음6 반영. 매수와 매도 로직을 분리한다.

[매수]
  - 대상 밴드 : 0.40 ~ 1.00 (전구간 분할 바스켓)
  - 당일 예산 : 500만원(바스켓 전체), 종목별 정수 계약으로 최대한 고르게 배분
  - 진입 판단 : 09:50 기준. 그 이후 첫 실제 분봉 시가로 매수(시가 0.40~1.00,
                진입 가능 시각 ≤ 14:30).
  - 제외 필터 : 첫 거래 이후 09:50까지 "장중 고점 대비 50% 이상" 하락한 종목은
                진입하지 않는다(이미 붕괴한 종목 회피).
  - 물타기(추가매수) : 이번 초기형에서는 사용하지 않는다(단일 진입).
  - 손절 : 없음(2배 익절 또는 잔량 소멸).

[매도]
  - 평균 매입단가의 2배 전량 익절(단일 진입이므로 진입가의 2배).
    종가가 진입가 2배 이상인 봉의 "다음 실제 분봉 시가"로 전량 매도.
  - 미도달 시 15:15 시간청산, 그래도 남으면 만기 소멸(0원).

표준 산출물: strategy5_v10_split_daily.csv / strategy5_v10_split_trades.csv,
요약 strategy5_v10_split_summary.json. 감사(cash chain·손익등식·수량보존) 포함.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import minimal_baseline_backtest as base
from count_opportunities import derive_expiry_groups

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# 손절 효과 비교용 변형. stop_frac = 종가가 진입가의 이 배율 이하이면 전량 손절.
# None 이면 무손절(초기형 그대로).
# stop_frac: 종가가 진입가의 이 배율 이하이면 전량 손절(None=무손절)
# call_put : "ALL"/"CALL"/"PUT" 진입 방향 필터
# filter   : 진입 제외 필터
#   "high_0950" = 첫거래~09:50 장중 고점 대비 50% 이상 하락 제외
#   "p0_half"   = 09:00~09:10 최초 관측가 P0 대비 절반 이하로 하락한 종목 제외
CONFIGS = (
    {"version": "S5-v1.0", "stem": "strategy5_v10_split",
     "label": "S5-v1.0 분할바스켓·무손절·2배전량(초기형)",
     "stop_frac": None, "call_put": "ALL", "filter": "high_0950"},
    {"version": "S5-v1.1", "stem": "strategy5_v11_stop30",
     "label": "S5-v1.1 분할바스켓·손절 −30%·2배전량",
     "stop_frac": 0.70, "call_put": "ALL", "filter": "high_0950"},
    {"version": "S5-v1.0-stop50", "stem": "strategy5_v10_stop50",
     "label": "S5-v1.0 손절 −50% 비교안",
     "stop_frac": 0.50, "call_put": "ALL", "filter": "high_0950"},
    {"version": "S5-v1.0-stop70", "stem": "strategy5_v10_stop70",
     "label": "S5-v1.0 손절 −70% 비교안",
     "stop_frac": 0.30, "call_put": "ALL", "filter": "high_0950"},
    {"version": "S5-v1.1-CALL", "stem": "strategy5_v11_stop30_call",
     "label": "S5-v1.1 손절−30%·콜옵션만",
     "stop_frac": 0.70, "call_put": "CALL", "filter": "high_0950"},
    {"version": "S5-v1.2", "stem": "strategy5_v12_p0half",
     "label": "S5-v1.2 손절−30%·P0대비 절반이상 유지 필터",
     "stop_frac": 0.70, "call_put": "ALL", "filter": "p0_half"},
    {"version": "S5-v1.2-CALL", "stem": "strategy5_v12_p0half_call",
     "label": "S5-v1.2 손절−30%·P0절반필터·콜옵션만",
     "stop_frac": 0.70, "call_put": "CALL", "filter": "p0_half"},
)

MORNING_START = 900
MORNING_END = 910

INITIAL_CASH = 20_000_000
DAILY_BUDGET = 5_000_000
BAND_LOW = 0.40
BAND_HIGH = 1.00
DECISION_MINUTE = 950
ENTRY_CUTOFF = 1430
DRAWDOWN_FLOOR = 0.50   # 고점 대비 50% 이상 하락 시 제외
EXIT_MULTIPLE = 2.0

MULT = base.MULTIPLIER
COMM = base.COMMISSION_PER_CONTRACT
SLIP = base.SLIPPAGE
PART = base.ENTRY_PARTICIPATION


def passes_drawdown_filter(contract: base.ContractGrid, decision_index: int) -> bool:
    """첫 거래 이후 decision_index(09:50)까지 고점 대비 50% 이상 하락이 없으면 True."""
    running_high = None
    traded_any = False
    for index in range(0, decision_index + 1):
        bar = contract.bars[index]
        if not bar.traded or bar.high is None:
            continue
        traded_any = True
        running_high = bar.high if running_high is None else max(running_high, bar.high)
        if (
            bar.low is not None
            and running_high is not None
            and running_high > 0
            and float(bar.low) <= DRAWDOWN_FLOOR * running_high + 1e-12
        ):
            return False
    return traded_any


def p0_observation(contract: base.ContractGrid):
    """09:00~09:10 최초 거래 분봉의 시가를 P0로 본다. 없으면 None."""
    for index in range(base.MINUTE_INDEX[MORNING_START],
                       base.MINUTE_INDEX[MORNING_END] + 1):
        bar = contract.bars[index]
        if bar.traded and bar.open is not None:
            return float(bar.open)
    return None


def passes_p0_half_filter(contract: base.ContractGrid, decision_index: int) -> bool:
    """09:00~09:10 최초 관측가 P0 대비, decision_index(09:50)까지 저가가
    절반(0.5×P0) 이하로 떨어진 적이 없으면 True."""
    p0 = p0_observation(contract)
    if p0 is None or p0 <= 0:
        return False
    floor = 0.5 * p0
    for index in range(0, decision_index + 1):
        bar = contract.bars[index]
        if bar.traded and bar.low is not None and float(bar.low) <= floor + 1e-12:
            return False
    return True


def match_call_put(contract: base.ContractGrid, want: str) -> bool:
    if want == "ALL":
        return True
    cp = contract.call_put.upper()
    if want == "CALL":
        return "CALL" in cp or "콜" in contract.call_put or cp == "C"
    if want == "PUT":
        return "PUT" in cp or "풋" in contract.call_put or cp == "P"
    return True


def entry_fill(contract: base.ContractGrid, decision_index: int):
    """09:50 이후 첫 실제 분봉. 시가 0.40~1.00·시각 ≤ 14:30이면 (idx, bar, fill_price)."""
    nb = base.next_traded_bar(contract, decision_index, max_wait=None)
    if nb is None:
        return None
    index, bar = nb
    if bar.open is None or bar.minute > ENTRY_CUTOFF:
        return None
    raw = float(bar.open)
    if not (BAND_LOW <= raw <= BAND_HIGH):
        return None
    return index, bar, raw + SLIP


def allocate_basket(candidates: list[dict], cash: int) -> list[dict]:
    """500만 예산을 종목별 정수 계약으로 최대한 고르게 배분(거래량 10%·현금 한도)."""
    for cand in candidates:
        cand["unit"] = cand["fill_price"] * MULT
        cand["max_qty"] = min(
            math.floor(cand["bar"].volume * PART),
            math.floor(DAILY_BUDGET / cand["unit"]) if cand["unit"] > 0 else 0,
        )
        cand["qty"] = 0
    remaining_budget = DAILY_BUDGET
    remaining_cash = cash
    while True:
        feasible = [
            cand
            for cand in candidates
            if cand["qty"] < cand["max_qty"]
            and cand["unit"] <= remaining_budget + 1e-9
            and (cand["unit"] + 2 * COMM) <= remaining_cash + 1e-9
        ]
        if not feasible:
            break
        # 투자금이 가장 적은 종목부터 1계약씩 채워 고르게 배분한다.
        pick = min(
            feasible,
            key=lambda c: (c["qty"] * c["unit"], c["code"]),
        )
        pick["qty"] += 1
        remaining_budget -= pick["unit"]
        remaining_cash -= pick["unit"] + 2 * COMM
    return [cand for cand in candidates if cand["qty"] > 0]


def sell_all(contract: base.ContractGrid, signal_index: int, quantity: int):
    """signal_index 다음 실제 분봉 시가로 전량 매도. (gross, fee, minute, price, idx)."""
    for index in range(signal_index + 1, len(contract.bars)):
        bar = contract.bars[index]
        if not bar.traded or bar.open is None:
            continue
        price = max(0.0, float(bar.open) - SLIP)
        gross = round(quantity * price * MULT)
        fee = quantity * COMM
        return gross, fee, bar.minute, round(price, 4), index
    return None


def simulate_position(version, stop_frac, date_value, contract, campaign,
                      entry_index, entry_bar, fill_price, qty):
    """단일 진입 후 2배 전량 익절 / (옵션)손절 / 시간청산 / 소멸."""
    buy_principal = round(qty * fill_price * MULT)
    buy_fee = qty * COMM
    trades = [{
        "strategy": version, "date": date_value, "campaign": campaign,
        "code": contract.code, "call_put": contract.call_put, "side": "BUY",
        "reason": "SPLIT_BASKET_0950", "minute": entry_bar.minute,
        "quantity": qty, "price": round(fill_price, 4),
        "cash_flow": -(buy_principal + buy_fee),
    }]
    target = fill_price * EXIT_MULTIPLE
    stop_price = fill_price * stop_frac if stop_frac is not None else None
    sold_qty = 0
    gross_total = 0
    fee_total = buy_fee
    index = entry_index
    reached_2x = False
    stopped = False

    def record_sell(signal_index, reason):
        nonlocal gross_total, fee_total, sold_qty
        sale = sell_all(contract, signal_index, qty)
        if sale is not None:
            gross, fee, minute, price, _ = sale
            gross_total += gross
            fee_total += fee
            sold_qty = qty
            trades.append({
                "strategy": version, "date": date_value, "campaign": campaign,
                "code": contract.code, "call_put": contract.call_put,
                "side": "SELL", "reason": reason, "minute": minute,
                "quantity": qty, "price": price, "cash_flow": gross - fee,
            })

    while index < len(contract.bars):
        bar = contract.bars[index]
        if bar.traded and bar.close is not None:
            close = float(bar.close)
            if close >= target:
                reached_2x = True
                record_sell(index, "TWO_X_ALL")
                break
            if stop_price is not None and close <= stop_price:
                stopped = True
                record_sell(index, "STOP_LOSS")
                break
        if bar.minute >= base.FORCE_EXIT_TIME:
            record_sell(index - 1, "TIME")
            break
        index += 1
    expired = qty - sold_qty
    if expired > 0:
        trades.append({
            "strategy": version, "date": date_value, "campaign": campaign,
            "code": contract.code, "call_put": contract.call_put,
            "side": "EXPIRE", "reason": "UNFILLED_ZERO_VALUE",
            "minute": base.SESSION_MINUTES[-1], "quantity": expired,
            "price": 0.0, "cash_flow": 0,
        })
    net = gross_total - fee_total - buy_principal
    return (trades, buy_principal, fee_total, gross_total, expired,
            reached_2x, stopped, net)


def simulate_day(version, stop_frac, call_put, filter_mode, date_value,
                 start_cash, contracts):
    cash = start_cash
    decision_index = base.MINUTE_INDEX[DECISION_MINUTE]
    candidates: list[dict] = []
    for contract in contracts.values():
        if not match_call_put(contract, call_put):
            continue
        if filter_mode == "p0_half":
            if not passes_p0_half_filter(contract, decision_index):
                continue
        else:
            if not passes_drawdown_filter(contract, decision_index):
                continue
        fill = entry_fill(contract, decision_index)
        if fill is None:
            continue
        index, bar, fill_price = fill
        candidates.append({
            "code": contract.code, "contract": contract,
            "entry_index": index, "bar": bar, "fill_price": fill_price,
        })
    candidates.sort(key=lambda c: c["code"])
    chosen = allocate_basket(candidates, cash)

    trades: list[dict] = []
    total_buy = 0
    total_fee = 0
    total_sell = 0
    expired_qty = 0
    reached_2x_count = 0
    stop_count = 0
    positions = 0
    for campaign, cand in enumerate(chosen, start=1):
        (pos_trades, buy_principal, fee_total, gross_total, expired,
         reached, stopped, _) = simulate_position(
            version, stop_frac, date_value, cand["contract"], campaign,
            cand["entry_index"], cand["bar"], cand["fill_price"], cand["qty"],
        )
        trades.extend(pos_trades)
        total_buy += buy_principal
        total_fee += fee_total
        total_sell += gross_total
        expired_qty += expired
        reached_2x_count += int(reached)
        stop_count += int(stopped)
        positions += 1
    # 현금은 거래장부 합으로 재계산(정확성 보장).
    end_cash = start_cash + sum(int(t["cash_flow"]) for t in trades)
    pnl = end_cash - start_cash
    daily = {
        "strategy": version,
        "date": date_value,
        "start_cash": start_cash,
        "end_cash": end_cash,
        "pnl": pnl,
        "trade_day": total_buy > 0,
        "signal_found": total_buy > 0,
        "buy_principal": total_buy,
        "fees": total_fee,
        "expired_quantity": expired_qty,
        "positions": positions,
        "reached_2x_positions": reached_2x_count,
        "stop_positions": stop_count,
    }
    return daily, trades


def internal_audit(daily, trades):
    errors = []
    by_date: dict[str, list[dict]] = {}
    for tr in trades:
        by_date.setdefault(str(tr["date"]), []).append(tr)
    prev_end = INITIAL_CASH
    for row in daily:
        d = str(row["date"])
        if int(row["start_cash"]) != prev_end:
            errors.append(f"{d}: 현금 연속성")
        if int(round(row["end_cash"])) != int(round(row["start_cash"] + row["pnl"])):
            errors.append(f"{d}: 손익 등식")
        if row["end_cash"] < 0:
            errors.append(f"{d}: 음수 현금")
        if int(row["buy_principal"]) > DAILY_BUDGET:
            errors.append(f"{d}: 예산 초과")
        day_trades = by_date.get(d, [])
        cf = sum(int(round(t["cash_flow"])) for t in day_trades)
        if abs(cf - int(round(row["pnl"]))) > 1:
            errors.append(f"{d}: 거래장부 손익({cf} vs {row['pnl']})")
        # 캠페인별 수량 보존
        qty_by: dict[int, int] = {}
        for t in day_trades:
            c = int(t["campaign"])
            if t["side"] == "BUY":
                qty_by[c] = qty_by.get(c, 0) + int(t["quantity"])
            else:
                qty_by[c] = qty_by.get(c, 0) - int(t["quantity"])
        if any(v != 0 for v in qty_by.values()):
            errors.append(f"{d}: 수량 미보존")
        prev_end = row["end_cash"]
    return {"passed": not errors, "error_count": len(errors),
            "error_samples": errors[:30]}


def summarize(daily, label):
    if not daily:
        return {"label": label, "expiry_days": 0}
    traded = [r for r in daily if r["trade_day"]]
    pnls = [r["pnl"] for r in daily]
    equity = [INITIAL_CASH] + [r["end_cash"] for r in daily]
    peak = equity[0]
    mdd_abs = 0
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        mdd_abs = min(mdd_abs, v - peak)
        if peak:
            mdd = min(mdd, (v - peak) / peak * 100)
    profits = sorted((p for p in pnls if p > 0), reverse=True)
    gp = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    return {
        "label": label,
        "period_start": daily[0]["date"],
        "period_end": daily[-1]["date"],
        "expiry_days": len(daily),
        "trade_days": len(traded),
        "win_rate_pct": round(sum(p > 0 for p in pnls) / len(traded) * 100, 2) if traded else 0.0,
        "total_positions": sum(int(r["positions"]) for r in daily),
        "reached_2x_positions": sum(int(r["reached_2x_positions"]) for r in daily),
        "stop_positions": sum(int(r.get("stop_positions", 0)) for r in daily),
        "total_pnl": sum(pnls),
        "total_return_pct": round(sum(pnls) / INITIAL_CASH * 100, 2),
        "profit_factor": round(gp / gl, 3) if gl else None,
        "max_drawdown": int(round(mdd_abs)),
        "max_drawdown_pct": round(mdd, 2),
        "pnl_excluding_best_3_days": sum(pnls) - sum(profits[:3]),
        "pnl_excluding_best_5_days": sum(pnls) - sum(profits[:5]),
        "total_fees": sum(int(r["fees"]) for r in daily),
        "expired_quantity": sum(int(r["expired_quantity"]) for r in daily),
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


def by_year(rows, year):
    return [r for r in rows if str(r["date"]).startswith(year)]


def main():
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = derive_expiry_groups(profile)
    files = base.target_file_map()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state = {cfg["version"]: {"cash": INITIAL_CASH, "daily": [], "trades": []}
             for cfg in CONFIGS}
    cfg_by_version = {cfg["version"]: cfg for cfg in CONFIGS}

    # 만기일 파일은 하루에 한 번만 로드하고 각 변형을 함께 돌린다.
    for group in groups:
        date_value = group["expiry_date"]
        if date_value not in files:
            continue
        contracts = base.load_expiry_day(files[date_value], group)
        for cfg in CONFIGS:
            st = state[cfg["version"]]
            daily, trades = simulate_day(
                cfg["version"], cfg["stop_frac"], cfg["call_put"],
                cfg["filter"], date_value, st["cash"], contracts,
            )
            st["cash"] = daily["end_cash"]
            st["daily"].append(daily)
            st["trades"].extend(trades)

    variants = {}
    overall_passed = True
    for cfg in CONFIGS:
        version = cfg["version"]
        st = state[version]
        audit = internal_audit(st["daily"], st["trades"])
        overall_passed = overall_passed and audit["passed"]
        write_csv_union(OUTPUT_DIR / f"{cfg['stem']}_daily.csv", st["daily"])
        write_csv_union(OUTPUT_DIR / f"{cfg['stem']}_trades.csv", st["trades"])
        variants[version] = {
            "version": version,
            "label": cfg["label"],
            "stop_frac": cfg["stop_frac"],
            "call_put": cfg["call_put"],
            "filter": cfg["filter"],
            "summary": summarize(st["daily"], "full"),
            "by_year": {y: summarize(by_year(st["daily"], y), y)
                        for y in ("2023", "2024", "2025", "2026")},
            "audit": audit,
        }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "initial_cash": INITIAL_CASH,
        "rules": {
            "band": [BAND_LOW, BAND_HIGH],
            "daily_budget": DAILY_BUDGET,
            "decision_minute": DECISION_MINUTE,
            "entry_cutoff": ENTRY_CUTOFF,
            "drawdown_exclusion": f"첫거래~09:50 고점대비 {int(DRAWDOWN_FLOOR*100)}% 이상 하락 제외",
            "averaging_down": "미사용(초기형)",
            "stop_loss": "변형별: 무손절 / 종가 −30% / −50% / −70% 전량 손절",
            "exit": "평균단가 2배 전량(종가 2배 봉 다음 시가), 미도달 시 15:15 시간청산",
        },
        "variants": variants,
        "limitations": [
            "분석구간 자료로 고정한 규칙이므로 미사용 홀드아웃 전진검증 전 실거래 금지.",
            "물타기·차트기반 청산 고도화는 후속 단계.",
            "최종 미체결 잔량은 0원으로 평가한다.",
        ],
    }
    (OUTPUT_DIR / "strategy5_v10_split_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    print(json.dumps(
        {v: {"stop_frac": d["stop_frac"], "call_put": d["call_put"],
             "filter": d["filter"],
             "total_return_pct": d["summary"].get("total_return_pct"),
             "win_rate_pct": d["summary"].get("win_rate_pct"),
             "trade_days": d["summary"].get("trade_days"),
             "total_positions": d["summary"].get("total_positions"),
             "max_drawdown_pct": d["summary"].get("max_drawdown_pct"),
             "by_year_return": {y: d["by_year"][y].get("total_return_pct")
                                for y in ("2023", "2024", "2025", "2026")},
             "audit_passed": d["audit"]["passed"]}
         for v, d in variants.items()},
        ensure_ascii=False, indent=2))
    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
