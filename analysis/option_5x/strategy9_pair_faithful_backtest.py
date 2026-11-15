"""S9-v0.1 — 원 요구사항(요구사항.md 1~4번)을 '그대로' 구현한 페어 전략.

S8(하프컷 리버스)은 "한쪽 반토막을 기다렸다가 반대편만 500 전액"으로 **선별 진입**했지만,
S9 는 원 요구사항의 문구를 **문자 그대로** 따른다:

  · 양쪽(콜·풋)을 '미리 모두' 진입한다(요구사항 3번: 양쪽 모두 진입, 각 250만원).
  · 매수 방식(요구사항 2번): 현재가 < P0 → 진입시점부터 분할 매수(물량 10%),
    현재가 >= P0 → 갭보정 120분 이평 상향돌파 시 매수.
  · 한쪽이 P0 절반(0.5*P0) 이하로 떨어지면(요구사항 1·3번) 그 옵션은 손절·매수중단,
    남은 예산을 **반대편(생존 측)에 몰아서** 계속 매수(= 몰빵/재배분).
  · 양쪽 모두 절반 이하로 떨어지면 둘 다 손절('꽝'인 날).
  · 11:00 이후 신규진입 금지. 절반 이하(0.5*P0) 가격에서는 매수 금지(요구사항 1번).
  · 청산: 보유 중 저가 <= 0.5*P0 → 0.5*P0 손절, 그 외는 만기결제(settle). (별도 n배 익절 없음)

※ 사용자/맹박사 검증대로 '양쪽 선매수 + 재배분'은 신호 없이 양쪽을 사므로 엣지가 약하다.
  S9 는 그 원안을 있는 그대로 재현해 결과를 비교·검증하기 위한 전략이다.

진입/분할 체결은 프로젝트 표준(분봉 거래량 10% 참여 분할)으로 근사하며, 반토막 판정·P0·
갭보정 120분 이평 등 공통 계산은 S6-v0.1 페어 엔진(strategy6_v01_pair_half_backtest)을 재사용한다.

출력: output/strategy9_v01_daily.csv / output/strategy9_v01_trades.csv
"""

from __future__ import annotations

import csv
import json

import minimal_baseline_backtest as base
from count_opportunities import PROFILE_PATH, derive_expiry_groups, target_file_map
import strategy6_v01_pair_half_backtest as pair

STRATEGY_ID = "S9-v0.1"
INITIAL_CASH = pair.INITIAL_CASH  # 20,000,000
DAILY_BUDGET = pair.DAILY_BUDGET  # 5,000,000
SIDE_BUDGET = DAILY_BUDGET // 2   # 양쪽 선진입: 각 250만원
MULT = base.MULTIPLIER
COMM = base.COMMISSION_PER_CONTRACT
SLIP = base.SLIPPAGE
PART = base.ENTRY_PARTICIPATION
HALF = pair.HALF
IDX = pair.IDX
CUTOFF_IDX = pair.CUTOFF_IDX
ENTRY_START = IDX.get(pair.OBS_END + 1, 0)  # 09:11

SIBLING_DAILY = base.OUTPUT_DIR / "strategy1_daily.csv"
REFRESH_SUMMARY = base.OUTPUT_DIR / "refresh_all_backtests_summary.json"

DAILY_FIELDS = [
    "strategy", "date", "start_cash", "end_cash", "pnl",
    "trade_day", "signal_found", "entry_filled",
    "code", "call_put", "initial_positions", "campaign_count",
    "entry_quantity", "buy_principal", "gross_sales", "fees",
    "expired_quantity", "status",
]
TRADES_FIELDS = [
    "strategy", "date", "campaign", "minute", "code", "call_put",
    "side", "reason", "stage", "quantity", "price",
    "principal", "fee", "cash_flow", "bar_volume",
]

SIGNAL_STATUSES = {
    "both_entry_no_half",
    "call_half_first_pile_put",
    "put_half_first_pile_call",
    "both_halved",
    "one_side_only",
}


# --------------------------------------------------------------------------- #
# 하루 백테스트 (양쪽 선진입 + 재배분)
# --------------------------------------------------------------------------- #
def _enrich(contracts) -> dict:
    enriched = {}
    for code, c in contracts.items():
        res = pair.p0_of(c)
        if res is None:
            continue
        p0, p0_idx = res
        enriched[code] = (code, c, p0, p0_idx)
    return enriched


def _buy_candidates(enriched, cp):
    items = [
        v for v in enriched.values()
        if v[1].call_put == cp and pair.BUY_LOW <= v[2] <= pair.BUY_HIGH
    ]
    items.sort(key=lambda v: v[2])  # P0 낮은 것부터
    return items


def _signal_rep(enriched, cp):
    items = [
        v for v in enriched.values()
        if v[1].call_put == cp and pair.BUY_LOW <= v[2] <= pair.P0_HIGH
    ]
    return max(items, key=lambda v: v[2]) if items else None


def _make_plan(candidates, ma_map):
    """종목별 진입 트리거 확정(요구사항 2번)."""
    plan = []
    for code, contract, p0, _p0_idx in candidates:
        start_price = (
            contract.bars[ENTRY_START].close
            if ENTRY_START < len(contract.bars)
            else None
        )
        if start_price is None:
            continue
        if start_price < p0:
            trigger_idx = ENTRY_START          # 현재가 < P0 → 진입시점부터 분할매수
            branch = "below_p0"
        else:
            trigger_idx = pair.cross_up_idx(contract, ma_map[code], ENTRY_START)
            if trigger_idx is None:            # 120 상향돌파 없으면 진입 안 함
                continue
            branch = "ma120_cross"
        if base.SESSION_MINUTES[trigger_idx] > pair.ENTRY_CUTOFF:
            continue
        plan.append((trigger_idx, code, contract, p0, branch))
    return plan


def build_day(group, expiry_path, prior_path):
    """하루치 메타(result)와 raw 포지션(contract 포함) 목록을 함께 반환."""
    contracts = base.load_expiry_day(expiry_path, group)
    prior = base.load_expiry_day(prior_path, group) if prior_path else {}
    enriched = _enrich(contracts)

    cands = {cp: _buy_candidates(enriched, cp) for cp in ("CALL", "PUT")}
    present = [cp for cp in ("CALL", "PUT") if cands[cp]]

    result = {
        "date": group["expiry_date"],
        "product_kind": group["product_kind"],
        "series": group["series"],
        "status": "",
    }
    if not present:
        result["status"] = "no_candidate"
        return result, []

    # 반토막(종가) 시점: 방향 대표(ATM)의 0.5*P0 종가 하회 분 idx
    def rep_half(cp):
        rep = _signal_rep(enriched, cp)
        if rep is None:
            return None
        _, c, p0, p0_idx = rep
        return pair.half_index(c, p0, p0_idx, field="close")

    half_idx = {cp: rep_half(cp) for cp in ("CALL", "PUT")}

    # 갭보정 120분 이평 맵(종목별)
    ma_map = {}
    for cp in present:
        for code, c, _p0, _pi in cands[cp]:
            ma_map[code] = pair.ma120_for_day(c, prior.get(code))

    plans = {cp: _make_plan(cands[cp], ma_map) for cp in present}

    # 사이드별 상태: 각 250만원, 11:00 또는 반토막 시 매수중단, 중단 시 잔여예산 반대편으로.
    state = {}
    for cp in present:
        stop_idx = half_idx[cp]  # None 이면 반토막 안 남
        state[cp] = {
            "plan": plans[cp],
            "budget": SIDE_BUDGET,
            "spent": 0.0,
            "active": True,
            "stop_idx": stop_idx,
        }
    # 한쪽만 후보면 그 쪽에 전액(양쪽 선진입이 불가능하므로)
    if len(present) == 1:
        state[present[0]]["budget"] = DAILY_BUDGET

    positions: list[dict] = []

    for t in range(ENTRY_START, CUTOFF_IDX + 1):
        # (1) 이번 분에 반토막난 사이드 매수중단 + 잔여예산 반대편으로 재배분(몰빵)
        for cp in present:
            s = state[cp]
            if s["active"] and s["stop_idx"] is not None and t >= s["stop_idx"]:
                s["active"] = False
                leftover = s["budget"] - s["spent"]
                others = [o for o in present if o != cp and state[o]["active"]]
                if others and leftover > 0:
                    state[others[0]]["budget"] += leftover
        # (2) 활성 사이드 매수(분봉 10% 분할, 0.5*P0 이하 매수금지)
        for cp in present:
            s = state[cp]
            if not s["active"]:
                continue
            remaining = s["budget"] - s["spent"]
            if remaining <= 0:
                continue
            for trigger_idx, code, contract, p0, branch in s["plan"]:
                if remaining <= 0:
                    break
                if t < trigger_idx or t >= len(contract.bars):
                    continue
                bar = contract.bars[t]
                if not bar.traded or bar.open is None:
                    continue
                fill_price = bar.open + SLIP
                if fill_price <= HALF * p0 + 1e-9:   # 손절선 이하 매수 금지
                    continue
                by_cash = int(remaining // (fill_price * MULT))
                by_vol = int(bar.volume * PART)
                qty = min(by_cash, by_vol)
                if qty < 1:
                    continue
                buy_cost = qty * fill_price * MULT + qty * COMM
                s["spent"] += buy_cost
                remaining -= buy_cost
                positions.append({
                    "code": code,
                    "call_put": contract.call_put,
                    "p0": p0,
                    "fill_idx": t,
                    "fill_minute": bar.minute,
                    "fill_price": fill_price,
                    "qty": qty,
                    "buy_cost": buy_cost,
                    "branch": branch,
                    "contract": contract,
                })

    # 상태 라벨
    ch, ph = half_idx.get("CALL"), half_idx.get("PUT")
    if len(present) == 1:
        result["status"] = "one_side_only"
    elif ch is not None and ph is not None:
        result["status"] = "both_halved"
    elif ch is not None:
        result["status"] = "call_half_first_pile_put"
    elif ph is not None:
        result["status"] = "put_half_first_pile_call"
    else:
        result["status"] = "both_entry_no_half"

    return result, positions


# --------------------------------------------------------------------------- #
# 실행 / CSV 내보내기
# --------------------------------------------------------------------------- #
def _catalog_end_date():
    if SIBLING_DAILY.exists():
        with SIBLING_DAILY.open("r", encoding="utf-8-sig", newline="") as f:
            dates = [r["date"] for r in csv.DictReader(f) if r.get("date")]
        if dates:
            return max(dates)
    if REFRESH_SUMMARY.exists():
        data = json.loads(REFRESH_SUMMARY.read_text(encoding="utf-8"))
        return str(data.get("latest_completed_expiry") or "") or None
    return None


def run() -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    groups = derive_expiry_groups(profile)
    cutoff = _catalog_end_date()
    if cutoff:
        groups = [g for g in groups if g["expiry_date"] <= cutoff]

    files = target_file_map()
    dates_sorted = sorted(files)

    def prior_file(group):
        exp = group["expiry_date"]
        for d in reversed([d for d in dates_sorted if d < exp]):
            if base.load_expiry_day(files[d], group):
                return files[d]
        return None

    daily_rows: list[dict] = []
    trade_rows: list[dict] = []

    for i, g in enumerate(groups, 1):
        exp = g["expiry_date"]
        if exp not in files:
            continue
        pf = prior_file(g)
        print(f"[{i}/{len(groups)}] {exp} {g['product_kind']} {g['series']}", flush=True)
        result, positions = build_day(g, files[exp], pf)

        day_pnl = 0.0
        day_qty = 0
        day_buy = 0.0
        day_gross = 0.0
        day_fees = 0.0
        day_expired = 0
        for p in positions:
            det = pair.exit_detail(p, "settle")  # 손절(0.5*P0) 또는 만기결제
            qty = p["qty"]
            fill = p["fill_price"]
            buy_principal = qty * fill * MULT
            buy_fee = qty * COMM
            sell_fee = qty * COMM
            buy_cash = -(buy_principal + buy_fee)
            gross_sales = qty * det["exit_price"] * MULT
            sell_cash = gross_sales - sell_fee

            day_pnl += det["pnl"]
            day_qty += qty
            day_buy += buy_principal
            day_gross += gross_sales
            day_fees += buy_fee + sell_fee
            if det["reason"] == "SETTLE":
                day_expired += qty

            trade_rows.append({
                "strategy": STRATEGY_ID, "date": exp, "campaign": 1,
                "minute": p["fill_minute"], "code": p["code"],
                "call_put": p["call_put"], "side": "BUY",
                "reason": f"PAIR_ENTRY_{p['branch'].upper()}", "stage": 1,
                "quantity": qty, "price": round(fill, 4),
                "principal": round(buy_principal), "fee": round(buy_fee),
                "cash_flow": round(buy_cash), "bar_volume": "",
            })
            trade_rows.append({
                "strategy": STRATEGY_ID, "date": exp, "campaign": 1,
                "minute": det["exit_minute"], "code": p["code"],
                "call_put": p["call_put"], "side": "SELL",
                "reason": det["reason"], "stage": 2,
                "quantity": qty, "price": round(det["exit_price"], 4),
                "principal": round(gross_sales), "fee": round(sell_fee),
                "cash_flow": round(sell_cash), "bar_volume": "",
            })

        has_trade = bool(positions)
        daily_rows.append({
            "date": exp,
            "pnl": round(day_pnl),
            "trade_day": has_trade,
            "signal_found": result["status"] in SIGNAL_STATUSES or has_trade,
            "entry_filled": has_trade,
            "initial_positions": len(positions),
            "campaign_count": len(positions),
            "entry_quantity": day_qty,
            "buy_principal": round(day_buy),
            "gross_sales": round(day_gross),
            "fees": round(day_fees),
            "expired_quantity": day_expired,
            "status": result["status"],
        })

    # 자본곡선 + CSV
    daily_rows.sort(key=lambda r: r["date"])
    cash = INITIAL_CASH
    daily_out: list[dict] = []
    for r in daily_rows:
        start_cash = cash
        cash += r["pnl"]
        daily_out.append({
            "strategy": STRATEGY_ID,
            "date": r["date"],
            "start_cash": start_cash,
            "end_cash": cash,
            "pnl": r["pnl"],
            "trade_day": r["trade_day"],
            "signal_found": r["signal_found"],
            "entry_filled": r["entry_filled"],
            "code": "",
            "call_put": "",
            "initial_positions": r["initial_positions"],
            "campaign_count": r["campaign_count"],
            "entry_quantity": r["entry_quantity"],
            "buy_principal": r["buy_principal"],
            "gross_sales": r["gross_sales"],
            "fees": r["fees"],
            "expired_quantity": r["expired_quantity"],
            "status": r["status"],
        })

    with (base.OUTPUT_DIR / "strategy9_v01_daily.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=DAILY_FIELDS)
        w.writeheader()
        w.writerows(daily_out)
    with (base.OUTPUT_DIR / "strategy9_v01_trades.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=TRADES_FIELDS)
        w.writeheader()
        w.writerows(trade_rows)

    total = cash - INITIAL_CASH
    tdays = sum(1 for r in daily_out if r["trade_day"])
    wins = sum(1 for r in daily_out if r["trade_day"] and r["pnl"] > 0)
    end = daily_out[-1]["date"] if daily_out else None
    print(f"\n=== {STRATEGY_ID} 페어(원 요구사항 그대로) ===  end={end} days={len(daily_out)} trade_days={tdays}")
    print(f"  win {wins}/{tdays} ({100*wins/tdays:.1f}%)" if tdays else "  no trades")
    print(f"  total_pnl {total:,.0f} | return {100*total/INITIAL_CASH:.1f}%")

    # 연도별
    from collections import defaultdict
    by_year = defaultdict(float)
    for r in daily_out:
        by_year[r["date"][:4]] += r["pnl"]
    print("-- 연도별 --")
    for y in sorted(by_year):
        print(f"  {y}: pnl={by_year[y]:>16,.0f}  return={100*by_year[y]/INITIAL_CASH:7.1f}%")


if __name__ == "__main__":
    run()
