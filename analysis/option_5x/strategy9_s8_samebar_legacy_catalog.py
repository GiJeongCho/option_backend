"""S9-v0.2~v0.4 — 과거 S8 동일 분봉 체결 재현용 비교 카탈로그.

S8은 종가 신호를 확인한 다음 분봉부터 주문하는 현재 규칙을 사용한다.
이 파일은 수정 전 S8의 계산을 독립적으로 재현한다.

과거 계산 방식:
  - 반토막 또는 120선 상향돌파를 해당 분봉 종가로 확인한다.
  - 그런데 같은 분봉의 시가+1틱에 체결한 것으로 처리한다.

종가를 알기 전에 그 분 시가로 주문할 수는 없으므로 이 전략은 실거래 후보가 아니다.
S8과의 성과 차이가 어느 정도인지 측정하기 위한 비교 기준으로만 유지한다.

출력:
  output/strategy9_v02_samebar_x2_daily.csv / _trades.csv
  output/strategy9_v03_samebar_x3_daily.csv / _trades.csv
  output/strategy9_v04_samebar_x4_daily.csv / _trades.csv
"""

from __future__ import annotations

import csv
import json

import minimal_baseline_backtest as base
from count_opportunities import PROFILE_PATH, derive_expiry_groups, target_file_map
import strategy6_v01_pair_half_backtest as pair


INITIAL_CASH = pair.INITIAL_CASH
MULT = base.MULTIPLIER
COMM = base.COMMISSION_PER_CONTRACT
REFRESH_SUMMARY = base.OUTPUT_DIR / "refresh_all_backtests_summary.json"
SIBLING_DAILY = base.OUTPUT_DIR / "strategy1_daily.csv"

# S9-v0.1은 원 요구사항의 양쪽 선진입·재배분 검증용으로 그대로 둔다.
# 아래 세 버전은 과거 S8과 1:1 비교하는 동일 분봉 체결 재현본이다.
VARIANTS = [
    (
        "S9-v0.2",
        "x2",
        2,
        "strategy9_v02_samebar_x2_daily.csv",
        "strategy9_v02_samebar_x2_trades.csv",
    ),
    (
        "S9-v0.3",
        "x3",
        3,
        "strategy9_v03_samebar_x3_daily.csv",
        "strategy9_v03_samebar_x3_trades.csv",
    ),
    (
        "S9-v0.4",
        "x4",
        4,
        "strategy9_v04_samebar_x4_daily.csv",
        "strategy9_v04_samebar_x4_trades.csv",
    ),
]

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
    "both_halved_call_first_buy_put",
    "both_halved_put_first_buy_call",
    "call_halved_buy_put",
    "put_halved_buy_call",
    "both_active",
    "one_side_only",
}


def legacy_buy_side(candidates, budget: int, entry_start: int, ma_map) -> list[dict]:
    """수정 전 S8의 동일 분봉 체결을 그대로 재현한다.

    판단은 `entry_start` 분봉 종가로 하지만, 체결은 같은 `entry_start` 분봉
    시가+1틱부터 시작한다. 이 시간 순서 오류가 S8과 S9 비교의 대상이다.
    """
    positions: list[dict] = []
    remaining = budget
    plan = []

    for code, contract, p0, _p0_idx in candidates:
        start_price = (
            contract.bars[entry_start].close
            if entry_start < len(contract.bars)
            else None
        )
        if start_price is None:
            continue
        if start_price < p0:
            trigger_idx = entry_start
            branch = "below_p0"
        else:
            trigger_idx = pair.cross_up_idx(
                contract, ma_map[code], entry_start
            )
            if trigger_idx is None:
                continue
            branch = "ma120_cross"
        if base.SESSION_MINUTES[trigger_idx] > pair.ENTRY_CUTOFF:
            continue
        plan.append((trigger_idx, code, contract, p0, branch))

    for minute_idx in range(entry_start, pair.CUTOFF_IDX + 1):
        if remaining <= 0:
            break
        for trigger_idx, code, contract, p0, branch in plan:
            if remaining <= 0:
                break
            if minute_idx < trigger_idx or minute_idx >= len(contract.bars):
                continue
            bar = contract.bars[minute_idx]
            if not bar.traded or bar.open is None:
                continue
            fill_price = bar.open + base.SLIPPAGE
            if fill_price <= pair.HALF * p0 + 1e-9:
                continue
            by_cash = int(remaining // (fill_price * MULT))
            by_volume = int(bar.volume * base.ENTRY_PARTICIPATION)
            quantity = min(by_cash, by_volume)
            if quantity < 1:
                continue
            buy_cost = quantity * fill_price * MULT + quantity * COMM
            remaining -= buy_cost
            positions.append(
                {
                    "code": code,
                    "call_put": contract.call_put,
                    "p0": p0,
                    "fill_idx": minute_idx,
                    "fill_minute": bar.minute,
                    "fill_price": fill_price,
                    "qty": quantity,
                    "buy_cost": buy_cost,
                    "branch": branch,
                    "contract": contract,
                }
            )
    return positions


def legacy_build_day(group, expiry_path, prior_path) -> tuple[dict, list[dict]]:
    """수정 전 S8의 방향 신호·진입 시점·체결 순서를 독립적으로 재현한다."""
    contracts = base.load_expiry_day(expiry_path, group)
    prior = base.load_expiry_day(prior_path, group) if prior_path else {}
    enriched = {}
    for code, contract in contracts.items():
        p0_result = pair.p0_of(contract)
        if p0_result is not None:
            p0, p0_idx = p0_result
            enriched[code] = (code, contract, p0, p0_idx)

    def buy_candidates(call_put: str):
        items = [
            value for value in enriched.values()
            if value[1].call_put == call_put
            and pair.BUY_LOW <= value[2] <= pair.BUY_HIGH
        ]
        return sorted(items, key=lambda value: value[2])

    def signal_rep(call_put: str):
        items = [
            value for value in enriched.values()
            if value[1].call_put == call_put
            and pair.BUY_LOW <= value[2] <= pair.P0_HIGH
        ]
        return max(items, key=lambda value: value[2]) if items else None

    calls = buy_candidates("CALL")
    puts = buy_candidates("PUT")
    candidates = {"CALL": calls, "PUT": puts}
    present = [side for side in ("CALL", "PUT") if candidates[side]]
    result = {
        "date": group["expiry_date"],
        "product_kind": group["product_kind"],
        "series": group["series"],
        "status": "",
    }
    if not present:
        result["status"] = "no_candidate"
        return result, []

    def rep_half(call_put: str):
        representative = signal_rep(call_put)
        if representative is None:
            return None
        _, contract, p0, p0_idx = representative
        return pair.half_index(contract, p0, p0_idx, field="close")

    call_half = rep_half("CALL")
    put_half = rep_half("PUT")
    call_halved = call_half is not None
    put_halved = put_half is not None

    if call_halved and put_halved:
        result["status"] = (
            "both_halved_call_first_buy_put"
            if call_half <= put_half
            else "both_halved_put_first_buy_call"
        )
    elif call_halved:
        result["status"] = "call_halved_buy_put"
    elif put_halved:
        result["status"] = "put_halved_buy_call"
    else:
        result["status"] = "both_active" if len(present) == 2 else "one_side_only"

    # 과거 코드: 양쪽 정상은 09:11부터, 반토막 신호는 신호가 확인된 바로 그 분부터 주문.
    entry_after_observation = pair.IDX.get(pair.OBS_END + 1, 0)
    active: list[tuple[list, int, int]] = []
    if call_halved and put_halved:
        if call_half <= put_half and puts:
            active.append((puts, pair.DAILY_BUDGET, call_half))
        elif put_half < call_half and calls:
            active.append((calls, pair.DAILY_BUDGET, put_half))
    elif call_halved and puts:
        active.append((puts, pair.DAILY_BUDGET, call_half))
    elif put_halved and calls:
        active.append((calls, pair.DAILY_BUDGET, put_half))
    elif not call_halved and not put_halved:
        budget = pair.DAILY_BUDGET // len(present)
        for side in present:
            active.append((candidates[side], budget, entry_after_observation))

    # 과거 코드도 11:00 이후에는 진입하지 않았다. 11:00 신호봉은 같은 봉 체결 가능으로 처리했다.
    active = [
        item
        for item in active
        if base.SESSION_MINUTES[item[2]] <= pair.ENTRY_CUTOFF
    ]
    if not active:
        result["status"] += "_late"
        return result, []

    positions: list[dict] = []
    for side_candidates, budget, entry_start in active:
        ma_map = {
            code: pair.ma120_for_day(contract, prior.get(code), legacy_close_anchor=True)
            for code, contract, _p0, _p0_idx in side_candidates
        }
        positions.extend(
            legacy_buy_side(side_candidates, budget, entry_start, ma_map)
        )
    return result, positions


def catalog_end_date() -> str | None:
    if SIBLING_DAILY.exists():
        with SIBLING_DAILY.open("r", encoding="utf-8-sig", newline="") as file:
            dates = [row["date"] for row in csv.DictReader(file) if row.get("date")]
        if dates:
            return max(dates)
    if REFRESH_SUMMARY.exists():
        payload = json.loads(REFRESH_SUMMARY.read_text(encoding="utf-8"))
        return str(payload.get("latest_completed_expiry") or "") or None
    return None


def run() -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    groups = derive_expiry_groups(profile)
    cutoff = catalog_end_date()
    if cutoff:
        groups = [group for group in groups if group["expiry_date"] <= cutoff]
    files = target_file_map()
    dates = sorted(files)

    def prior_file(group):
        for date in reversed([date for date in dates if date < group["expiry_date"]]):
            if base.load_expiry_day(files[date], group):
                return files[date]
        return None

    daily_by_variant: dict[str, list[dict]] = {variant[0]: [] for variant in VARIANTS}
    trades_by_variant: dict[str, list[dict]] = {variant[0]: [] for variant in VARIANTS}

    for ordinal, group in enumerate(groups, 1):
        expiry_date = group["expiry_date"]
        if expiry_date not in files:
            continue
        print(f"[{ordinal}/{len(groups)}] {expiry_date} {group['product_kind']} {group['series']}", flush=True)
        result, positions = legacy_build_day(group, files[expiry_date], prior_file(group))

        for strategy_id, variant, multiple, _daily_name, _trades_name in VARIANTS:
            day_pnl = 0.0
            day_quantity = 0
            day_buy = 0.0
            day_gross = 0.0
            day_fees = 0.0
            expired_quantity = 0
            trade_rows = trades_by_variant[strategy_id]
            for position in positions:
                detail = pair.exit_detail(position, variant)
                quantity = position["qty"]
                fill = position["fill_price"]
                buy_principal = quantity * fill * MULT
                buy_fee = quantity * COMM
                sell_fee = quantity * COMM
                gross_sales = quantity * detail["exit_price"] * MULT
                day_pnl += detail["pnl"]
                day_quantity += quantity
                day_buy += buy_principal
                day_gross += gross_sales
                day_fees += buy_fee + sell_fee
                if detail["reason"] == "SETTLE":
                    expired_quantity += quantity
                exit_reason = (
                    f"TARGET_{multiple}X_ALL"
                    if detail["reason"].startswith("TARGET_")
                    else detail["reason"]
                )
                trade_rows.extend(
                    [
                        {
                            "strategy": strategy_id, "date": expiry_date, "campaign": 1,
                            "minute": position["fill_minute"], "code": position["code"],
                            "call_put": position["call_put"], "side": "BUY",
                            "reason": f"PAIR_ENTRY_{position['branch'].upper()}", "stage": 1,
                            "quantity": quantity, "price": round(fill, 4),
                            "principal": round(buy_principal), "fee": round(buy_fee),
                            "cash_flow": round(-(buy_principal + buy_fee)), "bar_volume": "",
                        },
                        {
                            "strategy": strategy_id, "date": expiry_date, "campaign": 1,
                            "minute": detail["exit_minute"], "code": position["code"],
                            "call_put": position["call_put"], "side": "SELL",
                            "reason": exit_reason, "stage": 2,
                            "quantity": quantity, "price": round(detail["exit_price"], 4),
                            "principal": round(gross_sales), "fee": round(sell_fee),
                            "cash_flow": round(gross_sales - sell_fee), "bar_volume": "",
                        },
                    ]
                )

            daily_by_variant[strategy_id].append(
                {
                    "date": expiry_date,
                    "pnl": round(day_pnl),
                    "trade_day": bool(positions),
                    "signal_found": result["status"] in SIGNAL_STATUSES or bool(positions),
                    "entry_filled": bool(positions),
                    "initial_positions": len(positions),
                    "campaign_count": len(positions),
                    "entry_quantity": day_quantity,
                    "buy_principal": round(day_buy),
                    "gross_sales": round(day_gross),
                    "fees": round(day_fees),
                    "expired_quantity": expired_quantity,
                    "status": result["status"],
                }
            )

    for strategy_id, variant, _multiple, daily_name, trades_name in VARIANTS:
        cash = INITIAL_CASH
        daily_output = []
        for row in sorted(daily_by_variant[strategy_id], key=lambda item: item["date"]):
            start_cash = cash
            cash += row["pnl"]
            daily_output.append(
                {
                    "strategy": strategy_id,
                    "date": row["date"],
                    "start_cash": start_cash,
                    "end_cash": cash,
                    **row,
                }
            )
        with (base.OUTPUT_DIR / daily_name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=DAILY_FIELDS)
            writer.writeheader()
            writer.writerows(daily_output)
        with (base.OUTPUT_DIR / trades_name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=TRADES_FIELDS)
            writer.writeheader()
            writer.writerows(trades_by_variant[strategy_id])
        trade_days = sum(1 for row in daily_output if row["trade_day"])
        wins = sum(1 for row in daily_output if row["trade_day"] and row["pnl"] > 0)
        print(f"\n=== {strategy_id} legacy same-bar {variant} ===")
        print(f"  win {wins}/{trade_days} ({100 * wins / trade_days:.1f}%)" if trade_days else "  no trades")
        print(f"  total_pnl {cash - INITIAL_CASH:,.0f} | return {100 * (cash - INITIAL_CASH) / INITIAL_CASH:.1f}%")


if __name__ == "__main__":
    run()
