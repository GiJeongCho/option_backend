"""S8 P0 짝을 유지한 실행 가능한 진입·손실관리 후보의 별도 연구.

운영 S8 결과를 덮어쓰지 않는다. 각 연도를 2천만원으로 새로 시작하며,
종가 조건은 다음 봉 시가에만 실행한다. 거래량 참여는 보수적 OHLC 근사다.
호가/체결 순번이 없어 실거래 수익을 확정할 수는 없다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import minimal_baseline_backtest as base
import strategy6_v01_pair_half_backtest as pair
from count_opportunities import PROFILE_PATH, derive_expiry_groups, target_file_map


@dataclass(frozen=True)
class Policy:
    name: str
    entry: str
    stop_fraction: float
    timeout: bool


POLICIES = (
    Policy("direct", "original", .50, False),
    Policy("direct_protected", "original", .25, True),
    Policy("ma120", "ma120", .25, True),
    Policy("breakout", "breakout", .25, True),
    Policy("reclaim_p0", "reclaim_p0", .25, True),
)
LABELS = {
    "direct": "기존 진입·진입가 50% 손절",
    "direct_protected": "기존 진입·25% 손절·시간실패",
    "ma120": "120선 돌파·25% 손절·시간실패",
    "breakout": "추세와 5분 고점 돌파·25% 손절·시간실패",
    "reclaim_p0": "P0 회복과 추세 확인·25% 손절·시간실패",
}
SEED = 20_000_000
DAY_RISK = .01
TRADE_RISK = .005
START = pair.IDX[920]
EXIT_SIGNAL = pair.IDX[1510]
LAST = len(base.SESSION_MINUTES) - 1


def prepare(group, current, prior):
    eligible = []
    for code, contract in current.items():
        observation = pair.p0_of(contract)
        if observation:
            eligible.append((code, contract, *observation))
    eligible.sort(key=lambda item: (item[2], item[0]))
    result = []
    for item in eligible:
        code, contract, p0, p0_idx = item
        opposite = pair.match_opposite_p0(item, eligible)
        if opposite is None:
            continue
        other_code, other, other_p0, other_idx = opposite
        signal_idx = pair.half_index(other, other_p0, other_idx, field="close")
        own_half_idx = pair.half_index(contract, p0, p0_idx, field="low")
        if signal_idx is None or signal_idx >= pair.CUTOFF_IDX:
            continue
        if own_half_idx is not None and own_half_idx <= signal_idx:
            continue
        result.append({
            "code": code, "contract": contract, "p0": p0,
            "signal_idx": signal_idx, "own_half_idx": own_half_idx,
            "ma120": pair.ma120_for_day(contract, prior.get(code)),
            "signal_code": other_code,
        })
    return result


def entry_index(candidate, policy):
    c = candidate["contract"]
    start = candidate["signal_idx"]
    p0 = candidate["p0"]
    ma = candidate["ma120"]
    if policy.entry == "original":
        if c.bars[start].close is None:
            return None
        if c.bars[start].close < p0:
            return start
        return pair.ma_cross_up(c, ma, start)
    for idx in range(max(start, START, 5), pair.CUTOFF_IDX):
        if candidate["own_half_idx"] is not None and idx >= candidate["own_half_idx"]:
            break
        bar, previous = c.bars[idx], c.bars[idx - 1]
        if not bar.traded or bar.close is None or previous.close is None:
            continue
        if ma[idx] is None or ma[idx - 1] is None:
            continue
        if policy.entry == "ma120":
            if previous.close <= ma[idx - 1] and bar.close > ma[idx]:
                return idx
            continue
        if c.ma5[idx] is None or c.ma20[idx] is None or c.ma5[idx - 1] is None:
            continue
        trend = bar.close > ma[idx] and c.ma5[idx] > c.ma20[idx] and c.ma5[idx] > c.ma5[idx - 1]
        if not trend:
            continue
        if policy.entry == "reclaim_p0":
            if previous.close < p0 <= bar.close:
                return idx
        else:
            previous_highs = [b.high for b in c.bars[idx - 5:idx]]
            if all(h is not None for h in previous_highs) and bar.close > max(previous_highs):
                return idx
    return None


def participation(contract, idx):
    """직전 봉까지 알던 거래량과 실행 봉 실제 거래량 중 작은 쪽의 10%만 허용."""
    if idx == 0:
        return 0
    return int(min(contract.bars[idx - 1].volume, contract.bars[idx].volume) * .1)


def simulate_day(prepared, policy, target_multiple, slip_ticks, cash):
    start_cash = cash
    daily_limit = min(5_000_000, max(0, cash) * .25)
    risk_limit = max(0, cash) * DAY_RISK
    risk_spent = 0.0
    buy_spent = 0.0
    completed = []
    position = None
    used_codes = set()
    fills = 0
    contracts_bought = 0
    plans = []
    slip = slip_ticks * .01
    for candidate in prepared:
        idx = entry_index(candidate, policy)
        if idx is not None and idx < pair.CUTOFF_IDX:
            plans.append((idx + 1, candidate))
    for idx in range(pair.IDX[911], LAST + 1):
        closed_this_bar = False
        if position is not None:
            c = position["contract"]
            bar = c.bars[idx]
            if bar.traded and bar.open is not None:
                price = None
                reason = None
                if position["pending"]:
                    price = max(.0, round(bar.open - slip, 8))
                    reason = position["pending"]
                elif bar.high is not None and bar.high >= position["target"] + slip:
                    price = position["target"]
                    reason = "target"
                if price is not None:
                    quantity = min(position["remaining"], participation(c, idx))
                    if quantity:
                        proceeds = quantity * price * base.MULTIPLIER - quantity * base.COMMISSION_PER_CONTRACT
                        cash += proceeds
                        position["net"] += proceeds
                        position["remaining"] -= quantity
                        position["exit_reason"] = reason
                if position["remaining"] == 0:
                    completed.append(position["net"])
                    position = None
                    closed_this_bar = True
                else:
                    position["peak"] = max(position["peak"], bar.high or .0)
                    if bar.close is not None and bar.close <= position["stop"]:
                        position["pending"] = "stop_close_next_open"
                    elif idx >= EXIT_SIGNAL:
                        position["pending"] = "time_close_next_open"
                    elif policy.timeout and idx - position["fill_idx"] >= 15 and position["peak"] < position["fill"] * 1.2:
                        position["pending"] = "failed_to_advance"
        if closed_this_bar or position is not None or fills >= 2 or idx > pair.CUTOFF_IDX:
            continue
        if sum(completed) <= -risk_limit or cash <= 0:
            continue
        for first_idx, candidate in sorted(plans, key=lambda item: (item[0], item[1]["p0"], item[1]["code"])):
            if candidate["code"] in used_codes or not first_idx <= idx <= first_idx + 2:
                continue
            if candidate["own_half_idx"] is not None and idx > candidate["own_half_idx"]:
                continue
            c = candidate["contract"]
            bar = c.bars[idx]
            if not bar.traded or bar.open is None or bar.high is None:
                continue
            fill = round(bar.open + slip, 8)
            # 실제 매수 가격도 0.40~1.20으로 제한해 낮은 가격의 큰 비용 비중과 추격을 피한다.
            if not .4 <= fill <= 1.2 or fill > bar.high:
                continue
            stop = max(candidate["p0"] * .5, fill * (1 - policy.stop_fraction))
            if stop >= fill:
                continue
            loss_per_contract = (fill - stop + slip) * base.MULTIPLIER + 2 * base.COMMISSION_PER_CONTRACT
            risk_budget = min(start_cash * TRADE_RISK, risk_limit - risk_spent)
            cost = fill * base.MULTIPLIER + base.COMMISSION_PER_CONTRACT
            quantity = min(
                int(max(0, risk_budget) // loss_per_contract),
                int(max(0, daily_limit - buy_spent) // cost),
                int(max(0, cash) // cost), participation(c, idx),
            )
            if quantity <= 0:
                continue
            spent = quantity * cost
            cash -= spent
            buy_spent += spent
            risk_spent += quantity * loss_per_contract
            fills += 1
            contracts_bought += quantity
            used_codes.add(candidate["code"])
            position = {
                "contract": c, "remaining": quantity, "fill": fill,
                "fill_idx": idx, "stop": stop, "target": fill * target_multiple,
                "pending": None, "net": -spent, "peak": max(fill, bar.high),
            }
            # 매수한 봉에서도 종가 손절을 확인하고 다음 봉에 실행한다.
            if bar.close is not None and bar.close <= stop:
                position["pending"] = "stop_close_next_open"
            break
    if position is not None:
        # 강제 청산 시도에도 거래량이 없어 남은 수량은 가치 0으로 보수 평가.
        completed.append(position["net"])
    return cash, {
        "pnl": round(cash - start_cash), "trades": len(completed),
        "buy_spent": round(buy_spent), "contracts_bought": contracts_bought,
        "extra_roundtrip_tick_cost": contracts_bought * 2 * .01 * base.MULTIPLIER,
    }


def summarize(days):
    cash = SEED
    peak = SEED
    maximum_drawdown = 0.0
    for row in days:
        cash += row["pnl"]
        peak = max(peak, cash)
        maximum_drawdown = max(maximum_drawdown, (peak - cash) / peak * 100)
    traded = [row for row in days if row["trades"]]
    gross_profit = sum(max(0, row["pnl"]) for row in days)
    gross_loss = -sum(min(0, row["pnl"]) for row in days)
    return {
        "return_pct": round((cash / SEED - 1) * 100, 2),
        "trade_days": len(traded), "trades": sum(row["trades"] for row in days),
        "win_rate_pct": round(sum(row["pnl"] > 0 for row in traded) / len(traded) * 100, 1) if traded else 0,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "maximum_drawdown_pct": round(maximum_drawdown, 2),
        "pnl": round(cash - SEED),
        "same_fills_extra_tick_return_pct": round(
            (cash - SEED - sum(row["extra_roundtrip_tick_cost"] for row in days)) / SEED * 100, 2
        ),
    }


def run():
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [g for g in derive_expiry_groups(profile) if g["expiry_date"] <= "20261029"]
    files = target_file_map()
    configs = [(p, target, slip) for p in POLICIES for target in (2, 3) for slip in (1, 2)]
    accounts = {}
    results = {}
    for policy, target, slip in configs:
        key = f"{policy.name}_x{target}_slip{slip}"
        results[key] = {"policy": policy.name, "label": LABELS[policy.name], "target": target, "slippage_ticks_each_side": slip, "years": {}, "daily": []}
    for ordinal, group in enumerate(groups, 1):
        date = group["expiry_date"]
        year = date[:4]
        current = base.load_expiry_day(files[date], group)
        prior = {}
        for previous_date in reversed([d for d in sorted(files) if d < date]):
            prior = base.load_expiry_day(files[previous_date], group)
            if prior:
                break
        prepared = prepare(group, current, prior)
        for policy, target, slip in configs:
            key = f"{policy.name}_x{target}_slip{slip}"
            balance = accounts.get((key, year), SEED)
            balance, row = simulate_day(prepared, policy, target, slip, balance)
            accounts[(key, year)] = balance
            results[key]["daily"].append({"date": date, **row})
        if ordinal % 25 == 0 or ordinal == len(groups):
            print(f"[{ordinal}/{len(groups)}] {date}", flush=True)
    for result in results.values():
        for year in ("2023", "2024", "2025", "2026"):
            result["years"][year] = summarize([d for d in result["daily"] if d["date"].startswith(year)])
        result["selection_pnl_pre2026"] = sum(d["pnl"] for d in result["daily"] if d["date"] < "20260101")
    selected = max(
        (r for r in results.values() if r["slippage_ticks_each_side"] == 1),
        key=lambda r: r["selection_pnl_pre2026"],
    )
    payload = {
        "period": {"start": groups[0]["expiry_date"], "end": groups[-1]["expiry_date"]},
        "notes": [
            "각 연도 2천만원으로 리셋. 잔고 이내 매수, 하루 최대 두 번 진입, 동시 한 종목 보유.",
            "하루 신규 원가 상한 min(500만원, 당일 시작 잔고 25%). 계획 손실은 일 1%, 진입당 0.5%.",
            "실제 청산은 호가 공백 등으로 계획 손실을 초과할 수 있음. 손절 종가 확인 후 다음 봉 시가 매도.",
            "비용은 계약당 편도 500원, 매수/시장청산에 편도 1틱 또는 2틱. 목표 지정가는 목표가 체결 가정.",
            "별도 동일 거래 비용 민감도는 주문 선택/수량/시각을 고정하고 매수 계약당 왕복 1틱씩 비용만 추가 공제. 이 민감도에는 재투자 규모 변화를 반영하지 않음.",
            "실행 수량은 직전 봉과 실행 봉 거래량 중 작은 값의 10%로 근사. 매수·매도 모두 적용.",
            "호가/체결순서 없음. 미청산 잔량 0원 평가. 실제 정산가를 사용한 실거래 손익이 아님.",
            "2026년은 이미 반복해서 확인한 자료이므로 미사용 검증셋이 아닌 사후 점검 구간.",
            "최고 사후 수익 후보를 확정하지 않음. 2023~2025 합산 순손익으로 한 후보를 선택해 2026을 별도 제시.",
        ],
        "selected_using_pre2026": {k:v for k,v in selected.items() if k != "daily"},
        "results": list(results.values()),
    }
    path = base.OUTPUT_DIR / "s8_live_research.json"
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    for key,result in results.items():
        print(json.dumps({"key":key,"pre2026_pnl":result["selection_pnl_pre2026"],"2026":result["years"]["2026"]}))
    print("saved:",path)


if __name__ == "__main__":
    run()
