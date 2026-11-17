"""S8 하프컷 리버스와 페어 비교 화면의 전체 기간 공통 엔진.

근거: docs/plan.md 27절 (전략 명세 + 요구사항.md), 사용자 회신 반영.

핵심 로직 (전략 명세 + 요구사항.md 정합)
------------------------------------
- P0: 09:00~09:10 최초 적격 1분봉을 시간순으로 시가→종가→고가→저가 순서로 보아
  0.40~1.20에 처음 걸리는 값(종목마다).
- 같은 만기·주차의 반대 방향에서 동일 P0 우선, 없으면 P0 차이가 최소인 종목을 고정.
- 종목별 반대 짝의 종가가 09:10 이후 그 짝의 0.5*P0 이하이면 매수 신호.
- 자신이 먼저/동시에 반토막 난 종목은 제외. 신호가 없으면 매수하지 않음.
- 하루 예산 500만원을 모든 짝이 공유하며 실제 신호 정보를 거래 파일에 저장.
- 매수 밴드: P0 0.40~1.20 안의 종목을 '전부' 대상(P0 낮은 것부터).
- 진입 판단은 판단봉 종가로 수행하고, 실제 주문은 다음 분봉부터 시도:
  종가 < P0 → 다음 분봉부터 지속매수(below_p0) ;
  종가 >= P0 → 갭보정 120분 이평 상향돌파를 종가로 확인한 다음 분봉부터 매수(ma120_cross).
- 체결: 신호 확인 다음 분봉부터 매 분봉 min(남은예산, 분봉거래량*10%) 만큼만 체결 → 물량 걸리면
  여러 분봉에 걸쳐 '천천히' 매집(11:00까지). 체결가 = 분봉 시가 + 1틱(SLIPPAGE).
- 손절선 아래 매수 금지(전제조건): 체결가가 이미 0.5*P0 이하면 '꽝'이라 그 분봉 매수 제외
  (요구사항.md "절반 이하 하락 시 매수X"). 손절선 아래 진입→즉시 손절 '가짜 이득' 차단.
- 손절: 보유 중 저가 <= 0.5*P0 → 0.5*P0에 전량 손절(페어 전략 고유, 체결 건별).
- 11:00 이후 신규진입 제외.
- 익절/정산 변형(비교): time(15:15) / 2~10배 / settle(만기결제=마지막 종가).
  목표가 체결은 고가 >= 목표가+1틱일 때, 아니면 settle.
"""

from __future__ import annotations

import json
from pathlib import Path

import minimal_baseline_backtest as base
from count_opportunities import (
    PROFILE_PATH,
    derive_expiry_groups,
    target_file_map,
)

YEAR = None  # None = 전체 기간, "2026" 등 지정 시 해당 연도만
OBS_START, OBS_END = 900, 910
P0_LOW, P0_HIGH = 0.40, 1.20          # P0 적격 밴드
BUY_LOW, BUY_HIGH = 0.40, 1.20         # 매수 밴드(P0 적격밴드 전체, 0.40~1.20 다 매수)
HALF = 0.5
ENTRY_CUTOFF = 1100                     # 11:00 이후 신규진입 금지
FORCE_EXIT = 1515
MA_PERIOD = 120
DAILY_BUDGET = 5_000_000
TICK = 0.01
MAX_WAIT = 3                            # 트리거 후 체결 대기(분)
TARGETS = (2, 3, 4, 5, 6, 7, 8, 9, 10)
VARIANTS = ("time", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "settle")
INITIAL_CASH = 20_000_000  # 수익률 산정 기준 시드

MULT = base.MULTIPLIER
COMM = base.COMMISSION_PER_CONTRACT
SLIP = base.SLIPPAGE
PART = base.ENTRY_PARTICIPATION
IDX = base.MINUTE_INDEX
BARS_LEN = len(base.SESSION_MINUTES)
CUTOFF_IDX = IDX[ENTRY_CUTOFF]
FORCE_IDX = IDX[FORCE_EXIT]

SIGNAL_FIELDS = (
    "signal_code", "signal_call_put", "signal_p0", "signal_minute", "signal_close",
    "own_p0", "own_p0_minute", "decision_close", "pair_rule",
    "pair_series", "pair_product_kind",
    "ma120_cross_minute", "ma120_cross_close", "ma120_cross_value",
    "order_ready_minute", "ma120_gap_anchor",
)


# --------------------------------------------------------------------------- #
# 기본 계산 헬퍼
# --------------------------------------------------------------------------- #
def p0_of(contract) -> tuple[float, int] | None:
    for idx, bar in enumerate(contract.bars):
        if bar.minute < OBS_START:
            continue
        if bar.minute > OBS_END:
            break
        if not bar.traded:
            continue
        for price in (bar.open, bar.close, bar.high, bar.low):
            if price is not None and P0_LOW <= price <= P0_HIGH:
                return price, idx
    return None


OBS_END_IDX = IDX[OBS_END]  # 09:10


def half_index(contract, p0: float, start_idx: int, field: str = "low") -> int | None:
    """P0 절반 이하 하락 시점. 관측창(09:10) 이후만 인정(개장 노이즈 배제).

    field='close' → 방향 신호(지속적 가격선 하락), field='low' → 손절 터치.
    """
    stop = HALF * p0
    begin = max(start_idx, OBS_END_IDX + 1)
    for idx in range(begin, len(contract.bars)):
        bar = contract.bars[idx]
        if not bar.traded:
            continue
        value = getattr(bar, field)
        if value is not None and value <= stop + 1e-9:
            return idx
    return None


def last_close(contract) -> float | None:
    for bar in reversed(contract.bars):
        if bar.traded and bar.close is not None:
            return bar.close
    return None


def _day_session_closes(contract) -> list[float | None]:
    """09:00 이후 체결 종가만 사용해 빈 분을 직전 주간 종가로 채운다."""
    values = []
    previous = None
    for bar in contract.bars:
        if bar.minute < OBS_START:
            continue
        if bar.traded and bar.close is not None:
            previous = bar.close
        values.append(previous)
    return values


def ma120_for_day(current, prior, *, legacy_close_anchor=False) -> list[float | None]:
    """갭보정 120분 이평(만기일 세션 분봉에 정렬). prior=전일 grid(또는 None).

    매매로직은 09:00부터만 사용하므로 08:45~08:59(장전 단일가)·야간은 MA 입력에서
    제외한다(야간은 애초에 day 세션만 로드되어 들어오지 않음). 반환 배열은
    current.bars 인덱스에 그대로 정렬되며, 09:00 이전 분은 항상 None 이다.
    전일 종가를 당일 첫 실제 주간 시가에 비율 보정한다. 주간 첫 체결 이전 빈 분에
    08:45~08:59 가격을 이월하지 않는다. legacy_close_anchor는 과거 S9 재현 전용이다.
    """
    cur_bars = current.bars
    # 09:00 이후 분만 MA 입력으로 사용. day_pos 로 원래 인덱스를 보존해 정렬 유지.
    day_pos = [i for i, b in enumerate(cur_bars) if b.minute >= OBS_START]
    day_closes = ([cur_bars[i].close for i in day_pos] if legacy_close_anchor
                  else _day_session_closes(current))
    cur_first = (next((c for c in day_closes if c is not None), None)
                 if legacy_close_anchor else next((b.open for b in cur_bars
                    if b.minute >= OBS_START and b.traded and b.open is not None), None))
    prior_adj: list[float | None] = []
    if prior is not None and cur_first:
        pc = ([b.close for b in prior.bars if b.minute >= OBS_START] if legacy_close_anchor
              else _day_session_closes(prior))
        prior_last = next((c for c in reversed(pc) if c is not None), None)
        if prior_last and prior_last > 0:
            factor = cur_first / prior_last
            prior_adj = [c * factor if c is not None else None for c in pc]
    series = prior_adj + day_closes
    offset = len(prior_adj)
    out: list[float | None] = [None] * len(cur_bars)
    for j in range(len(day_closes)):
        end = offset + j
        if end - MA_PERIOD + 1 < 0:
            continue
        window = series[end - MA_PERIOD + 1 : end + 1]
        if any(v is None for v in window):
            continue
        out[day_pos[j]] = sum(window) / MA_PERIOD
    return out


def ma_cross_up(contract, ma: list[float | None], start_idx: int) -> int | None:
    for idx in range(max(start_idx, 1), CUTOFF_IDX + 1):
        bar = contract.bars[idx]
        prev = contract.bars[idx - 1]
        if (
            bar.traded
            and bar.close is not None
            and prev.close is not None
            and ma[idx] is not None
            and ma[idx - 1] is not None
            and prev.close <= ma[idx - 1]
            and bar.close > ma[idx]
        ):
            return idx
    return None


def fill_bar(contract, idx: int):
    """idx(포함)부터 3분내 첫 실제봉."""
    bar = contract.bars[idx]
    if bar.traded and bar.open is not None:
        return idx, bar
    return base.next_traded_bar(contract, idx, MAX_WAIT)


# --------------------------------------------------------------------------- #
# side 매수 사다리 (시점별 분할 체결, 물량 제한이면 천천히)
# --------------------------------------------------------------------------- #
def cross_up_idx(contract, ma: list[float | None], start_idx: int) -> int | None:
    """start_idx 이후 가격이 갭보정 120분 이평을 상향돌파한 최초 분 idx(없으면 None)."""
    return ma_cross_up(contract, ma, start_idx)


def buy_side(
    candidates, budget: int, entry_start: int, ma_map,
    *, entry_starts: dict[str, int] | None = None,
    signal_details: dict[str, dict] | None = None,
) -> list[dict]:
    """candidates: [(code, contract, p0, p0_idx)] P0 오름차순. P0 낮은 것부터 매수.

    entry_start는 신호/판단 봉의 인덱스다. 그 봉 종가로 분기를 정하고,
    실제 주문은 반드시 다음 봉부터 시도한다(종가 신호를 같은 봉 시가에 소급 체결하지 않음).
      · entry_start 종가 < P0 → 다음 봉부터 '지속매수'(below_p0).
      · entry_start 종가 >= P0 → 갭보정 120분 이평 상향돌파를 종가로 확인한
        다음 봉부터 매수(ma120_cross).
    트리거 이후 매 분봉 체결수량 = min(남은예산, 분봉거래량*10%) →
      물량이 걸리면 자동으로 여러 분봉에 걸쳐 '천천히' 분할 체결(11:00까지).
    체결가 = 분봉 시가 + 1틱. 손절선(0.5*P0) 이하 분봉은 매수 금지.
    """
    positions: list[dict] = []
    remaining = budget

    # entry_start는 판단 봉, trigger_idx는 실제 첫 주문 가능 봉이다.
    first_order_idx = entry_start + 1
    if first_order_idx > CUTOFF_IDX:
        return positions

    # 종목별 트리거 시점·분기 확정(진입 가능한 후보만)
    plan = []  # (trigger_idx, code, contract, p0, branch)
    plan_details = {}
    for code, contract, p0, p0_idx in candidates:
        decision_idx = entry_starts[code] if entry_starts is not None else entry_start
        start_price = (
            contract.bars[decision_idx].close
            if decision_idx < len(contract.bars)
            else None
        )
        if start_price is None:
            continue
        if start_price < p0:
            cross_idx = None
            trigger_idx = decision_idx + 1
            branch = "below_p0"
        else:
            cross_idx = cross_up_idx(contract, ma_map[code], decision_idx)
            if cross_idx is None:
                continue
            # 상향돌파는 해당 분 종가에서 확정되므로 다음 분부터 주문 가능.
            trigger_idx = cross_idx + 1
            if trigger_idx > CUTOFF_IDX:
                continue
            branch = "ma120_cross"
        if base.SESSION_MINUTES[trigger_idx] > ENTRY_CUTOFF:
            continue
        plan_details[code] = {
            **(signal_details.get(code, {}) if signal_details else {}),
            "ma120_cross_minute": contract.bars[cross_idx].minute if cross_idx is not None else "",
            "ma120_cross_close": contract.bars[cross_idx].close if cross_idx is not None else "",
            "ma120_cross_value": ma_map[code][cross_idx] if cross_idx is not None else "",
            "order_ready_minute": base.SESSION_MINUTES[trigger_idx],
            "ma120_gap_anchor": "first_day_open",
        }
        plan.append((trigger_idx, code, contract, p0, branch))
    if not plan:
        return positions

    # 분봉을 바깥 루프로 — 트리거가 열린 종목들에 P0 낮은 것부터 조금씩(물량 10%) 체결
    for t in range(first_order_idx, CUTOFF_IDX + 1):
        if remaining <= 0:
            break
        for trigger_idx, code, contract, p0, branch in plan:
            if remaining <= 0:
                break
            if t < trigger_idx or t >= len(contract.bars):
                continue
            bar = contract.bars[t]
            if not bar.traded or bar.open is None:
                continue
            detail = plan_details[code]
            # 매수 종목의 반토막 종가가 이미 확인된 뒤에는 추가 매수하지 않는다.
            blocked_idx = detail.get("own_half_idx")
            if blocked_idx is not None and t > blocked_idx:
                continue
            fill_price = bar.open + SLIP
            # 손절선 아래 매수 금지(전제조건): 이미 0.5*P0 이하면 '꽝'이라 제외
            if fill_price <= HALF * p0 + 1e-9:
                continue
            by_cash = int(remaining // (fill_price * MULT))
            by_vol = int(bar.volume * PART)
            qty = min(by_cash, by_vol)
            if qty < 1:
                continue
            buy_cost = qty * fill_price * MULT + qty * COMM
            remaining -= buy_cost
            positions.append(
                {
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
                    **detail,
                }
            )
    return positions


# --------------------------------------------------------------------------- #
# 포지션 청산 (변형별)
# --------------------------------------------------------------------------- #
def last_traded_idx(contract) -> int | None:
    for idx in range(len(contract.bars) - 1, -1, -1):
        bar = contract.bars[idx]
        if bar.traded and bar.close is not None:
            return idx
    return None


def exit_detail(pos: dict, variant: str) -> dict:
    """청산 상세(시점·가격·사유·손익). exit_pnl 은 이 함수의 pnl 만 사용."""
    contract = pos["contract"]
    p0 = pos["p0"]
    fill_idx = pos["fill_idx"]
    fill = pos["fill_price"]
    qty = pos["qty"]
    stop_idx = half_index(contract, p0, fill_idx + 1)

    target_idx = None
    target_price = None
    if variant.startswith("x"):
        mult = int(variant[1:])
        target_price = mult * fill
        for idx in range(fill_idx + 1, len(contract.bars)):
            bar = contract.bars[idx]
            if bar.traded and bar.high is not None and bar.high + 1e-9 >= target_price + TICK:
                target_idx = idx
                break
    elif variant == "time":
        target_idx = FORCE_IDX  # 15:15 시간청산

    # 손절 vs (목표/시간) 중 먼저 오는 것
    if target_idx is not None and (stop_idx is None or target_idx <= stop_idx):
        if variant == "time":
            bar = contract.bars[target_idx]
            if bar.traded and bar.close is not None:
                exit_price = bar.close
                exit_idx = target_idx
            else:
                exit_price = last_close(contract)
                exit_idx = last_traded_idx(contract)
            reason = "TIME_EXIT"
        else:
            exit_price = target_price
            exit_idx = target_idx
            reason = f"TARGET_{int(variant[1:])}X"
    elif stop_idx is not None:
        exit_price = HALF * p0
        exit_idx = stop_idx
        reason = "STOP_HALF"
    else:
        exit_price = last_close(contract)  # settle(만기결제)
        exit_idx = last_traded_idx(contract)
        reason = "SETTLE"
    if exit_price is None:
        exit_price = 0.0
    if exit_idx is None:
        exit_idx = fill_idx
    exit_minute = contract.bars[exit_idx].minute
    sell = qty * exit_price * MULT - qty * COMM
    return {
        "exit_idx": exit_idx,
        "exit_minute": exit_minute,
        "exit_price": exit_price,
        "reason": reason,
        "sell": sell,
        "pnl": sell - pos["buy_cost"],
    }


def exit_pnl(pos: dict, variant: str) -> float:
    return exit_detail(pos, variant)["pnl"]


# --------------------------------------------------------------------------- #
# 하루 백테스트
# --------------------------------------------------------------------------- #


def match_opposite_p0(candidate, enriched):
    """같은 만기 그룹에서 반대 방향의 P0가 가장 가까운 종목을 고른다.

    정확히 같은 값 우선. 거리 동률이면 낮은 P0, 종목코드 순으로 고정한다.
    장중 고가·손익·반토막 발생 여부는 짝 선정에 사용하지 않는다.
    """
    opposite = [item for item in enriched if item[1].call_put != candidate[1].call_put]
    return min(
        opposite,
        key=lambda item: (round(abs(item[2] - candidate[2]), 8), item[2], item[0]),
    ) if opposite else None


def build_day(group, expiry_path, prior_path) -> tuple[dict, list[dict]]:
    """종목별 동일/근접 P0 반대 옵션을 고정하고 그 짝의 신호만 사용한다."""
    contracts = base.load_expiry_day(expiry_path, group)
    prior = base.load_expiry_day(prior_path, group) if prior_path else {}
    enriched = []
    for code, contract in contracts.items():
        observed = p0_of(contract)
        if observed is not None:
            p0, p0_idx = observed
            enriched.append((code, contract, p0, p0_idx))
    enriched.sort(key=lambda item: (item[2], item[0]))
    half_indices = {
        code: half_index(contract, p0, p0_idx, field="close")
        for code, contract, p0, p0_idx in enriched
    }
    result = {
        "date": group["expiry_date"],
        "product_kind": group["product_kind"], "series": group["series"],
        "call_halved": any(half_indices[item[0]] is not None for item in enriched if item[1].call_put == "CALL"),
        "put_halved": any(half_indices[item[0]] is not None for item in enriched if item[1].call_put == "PUT"),
        "status": "no_candidate" if not enriched else "no_pair_signal",
        "pairing_rule": "nearest_p0_same_expiry",
        "signal_found": False, "positions": [],
        "pnl": {variant: 0.0 for variant in VARIANTS}, "invested": 0.0,
    }
    candidates = []
    entry_starts = {}
    details = {}
    for candidate in enriched:
        code, contract, p0, p0_idx = candidate
        opposite = match_opposite_p0(candidate, enriched)
        if opposite is None:
            continue
        signal_code, signal_contract, signal_p0, _signal_p0_idx = opposite
        signal_idx = half_indices[signal_code]
        own_half_idx = half_indices[code]
        # 매수 종목이 먼저/동시에 반토막 난 짝은 제외한다. 방향은 종목별 짝 안에서 결정한다.
        if signal_idx is None or (own_half_idx is not None and own_half_idx <= signal_idx):
            continue
        if signal_idx >= CUTOFF_IDX:
            continue
        result["signal_found"] = True
        candidates.append(candidate)
        entry_starts[code] = signal_idx
        details[code] = {
            "signal_code": signal_code,
            "signal_call_put": signal_contract.call_put,
            "signal_p0": signal_p0,
            "signal_minute": signal_contract.bars[signal_idx].minute,
            "signal_close": signal_contract.bars[signal_idx].close,
            "own_p0": p0,
            "own_p0_minute": contract.bars[p0_idx].minute,
            "decision_close": contract.bars[signal_idx].close,
            "pair_rule": "exact_p0" if abs(p0 - signal_p0) < 1e-9 else "nearest_p0",
            "pair_series": group["series"],
            "pair_product_kind": group["product_kind"],
            "own_half_idx": own_half_idx,
        }
    if not candidates:
        return result, []
    ma_map = {
        code: ma120_for_day(contract, prior.get(code))
        for code, contract, _p0, _p0_idx in candidates
    }
    positions = buy_side(
        candidates, DAILY_BUDGET, min(entry_starts.values()), ma_map,
        entry_starts=entry_starts, signal_details=details,
    )
    sides = {position["call_put"] for position in positions}
    result["status"] = (
        "matched_buy_both" if len(sides) == 2 else
        "matched_buy_call" if "CALL" in sides else
        "matched_buy_put" if "PUT" in sides else "pair_signal_unfilled"
    )
    return result, positions


def run_day(group, expiry_path, prior_path) -> dict:
    result, positions = build_day(group, expiry_path, prior_path)
    return day_report(result, positions)


def day_report(result: dict, positions: list[dict]) -> dict:
    result["invested"] = sum(p["buy_cost"] for p in positions)
    for v in VARIANTS:
        result["pnl"][v] = sum(exit_pnl(p, v) for p in positions)
    result["positions"] = [
        {
            "code": p["code"],
            "call_put": p["call_put"],
            "p0": round(p["p0"], 3),
            "fill_minute": p["fill_minute"],
            "fill_price": round(p["fill_price"], 3),
            "qty": p["qty"],
            "buy_cost": round(p["buy_cost"]),
            "branch": p["branch"],
            "pnl_settle": round(exit_pnl(p, "settle")),
            **{field: p.get(field, "") for field in SIGNAL_FIELDS},
        }
        for p in positions
    ]
    return result


# --------------------------------------------------------------------------- #
def _summarize(daily: list[dict]) -> dict:
    summary = {v: {"pnl": 0.0, "win": 0, "loss": 0, "trade_days": 0} for v in VARIANTS}
    for d in daily:
        if not d["positions"]:
            continue
        for v in VARIANTS:
            pnl = d["pnl"][v]
            summary[v]["pnl"] += pnl
            summary[v]["trade_days"] += 1
            if pnl > 0:
                summary[v]["win"] += 1
            elif pnl < 0:
                summary[v]["loss"] += 1
    return {
        v: {
            "pnl": round(summary[v]["pnl"]),
            "win_days": summary[v]["win"],
            "loss_days": summary[v]["loss"],
            "trade_days": summary[v]["trade_days"],
            "win_rate_pct": round(100 * summary[v]["win"] / summary[v]["trade_days"], 1)
            if summary[v]["trade_days"] else 0.0,
            "return_pct": round(100 * summary[v]["pnl"] / INITIAL_CASH, 1),
        }
        for v in VARIANTS
    }


def run() -> dict:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    groups = derive_expiry_groups(profile)
    if YEAR:
        groups = [g for g in groups if g["expiry_date"].startswith(YEAR)]
    files = target_file_map()
    dates_sorted = sorted(files)

    def prior_file(group):
        exp = group["expiry_date"]
        for d in reversed([d for d in dates_sorted if d < exp]):
            path = files[d]
            grids = base.load_expiry_day(path, group)
            if grids:
                return path
        return None

    daily = []
    for i, g in enumerate(groups, 1):
        exp = g["expiry_date"]
        if exp not in files:
            continue
        pf = prior_file(g)
        print(f"[{i}/{len(groups)}] {exp} {g['product_kind']} {g['series']}", flush=True)
        daily.append(run_day(g, files[exp], pf))

    return write_report(daily)


def write_report(daily: list[dict]) -> dict:
    status_count: dict[str, int] = {}
    total_trades = 0
    branch_count = {"below_p0": 0, "ma120_cross": 0}
    branch_pnl = {"below_p0": 0.0, "ma120_cross": 0.0}
    for d in daily:
        status_count[d["status"]] = status_count.get(d["status"], 0) + 1
        total_trades += len(d["positions"])
        for p in d["positions"]:
            branch_count[p["branch"]] = branch_count.get(p["branch"], 0) + 1
            branch_pnl[p["branch"]] = branch_pnl.get(p["branch"], 0.0) + p["pnl_settle"]

    # 연도별 분해
    by_year: dict[str, list[dict]] = {}
    for d in daily:
        by_year.setdefault(d["date"][:4], []).append(d)

    out = {
        "strategy": "S6-v0.1-pair-half",
        "pairing_rule": "nearest_p0_same_expiry",
        "year": YEAR or "ALL",
        "expiry_days": len(daily),
        "total_positions": total_trades,
        "status_count": status_count,
        "branch_count": branch_count,
        "branch_pnl_settle": {k: round(v) for k, v in branch_pnl.items()},
        "initial_cash": INITIAL_CASH,
        "summary": _summarize(daily),
        "by_year": {y: _summarize(ds) for y, ds in sorted(by_year.items())},
        "daily": daily,
    }
    out_path = base.OUTPUT_DIR / "strategy6_v01_pair.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== S6-v0.1 pair-half [{out['year']}] ===")
    print("expiry_days", out["expiry_days"], "positions", total_trades)
    print("status", status_count)
    print("branch", branch_count, "branch_pnl(settle)", out["branch_pnl_settle"])
    print(f"  {'변형':6s} {'총손익':>14s} {'수익률%':>9s} {'승률%':>7s} {'승':>3s} {'패':>3s}")
    for v in VARIANTS:
        s = out["summary"][v]
        print(f"  {v:6s} {s['pnl']:>14,} {s['return_pct']:>9.1f} {s['win_rate_pct']:>7.1f} {s['win_days']:>3} {s['loss_days']:>3}")
    print("\n-- 연도별(settle) --")
    for y, ys in out["by_year"].items():
        s = ys["settle"]
        print(f"  {y}: pnl={s['pnl']:>14,} 수익률={s['return_pct']:>8.1f}% 승률={s['win_rate_pct']:>5.1f}% ({s['win_days']}/{s['trade_days']})")
    print("saved:", out_path)
    return out


if __name__ == "__main__":
    run()
