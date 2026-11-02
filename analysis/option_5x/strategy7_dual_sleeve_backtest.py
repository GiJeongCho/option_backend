# -*- coding: utf-8 -*-
"""
S7 - 국면 듀얼 슬리브 (안정 + 복권) 와 청산목표 변형 비교

상관/편상관 분석에서 도출한 조합을 정식 전략으로 고정하고,
같은 진입 규칙에서 "청산 목표"만 바꾼 세 변형을 한 번에 생성한다.

  공통 진입(두 슬리브)
    - 안정 슬리브 : 프리미엄 1.00~3.00, 당일자금 5%(최대 500만)
    - 복권 슬리브 : 프리미엄 0.40~0.80, 당일자금 2%(최대 100만)
    - 풋 우선, 전일 VKOSPI ≥ 5일선(변동성 미붕괴), 09:30~12:59, 당일 재진입 없음

  청산 변형
    - S7-v0.1 듀얼 사다리 : 안정=1/3 2배+1/3 5배+나머지 10배, 복권=전량 10배
    - S7-v0.2 전량 2배 익절 : 두 슬리브 모두 2배에서 전량 청산
    - S7-v0.3 전량 10배 홀드 : 두 슬리브 모두 10배까지 전량 홀드

청산엔진은 검증된 strategy6_v01.execute_exits 를 재사용한다.
표준 산출물(버전별): <stem>_daily.csv / <stem>_trades.csv, 공용 strategy7_summary.json
기간: COMMON_START(전체) ~ 최신 만기일. (요약에 2025-03 이후 부분성과도 함께 기록)
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
COMMON_START = v01.COMMON_START
RECENT_SPLIT = "20250301"  # 최근 국면 부분성과 기준

CORE = base.PremiumBand("core_100_300", 1.00, 3.00)
LOTTERY = base.PremiumBand("lottery_040_080", 0.40, 0.80)

INITIAL_CASH = v01.INITIAL_CASH
MULT = base.MULTIPLIER
COMM = base.COMMISSION_PER_CONTRACT
T = v01.Tranche

CORE_FRACTION = 0.05
CORE_MAX = 5_000_000
LOTTERY_FRACTION = 0.02
LOTTERY_MAX = 1_000_000

# 조건부 러너(v0.4) 기준: '빠른 더블'이면 10배 홀드, 아니면 2배 전량 익절
FAST_MINUTES = 30     # 진입→2배 소요 이 이하이면 빠른 더블
EARLY_CUTOFF = 1200   # 2배 도달 시각이 이 시각(12:00) 전이어야 빠른 더블


def minutes_of(hhmm):
    """HHMM 정수를 장중 분으로 변환 (930 -> 570)."""
    return (int(hhmm) // 100) * 60 + (int(hhmm) % 100)


def first_2x_minute(contract, event):
    """진입가(fill_price) 기준 2배를 처음 찍는 봉의 시각(HHMM). 없으면 None."""
    target = event.fill_price * 2.0
    for i in range(event.fill_index + 1, len(contract.bars)):
        bar = contract.bars[i]
        if bar.minute > base.FORCE_EXIT_TIME:
            break
        if bar.traded and bar.high is not None and float(bar.high) + 1e-9 >= target:
            return bar.minute
    return None


# --- 청산 트랜치 빌더 (contract, event, qty) --------------------------------
def ladder_core(contract, event, qty):
    """1/3 2배 + 1/3 5배 + 나머지 10배."""
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


def all_2x(contract, event, qty):
    return [T(1, 2, qty, qty)]


def all_10x(contract, event, qty):
    return [T(1, 10, qty, qty)]


def lottery_runner(contract, event, qty):
    return [T(1, 10, qty, qty)]


def conditional_runner(contract, event, qty):
    """2배 도달 속도로 분기: 빠른 더블이면 전량 10배 홀드, 아니면 전량 2배 익절."""
    t2 = first_2x_minute(contract, event)
    if t2 is not None:
        fast = (minutes_of(t2) - minutes_of(event.fill_minute) <= FAST_MINUTES
                and t2 < EARLY_CUTOFF)
        if fast:
            return [T(1, 10, qty, qty)]
    return [T(1, 2, qty, qty)]


CONFIGS = (
    {
        "version": "S7-v0.1",
        "stem": "strategy7",
        "label": "S7-v0.1 듀얼 슬리브 사다리(2·5·10배)",
        "core": ladder_core,
        "lottery": lottery_runner,
    },
    {
        "version": "S7-v0.2",
        "stem": "strategy7_v02",
        "label": "S7-v0.2 듀얼 슬리브 전량 2배 익절",
        "core": all_2x,
        "lottery": all_2x,
    },
    {
        "version": "S7-v0.3",
        "stem": "strategy7_v03",
        "label": "S7-v0.3 듀얼 슬리브 전량 10배 홀드",
        "core": all_10x,
        "lottery": all_10x,
    },
    {
        "version": "S7-v0.4",
        "stem": "strategy7_v04",
        "label": "S7-v0.4 조건부 러너(빠른더블 10배·느린더블 2배)",
        "core": conditional_runner,
        "lottery": conditional_runner,
    },
)


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


def simulate_day(version, date_value, start_cash, contracts, snap,
                 core_fn, lottery_fn):
    cash = start_cash
    total_buy = 0
    total_sell = 0
    total_fee = 0
    expired_qty = 0
    trades = []
    info = {
        "core_code": "", "core_qty": 0,
        "core_2x": False, "core_5x": False, "core_10x": False,
        "lottery_code": "", "lottery_qty": 0, "lottery_10x": False,
    }
    gate_ok = entry_gate(snap)
    slope120 = snap.slope120 if snap else math.nan

    sleeves = [
        ("core", CORE, CORE_FRACTION, CORE_MAX, core_fn),
        ("lottery", LOTTERY, LOTTERY_FRACTION, LOTTERY_MAX, lottery_fn),
    ]
    for name, band, frac, cap, exit_fn in sleeves:
        if not gate_ok or not contracts:
            continue
        event = select_band(contracts, band, slope120)
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
        total_buy += buy_principal
        total_fee += buy_fee
        trades.append({
            "strategy": version, "date": date_value, "sleeve": name,
            "code": event.code, "call_put": contract.call_put, "side": "BUY",
            "minute": event.fill_minute, "quantity": qty,
            "price": round(event.fill_price, 4),
            "cash_flow": -(buy_principal + buy_fee),
        })
        positions = exit_fn(contract, event, qty)
        gross, sell_fees, sell_trades, reached = v01.execute_exits(
            date_value, contract, event, positions
        )
        cash += gross - sell_fees
        total_sell += gross
        total_fee += sell_fees
        sold_qty = sum(int(tr.get("quantity", 0)) for tr in sell_trades)
        expired_qty += max(0, qty - sold_qty)
        for tr in sell_trades:
            tr["strategy"] = version
            tr["sleeve"] = name
        trades.extend(sell_trades)
        if name == "core":
            info.update({
                "core_code": event.code, "core_qty": qty,
                "core_2x": bool(reached.get(2, False)),
                "core_5x": bool(reached.get(5, False)),
                "core_10x": bool(reached.get(10, False)),
            })
        else:
            info.update({
                "lottery_code": event.code, "lottery_qty": qty,
                "lottery_10x": bool(reached.get(10, False)),
            })

    end_cash = start_cash - (total_buy + total_fee) + total_sell
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
        "reached_10x": bool(info["core_10x"] or info["lottery_10x"]),
        "vix_prev_close": snap.close if snap else None,
        **info,
    }
    return daily, trades


def internal_audit(daily, trades):
    errors = []
    by_date = {}
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
        cf = sum(int(round(t["cash_flow"])) for t in by_date.get(d, []))
        if abs(cf - int(round(row["pnl"]))) > 1:
            errors.append(f"{d}: 거래장부 손익({cf} vs {row['pnl']})")
        prev_end = row["end_cash"]
    return {"passed": not errors, "error_count": len(errors), "error_samples": errors[:30]}


def summarize(daily, label):
    traded = [r for r in daily if r["trade_day"]]
    pnls = [r["pnl"] for r in daily]
    if not daily:
        return {"label": label, "expiry_days": 0}
    equity = [INITIAL_CASH] + [r["end_cash"] for r in daily]
    peak = equity[0]
    mdd = 0.0
    mdd_abs = 0
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
        "core_10x_days": sum(bool(r["core_10x"]) for r in daily),
        "lottery_10x_days": sum(bool(r["lottery_10x"]) for r in daily),
        "total_pnl": sum(pnls),
        "total_return_pct": round(sum(pnls) / INITIAL_CASH * 100, 2),
        "profit_factor": round(gp / gl, 3) if gl else None,
        "max_drawdown": int(round(mdd_abs)),
        "max_drawdown_pct": round(mdd, 2),
        "pnl_excluding_best_3_days": sum(pnls) - sum(profits[:3]),
        "pnl_excluding_best_5_days": sum(pnls) - sum(profits[:5]),
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
              if g["expiry_date"] >= COMMON_START]
    files = base.target_file_map()
    vix_dates, snapshots = v02.vix_snapshots()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 각 설정별 현금/산출물을 유지하며, 만기일 파일은 하루에 한 번만 로드한다.
    state = {
        cfg["version"]: {"cash": INITIAL_CASH, "daily": [], "trades": []}
        for cfg in CONFIGS
    }
    config_by_version = {cfg["version"]: cfg for cfg in CONFIGS}

    for group in groups:
        date_value = group["expiry_date"]
        contracts = (base.load_expiry_day(files[date_value], group)
                     if date_value in files else {})
        snap = v02.previous_snapshot(date_value, vix_dates, snapshots)
        for version, st in state.items():
            cfg = config_by_version[version]
            daily, trades = simulate_day(
                version, date_value, st["cash"], contracts, snap,
                cfg["core"], cfg["lottery"],
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
        recent = [r for r in st["daily"] if str(r["date"]) >= RECENT_SPLIT]
        write_csv_union(OUTPUT_DIR / f"{cfg['stem']}_daily.csv", st["daily"])
        write_csv_union(OUTPUT_DIR / f"{cfg['stem']}_trades.csv", st["trades"])
        variants[version] = {
            "version": version,
            "label": cfg["label"],
            "summary": summarize(st["daily"], "full"),
            "recent_2025_03_on": summarize(recent, "2025-03~"),
            "audit": audit,
        }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "initial_cash": INITIAL_CASH,
        "common_rules": {
            "core_band": [CORE.lower, CORE.upper],
            "core_budget": [CORE_FRACTION, CORE_MAX],
            "lottery_band": [LOTTERY.lower, LOTTERY.upper],
            "lottery_budget": [LOTTERY_FRACTION, LOTTERY_MAX],
            "gate": "전일 VKOSPI ≥ 5일선, 풋 우선, 09:30~12:59, 당일 재진입 없음",
            "exit_engine": "strategy6_v01.execute_exits 재사용",
        },
        "variants": variants,
        "limitations": [
            "분석 구간 자료로 고정한 규칙이므로 미사용 홀드아웃 전진검증 전 실거래 금지.",
            "복권 슬리브는 승률이 낮고 꼬리수익 비중이 높다.",
            "최종 미체결 잔량은 0원으로 평가한다.",
        ],
    }
    (OUTPUT_DIR / "strategy7_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")

    print(json.dumps(
        {v: {"full": d["summary"], "recent": d["recent_2025_03_on"],
             "audit": d["audit"]}
         for v, d in variants.items()},
        ensure_ascii=False, indent=2))
    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
