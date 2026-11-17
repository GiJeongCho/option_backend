"""S8 하프컷 리버스(Half-Cut Reverse) — 진입 동일, 청산 배수만 다른 변형 내보내기.

진입/반토막 판정/매수 사다리/손절은 S6-v0.1 페어전략과 동일하고, **익절 배수**만 다르다.
버전 끝자리 = 익절 배수(안헷갈리게):
  - S8-v0.2 : 진입가 2배 전량익절 (미달 시 만기결제)
  - S8-v0.3 : 진입가 3배 전량익절
  - S8-v0.4 : 진입가 4배 전량익절

한 번의 데이터 패스로 세 변형의 daily/trades CSV 를 모두 생성한다.
종료일은 다른 전략과 동일하게 형제 일별 CSV(strategy1_daily.csv) 최신 날짜에 맞춘다.

출력(각 변형별 2개):
  output/strategy8_v02_x2_daily.csv  / _trades.csv
  output/strategy8_v03_x3_daily.csv  / _trades.csv
  output/strategy8_v04_x4_daily.csv  / _trades.csv
"""

from __future__ import annotations

import csv
import json

import minimal_baseline_backtest as base
from count_opportunities import PROFILE_PATH, derive_expiry_groups, target_file_map
import strategy6_v01_pair_half_backtest as pair

INITIAL_CASH = pair.INITIAL_CASH  # 20,000,000
MULT = base.MULTIPLIER
COMM = base.COMMISSION_PER_CONTRACT

REFRESH_SUMMARY = base.OUTPUT_DIR / "refresh_all_backtests_summary.json"
SIBLING_DAILY = base.OUTPUT_DIR / "strategy1_daily.csv"

# (전략ID, variant, multiple, daily파일, trades파일) — 버전 끝자리 = 배수
VARIANTS = [
    ("S8-v0.2", "x2", 2, "strategy8_v02_x2_daily.csv", "strategy8_v02_x2_trades.csv"),
    ("S8-v0.3", "x3", 3, "strategy8_v03_x3_daily.csv", "strategy8_v03_x3_trades.csv"),
    ("S8-v0.4", "x4", 4, "strategy8_v04_x4_daily.csv", "strategy8_v04_x4_trades.csv"),
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
    *pair.SIGNAL_FIELDS,
]

SIGNAL_STATUSES = {
    "both_halved_call_first_buy_put",
    "both_halved_put_first_buy_call",
    "call_halved_buy_put",
    "put_halved_buy_call",
    "both_active",
    "one_side_only",
}


def _catalog_end_date() -> str | None:
    """다른 전략들이 끝나는 종료일. 형제 일별 CSV(최신 날짜) → 리프레시 요약 순."""
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

    # variant 별 누적 저장소
    daily_by_variant: dict[str, list[dict]] = {v[0]: [] for v in VARIANTS}
    trades_by_variant: dict[str, list[dict]] = {v[0]: [] for v in VARIANTS}
    pair_daily: list[dict] = []

    for i, g in enumerate(groups, 1):
        exp = g["expiry_date"]
        if exp not in files:
            continue
        pf = prior_file(g)
        print(f"[{i}/{len(groups)}] {exp} {g['product_kind']} {g['series']}", flush=True)
        result, positions = pair.build_day(g, files[exp], pf)
        pair_daily.append(pair.day_report(result, positions))

        for sid, variant, mult, _, _ in VARIANTS:
            day_pnl = 0.0
            day_qty = 0
            day_buy = 0.0
            day_gross = 0.0
            day_fees = 0.0
            day_expired = 0
            trows = trades_by_variant[sid]

            for p in positions:
                det = pair.exit_detail(p, variant)
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

                exit_reason = (
                    f"TARGET_{mult}X_ALL"
                    if det["reason"].startswith("TARGET_")
                    else det["reason"]
                )
                trows.append({
                    "strategy": sid, "date": exp, "campaign": 1,
                    "minute": p["fill_minute"], "code": p["code"],
                    "call_put": p["call_put"], "side": "BUY",
                    "reason": f"PAIR_ENTRY_{p['branch'].upper()}", "stage": 1,
                    "quantity": qty, "price": round(fill, 4),
                    "principal": round(buy_principal), "fee": round(buy_fee),
                    "cash_flow": round(buy_cash), "bar_volume": "",
                    **{field: p.get(field, "") for field in pair.SIGNAL_FIELDS},
                })
                trows.append({
                    "strategy": sid, "date": exp, "campaign": 1,
                    "minute": det["exit_minute"], "code": p["code"],
                    "call_put": p["call_put"], "side": "SELL",
                    "reason": exit_reason, "stage": 2,
                    "quantity": qty, "price": round(det["exit_price"], 4),
                    "principal": round(gross_sales), "fee": round(sell_fee),
                    "cash_flow": round(sell_cash), "bar_volume": "",
                    **{field: p.get(field, "") for field in pair.SIGNAL_FIELDS},
                })

            has_trade = bool(positions)
            daily_by_variant[sid].append({
                "date": exp,
                "pnl": round(day_pnl),
                "trade_day": has_trade,
                "signal_found": result.get("signal_found", False) or has_trade,
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

    # variant 별 자본곡선 구성 + CSV 기록
    for sid, variant, mult, daily_name, trades_name in VARIANTS:
        records = sorted(daily_by_variant[sid], key=lambda r: r["date"])
        cash = INITIAL_CASH
        daily_out: list[dict] = []
        for r in records:
            start_cash = cash
            cash += r["pnl"]
            daily_out.append({
                "strategy": sid,
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

        with (base.OUTPUT_DIR / daily_name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DAILY_FIELDS)
            w.writeheader()
            w.writerows(daily_out)
        with (base.OUTPUT_DIR / trades_name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRADES_FIELDS)
            w.writeheader()
            w.writerows(trades_by_variant[sid])

        total = cash - INITIAL_CASH
        tdays = sum(1 for r in daily_out if r["trade_day"])
        wins = sum(1 for r in daily_out if r["trade_day"] and r["pnl"] > 0)
        end = daily_out[-1]["date"] if daily_out else None
        print(f"\n=== {sid} pair {variant} ===  end={end} days={len(daily_out)} trade_days={tdays}")
        print(f"  win {wins}/{tdays} ({100*wins/tdays:.1f}%)" if tdays else "  no trades")
        print(f"  total_pnl {total:,.0f} | return {100*total/INITIAL_CASH:.1f}%")

    # 페어 비교 화면도 S8과 같은 날짜·짝·신호로 함께 갱신한다.
    pair.write_report(pair_daily)

if __name__ == "__main__":
    run()
