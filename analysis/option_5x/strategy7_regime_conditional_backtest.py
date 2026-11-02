# -*- coding: utf-8 -*-
"""
S7 - 국면조건부(regime-conditional) 전략 백테스트

상관/편상관 분석 결론을 그대로 규칙화한 전략.
  - 진입 게이트 : 전일 VKOSPI가 5일선 위 (변동성 미붕괴)  ← '단기 5일선' 독립 신호
  - 국면 분기   : 전일 VKOSPI 레벨 ≥ 30 → 고변동성 / 미만 → 저변동성
  - 가격대 선택 : 저변동성 → 싼 밴드(0.5~1.0) / 고변동성 → 중간 밴드(1.0~2.5)  ← 핵심 결론
  - 방향        : 풋(PUT) 우선 (풋이 10배를 더 자주 터뜨림)
  - 청산        : 절반 2배 익절 + 나머지 10배 러너 (strategy6_v01 로직 재사용)
  - 진입 시간   : 09:30~13:00 (13시 이후 신규 진입 금지; v01 엔진 설정)

검증된 strategy6_v01 의 자금 시뮬레이션/청산/감사 기계를 그대로 재사용하고,
'진입 대상 선택(select_event)' 만 위 규칙으로 교체한다.

대조군(국면 분기 효과 분리):
  - regime   : 위 규칙(국면별 밴드 전환)
  - mid_only : 국면 무관 항상 중간 밴드(1.0~2.5)
  - cheap_only: 국면 무관 항상 싼 밴드(0.5~1.0)
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import minimal_baseline_backtest as base
import strategy6_v01_vix_rv_put_backtest as v01
import strategy6_v02_vix_ma5_rising_backtest as v02

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STRATEGY_VERSION = "S7-v0.1"

VIX_HIGH_THRESHOLD = 30.0  # VKOSPI 'fear' 레벨; 이상이면 고변동성 국면
CHEAP = base.PremiumBand("CHEAP_050_100", 0.50, 1.00)
MID = base.PremiumBand("MID_100_250", 1.00, 2.50)

MODES = ("regime", "mid_only", "cheap_only")


def is_put(contract: base.ContractGrid) -> bool:
    return "PUT" in contract.call_put.upper() or "풋" in contract.call_put


def entry_gate(snap: "v02.VixSnapshot | None") -> bool:
    """전일 VKOSPI가 5일선 위 = 변동성 미붕괴. (5일선 계산 전이면 레벨만 유효하면 허용)"""
    if snap is None or not math.isfinite(snap.close):
        return False
    if math.isfinite(snap.ma5):
        return snap.close >= snap.ma5
    return True


def classify_regime(snap: "v02.VixSnapshot") -> str:
    return "HIGH" if snap.close >= VIX_HIGH_THRESHOLD else "LOW"


def band_for(mode: str, snap: "v02.VixSnapshot") -> base.PremiumBand:
    if mode == "mid_only":
        return MID
    if mode == "cheap_only":
        return CHEAP
    # regime
    return CHEAP if classify_regime(snap) == "LOW" else MID


def select_event(
    contracts: dict[str, base.ContractGrid],
    snap: "v02.VixSnapshot | None",
    mode: str,
) -> "v01.EntryEvent | None":
    if not entry_gate(snap):
        return None
    assert snap is not None
    band = band_for(mode, snap)

    candidates: list[tuple[int, v01.EntryEvent]] = []
    for contract in contracts.values():
        event = v01.first_band_event(contract, band, snap.slope120)
        if event is None:
            continue
        # 풋 우선(0) → 콜(1), 이후 이른 신호, 코드 순
        put_rank = 0 if is_put(contract) else 1
        candidates.append((put_rank, event))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1].signal_minute, t[1].code))
    return candidates[0][1]


def run_mode(
    mode: str,
    groups: list[dict[str, str]],
    files: dict[str, Path],
    vix_dates: list[str],
    snapshots: dict[str, "v02.VixSnapshot"],
) -> v01.Account:
    account = v01.Account()
    assert account.daily is not None
    assert account.trades is not None
    assert account.positions is not None

    for group in groups:
        date_value = group["expiry_date"]
        contracts = (
            base.load_expiry_day(files[date_value], group)
            if date_value in files
            else {}
        )
        snap = v02.previous_snapshot(date_value, vix_dates, snapshots)
        event = select_event(contracts, snap, mode) if contracts else None
        daily, trades, position = v01.simulate_day(
            date_value, account, contracts, event
        )
        # 국면/밴드 기록 추가
        daily["mode"] = mode
        daily["vix_prev_close"] = snap.close if snap else None
        daily["regime"] = (
            classify_regime(snap) if snap and math.isfinite(snap.close) else ""
        )
        account.cash = int(daily["end_cash"])
        account.daily.append(daily)
        account.trades.extend(trades)
        if position is not None:
            account.positions.append(position)
    return account


def main() -> None:
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = [
        g for g in base.derive_expiry_groups(profile)
        if g["expiry_date"] >= v01.COMMON_START
    ]
    files = base.target_file_map()
    vix_dates, snapshots = v02.vix_snapshots()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison: list[dict[str, object]] = []
    all_daily: list[dict[str, object]] = []

    for mode in MODES:
        account = run_mode(mode, groups, files, vix_dates, snapshots)
        assert account.daily is not None and account.trades is not None

        summary = v01.summarize(account.daily)
        audit = v01.internal_audit(account.daily, account.trades)

        traded = [r for r in account.daily if bool(r["trade_day"])]
        reached_2x = sum(bool(r["reached_2x"]) for r in traded)
        reached_10x = sum(bool(r["reached_10x"]) for r in traded)

        # 개발/검증 시간분할 (v01 기준일 사용)
        development = [r for r in account.daily if str(r["date"]) <= v01.DEVELOPMENT_END]
        validation = [r for r in account.daily if str(r["date"]) > v01.DEVELOPMENT_END]

        comparison.append(
            {
                "mode": mode,
                "trade_days": summary["trade_days"],
                "win_rate_pct": summary["win_rate_pct"],
                "reached_2x_days": reached_2x,
                "reached_10x_days": reached_10x,
                "total_pnl": summary["total_pnl"],
                "total_return_pct": summary["total_return_pct"],
                "profit_factor": summary["profit_factor"],
                "max_drawdown_pct": summary["max_drawdown_pct"],
                "pnl_excl_best_3": summary["pnl_excluding_best_3_days"],
                "dev_pnl": v01.summarize(development)["total_pnl"] if development else 0,
                "val_pnl": v01.summarize(validation)["total_pnl"] if validation else 0,
                "audit_passed": audit["passed"],
            }
        )
        for r in account.daily:
            all_daily.append(r)
        base.write_csv(OUTPUT_DIR / f"strategy7_{mode}_daily.csv", account.daily)
        base.write_csv(OUTPUT_DIR / f"strategy7_{mode}_trades.csv", account.trades)

    result = {
        "strategy_version": STRATEGY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "initial_cash": v01.INITIAL_CASH,
        "rules": {
            "entry_gate": "전일 VKOSPI ≥ 5일선 (변동성 미붕괴)",
            "regime_split": f"전일 VKOSPI ≥ {VIX_HIGH_THRESHOLD} → 고변동성",
            "band_low_vol": [CHEAP.lower, CHEAP.upper],
            "band_high_vol": [MID.lower, MID.upper],
            "direction": "풋 우선",
            "exit": "절반 2배 익절 + 나머지 10배 러너 (strategy6_v01)",
            "entry_window": [v01.ENTRY_START, v01.ENTRY_CUTOFF],
        },
        "comparison": comparison,
    }
    (OUTPUT_DIR / "strategy7_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    base.write_csv(OUTPUT_DIR / "strategy7_comparison.csv", comparison)

    # ---- 콘솔 출력 ----
    print(f"분석 만기일: {len(groups)}일  (초기자금 {v01.INITIAL_CASH:,}원)\n")
    import pandas as pd  # noqa: WPS433
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))
    print("\n저장: output/strategy7_comparison.json / _comparison.csv / strategy7_<mode>_daily.csv")


if __name__ == "__main__":
    main()
