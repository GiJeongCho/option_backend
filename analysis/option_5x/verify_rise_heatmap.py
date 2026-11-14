# -*- coding: utf-8 -*-
"""
맹박사 '매수 후 K배 상승' 히트맵 독립 검증.

재현검증 명세((대화내역)위클리옵션_전략접근_재현검증명세_...20261030.md) 3·4·7.1절을
코드로 재현해 집계 정확성을 검증한다. (LLM 집계 오류 가능성 차단)

정의
  - 가격대 A[0.40,0.60] / B[0.70,0.90] / C[1.00,1.20] (양끝 포함). 세 가격대 합산.
  - P0 : 만기일 09:00~09:10 첫 실제 관측 OHLC가 가격대에 처음 든 값.
         같은 분봉은 OPEN→CLOSE→HIGH→LOW 순, 이른 분봉 우선.
  - 신호 : 정확히 t0+T 라벨의 종가 C ≥ M×P0.  (해당 분봉 미거래면 관측 불가=분모 제외)
  - 매수가 B : 신호 라벨 다음 실제 거래 분봉 OPEN. 신호에서 3분 이내일 때만 유효.
  - 상승 판정 : 매수봉 제외 이후~15:19(=15:20 경계)에 HIGH ≥ K×B.  (K=2,3,4,5)
  - 셀(T,M) : 유효 매수 중 K배 도달 비율.  분자/분모/고유 만기일 수 보존.

사용:  python verify_rise_heatmap.py            # 전체+연도별
       python verify_rise_heatmap.py 2026       # 특정 연도만 콘솔 상세
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import minimal_baseline_backtest as base

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

BANDS = (("A", 0.40, 0.60), ("B", 0.70, 0.90), ("C", 1.00, 1.20))
T_VALUES = list(range(5, 121, 5))        # 5..120 (5분 간격)
M_VALUES = [round(1.0 + 0.5 * i, 1) for i in range(9)]  # 1.0..5.0 (0.5 간격)
K_VALUES = (2, 3, 4, 5)
EPS = 1e-9
OBS_START, OBS_END = 900, 910             # 초기 관측창 09:00~09:10
LAST_MINUTE = 1519                         # 그리드 마지막(=15:20 경계 전)


def to_min(hhmm: int) -> int:
    return (hhmm // 100) * 60 + (hhmm % 100)


def to_hhmm(total: int) -> int:
    return (total // 60) * 100 + (total % 60)


def band_of(price: float):
    for name, lo, hi in BANDS:
        if lo - EPS <= price <= hi + EPS:
            return name
    return None


def find_p0(contract):
    """09:00~09:10 첫 관측에서 가격대에 처음 든 (P0, t0_minute, band).

    '세 가격대 합산' = P0가 A/B/C 중 하나에 들면 해당 종목을 포함(종목당 1관측).
    같은 분봉은 OPEN→CLOSE→HIGH→LOW 순, 이른 분봉 우선.
    """
    for bar in contract.bars:
        if bar.minute < OBS_START or bar.minute > OBS_END:
            continue
        if not bar.traded:
            continue
        for field in (bar.open, bar.close, bar.high, bar.low):
            if field is None:
                continue
            b = band_of(float(field))
            if b is not None:
                return float(field), bar.minute, b
    return None


def _cells_for_p0(contract, p0, t0, t0_idx):
    """특정 (P0, t0) 관측에 대한 (T, M) -> {reach: {K:bool}, crashed: bool}.

    crashed = P0 관측시점(t0)부터 신호봉(t0+T)까지 사이에 실제 거래 저가가
    0.5*P0 이하로 내려간 적이 있는지. '절반 이하로 하락하지 않은 옵션' 필터용.
    """
    bars = contract.bars
    idx = base.MINUTE_INDEX
    half = 0.5 * p0
    results = {}
    for T in T_VALUES:
        sig_clock = to_hhmm(to_min(t0) + T)
        sig_idx = idx.get(sig_clock)
        if sig_idx is None:
            continue
        sig_bar = bars[sig_idx]
        if not sig_bar.traded or sig_bar.close is None:
            continue  # 신호 종가 관측 불가 → 분모 제외
        close = float(sig_bar.close)
        # 다음 실제 거래 분봉 OPEN (신호 라벨에서 3분 이내)
        buy_price = buy_idx = None
        for j in range(sig_idx + 1, len(bars)):
            b = bars[j]
            if to_min(b.minute) - to_min(sig_clock) > 3:
                break
            if b.traded and b.open is not None:
                buy_price = float(b.open)
                buy_idx = j
                break
        if buy_price is None or buy_price <= 0:
            continue  # 매수 불가 → 분모 제외
        # 절반 이하 하락 여부: t0 관측봉 ~ 신호봉 사이 실제 거래 저가
        crashed = False
        for j in range(t0_idx, sig_idx + 1):
            b = bars[j]
            if b.traded and b.low is not None and float(b.low) <= half + EPS:
                crashed = True
                break
        # 매수봉 제외 이후 ~ 15:20 최고가
        max_high = 0.0
        for j in range(buy_idx + 1, len(bars)):
            b = bars[j]
            if b.minute > LAST_MINUTE:
                break
            if b.traded and b.high is not None:
                if float(b.high) > max_high:
                    max_high = float(b.high)
        for M in M_VALUES:
            if close + EPS >= M * p0:
                reach = {K: (max_high + EPS >= K * buy_price) for K in K_VALUES}
                results[(T, M)] = {"reach": reach, "crashed": crashed}
    return results


def contract_observations(contract):
    """이 종목이 기여하는 (results, band) 목록(종목당 단일 P0 관측)."""
    p0info = find_p0(contract)
    if p0info is None:
        return []
    p0, t0, band = p0info
    t0_idx = base.MINUTE_INDEX[t0]
    return [(_cells_for_p0(contract, p0, t0, t0_idx), band)]


def blank_grid():
    # grid[K][(T,M)] = [reached, valid, set(expiry_days)]
    return {K: defaultdict(lambda: [0, 0, set()]) for K in K_VALUES}


def accumulate(grid, results, expiry_date, require_no_crash=False):
    for (T, M), info in results.items():
        if require_no_crash and info["crashed"]:
            continue  # P0 대비 절반 이하로 떨어진 적 있으면 제외
        reach = info["reach"]
        for K in K_VALUES:
            cell = grid[K][(T, M)]
            cell[1] += 1
            cell[2].add(expiry_date)
            if reach[K]:
                cell[0] += 1


def grid_to_records(grid):
    out = {}
    for K in K_VALUES:
        rows = []
        for (T, M), (reached, valid, days) in sorted(grid[K].items()):
            rows.append({
                "T": T, "M": M,
                "reached": reached, "valid": valid,
                "pct": round(reached / valid * 100, 1) if valid else None,
                "expiry_days": len(days),
            })
        out[str(K)] = rows
    return out


def run():
    profile = json.loads(base.PROFILE_PATH.read_text(encoding="utf-8"))
    groups = base.derive_expiry_groups(profile)
    files = base.target_file_map()

    # 집계 버킷: 전체/연도별 × 합산/콜/풋
    buckets = {}

    def get_bucket(key, sub):
        return buckets.setdefault((key, sub), blank_grid())

    stats = defaultdict(lambda: {"contracts": 0, "with_p0": 0, "expiry": set()})

    for group in groups:
        date = group["expiry_date"]
        if date not in files:
            continue
        year = date[:4]
        contracts = base.load_expiry_day(files[date], group)
        for contract in contracts.values():
            is_put = "PUT" in contract.call_put.upper() or "풋" in contract.call_put
            sub_dir = "put" if is_put else "call"
            for scope in ("all", year):
                stats[scope]["contracts"] += 1
                stats[scope]["expiry"].add(date)
            observations = contract_observations(contract)
            if not observations:
                continue
            for scope in ("all", year):
                stats[scope]["with_p0"] += 1
            for results, _band in observations:
                for scope in ("all", year):
                    accumulate(get_bucket(scope, "combined"), results, date)
                    accumulate(get_bucket(scope, sub_dir), results, date)
                    # P0 대비 절반 이하로 하락한 적 없는 종목만 (nohalf)
                    accumulate(
                        get_bucket(scope, "combined_nohalf"), results, date,
                        require_no_crash=True)
                    accumulate(
                        get_bucket(scope, sub_dir + "_nohalf"), results, date,
                        require_no_crash=True)

    document = {"definition": {
        "bands": [list(b) for b in BANDS], "T": T_VALUES, "M": M_VALUES,
        "K": list(K_VALUES), "obs_window": [OBS_START, OBS_END],
        "signal": "close at t0+T >= M*P0", "buy": "next traded open within 3min",
        "measure": "high >= K*buy from after buy bar to 15:20",
    }, "scopes": {}}

    for (scope, sub), grid in buckets.items():
        document["scopes"].setdefault(scope, {})[sub] = grid_to_records(grid)
    for scope, s in stats.items():
        document.setdefault("sample", {})[scope] = {
            "contract_days": s["contracts"],
            "contract_days_with_p0": s["with_p0"],
            "expiry_days": len(s["expiry"]),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "verify_rise_heatmap.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


def print_grid(document, scope, sub, K):
    rows = document["scopes"][scope][sub][str(K)]
    cell = {(r["T"], r["M"]): r for r in rows}
    print(f"\n[{scope} · {sub} · 매수후 {K}배]  행=신호배수M, 열=경과분T  (값%%, 분모n)")
    header = "  M\\T " + "".join(f"{T:>7}" for T in T_VALUES)
    print(header)
    for M in reversed(M_VALUES):
        line = f"{M:>5} "
        for T in T_VALUES:
            r = cell.get((T, M))
            if r is None or r["valid"] == 0:
                line += f"{'·':>7}"
            else:
                line += f"{r['pct']:>5.0f}/{r['valid']:<1}"[:7].rjust(7)
        print(line)


def main():
    doc = run()
    print("표본 요약:")
    for scope, s in sorted(doc["sample"].items()):
        print(f"  {scope}: 만기일 {s['expiry_days']}  "
              f"종목일 {s['contract_days']}  P0확보 {s['contract_days_with_p0']}")
    focus = sys.argv[1] if len(sys.argv) > 1 else "2026"
    if focus in doc["scopes"]:
        print_grid(doc, focus, "combined", 2)
    print("\n저장: output/verify_rise_heatmap.json")


if __name__ == "__main__":
    main()
