"""依目標區間判斷每格狀態、挑選影片字幕、產生報告用的單一建議。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snowcoach.coach.targets import CATEGORIES, CUES, METRICS, TARGETS
from snowcoach.turns import Turn

NA, OK, WARN, BAD = 0, 1, 2, 3
STATUS_ZH = {NA: "無法判斷", OK: "良好", WARN: "注意", BAD: "需要改善"}


def judge(value: float, target: dict) -> tuple[int, int]:
    """回傳 (狀態, 方向)。方向：-1 偏低、+1 偏高、0 在區間內。"""
    if value is None or np.isnan(value):
        return NA, 0
    lo, hi = target["ok"]
    if lo <= value <= hi:
        return OK, 0
    if value < lo:
        bad = target.get("bad_lo")
        return (BAD if bad is not None and value < bad else WARN), -1
    bad = target.get("bad_hi")
    return (BAD if bad is not None and value > bad else WARN), 1


def turn_peak(incl: np.ndarray, turn: Turn) -> float:
    """一個轉彎的代表傾角：|傾角| 的第 80 百分位（避開單格雜訊）。"""
    seg = np.abs(incl[turn.start:turn.end])
    seg = seg[~np.isnan(seg)]
    return float(np.percentile(seg, 80)) if len(seg) else np.nan


def judged_values(series: dict[str, np.ndarray], turns: list[Turn], turn_id: np.ndarray) -> dict[str, np.ndarray]:
    """轉成評分用的數值：傾角用所在轉彎的代表值、折髖只看轉彎中、兩膝差取絕對值。"""
    n = len(turn_id)
    out = {k: series[k].copy() for k in METRICS if k in series}
    peaks = np.full(n, np.nan)
    for t in turns:
        peaks[t.start:t.end] = turn_peak(series["inclination"], t)
    out["inclination"] = peaks
    ang = series["angulation"].copy()
    ang[turn_id < 0] = np.nan
    out["angulation"] = ang
    out["knee_diff"] = np.abs(series["knee_diff"])
    return out


@dataclass
class FrameJudgement:
    status: dict[str, np.ndarray]     # 每個指標每格的狀態碼
    direction: dict[str, np.ndarray]  # -1 / 0 / +1


def judge_frames(values: dict[str, np.ndarray], level: str) -> FrameJudgement:
    status, direction = {}, {}
    for k, v in values.items():
        t = TARGETS[level][k]
        res = [judge(x, t) for x in v]
        status[k] = np.array([r[0] for r in res])
        direction[k] = np.array([r[1] for r in res])
    return FrameJudgement(status, direction)


def fill_gaps(status: np.ndarray, direction: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """短暫無法判斷（<= max_gap 格）時沿用前一個狀態，避免遮擋一下就把持續時間歸零。"""
    st, dr = status.copy(), direction.copy()
    gap = 0
    for i in range(1, len(st)):
        if st[i] == NA and st[i - 1] != NA and gap < max_gap:
            st[i], dr[i] = st[i - 1], dr[i - 1]
            gap += 1
        elif st[i] != NA:
            gap = 0
    return st, dr


def _cue_key(metric: str, direction: int) -> tuple[str, str] | None:
    key = (metric, "low" if direction < 0 else "high")
    return key if key in CUES else None


def live_cues(fj: FrameJudgement, fps: float, skip: set[str] = frozenset(),
              hold_warn: float = 1.5, min_show: float = 0.6,
              max_talk: float = 3.0, rest: float = 1.5) -> list[tuple[str, str] | None]:
    """每格要顯示的字幕（一次只有一句）。

    - 候選：依「平衡 → 壓力 → 立刃 → 旋轉」順序，第一個達到紅色、
      或黃色持續 hold_warn 秒以上的指標。
    - 同一候選至少持續 min_show 秒才會顯示，避免閃爍。
    - 字幕整體連續講 max_talk 秒後休息 rest 秒，不讓畫面一直有字。
    """
    metrics = [m for c in CATEGORIES for m in METRICS if METRICS[m][2] == c and m in fj.status and m not in skip]
    n = len(next(iter(fj.status.values())))
    filled = {m: fill_gaps(fj.status[m], fj.direction[m], int(round(0.5 * fps))) for m in metrics}
    warn_run = {m: 0 for m in metrics}
    cand: list[tuple[str, str] | None] = []
    for i in range(n):
        pick = None
        for m in metrics:
            st, dr = filled[m]
            s = st[i]
            warn_run[m] = warn_run[m] + 1 if s in (WARN, BAD) else 0
            if pick is None and (s == BAD or (s == WARN and warn_run[m] >= hold_warn * fps)):
                pick = _cue_key(m, dr[i])
        cand.append(pick)

    # 同一候選連續至少 min_show 秒才算數（整段都算，不會延遲出現）
    need = max(1, int(round(min_show * fps)))
    confirmed: list[tuple[str, str] | None] = [None] * n
    i = 0
    while i < n:
        j = i
        while j < n and cand[j] == cand[i]:
            j += 1
        if cand[i] is not None and j - i >= need:
            confirmed[i:j] = cand[i:j]
        i = j

    out: list[tuple[str, str] | None] = [None] * n
    talk_max, rest_len = int(round(max_talk * fps)), int(round(rest * fps))
    talking = resting = 0
    for i in range(n):
        if resting > 0:
            resting -= 1
            continue
        if confirmed[i] is not None:
            out[i] = confirmed[i]
            talking += 1
            if talking >= talk_max:
                talking, resting = 0, rest_len
        else:
            talking = 0
    return out


def enough_data(fj: FrameJudgement, turns: list[Turn], fps: float) -> dict[str, bool]:
    """各指標的可判斷資料是否足以下結論（轉彎類至少 2 個彎，其餘至少 1 秒）。"""
    out = {}
    for m, st in fj.status.items():
        if m == "inclination":
            out[m] = sum(st[t.start] != NA for t in turns) >= 2
        else:
            out[m] = int((st != NA).sum()) >= int(fps * 1.0)
    return out


@dataclass
class Finding:
    metric: str
    direction: str       # "low" / "high"
    share: float         # 超出目標的比例
    bad_share: float
    samples: int
    unit: str            # "格" 或 "個轉彎"


def metric_findings(values: dict[str, np.ndarray], fj: FrameJudgement, turns: list[Turn],
                    fps: float) -> dict[str, Finding | None]:
    """每個指標是否構成問題（超出目標夠多、樣本夠多）。"""
    out: dict[str, Finding | None] = {}
    min_frames = int(fps * 1.0)
    for m in fj.status:
        st, dr = fj.status[m], fj.direction[m]
        if m == "inclination":
            # 以轉彎為單位
            idx = [t.start for t in turns if st[t.start] != NA]
            st, dr, need, unit = st[idx], dr[idx], 2, "個轉彎"
        else:
            valid = st != NA
            st, dr, need, unit = st[valid], dr[valid], min_frames, "格"
        if len(st) < need:
            out[m] = None
            continue
        off = (st == WARN) | (st == BAD)
        share, bad_share = float(off.mean()), float((st == BAD).mean())
        if not (bad_share >= 0.15 or share >= 0.4):
            out[m] = None
            continue
        low, high = int((off & (dr < 0)).sum()), int((off & (dr > 0)).sum())
        direction = "low" if low >= high else "high"
        if (m, direction) not in CUES:
            out[m] = None
            continue
        out[m] = Finding(m, direction, share, bad_share, len(st), unit)
    return out


def top_finding(findings: dict[str, Finding | None], skip: set[str] = frozenset()) -> Finding | None:
    """依優先順序挑出唯一一個最該先處理的問題。"""
    for c in CATEGORIES:
        cands = [f for m, f in findings.items() if f is not None and METRICS[m][2] == c and m not in skip]
        if cands:
            return max(cands, key=lambda f: (f.bad_share, f.share))
    return None
