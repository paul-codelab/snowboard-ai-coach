"""站姿推斷與轉彎切分。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from snowcoach.geometry import MIN_CONF, board_frame
from snowcoach.joints import IDX
from snowcoach.pose.base import PoseFrame


@dataclass
class StanceResult:
    """慣用站姿，以及每格「目前哪隻腳在前」（單板可以 switch 滑行，前腳會換邊）。"""
    stance: str | None     # 慣用站姿 "regular" / "goofy" / None（無法確定）
    confidence: float      # 0~1
    votes: int             # 有方向證據的影格數
    source: str            # "user" / "auto"
    lead: list[str | None] = field(default_factory=list)  # 每格前腳 "left" / "right" / None
    segments: list[tuple[int, int, str]] = field(default_factory=list)  # (起, 迄, 前腳)，偵測到的方向區段

    def frame_stance(self, i: int) -> str | None:
        """第 i 格當下的站姿方向（左腳在前 = regular 方向）。"""
        if not self.lead:
            return self.stance
        lead = self.lead[i]
        return None if lead is None else ("regular" if lead == "left" else "goofy")

    def is_switch(self, i: int) -> bool | None:
        cur = self.frame_stance(i)
        if cur is None or self.stance is None:
            return None
        return cur != self.stance

    @property
    def known(self) -> bool:
        return self.stance is not None or any(x is not None for x in self.lead)


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def direction_vote(pose: PoseFrame) -> float | None:
    """單格投票：正值 = 左腳在前（朝左腳方向滑），負值 = 右腳在前。

    線索（都投影到水平面、以「右腳→左腳」方向為正）：
    - 頭部朝向：騎手看向前進方向。
    - 胸口朝向：肩膀會往板頭方向打開。
    雙手位置在測試影片上會投反（自拍時唯一可用的手是後手，常伸向板尾），所以不採用。
    """
    if pose.world is None:
        return None
    P = pose.world
    bf, _ = board_frame(pose, P, None, pose.up)
    if bf is None:
        return None
    # 左右標記一致性：面向腳尖側時，左腳應在騎手左方（ẑ × ŷ），否則此格左右可能標反
    if np.dot(P[IDX["left_ankle"]] - P[IDX["right_ankle"]], np.cross(bf.z, bf.y)) <= 0:
        return None
    toward_left = -bf.x  # 沒有站姿時 x = 左腳→右腳

    def flat(v: np.ndarray) -> np.ndarray:
        return v - np.dot(v, bf.z) * bf.z

    def usable(*names: str) -> bool:
        return all(pose.conf[IDX[n]] >= MIN_CONF for n in names)

    votes = []
    if usable("nose", "left_ear", "right_ear"):
        d = _unit(flat(P[IDX["nose"]] - (P[IDX["left_ear"]] + P[IDX["right_ear"]]) / 2))
        if d is not None:
            votes.append(float(np.dot(d, toward_left)))
    if usable("left_shoulder", "right_shoulder"):
        chest = _unit(np.cross(bf.z, flat(P[IDX["right_shoulder"]] - P[IDX["left_shoulder"]])))
        if chest is not None:
            if np.dot(chest, bf.y) < 0:
                chest = -chest
            votes.append(float(np.dot(chest, toward_left)))
    return float(np.mean(votes)) if votes else None


def _runs(labels: list[str | None]) -> list[tuple[int, int, str | None]]:
    out, i = [], 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        out.append((i, j, labels[i]))
        i = j
    return out


def infer_direction(poses: list[PoseFrame | None], fps: float, natural: str | None = None,
                    switch: bool = True, window: float = 1.0, min_segment: float = 2.0,
                    threshold: float = 0.25) -> StanceResult:
    """判斷每格前腳，並推斷慣用站姿（使用者有指定時以指定為準）。

    1. 每格投票（頭部＋胸口），以前後 window 秒平均；證據不足 0.3 秒的位置不判斷。
    2. 平均值超過 ±threshold 才算有方向；同一方向持續 min_segment 秒以上才成立為一個區段
       （避免把雜訊當成 switch）。
    3. 區段之間的空檔：兩側同方向就補上；兩側不同（換邊的過渡）維持未知。
    4. 慣用站姿：使用者指定優先；否則取區段總長的多數方向（需 >= 3 秒且佔 70% 以上）。
    """
    n = len(poses)
    raw = np.array([np.nan if p is None else (direction_vote(p) if direction_vote(p) is not None else np.nan)
                    for p in poses])
    votes = int(np.isfinite(raw).sum())
    user = natural is not None
    front = {"regular": "left", "goofy": "right"}

    if not switch:
        lead = [front[natural]] * n if natural else [None] * n
        return StanceResult(natural, 1.0 if user else 0.0, votes, "user" if user else "auto", lead, [])

    w = max(1, int(round(window * fps)))
    need = max(1, int(round(0.3 * fps)))
    labels: list[str | None] = []
    for i in range(n):
        seg = raw[max(0, i - w):i + w + 1]
        seg = seg[np.isfinite(seg)]
        m = seg.mean() if len(seg) >= need else np.nan
        labels.append("left" if m > threshold else "right" if m < -threshold else None)

    min_len = int(round(min_segment * fps))
    segments = [(a, b, lab) for a, b, lab in _runs(labels) if lab is not None and b - a >= min_len]

    lead: list[str | None] = [None] * n
    for a, b, lab in segments:
        lead[a:b] = [lab] * (b - a)

    def fill(a: int, b: int, lab: str) -> None:
        # 空檔裡只要出現相反方向的跡象就不補（可能是換邊）
        if all(x in (None, lab) for x in labels[a:b]):
            lead[a:b] = [lab] * (b - a)

    if segments:
        fill(0, segments[0][0], segments[0][2])
        fill(segments[-1][1], n, segments[-1][2])
        for (a0, b0, l0), (a1, b1, l1) in zip(segments, segments[1:]):
            if l0 == l1:
                fill(b0, a1, l0)

    if user:
        conf = 1.0
    else:
        dur = {"left": 0, "right": 0}
        for a, b, lab in segments:
            dur[lab] += b - a
        total = sum(dur.values())
        best = max(dur, key=dur.get)
        conf = dur[best] / total if total else 0.0
        if total >= 3 * fps and conf >= 0.7:
            natural = "regular" if best == "left" else "goofy"
        else:
            natural = None
    if not segments and natural is not None:
        # 完全沒有方向證據：假設以慣用站姿滑行
        lead = [front[natural]] * n
    return StanceResult(natural, conf, votes, "user" if user else "auto", lead, segments)


# ---------------------------------------------------------------------------
# 轉彎切分：重心越過板子（傾角正負號改變）就是換刃
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    index: int
    side: str           # "toe" / "heel"
    start: int          # 影格序（在分析序列中的位置，含）
    end: int            # 不含
    start_time: float
    end_time: float
    stats: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def edge_labels(incl: np.ndarray, on: float = 6.0, off: float = 3.0,
                max_gap: int = 3) -> np.ndarray:
    """每格的立刃狀態：1 = toe、-1 = heel、0 = 平板、NaN 缺值。

    使用遲滯（超過 on 才進入立刃、低於 off 才離開），短暫缺值（<= max_gap 格）沿用前一格。
    """
    out = np.full(len(incl), np.nan)
    state = 0
    gap = 0
    for i, v in enumerate(incl):
        if np.isnan(v):
            gap += 1
            if gap <= max_gap and i > 0 and not np.isnan(out[i - 1]):
                out[i] = out[i - 1]
            continue
        gap = 0
        if state == 0:
            if v >= on:
                state = 1
            elif v <= -on:
                state = -1
        elif state == 1 and v < off:
            state = -1 if v <= -on else 0
        elif state == -1 and v > -off:
            state = 1 if v >= on else 0
        out[i] = state
    return out


def segment_turns(incl: np.ndarray, times: np.ndarray, min_duration: float = 0.5) -> tuple[list[Turn], np.ndarray]:
    """回傳轉彎清單，以及每格的轉彎編號（不在轉彎中為 -1）。"""
    labels = edge_labels(incl)
    turn_id = np.full(len(incl), -1)
    turns: list[Turn] = []
    i = 0
    n = len(labels)
    while i < n:
        lab = labels[i]
        if np.isnan(lab) or lab == 0:
            i += 1
            continue
        j = i
        while j < n and labels[j] == lab:
            j += 1
        t0, t1 = times[i], times[j - 1]
        if t1 - t0 >= min_duration:
            turn = Turn(len(turns), "toe" if lab > 0 else "heel", i, j, float(t0), float(t1))
            turn_id[i:j] = turn.index
            turns.append(turn)
        i = j
    return turns, turn_id
