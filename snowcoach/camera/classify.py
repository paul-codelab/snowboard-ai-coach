"""判斷影片是自拍（自拍桿/隱形自拍桿）還是他人拍攝（跟拍、定點）。

依據三個特徵（每個轉成 -1~1 分，取平均）：
1. 騎手在畫面中的大小：自拍時相機距離約一臂加桿長，騎手很大；他人拍攝通常較小。
2. 騎手大小的穩定度：自拍時相機和人一起移動，大小幾乎不變；他人拍攝變化大。
3. 有一隻手伸向鏡頭：自拍時持桿的手腕明顯比肩膀更靠近相機（3D 深度）。

分數 > 0.25 判為自拍，< -0.25 判為他人拍攝，中間視為不確定（以他人拍攝處理並提示用參數指定）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from snowcoach.joints import IDX
from snowcoach.pose.base import PoseFrame


@dataclass
class SourceResult:
    source: str                 # "selfie" / "follow"
    confidence: float           # 0~1
    decided_by: str             # "user" / "auto" / "auto-uncertain"
    stick_arm: str | None = None  # 持桿手（"left"/"right"），僅自拍
    features: dict = field(default_factory=dict)


def _clip(v: float) -> float:
    return float(np.clip(v, -1.0, 1.0))


def _wrist_forward(p: PoseFrame) -> dict[str, float] | None:
    """各手腕比肩膀中點靠近相機多少（公尺，正 = 較近）。"""
    if p.world is None:
        return None
    W = p.world
    sh_z = (W[IDX["left_shoulder"], 2] + W[IDX["right_shoulder"], 2]) / 2
    return {s: float(sh_z - W[IDX[f"{s}_wrist"], 2]) for s in ("left", "right")}


def classify_source(poses: list[PoseFrame | None], frame_height: int) -> SourceResult:
    heights, fwd = [], {"left": [], "right": []}
    for p in poses:
        if p is None:
            continue
        b = p.bbox(0.3)
        if b is None:
            continue
        heights.append((b[3] - b[1]) / frame_height)
        wf = _wrist_forward(p)
        if wf:
            for s in wf:
                fwd[s].append(wf[s])
    if len(heights) < 10:
        return SourceResult("follow", 0.0, "auto-uncertain", features={"frames": len(heights)})

    h = np.array(heights)
    h_med = float(np.median(h))
    h_cv = float(h.std() / max(h.mean(), 1e-6))
    feats = {"rider_height": round(h_med, 3), "height_cv": round(h_cv, 3)}
    scores = [
        _clip((h_med - 0.18) / 0.12),   # 0.30 以上 → +1，0.06 以下 → -1
        _clip((0.35 - h_cv) / 0.2),     # 大小穩定 → +1
    ]
    stick_arm = None
    if fwd["left"]:
        med = {s: float(np.median(v)) for s, v in fwd.items()}
        arm = max(med, key=med.get)
        feats["wrist_forward_m"] = round(med[arm], 3)
        scores.append(_clip((med[arm] - 0.2) / 0.12))  # 0.32 m 以上 → +1
        stick_arm = arm
    score = float(np.mean(scores))
    feats["score"] = round(score, 3)
    if score > 0.25:
        return SourceResult("selfie", min(1.0, score), "auto", stick_arm, feats)
    if score < -0.25:
        return SourceResult("follow", min(1.0, -score), "auto", None, feats)
    return SourceResult("follow", abs(score), "auto-uncertain", None, feats)


def stick_arm_for(poses: list[PoseFrame | None]) -> str | None:
    """使用者指定自拍時，仍要找出持桿手。"""
    fwd = {"left": [], "right": []}
    for p in poses:
        wf = _wrist_forward(p) if p is not None else None
        if wf:
            for s in wf:
                fwd[s].append(wf[s])
    if not fwd["left"]:
        return None
    med = {s: float(np.median(v)) for s, v in fwd.items()}
    return max(med, key=med.get)
