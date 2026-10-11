"""板子座標系與每格指標。

板子本身不偵測，而是由雙腳推算（兩腳都固定在板上）：

- x̂（板軸）：後腳中心 → 前腳中心，指向板頭。
- ẑ（上方）：重力反方向扣掉沿板軸的分量。3D 後端若提供重力方向（``PoseFrame.up``，
  MediaPipe 假設相機水平：Insta360 水平鎖定、手機直拿時成立）就用它；
  沒有時改用雙腳推算的板面法線。
- ŷ（腳尖側）：與 x̂、ẑ 垂直，正負號由「腳跟→腳尖」與「膝蓋彎曲方向」決定
  （膝蓋只會往腳尖側彎，比單眼 3D 估出的腳尖點穩定）。

傾角、折髖以重力為基準（教學上的標準定義）；前後重心、肩膀對齊以板軸為基準，
與相機位置無關。
有 3D 座標時（MediaPipe world landmarks，之後的 SAM 3D Body）走 3D 算法；
只有 2D 時（YOLO-Pose）走 2D 備援算法，並依拍攝角度判斷哪些指標量得到。

每個指標都回傳 ``Measure``：量得到時有數值，量不到時 value 為 None 並附中文原因，
絕不用猜的補值。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snowcoach.joints import IDX, ZH
from snowcoach.pose.base import PoseFrame

MIN_CONF = 0.5
MAX_LEAN = 75.0  # 超過此傾角（度）視為姿勢估計失敗
MAX_ANGULATION = 60.0
MIN_RIDER_PX = 160  # 騎手（可見關節範圍）高度低於此像素數時，姿勢估計不可靠

# 基本指標（直接由姿勢算出）
BASE_METRICS = ["knee_left", "knee_right", "inclination", "angulation", "fore_aft", "shoulder"]


@dataclass
class Measure:
    value: float | None
    reason: str | None = None  # value 為 None 時的原因

    @property
    def ok(self) -> bool:
        return self.value is not None


def _missing_reason(pose: PoseFrame, names: list[str]) -> str | None:
    """names 中有任何關節不可用時，回傳中文原因（區分未入鏡與被遮擋）。"""
    out_frame, occluded = [], []
    for n in names:
        i = IDX[n]
        if pose.conf[i] >= MIN_CONF:
            continue
        if pose.in_frame is not None and not pose.in_frame[i]:
            out_frame.append(ZH[n])
        else:
            occluded.append(ZH[n])
    parts = []
    if out_frame:
        parts.append("、".join(out_frame) + "未入鏡")
    if occluded:
        parts.append("、".join(occluded) + "被遮擋或看不清")
    return "；".join(parts) if parts else None


def _usable(pose: PoseFrame, name: str) -> bool:
    return pose.conf[IDX[name]] >= MIN_CONF


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """b 點的夾角（度）。"""
    u, v = a - b, c - b
    cosv = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))


def front_side(stance: str) -> str:
    """regular：左腳在前；goofy：右腳在前。"""
    return "left" if stance == "regular" else "right"


@dataclass
class BoardFrame:
    center: np.ndarray   # 兩腳中心的中點
    x: np.ndarray        # 板軸（有站姿時指向板頭；沒有時為左腳→右腳）
    y: np.ndarray        # 腳尖側方向（垂直於板軸）
    z: np.ndarray        # 參考「上方」：重力反方向（預設）或板面法線
    stance_width: float
    oriented: bool       # x 是否已依站姿定向
    reference: str       # "gravity" / "board"


def _foot_center(P: np.ndarray, pose: PoseFrame, side: str) -> np.ndarray | None:
    pts = [P[IDX[f"{side}_{p}"]] for p in ("ankle", "heel", "toe") if _usable(pose, f"{side}_{p}")]
    return np.mean(pts, axis=0) if pts else None


def _toe_evidence(P: np.ndarray, pose: PoseFrame) -> np.ndarray | None:
    """腳尖側方向的證據：腳跟→腳尖，以及膝蓋彎曲的方向（膝蓋只會往腳尖側彎）。"""
    ev = []
    for s in ("left", "right"):
        if _usable(pose, f"{s}_toe") and _usable(pose, f"{s}_heel"):
            v = _unit(P[IDX[f"{s}_toe"]] - P[IDX[f"{s}_heel"]])
            if v is not None:
                ev.append(v)
        if all(_usable(pose, f"{s}_{j}") for j in ("hip", "knee", "ankle")):
            hip, knee, ank = P[IDX[f"{s}_hip"]], P[IDX[f"{s}_knee"]], P[IDX[f"{s}_ankle"]]
            v = _unit(knee - (hip + ank) / 2)
            if v is not None and 180.0 - _angle_at(hip, knee, ank) > 10.0:
                ev.append(v)
    return np.sum(ev, axis=0) if ev else None


def board_frame(pose: PoseFrame, P: np.ndarray, stance: str | None,
                up: np.ndarray | None = None) -> tuple[BoardFrame | None, str | None]:
    """由雙腳建立板子座標系。P 為 (J, 3) 座標；up 為重力反方向（None 時用板面法線）。"""
    reason = _missing_reason(pose, ["left_hip", "right_hip"])
    if reason:
        return None, reason
    left, right = _foot_center(P, pose, "left"), _foot_center(P, pose, "right")
    if left is None or right is None:
        missing = [n for n, c in (("左腳", left), ("右腳", right)) if c is None]
        outside = all(
            pose.in_frame is not None and not pose.in_frame[IDX[f"{'left' if n == '左腳' else 'right'}_{j}"]]
            for n in missing for j in ("ankle", "heel", "toe"))
        return None, "、".join(missing) + ("未入鏡" if outside else "被遮擋或看不清")
    if stance is None:
        axis = right - left
    else:
        axis = (left - right) if front_side(stance) == "left" else (right - left)
    width = float(np.linalg.norm(axis))
    x = _unit(axis)
    if x is None:
        return None, "兩腳位置重疊，無法判斷板軸"

    hip_mid = (P[IDX["left_hip"]] + P[IDX["right_hip"]]) / 2
    center = (left + right) / 2
    leg = np.mean([np.linalg.norm(hip_mid - left), np.linalg.norm(hip_mid - right)])
    if not 0.2 < width / max(leg, 1e-6) < 1.2:
        return None, "姿勢估計不合理（兩腳距離與腿長不符）"

    ev = _toe_evidence(P, pose)
    if ev is None:
        return None, "腳尖方向與膝蓋都看不清，無法判斷 toe/heel 側"

    if up is not None:
        z = _unit(up - np.dot(up, x) * x)
        if z is None:
            return None, "板軸接近垂直，無法判斷"
        y = np.cross(z, x)
        if np.dot(y, ev) < 0:
            y = -y
        return BoardFrame(center, x, y, z, width, stance is not None, "gravity"), None

    y = _unit(ev - np.dot(ev, x) * x)
    if y is None:
        return None, "腳尖方向與板軸平行，無法判斷板子朝向"
    z = np.cross(x, y)
    if np.dot(hip_mid - center, z) < 0:
        z = -z
    return BoardFrame(center, x, y, z, width, stance is not None, "board"), None


def center_of_mass(P: np.ndarray) -> np.ndarray:
    """軀幹質心近似：髖部中點往肩膀方向 30%。"""
    hip = (P[IDX["left_hip"]] + P[IDX["right_hip"]]) / 2
    sh = (P[IDX["left_shoulder"]] + P[IDX["right_shoulder"]]) / 2
    return hip + 0.3 * (sh - hip)


def _lean(v: np.ndarray, bf: BoardFrame) -> float:
    """向量在垂直板軸的平面上、相對參考上方的傾角（度），正 = 往腳尖側。"""
    return float(np.degrees(np.arctan2(np.dot(v, bf.y), np.dot(v, bf.z))))


def metrics_3d(pose: PoseFrame, stance: str | None) -> dict[str, Measure]:
    assert pose.world is not None
    P = pose.world
    out: dict[str, Measure] = {}

    for side in ("left", "right"):
        names = [f"{side}_hip", f"{side}_knee", f"{side}_ankle"]
        r = _missing_reason(pose, names)
        if r:
            out[f"knee_{side}"] = Measure(None, r)
        else:
            ang = _angle_at(P[IDX[names[0]]], P[IDX[names[1]]], P[IDX[names[2]]])
            out[f"knee_{side}"] = Measure(180.0 - ang)

    bf, r = board_frame(pose, P, stance, pose.up)
    if bf is None:
        for k in ("inclination", "angulation", "fore_aft", "shoulder"):
            out[k] = Measure(None, r)
        return out
    torso_r = _missing_reason(pose, ["left_shoulder", "right_shoulder"])
    if torso_r:
        for k in ("inclination", "angulation", "fore_aft", "shoulder"):
            out[k] = Measure(None, torso_r)
        return out

    com = center_of_mass(P)
    hip = (P[IDX["left_hip"]] + P[IDX["right_hip"]]) / 2
    sh = (P[IDX["left_shoulder"]] + P[IDX["right_shoulder"]]) / 2
    incl = _lean(com - bf.center, bf)
    lower, upper = _lean(hip - bf.center, bf), _lean(sh - hip, bf)
    if abs(incl) > MAX_LEAN or abs(lower) > MAX_LEAN or abs(upper) > MAX_LEAN + 20:
        bad = "姿勢估計不合理（傾角超出人體可能範圍）"
        out["inclination"] = Measure(None, bad)
        out["angulation"] = Measure(None, bad)
    else:
        out["inclination"] = Measure(incl)
        if abs(lower) < 5.0:
            out["angulation"] = Measure(None, "身體沒有傾倒，折髖無從比較")
        else:
            # 正值 = 上半身比下半身直（有折髖）；負值 = 上半身倒得比腿還多（整個人往內倒）
            ang = float(np.sign(lower) * (lower - upper))
            out["angulation"] = Measure(ang) if abs(ang) <= MAX_ANGULATION else \
                Measure(None, "姿勢估計不合理（折髖角度超出人體可能範圍）")

    if stance is None:
        out["fore_aft"] = Measure(None, "站姿未確定（請用 --stance 指定）")
        out["shoulder"] = Measure(None, "站姿未確定（請用 --stance 指定）")
        return out

    fa = float(np.dot(com - bf.center, bf.x) / bf.stance_width)
    out["fore_aft"] = Measure(fa) if abs(fa) <= 0.8 else Measure(None, "姿勢估計不合理（重心超出板子範圍）")
    f = front_side(stance)
    b = "right" if f == "left" else "left"
    s = P[IDX[f"{f}_shoulder"]] - P[IDX[f"{b}_shoulder"]]
    # 投影到水平面；開肩（胸口轉向板頭）時前肩往腳跟側移動，定義為正
    out["shoulder"] = Measure(float(np.degrees(np.arctan2(-np.dot(s, bf.y), np.dot(s, bf.x)))))
    return out


# ---------------------------------------------------------------------------
# 2D 備援算法（沒有 3D 座標的後端）
# ---------------------------------------------------------------------------

# 側面看板子（相機正對胸口或背部）時，兩腳距離 / 肩到腳的長度 約為此值
BROADSIDE_RATIO = 0.45


def view_angle_2d(pose: PoseFrame) -> float | None:
    """估計拍攝角度：0° = 從板子側面拍，90° = 從板頭/板尾方向拍。"""
    need = ["left_ankle", "right_ankle", "left_shoulder", "right_shoulder"]
    if _missing_reason(pose, need):
        return None
    P = pose.xy
    feet = P[IDX["right_ankle"]] - P[IDX["left_ankle"]]
    c = (P[IDX["right_ankle"]] + P[IDX["left_ankle"]]) / 2
    sh = (P[IDX["left_shoulder"]] + P[IDX["right_shoulder"]]) / 2
    body = float(np.linalg.norm(sh - c))
    if body < 1e-6:
        return None
    r = min(1.0, float(np.linalg.norm(feet)) / body / BROADSIDE_RATIO)
    return float(np.degrees(np.arccos(r)))


def metrics_2d(pose: PoseFrame, stance: str | None) -> dict[str, Measure]:
    P = pose.xy
    out: dict[str, Measure] = {}
    view = view_angle_2d(pose)
    for side in ("left", "right"):
        names = [f"{side}_hip", f"{side}_knee", f"{side}_ankle"]
        r = _missing_reason(pose, names)
        if r:
            out[f"knee_{side}"] = Measure(None, r)
        elif view is None or view < 45:
            out[f"knee_{side}"] = Measure(None, "拍攝角度：膝蓋朝向鏡頭，2D 量不到彎曲")
        else:
            out[f"knee_{side}"] = Measure(180.0 - _angle_at(P[IDX[names[0]]], P[IDX[names[1]]], P[IDX[names[2]]]))

    no3d = "2D 姿勢無法判斷（需要 3D 後端或腳跟/腳尖點）"
    out["inclination"] = Measure(None, no3d)
    out["angulation"] = Measure(None, no3d)
    out["shoulder"] = Measure(None, "2D 姿勢無法判斷肩膀旋轉")

    r = _missing_reason(pose, ["left_ankle", "right_ankle", "left_hip", "right_hip",
                               "left_shoulder", "right_shoulder"])
    if r:
        out["fore_aft"] = Measure(None, r)
    elif stance is None:
        out["fore_aft"] = Measure(None, "站姿未確定（請用 --stance 指定）")
    elif view is None or view > 45:
        out["fore_aft"] = Measure(None, "拍攝角度：從板頭/板尾方向拍，看不到前後")
    else:
        f = front_side(stance)
        b = "right" if f == "left" else "left"
        F, B = P[IDX[f"{f}_ankle"]], P[IDX[f"{b}_ankle"]]
        axis = F - B
        com = center_of_mass(np.c_[P, np.zeros(len(P))])[:2]
        out["fore_aft"] = Measure(float(np.dot(com - (F + B) / 2, axis) / np.dot(axis, axis)))
    return out


def compute_metrics(pose: PoseFrame | None, stance: str | None,
                    min_rider_px: float = MIN_RIDER_PX) -> dict[str, Measure]:
    if pose is None:
        return {k: Measure(None, "這一格沒有偵測到騎手") for k in BASE_METRICS}
    b = pose.bbox(0.3)
    if b is None or b[3] - b[1] < min_rider_px:
        return {k: Measure(None, f"騎手在畫面中太小（不到 {min_rider_px:.0f} 像素高），估計不可靠")
                for k in BASE_METRICS}
    if pose.world is not None:
        return metrics_3d(pose, stance)
    return metrics_2d(pose, stance)
