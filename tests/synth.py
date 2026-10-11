"""產生已知答案的合成 3D 姿勢，用來驗證幾何計算。"""

from __future__ import annotations

import numpy as np

from snowcoach.joints import IDX, NUM_JOINTS
from snowcoach.pose.base import PoseFrame


def rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def random_rotation(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def make_pose(stance: str = "regular", knee: float = 40.0, lean: float = 0.0,
              upper_lean: float | None = None, fore_aft: float = 0.0, shoulder: float = 0.0,
              width: float = 0.5, R: np.ndarray | None = None, head: str = "nose") -> PoseFrame:
    """在板子座標（x = 板頭、y = 腳尖側、z = 上）建立姿勢，再用 R 轉到任意相機座標。

    lean：整個人（含板子）繞板軸傾倒的角度（+ = 腳尖側），即期望的身體傾倒。
    upper_lean：上半身相對鉛直的傾角；None 表示與下半身相同（沒有折髖）。
    head：騎手頭部轉向，"nose" = 看向板頭。
    """
    P = np.zeros((NUM_JOINTS, 3))
    # 面向 +y（腳尖側）、頭頂 +z 時，騎手的左邊是 -x
    xs = {"left": -width / 2, "right": width / 2}
    front = "left" if stance == "regular" else "right"
    nose_dir = np.array([np.sign(xs[front]), 0.0, 0.0])
    L = 0.45
    a = np.radians(knee / 2)
    hips = {}
    for s in ("left", "right"):
        ankle = np.array([xs[s], 0.0, 0.08])
        P[IDX[f"{s}_ankle"]] = ankle
        P[IDX[f"{s}_heel"]] = [xs[s], -0.12, 0.02]
        P[IDX[f"{s}_toe"]] = [xs[s], 0.12, 0.02]
        kneep = ankle + L * np.array([0, np.sin(a), np.cos(a)])
        P[IDX[f"{s}_knee"]] = kneep
        hips[s] = kneep + L * np.array([0, -np.sin(a), np.cos(a)])
    hip_h = hips["left"][2]
    for s in ("left", "right"):
        P[IDX[f"{s}_hip"]] = hips[s]
    shift = fore_aft * width * nose_dir
    hip_mid = np.array([0, 0, hip_h])

    # 上半身：先在直立狀態擺好，再繞髖部依折髖旋轉
    phi = np.radians(shoulder)
    half = 0.2
    up_torso = np.array([0, 0, 0.5])
    sline = half * (np.cos(phi) * nose_dir + np.array([0, -np.sin(phi), 0]))
    back = "right" if front == "left" else "left"
    sh = {front: hip_mid + up_torso + sline, back: hip_mid + up_torso - sline}
    for s in ("left", "right"):
        P[IDX[f"{s}_shoulder"]] = sh[s]
        P[IDX[f"{s}_elbow"]] = sh[s] + [0, 0.1, -0.25]
        P[IDX[f"{s}_wrist"]] = sh[s] + [0, 0.2, -0.45]
    head_c = hip_mid + np.array([0, 0, 0.75])
    look = nose_dir if head == "nose" else np.array([0, 1.0, 0])
    P[IDX["nose"]] = head_c + 0.1 * look
    # 耳朵在頭部兩側：看向 look、頭頂 +z 時，左邊為 z × look
    left = np.cross([0, 0, 1], look)
    P[IDX["left_ear"]] = head_c + 0.08 * left
    P[IDX["right_ear"]] = head_c - 0.08 * left
    P[IDX["left_eye"]] = P[IDX["nose"]] + 0.03 * left
    P[IDX["right_eye"]] = P[IDX["nose"]] - 0.03 * left

    upper_ids = [IDX[n] for n in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                  "left_wrist", "right_wrist", "nose", "left_ear", "right_ear",
                                  "left_eye", "right_eye")]
    # 前後重心：整個上半身與髖部沿板軸平移
    for n in ["left_hip", "right_hip"]:
        P[IDX[n]] = P[IDX[n]] + shift
    for i in upper_ids:
        P[i] = P[i] + shift
    hip_mid = hip_mid + shift
    if upper_lean is not None:
        # 上半身相對下半身的額外旋轉 = upper_lean - lean
        Ru = rot_x(-(upper_lean - lean))  # rot_x(+) 會把 +z 轉向 -y；取負號讓 + = 腳尖側
        for i in upper_ids:
            P[i] = hip_mid + Ru @ (P[i] - hip_mid)
    Rl = rot_x(-lean)
    P = P @ Rl.T
    up = np.array([0.0, 0.0, 1.0])
    if R is not None:
        P = P @ R.T
        up = R @ up
    xy = P[:, :2] * 500 + 500
    return PoseFrame(xy=xy, conf=np.ones(NUM_JOINTS), world=P,
                     in_frame=np.ones(NUM_JOINTS, dtype=bool), up=up, backend="synthetic")
