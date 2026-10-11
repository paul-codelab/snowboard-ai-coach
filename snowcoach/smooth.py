"""時間序列清理：去除單格跳動（Hampel）＋ 短窗平滑。缺值保持缺值，不跨缺口內插。"""

from __future__ import annotations

import numpy as np


def hampel(x: np.ndarray, half: int = 3, n_sigma: float = 3.0) -> np.ndarray:
    """偏離鄰近中位數超過 n_sigma 倍 MAD 的點視為跳動，設為 NaN。"""
    x = np.asarray(x, dtype=float)
    out = x.copy()
    n = len(x)
    for i in range(n):
        if np.isnan(x[i]):
            continue
        w = x[max(0, i - half):min(n, i + half + 1)]
        w = w[~np.isnan(w)]
        if len(w) < 3:
            continue
        med = np.median(w)
        mad = 1.4826 * np.median(np.abs(w - med))
        if mad > 0 and abs(x[i] - med) > n_sigma * mad:
            out[i] = np.nan
    return out


def smooth(x: np.ndarray, half: int = 2, min_count: int = 2) -> np.ndarray:
    """忽略 NaN 的置中移動平均；原本就是 NaN 的位置維持 NaN。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(x[i]):
            continue
        k = min(half, i, n - 1 - i)  # 頭尾縮成對稱窗口，避免偏差
        w = x[i - k:i + k + 1]
        w = w[~np.isnan(w)]
        out[i] = w.mean() if len(w) >= min(min_count, 1 + 2 * k) else x[i]
    return out


def clean(x: np.ndarray, fps: float) -> np.ndarray:
    """依影格率決定視窗：Hampel 約 0.25 秒、平滑約 0.1 秒。"""
    h = max(2, int(round(fps * 0.125)))
    s = max(1, int(round(fps * 0.05)))
    return smooth(hampel(x, half=h), half=s)
