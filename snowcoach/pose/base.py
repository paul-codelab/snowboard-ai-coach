"""姿勢後端的共同介面。

新增後端（例如之後的 SAM 3D Body）只需要：
1. 實作 ``PoseBackend.estimate``，回傳 ``PoseFrame`` 候選清單；
2. 把關節轉成 ``snowcoach.joints.JOINTS`` 的順序；
3. 若有 3D 輸出，填入 ``PoseFrame.world``（單位公尺，右手座標系，任意相機朝向皆可，
   指標都在以雙腳建立的板子座標系中計算，與相機方向無關）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from snowcoach.joints import NUM_JOINTS

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1（像素）


@dataclass
class PoseFrame:
    xy: np.ndarray          # (J, 2) 畫面像素座標；未預測為 NaN
    conf: np.ndarray        # (J,) 0~1；畫面外的關節一律為 0
    world: np.ndarray | None = None  # (J, 3) 3D 座標（公尺）或 None
    in_frame: np.ndarray | None = None  # (J,) bool；用來區分「未入鏡」和「被遮擋」
    up: np.ndarray | None = None  # world 座標中的重力反方向；None = 未知
    score: float = 1.0
    backend: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PoseFrame":
        return cls(
            xy=np.full((NUM_JOINTS, 2), np.nan),
            conf=np.zeros(NUM_JOINTS),
            in_frame=np.zeros(NUM_JOINTS, dtype=bool),
        )

    def bbox(self, min_conf: float = 0.3) -> BBox | None:
        ok = self.conf >= min_conf
        if ok.sum() < 3:
            return None
        pts = self.xy[ok]
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))


class PoseBackend:
    """姿勢後端。``roi`` 為 None 時代表整張畫面。回傳值使用整張畫面的像素座標。"""

    name = "base"
    provides_3d = False
    has_feet = False  # 是否提供腳跟/腳尖

    def estimate(self, image: np.ndarray, roi: BBox | None) -> list[PoseFrame]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PersonDetector:
    """人物偵測器，用於追蹤遺失時重新找到騎手。"""

    def detect(self, image: np.ndarray) -> list[tuple[BBox, float]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


def in_frame_mask(xy: np.ndarray, width: int, height: int, margin: float = 0.005) -> np.ndarray:
    """關節是否在畫面內（扣掉邊界 margin，貼邊的點通常是被裁切後外插出來的）。"""
    mx, my = margin * width, margin * height
    with np.errstate(invalid="ignore"):
        inside = (
            (xy[:, 0] >= mx) & (xy[:, 0] <= width - 1 - mx)
            & (xy[:, 1] >= my) & (xy[:, 1] <= height - 1 - my)
        )
    return inside & ~np.isnan(xy).any(axis=1)
