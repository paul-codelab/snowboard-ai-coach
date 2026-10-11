"""鎖定主要騎手：以前一格的位置為中心偵測，遺失時再用人物偵測器重新尋找。

路人、剪輯加上的子母畫面小窗都靠「與前一格重疊度」排除；一開始沒有前一格時選最大的人。
另外每隔幾格檢查一次畫面中是否出現明顯更大的人（連續兩次確認才切換），
避免一開始鎖錯到子母畫面或遠處路人後就一直跟錯。
"""

from __future__ import annotations

import numpy as np

from snowcoach.pose.base import BBox, PersonDetector, PoseBackend, PoseFrame


def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def select_primary(cands: list[PoseFrame], prev: BBox | None) -> PoseFrame | None:
    """從候選中挑出主要騎手。有前一格時要求重疊，否則挑面積最大的。"""
    scored = []
    for c in cands:
        b = c.bbox()
        if b is None or c.score < 0.2:
            continue
        if prev is not None:
            ov = iou(b, prev)
            if ov < 0.1:
                continue
            scored.append((ov + 0.2 * c.score, c))
        else:
            scored.append((area(b) * (0.5 + c.score), c))
    if not scored:
        return None
    return max(scored, key=lambda t: t[0])[1]


def pick_box(boxes: list[tuple[BBox, float]], prev: BBox | None) -> BBox | None:
    if not boxes:
        return None
    if prev is not None:
        best = max(boxes, key=lambda bs: iou(bs[0], prev))
        if iou(best[0], prev) > 0.05:
            return best[0]
    return max(boxes, key=lambda bs: area(bs[0]) * bs[1])[0]


class RiderTracker:
    def __init__(self, backend: PoseBackend, detector: PersonDetector | None,
                 max_lost: int = 15, recheck_every: int = 10, switch_ratio: float = 2.5):
        self.backend = backend
        self.detector = detector
        self.max_lost = max_lost
        self.recheck_every = recheck_every
        self.switch_ratio = switch_ratio
        self.prev: BBox | None = None
        self.lost = 0
        self._n = 0
        self._pending = 0

    def _larger_rider(self, frame: np.ndarray) -> PoseFrame | None:
        """畫面中有明顯更大、且不重疊的人時，回傳他的姿勢。"""
        assert self.prev is not None and self.detector is not None
        big = [b for b, _ in self.detector.detect(frame)
               if area(b) > self.switch_ratio * area(self.prev) and iou(b, self.prev) < 0.1]
        if not big:
            return None
        box = max(big, key=area)
        return select_primary(self.backend.estimate(frame, box), box)

    def update(self, frame: np.ndarray) -> PoseFrame | None:
        self._n += 1
        pose = None
        if self.prev is not None and self.detector is not None and self._n % self.recheck_every == 0:
            cand = self._larger_rider(frame)
            self._pending = self._pending + 1 if cand is not None else 0
            if self._pending >= 2:
                self._pending = 0
                self.lost = 0
                self.prev = cand.bbox()
                return cand
        if self.prev is not None:
            pose = select_primary(self.backend.estimate(frame, self.prev), self.prev)
        if pose is None:
            ref = self.prev if self.lost <= self.max_lost else None
            box = pick_box(self.detector.detect(frame), ref) if self.detector else None
            if box is not None:
                pose = select_primary(self.backend.estimate(frame, box), box)
            if pose is None:
                pose = select_primary(self.backend.estimate(frame, None), ref)
        if pose is None:
            self.lost += 1
            if self.lost > self.max_lost:
                self.prev = None
            return None
        self.lost = 0
        self.prev = pose.bbox()
        return pose
