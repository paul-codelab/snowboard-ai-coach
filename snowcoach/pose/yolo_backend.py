"""Ultralytics YOLO11 後端（AGPL-3.0，選用）。

- ``YoloPose``：COCO 17 點 2D 姿勢，沒有腳跟/腳尖，也沒有 3D，指標會走 2D 備援算法。
- ``YoloPersonDetector``：只用來找人，搭配 MediaPipe 做姿勢，適合騎手很小的畫面。
"""

from __future__ import annotations

import numpy as np

from snowcoach.joints import COCO_INDEX, IDX, NUM_JOINTS
from snowcoach.models import model_dir
from snowcoach.pose.base import BBox, PersonDetector, PoseBackend, PoseFrame, in_frame_mask


def _load(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover
        raise SystemExit("需要安裝 ultralytics：pip install 'snowcoach[yolo]'") from e
    # 權重放在模型快取資料夾（不存在時 ultralytics 會自動下載到這個路徑）
    path = model_dir() / weights
    path.parent.mkdir(parents=True, exist_ok=True)
    return YOLO(str(path))


class YoloPose(PoseBackend):
    name = "yolo-pose"
    provides_3d = False
    has_feet = False

    def __init__(self, weights: str = "yolo11s-pose.pt"):
        self._model = _load(weights)

    def estimate(self, image: np.ndarray, roi: BBox | None) -> list[PoseFrame]:
        h, w = image.shape[:2]
        res = self._model.predict(image, verbose=False, device="cpu", imgsz=960)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return []
        kxy = res.keypoints.xy.cpu().numpy()
        kconf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])
        boxes = res.boxes.conf.cpu().numpy()
        out = []
        for p in range(len(kxy)):
            xy = np.full((NUM_JOINTS, 2), np.nan)
            conf = np.zeros(NUM_JOINTS)
            for name, ci in COCO_INDEX.items():
                xy[IDX[name]] = kxy[p, ci]
                conf[IDX[name]] = kconf[p, ci]
            xy[(xy == 0).all(axis=1)] = np.nan  # YOLO 以 (0, 0) 表示沒有預測
            inside = in_frame_mask(xy, w, h)
            conf = np.where(inside, conf, 0.0)
            out.append(PoseFrame(xy=xy, conf=conf, in_frame=inside,
                                 score=float(boxes[p]), backend=self.name))
        return out


class YoloPersonDetector(PersonDetector):
    def __init__(self, weights: str = "yolo11s.pt", score_threshold: float = 0.25):
        self._model = _load(weights)
        self._thr = score_threshold

    def detect(self, image: np.ndarray) -> list[tuple[BBox, float]]:
        res = self._model.predict(image, verbose=False, device="cpu", classes=[0], imgsz=960)[0]
        out = []
        for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            if c >= self._thr:
                out.append(((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c)))
        return out
