"""MediaPipe Pose Landmarker 後端（Apache-2.0）。

- 33 點，含腳跟與腳尖，能直接建立板子座標系。
- 同時輸出 world landmarks（以髖部為原點的 3D 座標，公尺），用來算 3D 指標。
- 以騎手為中心裁切後再偵測，讓畫面中較小的騎手也能偵測。
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from snowcoach.joints import JOINTS, MEDIAPIPE_INDEX
from snowcoach.models import ensure_model
from snowcoach.pose.base import BBox, PersonDetector, PoseBackend, PoseFrame, in_frame_mask

_MP_IDX = np.array([MEDIAPIPE_INDEX[n] for n in JOINTS])
_UP = np.array([0.0, -1.0, 0.0])


def square_crop(image: np.ndarray, roi: BBox, scale: float = 1.6,
                min_side: int = 160, max_input: int = 640) -> tuple[np.ndarray, float, float, float]:
    """以 roi 為中心切出正方形（超出畫面的部分補黑）。

    回傳 (crop, x0, y0, s)：crop 中的 (u, v) 對應原圖 (x0 + u * s, y0 + v * s)。
    """
    h, w = image.shape[:2]
    cx, cy = (roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2
    side = max(roi[2] - roi[0], roi[3] - roi[1]) * scale
    side = int(round(max(side, min_side)))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    crop = np.zeros((side, side, 3), dtype=image.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    s = 1.0
    if side > max_input:
        s = side / max_input
        crop = cv2.resize(crop, (max_input, max_input), interpolation=cv2.INTER_AREA)
    return crop, float(x0), float(y0), s


class MediaPipePose(PoseBackend):
    name = "mediapipe"
    provides_3d = True
    has_feet = True

    def __init__(self, model: str = "full", num_poses: int = 2):
        path = ensure_model(f"pose_landmarker_{model}.task")
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opts)

    def estimate(self, image: np.ndarray, roi: BBox | None) -> list[PoseFrame]:
        h, w = image.shape[:2]
        if roi is None:
            crop, x0, y0, s = image, 0.0, 0.0, 1.0
            if max(h, w) > 1280:
                s = max(h, w) / 1280
                crop = cv2.resize(image, (int(w / s), int(h / s)), interpolation=cv2.INTER_AREA)
        else:
            crop, x0, y0, s = square_crop(image, roi)
        ch, cw = crop.shape[:2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        out: list[PoseFrame] = []
        for i, lms in enumerate(res.pose_landmarks):
            arr = np.array([[lm.x, lm.y, lm.visibility if lm.visibility is not None else 0.0,
                             lm.presence if lm.presence is not None else 0.0] for lm in lms])[_MP_IDX]
            xy = np.stack([x0 + arr[:, 0] * cw * s, y0 + arr[:, 1] * ch * s], axis=1)
            conf = np.minimum(arr[:, 2], arr[:, 3]) if arr[:, 3].any() else arr[:, 2]
            inside = in_frame_mask(xy, w, h)
            conf = np.where(inside, conf, 0.0)
            world = None
            if i < len(res.pose_world_landmarks):
                world = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[i]])[_MP_IDX]
            # world 座標軸與相機一致（y 朝下）；假設相機水平，重力反方向即 -y
            out.append(PoseFrame(xy=xy, conf=conf, world=world, in_frame=inside, up=_UP,
                                 score=float(np.mean(conf)), backend=self.name))
        return out

    def close(self) -> None:
        self._lm.close()


class MediaPipePersonDetector(PersonDetector):
    """EfficientDet-Lite0 人物偵測（Apache-2.0），追蹤遺失時用來重新找騎手。"""

    def __init__(self, score_threshold: float = 0.25):
        path = ensure_model("efficientdet_lite0.tflite")
        opts = vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=score_threshold,
            category_allowlist=["person"],
            max_results=10,
        )
        self._det = vision.ObjectDetector.create_from_options(opts)

    def detect(self, image: np.ndarray) -> list[tuple[BBox, float]]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = self._det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        out = []
        for d in res.detections:
            b = d.bounding_box
            out.append(((float(b.origin_x), float(b.origin_y),
                         float(b.origin_x + b.width), float(b.origin_y + b.height)),
                        float(d.categories[0].score)))
        return out

    def close(self) -> None:
        self._det.close()

