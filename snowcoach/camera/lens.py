"""鏡頭變形校正。

- ``none``：不校正。Insta360 X 系列重新構圖（reframe）後輸出、手機一般鏡頭都用這個。
- ``gopro-wide`` / ``gopro-superview``：以等距魚眼模型近似 GoPro 超廣角（參數為近似值）。
- ``fisheye:<對角視角度數>``：自訂等距魚眼，例如 ``fisheye:150``。
- ``calib:<檔案.json>``：用 OpenCV 校正得到的參數
  ``{"model": "fisheye"|"pinhole", "K": 3x3, "D": [...], "size": [w, h]}``。

校正作用在整張畫面上（姿勢模型的 3D 推估也需要看到校正後的影像），
輸出與輸入同尺寸，之後的偵測、指標、疊圖都在校正後的畫面上進行。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import cv2
import numpy as np

PRESETS = {
    "gopro-wide": 150.0,
    "gopro-superview": 160.0,
}


@dataclass
class Lens:
    name: str
    model: str = "none"          # none / fisheye / pinhole
    dfov: float | None = None    # 對角視角（度），fisheye 預設用
    K: np.ndarray | None = None
    D: np.ndarray | None = None
    calib_size: tuple[int, int] | None = None
    _maps: tuple | None = None

    @property
    def active(self) -> bool:
        return self.model != "none"

    def prepare(self, width: int, height: int) -> None:
        if not self.active:
            return
        K, D = self._intrinsics(width, height)
        size = (width, height)
        if self.model == "fisheye":
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, size, np.eye(3), balance=0.0)
            self._maps = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, size, cv2.CV_16SC2)
        else:
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, size, 0.0)
            self._maps = cv2.initUndistortRectifyMap(K, D, None, newK, size, cv2.CV_16SC2)

    def _intrinsics(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        if self.K is not None:
            K = self.K.astype(float).copy()
            if self.calib_size and tuple(self.calib_size) != (width, height):
                sx, sy = width / self.calib_size[0], height / self.calib_size[1]
                K[0] *= sx
                K[1] *= sy
            D = self.D if self.D is not None else np.zeros(4)
            return K, np.asarray(D, dtype=float)
        # 等距魚眼：r = f·θ，對角半長對應半視角
        assert self.dfov is not None
        half_diag = math.hypot(width, height) / 2
        f = half_diag / math.radians(self.dfov / 2)
        K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=float)
        return K, np.zeros(4)

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if not self.active:
            return frame
        if self._maps is None:
            self.prepare(frame.shape[1], frame.shape[0])
        assert self._maps is not None
        return cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR)


def parse_lens(spec: str) -> Lens:
    spec = spec.strip()
    if spec in ("none", "", "insta360", "phone"):
        return Lens(name=spec or "none")
    if spec in PRESETS:
        return Lens(name=spec, model="fisheye", dfov=PRESETS[spec])
    if spec.startswith("fisheye:"):
        return Lens(name=spec, model="fisheye", dfov=float(spec.split(":", 1)[1]))
    if spec.startswith("calib:"):
        data = json.loads(open(spec.split(":", 1)[1], encoding="utf-8").read())
        return Lens(
            name=spec,
            model=data.get("model", "fisheye"),
            K=np.array(data["K"], dtype=float),
            D=np.array(data.get("D", [0, 0, 0, 0]), dtype=float),
            calib_size=tuple(data["size"]) if "size" in data else None,
        )
    raise ValueError(f"無法辨識的鏡頭設定：{spec}（可用 none、gopro-wide、gopro-superview、fisheye:度數、calib:檔案）")
