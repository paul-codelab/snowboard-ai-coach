"""SAM 3D Body 後端（尚未實作，保留介面）。

SAM 3D Body 需要 CUDA GPU 與另外安裝（模型權重需向 Meta 申請，授權另計），
目前環境只有 CPU，所以這裡只定義接法。實作時要做的事：

1. 載入模型一次，跨影格重複使用。
2. ``estimate(image, roi)``：以 roi（沒有時先用人物偵測）當提示框跑模型，
   回傳 ``PoseFrame``：
   - ``xy``：把 3D 關節投影回畫面的 2D 座標（像素），轉成 ``snowcoach.joints.JOINTS`` 的順序；
     MHR70 有腳跟與腳尖，對應 ``left_heel`` / ``left_toe`` 等。
   - ``conf``：沒有逐點信心時，可用「關節是否在畫面內且未被遮擋」估計，畫面外一律 0。
   - ``world``：3D 關節（公尺），相機座標即可。
   - ``up``：若相機水平可用相機座標的「上」；模型若提供重力方向則用它。
3. ``provides_3d = True``、``has_feet = True``，指標計算、站姿推斷、轉彎切分都不用改。

使用方式：``snowcoach analyze VIDEO --backend sam3d``
"""

from __future__ import annotations

import numpy as np

from snowcoach.pose.base import BBox, PoseBackend, PoseFrame


class Sam3DBodyBackend(PoseBackend):
    name = "sam3d"
    provides_3d = True
    has_feet = True

    def __init__(self) -> None:
        try:
            import sam_3d_body  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "SAM 3D Body 後端尚未實作，且需要 GPU 與另外安裝 sam_3d_body。"
                "目前請使用 --backend mediapipe。") from e
        raise SystemExit("SAM 3D Body 後端尚未實作；介面說明見 snowcoach/pose/sam3d_backend.py。")

    def estimate(self, image: np.ndarray, roi: BBox | None) -> list[PoseFrame]:  # pragma: no cover
        raise NotImplementedError
