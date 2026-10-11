"""模型檔下載與快取。預設存在 ~/.cache/snowcoach，可用 SNOWCOACH_MODELS 改位置。"""

from __future__ import annotations

import os
import shutil
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

_MP = "https://storage.googleapis.com/mediapipe-models"
MODEL_URLS: dict[str, str] = {
    "pose_landmarker_lite.task": f"{_MP}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": f"{_MP}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": f"{_MP}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "efficientdet_lite0.tflite": f"{_MP}/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite",
}


# 預設設定（MediaPipe full 姿勢模型＋找人用的偵測器）需要的模型；先下載好就能離線使用
DEFAULT_MODELS = ["pose_landmarker_full.task", "efficientdet_lite0.tflite"]


def model_dir() -> Path:
    return Path(os.environ.get("SNOWCOACH_MODELS", Path.home() / ".cache" / "snowcoach"))


def _open_url(url: str):
    try:
        return urllib.request.urlopen(url, timeout=60)
    except urllib.error.URLError as e:
        # 部分 Python 發行版（例如 macOS 上的）找不到系統憑證，改用 certifi 的憑證再試一次
        if not isinstance(getattr(e, "reason", None), ssl.SSLError):
            raise
        try:
            import certifi
        except ImportError:
            raise e from None
        ctx = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(url, timeout=60, context=ctx)


def ensure_model(name: str) -> Path:
    """回傳模型檔路徑，不存在時下載。"""
    path = model_dir() / name
    if path.exists() and path.stat().st_size > 0:
        return path
    if name not in MODEL_URLS:
        raise FileNotFoundError(f"找不到模型檔 {path}，也沒有已知的下載網址")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"下載模型 {name} ...", file=sys.stderr)
    with _open_url(MODEL_URLS[name]) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(path)
    return path


def download_defaults() -> list[Path]:
    return [ensure_model(n) for n in DEFAULT_MODELS]
