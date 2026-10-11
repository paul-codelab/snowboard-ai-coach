"""影片讀取與寫出。"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int          # 轉正後（播放時看到的）寬度
    height: int         # 轉正後的高度
    fps: float
    frame_count: int
    rotation: int = 0   # 讀到的影格還需順時針旋轉的角度（手機直拍常見 90 或 270）

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _open(path: str | os.PathLike) -> tuple[cv2.VideoCapture, bool]:
    """開啟影片並關掉 OpenCV 的自動旋轉，由我們自己依旋轉標記轉正（各版本行為不一致）。"""
    cap = cv2.VideoCapture(str(path))
    auto_off = bool(cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0))
    return cap, auto_off


def _ffmpeg_rotation(path: str | os.PathLike) -> int:
    """OpenCV 讀不到旋轉標記時，改用 ffmpeg 讀（displaymatrix 或舊式 rotate 標籤）。"""
    exe = _ffmpeg_exe()
    if not exe:
        return 0
    try:
        err = subprocess.run([exe, "-hide_banner", "-i", str(path)], capture_output=True,
                             text=True, errors="replace", timeout=30).stderr
    except Exception:
        return 0
    m = re.search(r"rotation of (-?[\d.]+) degrees", err)
    if m:  # displaymatrix 是逆時針角度
        return int(round(-float(m.group(1)))) % 360
    m = re.search(r"^\s*rotate\s*:\s*(-?\d+)", err, re.M)
    return int(m.group(1)) % 360 if m else 0


def _rotation(path: str | os.PathLike, cap: cv2.VideoCapture) -> int:
    deg = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)) % 360
    if deg == 0:
        deg = _ffmpeg_rotation(path)
    return deg if deg in _ROTATE else 0


def _to_display(frame: np.ndarray, info: VideoInfo) -> np.ndarray:
    """把讀到的影格轉成播放時的方向。"""
    return cv2.rotate(frame, _ROTATE[info.rotation]) if info.rotation else frame


def probe(path: str | os.PathLike) -> VideoInfo:
    cap, auto_off = _open(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片：{path}")
    rot = _rotation(path, cap)
    ok, frame = cap.read()
    if ok:  # 以實際讀到的影格尺寸為準
        fh, fw = frame.shape[:2]
    else:
        fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 無法關掉 OpenCV 自動旋轉時，讀到的影格已經轉正，不必再轉
    rot = rot if auto_off else 0
    w, h = (fh, fw) if rot in (90, 270) else (fw, fh)
    info = VideoInfo(
        path=str(path),
        width=w,
        height=h,
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        rotation=rot,
    )
    cap.release()
    return info


def frame_indices(info: VideoInfo, start: float | None, end: float | None, stride: int) -> list[int]:
    first = int(round((start or 0.0) * info.fps))
    last = info.frame_count if end is None else min(info.frame_count, int(round(end * info.fps)))
    return list(range(max(first, 0), last, max(stride, 1)))


def read_frames(path: str, indices: list[int], info: VideoInfo | None = None
                ) -> Iterator[tuple[int, np.ndarray]]:
    """依序讀取指定的影格（indices 需遞增），並轉成播放時的方向。"""
    if not indices:
        return
    info = info or probe(path)
    wanted = set(indices)
    cap, _ = _open(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    idx = indices[0]
    last = indices[-1]
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            yield idx, _to_display(frame, info)
        idx += 1
    cap.release()


class VideoWriter:
    """先用 OpenCV 寫 mp4v，若有 ffmpeg（系統或 imageio-ffmpeg）再轉成 H.264 方便手機播放。"""

    def __init__(self, path: str | os.PathLike, fps: float, size: tuple[int, int]):
        self.path = Path(path)
        self._tmp = self.path.with_name(self.path.stem + ".mp4v.mp4")
        self._w = cv2.VideoWriter(str(self._tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._w.isOpened():
            raise RuntimeError(f"無法寫出影片：{path}")

    def write(self, frame: np.ndarray) -> None:
        self._w.write(frame)

    def close(self) -> str:
        self._w.release()
        exe = _ffmpeg_exe()
        if exe:
            cmd = [exe, "-y", "-loglevel", "error", "-i", str(self._tmp), "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast",
                   "-movflags", "+faststart", str(self.path)]
            if subprocess.run(cmd).returncode == 0:
                self._tmp.unlink(missing_ok=True)
                return "h264"
        self._tmp.replace(self.path)
        return "mp4v"


def _ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
