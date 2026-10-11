"""手機直拍影片常把畫面存成橫的並加上旋轉標記，讀取時要轉正。"""

import subprocess

import cv2
import numpy as np
import pytest

from snowcoach.video import _ffmpeg_exe, frame_indices, probe, read_frames


def _make(path, w, h, rotate_ccw=None):
    """產生 w×h 的測試影片（左上角白色方塊），可選擇加上 displaymatrix 旋轉標記。"""
    raw = path.with_name(path.stem + "_raw.mp4")
    wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for _ in range(5):
        f = np.zeros((h, w, 3), np.uint8)
        f[:40, :60] = 255
        wr.write(f)
    wr.release()
    if rotate_ccw is None:
        return raw
    exe = _ffmpeg_exe()
    cmd = [exe, "-y", "-loglevel", "error", "-display_rotation:v:0", str(rotate_ccw), "-i", str(raw),
           "-c", "copy", str(path)]
    subprocess.run(cmd, check=True)
    return path


def test_plain_portrait_video(tmp_path):
    info = probe(_make(tmp_path / "p.mp4", 180, 320))
    assert (info.width, info.height, info.rotation) == (180, 320, 0)


def _ffmpeg_first_frame(path, w, h):
    """ffmpeg 預設會依旋轉標記自動轉正，當作正確答案。"""
    raw = subprocess.run([_ffmpeg_exe(), "-loglevel", "error", "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


@pytest.mark.skipif(_ffmpeg_exe() is None, reason="需要 ffmpeg")
@pytest.mark.parametrize("ccw,size", [(90, (180, 320)), (-90, (180, 320)), (180, (320, 180))])
def test_rotated_phone_video_is_read_upright(tmp_path, ccw, size):
    # 畫面存成橫的 320×180，加上旋轉標記；播放時是直式（或上下顛倒）
    path = _make(tmp_path / "r.mp4", 320, 180, rotate_ccw=ccw)
    info = probe(path)
    assert (info.width, info.height) == size
    frames = [f for _, f in read_frames(str(path), frame_indices(info, None, None, 1), info)]
    assert frames and all(f.shape[:2] == (size[1], size[0]) for f in frames)
    # 方向也要對：和 ffmpeg 轉正的畫面一致
    ref = _ffmpeg_first_frame(path, *size)
    assert np.abs(frames[0].astype(int) - ref.astype(int)).mean() < 2
