"""跑完整流程（需要 MediaPipe 模型，會自動下載）。

repo 不附影片：一律用合成影片測試「沒有人」時的輸出；根目錄有自己的 .mp4 時再多跑一次真實影片。
"""

import csv
import glob
import os

import numpy as np
import pytest

VIDEOS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "*.mp4")))


@pytest.mark.slow
@pytest.mark.skipif(not VIDEOS, reason="沒有測試影片")
def test_pipeline_on_repo_video(tmp_path):
    pytest.importorskip("mediapipe")
    from snowcoach.pipeline import Options, run

    video = next((v for v in VIDEOS if "selfie" in v), VIDEOS[0])
    try:
        a = run(Options(video=video, out=str(tmp_path), start=1.0, end=3.0, stride=2))
    except OSError as e:  # 例如缺 libEGL 或無法下載模型
        pytest.skip(f"MediaPipe 無法執行：{e}")
    for name in ("metrics.csv", "keypoints.csv", "report.md", "report.html", "summary.json", "overlay.mp4"):
        assert (tmp_path / name).exists(), name
    rows = list(csv.DictReader(open(tmp_path / "metrics.csv", encoding="utf-8-sig")))
    assert len(rows) == len(a.indices)
    assert sum(r["rider_detected"] == "1" for r in rows) > len(rows) * 0.5
    # 每個缺值都要有原因
    for r in rows:
        for k in ("knee_left", "inclination", "fore_aft"):
            assert r[k] or r[f"{k}_note"]
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## 最優先的建議" in report
    from snowcoach.disclaimer import LONG
    assert LONG in report
    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "最優先的建議" in page and "<svg class='plot'" in page
    assert "非專業建議，僅供參考" in page
    data = page.split("<script type='application/json' id='data'>")[1].split("</script>")[0]
    import json
    parsed = json.loads(data)
    assert len(parsed["t"]) == len(a.indices) and len(parsed["charts"]) == 5


def _blank_clip(path, n=45, fps=30):
    """沒有人的合成影片：只有一個移動的圓。"""
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (480, 270))
    for i in range(n):
        f = np.full((270, 480, 3), 230, np.uint8)
        cv2.circle(f, (60 + i * 6, 135), 25, (50, 50, 200), -1)
        w.write(f)
    w.release()


@pytest.mark.slow
def test_pipeline_without_rider(tmp_path):
    """看不到人時要完整輸出，每格都說明原因，而且不給建議。"""
    pytest.importorskip("mediapipe")
    from snowcoach.disclaimer import LONG
    from snowcoach.pipeline import Options, run

    video = tmp_path / "blank.mp4"
    _blank_clip(video)
    out = tmp_path / "out"
    try:
        a = run(Options(video=str(video), out=str(out)))
    except OSError as e:
        pytest.skip(f"MediaPipe 無法執行：{e}")
    for name in ("metrics.csv", "keypoints.csv", "report.md", "report.html", "summary.json", "overlay.mp4"):
        assert (out / name).exists(), name
    rows = list(csv.DictReader(open(out / "metrics.csv", encoding="utf-8-sig")))
    assert len(rows) == len(a.indices) and all(r["rider_detected"] == "0" for r in rows)
    assert all(r["inclination_note"] for r in rows)
    assert a.top is None
    report = (out / "report.md").read_text(encoding="utf-8")
    assert LONG in report and "無法給出建議" in report
