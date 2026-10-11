import glob
import os

import pytest

from snowcoach.gui import _out_dir, run_analysis

VIDEOS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "*.mp4")))


def test_out_dir_is_safe(tmp_path):
    d = _out_dir("/x/#insta360 #snow day 滑雪.mp4", tmp_path)
    assert d.parent == tmp_path
    assert "#" not in d.name and " " not in d.name
    assert d.name.startswith("insta360_snow_day_滑雪_")


def test_rejects_missing_video_and_bad_range():
    with pytest.raises(ValueError):
        run_analysis("")
    with pytest.raises(ValueError):
        run_analysis("x.mp4", start=5, end=3)


def test_build_app():
    pytest.importorskip("gradio")
    from snowcoach.gui import build_app
    assert build_app() is not None


@pytest.mark.slow
@pytest.mark.skipif(not VIDEOS, reason="沒有測試影片")
def test_run_analysis_end_to_end(tmp_path):
    pytest.importorskip("mediapipe")
    seen = []
    r = run_analysis(VIDEOS[0], end=1.5, make_video=False, out_base=tmp_path,
                     progress=lambda f, m: seen.append(f))
    assert r.report_html.exists() and r.zip_path.exists() and r.video is None
    assert "非專業" in r.summary_md
    assert seen and 0 <= min(seen) and max(seen) <= 1 and seen == sorted(seen)
