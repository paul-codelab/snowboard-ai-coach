import numpy as np
import pytest

from snowcoach.overlay import OverlayRenderer, find_font, fmt_value
from synth import make_pose


def test_render_keeps_size():
    r = OverlayRenderer(540, 960, "intermediate", "自拍 · Regular · 中級", 30)
    frame = np.zeros((960, 540, 3), np.uint8)
    values = {"fore_aft": 0.05, "front_load": 55, "knee_front": 30, "knee_back": np.nan,
              "inclination": -20, "angulation": 10, "shoulder": 5}
    out = r.render(frame, make_pose(), -1, values, {"fore_aft": 1}, {}, np.array([1.0, -5, -20]), "重心太後面")
    assert out.shape == frame.shape
    assert out.any()


def test_fmt_value():
    assert fmt_value("inclination", 20) == "20° 腳尖側"
    assert fmt_value("inclination", -20) == "20° 腳跟側"
    assert fmt_value("inclination", 1) == "0° 平"
    assert fmt_value("front_load", 43.2) == "43%"


@pytest.mark.skipif(find_font() is None, reason="系統沒有中文字型")
def test_font_has_cjk_glyphs():
    from PIL import ImageFont
    f = ImageFont.truetype(find_font(), 20)
    assert f.getbbox("傾")[2] > 5
