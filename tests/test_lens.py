import numpy as np
import pytest

from snowcoach.camera.lens import parse_lens


def test_none_is_passthrough():
    lens = parse_lens("none")
    img = np.random.default_rng(0).integers(0, 255, (40, 30, 3), dtype=np.uint8)
    assert lens.undistort(img) is img
    assert not parse_lens("insta360").active


@pytest.mark.parametrize("spec", ["gopro-wide", "fisheye:150"])
def test_fisheye_keeps_size_and_center(spec):
    lens = parse_lens(spec)
    h, w = 120, 90
    img = np.zeros((h, w, 3), np.uint8)
    img[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3] = 255
    out = lens.undistort(img)
    assert out.shape == img.shape
    assert out[h // 2, w // 2].max() > 200  # 中心點不動


def test_calib_file(tmp_path):
    f = tmp_path / "cam.json"
    f.write_text('{"model": "fisheye", "K": [[300,0,160],[0,300,120],[0,0,1]], "D": [0.1,0,0,0], "size": [320,240]}')
    lens = parse_lens(f"calib:{f}")
    out = lens.undistort(np.zeros((480, 640, 3), np.uint8))
    assert out.shape == (480, 640, 3)


def test_bad_spec():
    with pytest.raises(ValueError):
        parse_lens("banana")
