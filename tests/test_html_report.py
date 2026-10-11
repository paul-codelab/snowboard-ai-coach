import base64

import numpy as np

from snowcoach.html_report import _crop_jpeg, _esc
from synth import make_pose


def test_crop_is_portrait_jpeg_around_rider():
    frame = np.zeros((1920, 1080, 3), np.uint8)
    b64 = _crop_jpeg(frame, make_pose(), width=200)
    raw = base64.b64decode(b64)
    assert raw[:2] == b"\xff\xd8"  # JPEG


def test_crop_without_pose():
    assert _crop_jpeg(np.zeros((100, 80, 3), np.uint8), None, width=40)


def test_escape_untrusted_filename():
    assert _esc("<script>x</script>.mp4") == "&lt;script&gt;x&lt;/script&gt;.mp4"
