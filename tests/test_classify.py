import numpy as np

from snowcoach.camera.classify import classify_source
from snowcoach.joints import IDX
from synth import make_pose


def _scaled(p, scale, dx=0.0, arm=0.0):
    p.xy = (p.xy - 500) * scale + 500 + dx
    p.world = p.world.copy()
    # 相機座標：z 越小越靠近相機
    p.world[:, 2] = p.world[:, 2] * 0  # 攤平深度
    p.world[IDX["right_wrist"], 2] = -arm
    return p


def test_selfie_detected():
    # 騎手約佔畫面 1/3 高、大小穩定、右手伸向鏡頭 0.4 m
    poses = [_scaled(make_pose(), 0.55, arm=0.4) for _ in range(30)]
    r = classify_source(poses, frame_height=1920)
    assert r.source == "selfie" and r.decided_by == "auto"
    assert r.stick_arm == "right"


def test_follow_detected():
    rng = np.random.default_rng(0)
    poses = [_scaled(make_pose(), s) for s in rng.uniform(0.05, 0.3, 30)]
    r = classify_source(poses, frame_height=1920)
    assert r.source == "follow"


def test_too_few_frames_is_uncertain():
    r = classify_source([None] * 50, frame_height=1920)
    assert r.decided_by == "auto-uncertain"
