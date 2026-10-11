import numpy as np
import pytest

from snowcoach.geometry import compute_metrics, metrics_2d
from snowcoach.joints import IDX
from synth import make_pose, random_rotation

ROTS = [None, random_rotation(1), random_rotation(7)]


@pytest.mark.parametrize("R", ROTS)
def test_knee_flexion(R):
    m = compute_metrics(make_pose(knee=55, R=R), "regular")
    assert m["knee_left"].value == pytest.approx(55, abs=0.5)
    assert m["knee_right"].value == pytest.approx(55, abs=0.5)


@pytest.mark.parametrize("R", ROTS)
@pytest.mark.parametrize("lean", [-35, -15, 15, 35])
def test_inclination_sign_and_size(R, lean):
    """正 = 腳尖側、負 = 腳跟側，且與相機角度無關。"""
    m = compute_metrics(make_pose(lean=lean, R=R), "regular")
    assert m["inclination"].value == pytest.approx(lean, abs=0.5)


@pytest.mark.parametrize("stance", ["regular", "goofy"])
@pytest.mark.parametrize("fa", [-0.2, 0.15])
def test_fore_aft(stance, fa):
    m = compute_metrics(make_pose(stance=stance, fore_aft=fa, R=random_rotation(3)), stance)
    assert m["fore_aft"].value == pytest.approx(fa, abs=0.01)


def test_wrong_stance_flips_fore_aft():
    m = compute_metrics(make_pose(stance="regular", fore_aft=0.2), "goofy")
    assert m["fore_aft"].value == pytest.approx(-0.2, abs=0.01)


@pytest.mark.parametrize("stance", ["regular", "goofy"])
@pytest.mark.parametrize("phi", [-20, 30])
def test_shoulder_alignment(stance, phi):
    m = compute_metrics(make_pose(stance=stance, shoulder=phi, R=random_rotation(4)), stance)
    assert m["shoulder"].value == pytest.approx(phi, abs=0.5)


@pytest.mark.parametrize("lean,upper", [(30, 10), (-30, -10)])
def test_angulation_positive_when_upper_body_more_upright(lean, upper):
    m = compute_metrics(make_pose(lean=lean, upper_lean=upper, R=random_rotation(2)), "regular")
    assert m["angulation"].value == pytest.approx(20, abs=0.5)


def test_banking_gives_negative_angulation():
    m = compute_metrics(make_pose(lean=20, upper_lean=35), "regular")
    assert m["angulation"].value < 0


def test_no_angulation_when_flat():
    m = compute_metrics(make_pose(lean=0), "regular")
    assert m["angulation"].value is None
    assert "沒有傾倒" in m["angulation"].reason


def test_missing_joint_out_of_frame_reason():
    p = make_pose()
    for n in ("left_knee", "left_ankle", "left_heel", "left_toe"):
        p.conf[IDX[n]] = 0.0
        p.in_frame[IDX[n]] = False
    m = compute_metrics(p, "regular")
    assert m["knee_left"].value is None and "未入鏡" in m["knee_left"].reason
    assert m["knee_right"].ok
    assert m["fore_aft"].value is None and "左腳" in m["fore_aft"].reason


def test_occluded_joint_reason():
    p = make_pose()
    p.conf[IDX["right_knee"]] = 0.2  # 在畫面內但信心低
    m = compute_metrics(p, "regular")
    assert m["knee_right"].value is None
    assert "被遮擋" in m["knee_right"].reason


def test_unknown_stance_withholds_fore_aft_and_shoulder():
    m = compute_metrics(make_pose(lean=20), None)
    assert m["inclination"].ok
    assert m["fore_aft"].value is None and "站姿" in m["fore_aft"].reason
    assert m["shoulder"].value is None


def test_implausible_pose_rejected():
    p = make_pose()
    p.world[IDX["right_ankle"]] += np.array([3.0, 0, 0])  # 兩腳距離 3 公尺
    p.world[IDX["right_heel"]] += np.array([3.0, 0, 0])
    p.world[IDX["right_toe"]] += np.array([3.0, 0, 0])
    m = compute_metrics(p, "regular")
    assert m["inclination"].value is None
    assert "不合理" in m["inclination"].reason


def test_no_pose():
    m = compute_metrics(None, "regular")
    assert all(not v.ok for v in m.values())


def test_2d_fallback_is_conservative():
    p = make_pose()
    p.world = None
    m = metrics_2d(p, "regular")
    assert m["inclination"].value is None
    assert m["shoulder"].value is None


def test_tiny_rider_is_not_measured():
    p = make_pose()
    p.xy = (p.xy - 500) * 0.1 + 500  # 約 60 像素高
    m = compute_metrics(p, "regular")
    assert all(not v.ok for v in m.values())
    assert "太小" in m["knee_left"].reason


def test_roll_correction():
    from snowcoach.pipeline import apply_roll
    # 從板頭方向拍：板軸朝鏡頭深度方向、腳尖側朝畫面右、上方朝畫面上（相機 y 朝下）
    to_cam = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], float)
    cam_roll = np.array([[np.cos(np.radians(10)), -np.sin(np.radians(10)), 0],
                         [np.sin(np.radians(10)), np.cos(np.radians(10)), 0], [0, 0, 1]])
    p = make_pose(lean=20, R=cam_roll @ to_cam)
    p.up = np.array([0.0, -1.0, 0.0])  # 以為相機是水平的
    off = compute_metrics(p, "regular")["inclination"].value
    apply_roll([p], 10)
    fixed = compute_metrics(p, "regular")["inclination"].value
    assert abs(off - 20) > 5
    assert fixed == pytest.approx(20, abs=0.5)
