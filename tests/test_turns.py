import numpy as np

from snowcoach.turns import direction_vote, edge_labels, infer_direction, segment_turns
from synth import make_pose, random_rotation


def test_edge_labels_hysteresis():
    incl = np.array([0, 7, 5, 4, 2, -7, -5, np.nan, -8, 0])
    lab = edge_labels(incl, on=6, off=3)
    assert list(lab[:5]) == [0, 1, 1, 1, 0]
    assert lab[5] == -1 and lab[6] == -1
    assert lab[7] == -1  # 短暫缺值沿用前一格
    assert lab[9] == 0


def test_segment_turns_alternating():
    fps = 30
    t = np.arange(0, 8, 1 / fps)
    incl = 25 * np.sin(2 * np.pi * t / 4)  # 每 2 秒換刃一次
    turns, turn_id = segment_turns(incl, t)
    sides = [x.side for x in turns]
    assert len(turns) == 4
    assert sides == ["toe", "heel", "toe", "heel"]
    assert all(1.2 < x.duration < 2.0 for x in turns)
    assert (turn_id[turns[1].start:turns[1].end] == 1).all()


def test_short_blips_are_not_turns():
    t = np.arange(0, 2, 1 / 30)
    incl = np.zeros_like(t)
    incl[10:14] = 20  # 0.13 秒
    turns, _ = segment_turns(incl, t)
    assert turns == []


FPS = 10


def _clip(stances, secs):
    poses = []
    for st, sec in zip(stances, secs):
        poses += [make_pose(st, shoulder=15, R=random_rotation(len(poses) + k)) for k in range(int(sec * FPS))]
    return poses


def test_direction_vote_sign():
    assert direction_vote(make_pose("regular", shoulder=15, R=random_rotation(1))) > 0.5   # 左腳在前
    assert direction_vote(make_pose("goofy", shoulder=15, R=random_rotation(2))) < -0.5    # 右腳在前


def test_infer_natural_stance():
    reg = infer_direction(_clip(["regular"], [4]), FPS)
    goof = infer_direction(_clip(["goofy"], [4]), FPS)
    assert reg.stance == "regular" and set(reg.lead) == {"left"}
    assert goof.stance == "goofy" and set(goof.lead) == {"right"}
    assert goof.source == "auto"


def test_short_clip_does_not_decide_natural_stance():
    res = infer_direction(_clip(["regular"], [2.5]), FPS)
    assert res.stance is None
    assert res.frame_stance(5) == "regular"  # 但仍知道這段是左腳在前


def test_switch_is_detected():
    poses = _clip(["goofy", "regular", "goofy"], [5, 3, 4])
    res = infer_direction(poses, FPS, natural="goofy")
    assert res.stance == "goofy" and res.source == "user"
    assert res.is_switch(int(6.5 * FPS)) is True     # 中段是 switch
    assert res.is_switch(int(2 * FPS)) is False
    assert res.is_switch(int(10 * FPS)) is False
    # 換邊的過渡期不猜
    assert any(res.lead[i] is None for i in range(int(4.5 * FPS), int(5.5 * FPS)))


def test_no_switch_option_uses_natural_stance_everywhere():
    poses = _clip(["goofy", "regular"], [4, 4])
    res = infer_direction(poses, FPS, natural="goofy", switch=False)
    assert set(res.lead) == {"right"}


def test_brief_contrary_blip_is_not_a_switch():
    poses = _clip(["goofy", "regular", "goofy"], [4, 0.8, 4])
    res = infer_direction(poses, FPS, natural="goofy")
    assert not any(res.is_switch(i) for i in range(len(poses)))


def test_no_evidence_with_user_stance_assumes_natural():
    res = infer_direction([None] * 30, FPS, natural="regular")
    assert set(res.lead) == {"left"}
    assert infer_direction([None] * 30, FPS).known is False
