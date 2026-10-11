import numpy as np

from snowcoach.coach.coach import (BAD, NA, OK, WARN, FrameJudgement, enough_data, fill_gaps, judge,
                                   live_cues, metric_findings, top_finding)
from snowcoach.coach.targets import CUES, LEVELS, METRICS, TARGETS, cue_text
from snowcoach.turns import Turn


def test_every_level_has_every_metric():
    assert set(LEVELS) == set(TARGETS)
    for lv in TARGETS.values():
        assert set(lv) == set(METRICS)
        for t in lv.values():
            assert t["ok"][0] <= t["ok"][1]


def test_every_judged_direction_has_a_cue():
    for lv in TARGETS.values():
        for m, t in lv.items():
            if t["bad_lo"] is not None:
                assert (m, "low") in CUES
            if t["bad_hi"] is not None:
                assert (m, "high") in CUES


def test_judge():
    t = {"ok": (20, 60), "bad_lo": 10, "bad_hi": 90}
    assert judge(40, t) == (OK, 0)
    assert judge(15, t) == (WARN, -1)
    assert judge(5, t) == (BAD, -1)
    assert judge(95, t) == (BAD, 1)
    assert judge(float("nan"), t) == (NA, 0)


def _fj(**metrics):
    status = {k: np.array(v[0]) for k, v in metrics.items()}
    direction = {k: np.array(v[1]) for k, v in metrics.items()}
    return FrameJudgement(status, direction)


def test_priority_balance_before_pressure():
    n = 60
    fj = _fj(fore_aft=([BAD] * n, [-1] * n), knee_front=([BAD] * n, [-1] * n))
    findings = metric_findings({}, fj, [], fps=30)
    assert top_finding(findings).metric == "fore_aft"
    assert top_finding(findings, skip={"fore_aft"}).metric == "knee_front"


def test_small_sample_is_not_a_finding():
    fj = _fj(fore_aft=([BAD] * 10 + [NA] * 50, [-1] * 60))
    assert metric_findings({}, fj, [], fps=30)["fore_aft"] is None


def test_live_cues_one_at_a_time_with_rest():
    fps = 10
    n = 100
    fj = _fj(fore_aft=([BAD] * n, [-1] * n), knee_front=([BAD] * n, [-1] * n))
    cues = live_cues(fj, fps, max_talk=3.0, rest=1.5)
    shown = [c for c in cues if c]
    assert set(shown) == {("fore_aft", "low")}  # 只講優先的那一個
    # 講 3 秒就休息 1.5 秒
    assert cues[0] is not None
    first_gap = next(i for i, c in enumerate(cues) if c is None)
    assert first_gap == 30
    assert all(c is None for c in cues[30:45])


def test_live_cues_ignore_brief_blips():
    fps = 30
    st = [OK] * 100
    st[50:55] = [BAD] * 5  # 0.17 秒
    fj = _fj(fore_aft=(st, [-1] * 100))
    assert all(c is None for c in live_cues(fj, fps))


def test_fill_gaps():
    st = np.array([WARN, NA, NA, WARN, NA, NA, NA, NA])
    dr = np.array([1, 0, 0, 1, 0, 0, 0, 0])
    fs, fd = fill_gaps(st, dr, 2)
    assert list(fs) == [WARN, WARN, WARN, WARN, WARN, WARN, NA, NA]
    assert fd[1] == 1


def test_enough_data_for_turns():
    fj = _fj(inclination=([OK] * 30, [0] * 30))
    one = [Turn(0, "toe", 0, 30, 0, 1)]
    two = [Turn(0, "toe", 0, 15, 0, 0.5), Turn(1, "heel", 15, 30, 0.5, 1)]
    assert not enough_data(fj, one, 30)["inclination"]
    assert enough_data(fj, two, 30)["inclination"]


def test_cue_relabel_without_stance():
    assert "左膝" in cue_text(("knee_front", "low"), stance_known=False)
    assert "前膝" in cue_text(("knee_front", "low"), stance_known=True)
    assert cue_text(("fore_aft", "low"), False) == CUES[("fore_aft", "low")][0]


def test_only_knee_labels_are_relabelled():
    from snowcoach.report import _name
    assert _name("knee_front", stance_known=False) == "左膝彎曲"
    assert _name("front_load", stance_known=False) == "前腳負重"
