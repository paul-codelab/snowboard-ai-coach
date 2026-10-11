import numpy as np

from snowcoach.pose.base import PersonDetector, PoseBackend
from snowcoach.track import RiderTracker, iou, select_primary
from synth import make_pose


def _pose_at(x0, y0, size):
    p = make_pose()
    xy = p.xy - p.xy.min(axis=0)
    p.xy = xy / xy.max() * size + np.array([x0, y0])
    return p


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_select_primary_prefers_continuity_then_size():
    small_near = _pose_at(100, 100, 50)
    big_far = _pose_at(600, 600, 300)
    assert select_primary([small_near, big_far], prev=(95, 95, 155, 155)) is small_near
    assert select_primary([small_near, big_far], prev=None) is big_far


class FakeBackend(PoseBackend):
    def __init__(self, people):
        self.people = people

    def estimate(self, image, roi):
        return list(self.people)


class FakeDetector(PersonDetector):
    def __init__(self, people):
        self.people = people

    def detect(self, image):
        return [(p.bbox(), 0.9) for p in self.people]


def test_tracker_switches_to_clearly_larger_rider():
    inset = _pose_at(800, 100, 150)   # 例如子母畫面裡的小人
    main = _pose_at(200, 600, 700)
    tr = RiderTracker(FakeBackend([inset]), FakeDetector([inset]), recheck_every=5)
    frame = np.zeros((10, 10, 3), np.uint8)
    assert tr.update(frame) is inset
    tr.backend.people = [inset, main]
    tr.detector.people = [inset, main]
    got = [tr.update(frame) for _ in range(12)]
    assert got[-1] is main
