"""統一的關節定義。所有姿勢後端都要轉成這組關節名稱與順序。"""

from __future__ import annotations

JOINTS: list[str] = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_toe", "right_toe",
]
IDX: dict[str, int] = {name: i for i, name in enumerate(JOINTS)}
NUM_JOINTS = len(JOINTS)

ZH: dict[str, str] = {
    "nose": "鼻子",
    "left_eye": "左眼", "right_eye": "右眼",
    "left_ear": "左耳", "right_ear": "右耳",
    "left_shoulder": "左肩", "right_shoulder": "右肩",
    "left_elbow": "左肘", "right_elbow": "右肘",
    "left_wrist": "左腕", "right_wrist": "右腕",
    "left_hip": "左髖", "right_hip": "右髖",
    "left_knee": "左膝", "right_knee": "右膝",
    "left_ankle": "左踝", "right_ankle": "右踝",
    "left_heel": "左腳跟", "right_heel": "右腳跟",
    "left_toe": "左腳尖", "right_toe": "右腳尖",
}

# MediaPipe Pose Landmarker（33 點）的索引
MEDIAPIPE_INDEX: dict[str, int] = {
    "nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30, "left_toe": 31, "right_toe": 32,
}

# COCO 17 點（YOLO-Pose）的索引；沒有腳跟、腳尖
COCO_INDEX: dict[str, int] = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}

BONES: list[tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_toe"), ("left_ankle", "left_toe"),
    ("right_ankle", "right_heel"), ("right_heel", "right_toe"), ("right_ankle", "right_toe"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]


def other_side(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name
