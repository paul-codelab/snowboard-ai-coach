from snowcoach.pose.base import PersonDetector, PoseBackend, PoseFrame


def make_backend(name: str, model: str = "full") -> PoseBackend:
    if name == "mediapipe":
        from snowcoach.pose.mediapipe_backend import MediaPipePose
        return MediaPipePose(model=model)
    if name == "yolo-pose":
        from snowcoach.pose.yolo_backend import YoloPose
        return YoloPose()
    if name == "sam3d":
        from snowcoach.pose.sam3d_backend import Sam3DBodyBackend
        return Sam3DBodyBackend()
    raise ValueError(f"未知的姿勢後端：{name}")


def make_detector(name: str) -> PersonDetector | None:
    if name == "none":
        return None
    if name == "mediapipe":
        from snowcoach.pose.mediapipe_backend import MediaPipePersonDetector
        return MediaPipePersonDetector()
    if name == "yolo":
        from snowcoach.pose.yolo_backend import YoloPersonDetector
        return YoloPersonDetector()
    raise ValueError(f"未知的人物偵測器：{name}")


__all__ = ["PoseBackend", "PoseFrame", "PersonDetector", "make_backend", "make_detector"]
