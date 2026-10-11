"""分析主流程：影片 → 姿勢 → 指標 → 轉彎 → 教練判斷 → 輸出。"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from snowcoach.camera.classify import SourceResult, classify_source, stick_arm_for
from snowcoach.camera.lens import Lens, parse_lens
from snowcoach.coach.coach import (FrameJudgement, judge_frames, judged_values, live_cues,
                                   metric_findings, top_finding)
from snowcoach.coach.targets import LEVELS, cue_text
from snowcoach.geometry import BASE_METRICS, compute_metrics, front_side
from snowcoach.joints import JOINTS
from snowcoach.pose import make_backend, make_detector
from snowcoach.pose.base import PoseFrame
from snowcoach.smooth import clean
from snowcoach.track import RiderTracker
from snowcoach.turns import StanceResult, Turn, edge_labels, infer_direction, segment_turns
from snowcoach.video import VideoInfo, frame_indices, probe, read_frames

SERIES = ["knee_left", "knee_right", "knee_front", "knee_back", "knee_diff",
          "inclination", "angulation", "fore_aft", "front_load", "shoulder"]


@dataclass
class Options:
    video: str
    out: str = "out"
    source: str = "auto"         # auto / selfie / follow
    stance: str = "auto"         # 慣用站姿：auto / regular / goofy
    switch: bool = True          # 偵測 switch（逆腳）滑行
    roll: float = 0.0            # 相機側傾（度）：畫面中水平線右低左高為正
    level: str = "intermediate"  # beginner / intermediate / advanced
    lens: str = "none"
    backend: str = "mediapipe"
    detector: str = "mediapipe"
    model: str = "full"
    start: float | None = None
    end: float | None = None
    stride: int = 1
    video_out: bool = True


@dataclass
class Analysis:
    opts: Options
    info: VideoInfo
    indices: list[int]
    times: np.ndarray
    fps: float                               # 分析序列的有效影格率
    poses: list[PoseFrame | None]
    source: SourceResult
    stance: StanceResult
    series: dict[str, np.ndarray]
    reasons: dict[str, list[str | None]]
    edge: np.ndarray
    turns: list[Turn]
    turn_id: np.ndarray
    judged: dict[str, np.ndarray]
    fj: FrameJudgement
    cues: list[tuple[str, str] | None]
    findings: dict
    top: object
    skip: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


# 進度回報：(0~1 的比例, 說明文字)；圖形介面用來顯示進度條
ProgressFn = Callable[[float, str], None]


def _noop(frac: float, msg: str) -> None:
    pass


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def detect_poses(opts: Options, info: VideoInfo, indices: list[int], lens: Lens,
                 progress: ProgressFn = _noop) -> list[PoseFrame | None]:
    backend = make_backend(opts.backend, opts.model)
    detector = make_detector(opts.detector)
    tracker = RiderTracker(backend, detector)
    poses: list[PoseFrame | None] = []
    t0 = time.time()
    step = max(1, len(indices) // 10)
    try:
        for k, (_, frame) in enumerate(read_frames(info.path, indices, info)):
            poses.append(tracker.update(lens.undistort(frame)))
            progress((k + 1) / len(indices), f"姿勢偵測 {k + 1}/{len(indices)} 格")
            if (k + 1) % step == 0:
                log(f"  姿勢偵測 {k + 1}/{len(indices)}（{time.time() - t0:.0f} 秒）")
    finally:
        backend.close()
        if detector is not None:
            detector.close()
    poses += [None] * (len(indices) - len(poses))
    return poses


def apply_roll(poses: list[PoseFrame | None], roll_deg: float) -> None:
    """依相機側傾角修正重力方向（world 座標與相機座標軸一致時適用）。"""
    if not roll_deg:
        return
    a = np.radians(roll_deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    for p in poses:
        if p is not None and p.up is not None:
            p.up = rz @ p.up


UNKNOWN_DIRECTION = "滑行方向不確定（無法判斷哪隻腳在前）"


def build_series(poses: list[PoseFrame | None], stances: list[str | None], fps: float
                 ) -> tuple[dict[str, np.ndarray], dict[str, list[str | None]]]:
    """stances：每格當下的站姿方向（前腳左 = regular），可隨 switch 改變。"""
    n = len(poses)
    any_known = any(st is not None for st in stances)
    raw = {k: np.full(n, np.nan) for k in BASE_METRICS}
    reasons: dict[str, list[str | None]] = {k: [None] * n for k in SERIES}
    for i, p in enumerate(poses):
        for k, m in compute_metrics(p, stances[i]).items():
            if m.ok:
                raw[k][i] = m.value
            elif stances[i] is None and any_known and m.reason and "站姿" in m.reason:
                reasons[k][i] = UNKNOWN_DIRECTION
            else:
                reasons[k][i] = m.reason

    series: dict[str, np.ndarray] = {}
    for k in BASE_METRICS:
        series[k] = clean(raw[k], fps)
        for i in np.where(~np.isnan(raw[k]) & np.isnan(series[k]))[0]:
            reasons[k][i] = "數值突然跳動，視為偵測誤差"

    if any_known:
        # 前膝/後膝依每格當下的前腳對應（switch 時會換邊）
        for key in ("knee_front", "knee_back"):
            series[key] = np.full(n, np.nan)
            reasons[key] = [UNKNOWN_DIRECTION] * n
        for i, st in enumerate(stances):
            if st is None:
                continue
            f = front_side(st)
            b = "right" if f == "left" else "left"
            series["knee_front"][i], series["knee_back"][i] = series[f"knee_{f}"][i], series[f"knee_{b}"][i]
            reasons["knee_front"][i], reasons["knee_back"][i] = reasons[f"knee_{f}"][i], reasons[f"knee_{b}"][i]
    else:
        # 整段都無法判斷方向：「前膝/後膝」欄位改放左膝/右膝，顯示時標成左右
        series["knee_front"], series["knee_back"] = series["knee_left"], series["knee_right"]
        reasons["knee_front"], reasons["knee_back"] = list(reasons["knee_left"]), list(reasons["knee_right"])
    series["knee_diff"] = series["knee_front"] - series["knee_back"]
    reasons["knee_diff"] = [reasons["knee_front"][i] or reasons["knee_back"][i] for i in range(n)]
    series["front_load"] = np.clip(50.0 + 100.0 * series["fore_aft"], 0.0, 100.0)
    reasons["front_load"] = list(reasons["fore_aft"])
    return series, reasons


def analyze(opts: Options, progress: ProgressFn = _noop) -> Analysis:
    if opts.level not in LEVELS:
        raise ValueError(f"未知的程度：{opts.level}")
    info = probe(opts.video)
    indices = frame_indices(info, opts.start, opts.end, opts.stride)
    if not indices:
        raise ValueError("指定的時間範圍內沒有影格")
    fps = info.fps / max(opts.stride, 1)
    times = np.array(indices) / info.fps
    lens = parse_lens(opts.lens)
    lens.prepare(info.width, info.height)
    log(f"影片：{info.width}x{info.height}，{info.fps:.1f} fps，分析 {len(indices)} 格")

    poses = detect_poses(opts, info, indices, lens, progress)
    detected = sum(p is not None for p in poses)
    log(f"  偵測到騎手：{detected}/{len(poses)} 格")

    notes: list[str] = []
    # 拍攝來源
    if opts.source == "auto":
        source = classify_source(poses, info.height)
        if source.decided_by == "auto-uncertain":
            notes.append("無法確定是自拍還是他人拍攝，先以他人拍攝處理；可用 --source 指定。")
    else:
        source = SourceResult(opts.source, 1.0, "user",
                              stick_arm_for(poses) if opts.source == "selfie" else None)
    apply_roll(poses, opts.roll)
    if opts.roll:
        notes.append(f"已依指定的相機側傾 {opts.roll:+.1f}° 修正重力方向。")
    # 慣用站姿與每格前腳（可能 switch）
    stance = infer_direction(poses, fps, None if opts.stance == "auto" else opts.stance, opts.switch)
    if stance.stance is None and not stance.known:
        notes.append("無法從影片判斷哪隻腳在前，前後重心、肩膀對齊等指標無法判斷，"
                     "膝蓋以左膝/右膝表示；請用 --stance 指定慣用站姿。")
    elif stance.stance is None:
        notes.append("無法確定慣用站姿（regular/goofy），只在方向明確的片段計算前後重心與肩膀；"
                     "請用 --stance 指定。")
    elif stance.source == "auto":
        notes.append(f"慣用站姿是自動判斷的（{stance.stance}）；如果不對，請用 --stance 指定。")
    n_switch = sum(bool(stance.is_switch(i)) for i in range(len(poses)))
    if n_switch:
        notes.append(f"偵測到 switch（逆腳）滑行約 {n_switch / fps:.1f} 秒，這些片段的前後腳已對調計算；"
                     "如果其實沒有 switch，可加 --no-switch。")
    unknown = sum(x is None for x in stance.lead) if stance.lead else 0
    if stance.known and unknown:
        notes.append(f"有 {unknown / fps:.1f} 秒無法判斷哪隻腳在前（頭部與胸口朝向的證據不足，或正在換邊），"
                     "這些片段的前後重心、前後膝與肩膀標示為無法判斷。")

    series, reasons = build_series(poses, [stance.frame_stance(i) for i in range(len(poses))], fps)
    edge = edge_labels(series["inclination"])
    turns, turn_id = segment_turns(series["inclination"], times)
    judged = judged_values(series, turns, turn_id)
    fj = judge_frames(judged, opts.level)

    skip: set[str] = set()
    if stance.stance is None:
        # 方向判斷錯誤會讓前後重心、肩膀的建議完全相反，慣用站姿不確定時只顯示不評分
        skip.update({"fore_aft", "shoulder"})
        if stance.known:
            notes.append("慣用站姿不確定，前後重心與肩膀對齊只顯示數值，不列入建議。")
    if source.source == "selfie":
        skip.add("shoulder")
        notes.append("自拍時持桿的手臂會帶動肩膀，肩膀對齊只顯示數值，不列入建議。")
    cues = live_cues(fj, fps, skip)
    findings = metric_findings(judged, fj, turns, fps)
    top = top_finding(findings, skip)
    return Analysis(opts, info, indices, times, fps, poses, source, stance, series, reasons, edge,
                    turns, turn_id, judged, fj, cues, findings, top, skip, notes)


# ---------------------------------------------------------------------------
# 輸出
# ---------------------------------------------------------------------------

def _num(v: float, nd: int = 2) -> str:
    return "" if v is None or np.isnan(v) else f"{v:.{nd}f}"


def write_metrics_csv(a: Analysis, path: Path) -> None:
    edge_zh = {1: "toe", -1: "heel", 0: "flat"}
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        head = ["frame", "time_s", "rider_detected", "lead_foot", "riding", "edge", "turn_id"]
        for k in SERIES:
            head += [k, f"{k}_note"]
        head += ["cue"]
        w.writerow(head)
        for i, idx in enumerate(a.indices):
            e = a.edge[i]
            lead = a.stance.lead[i] if a.stance.lead else None
            sw = a.stance.is_switch(i)
            row = [idx, f"{a.times[i]:.3f}", int(a.poses[i] is not None), lead or "",
                   "" if sw is None else ("switch" if sw else "normal"),
                   "" if np.isnan(e) else edge_zh[int(e)], int(a.turn_id[i]) if a.turn_id[i] >= 0 else ""]
            for k in SERIES:
                v = a.series[k][i]
                row += [_num(v, 3 if k == "fore_aft" else 1), "" if not np.isnan(v) else (a.reasons[k][i] or "")]
            cue = a.cues[i]
            row.append(cue_text(cue, a.stance.known) if cue else "")
            w.writerow(row)


def write_keypoints_csv(a: Analysis, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        head = ["frame", "time_s"]
        for j in JOINTS:
            head += [f"{j}_x", f"{j}_y", f"{j}_conf"]
        w.writerow(head)
        for i, idx in enumerate(a.indices):
            p = a.poses[i]
            row = [idx, f"{a.times[i]:.3f}"]
            for k in range(len(JOINTS)):
                if p is None or np.isnan(p.xy[k]).any():
                    row += ["", "", "0"]
                else:
                    row += [f"{p.xy[k, 0]:.1f}", f"{p.xy[k, 1]:.1f}", f"{p.conf[k]:.2f}"]
            w.writerow(row)


def write_summary_json(a: Analysis, path: Path) -> None:
    from snowcoach.report import metric_stats
    data = {
        "video": a.info.path,
        "frames_analyzed": len(a.indices),
        "frames_with_rider": int(sum(p is not None for p in a.poses)),
        "source": asdict(a.source),
        "stance": {"natural": a.stance.stance, "confidence": a.stance.confidence,
                   "source": a.stance.source, "votes": a.stance.votes,
                   "segments": [{"start_s": round(float(a.times[s0]), 2),
                                 "end_s": round(float(a.times[min(s1, len(a.times)) - 1]), 2),
                                 "lead_foot": lab} for s0, s1, lab in a.stance.segments]},
        "roll_deg": a.opts.roll,
        "level": a.opts.level,
        "lens": a.opts.lens,
        "backend": a.opts.backend,
        "turns": [{"index": t.index, "side": t.side, "start_s": round(t.start_time, 2),
                   "end_s": round(t.end_time, 2)} for t in a.turns],
        "metrics": metric_stats(a),
        "top_finding": asdict(a.top) if a.top else None,
        "notes": a.notes,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def render_video(a: Analysis, path: Path, progress: ProgressFn = _noop) -> str:
    from snowcoach.coach.targets import LEVELS
    from snowcoach.overlay import OverlayRenderer
    from snowcoach.video import VideoWriter

    lens = parse_lens(a.opts.lens)
    lens.prepare(a.info.width, a.info.height)
    src = {"selfie": "自拍", "follow": "他人拍攝"}[a.source.source]
    st = {"regular": "Regular", "goofy": "Goofy", None: "站姿未定"}[a.stance.stance]
    lv = LEVELS[a.opts.level]["zh"]
    r = OverlayRenderer(a.info.width, a.info.height, a.opts.level, "", a.fps,
                        stance_known=a.stance.known)
    writer = VideoWriter(path, a.fps, (a.info.width, a.info.height))
    hist = int(round(a.fps * 3))
    for i, (_, frame) in enumerate(read_frames(a.info.path, a.indices, a.info)):
        values = {k: a.series[k][i] for k in a.series}
        status = {k: int(a.fj.status[k][i]) for k in a.fj.status}
        reasons = {k: a.reasons[k][i] for k in a.reasons}
        cue = a.cues[i]
        sw = a.stance.is_switch(i)
        cur = a.stance.frame_stance(i)
        if sw:
            r.header = f"{src} · {st} · Switch · {lv}"
        elif a.stance.stance is None and cur is not None:
            r.header = f"{src} · {'左' if cur == 'regular' else '右'}腳在前 · {lv}"
        else:
            r.header = f"{src} · {st} · {lv}"
        img = r.render(lens.undistort(frame), a.poses[i], a.edge[i], values, status, reasons,
                       a.series["inclination"][max(0, i - hist + 1):i + 1],
                       cue_text(cue, a.stance.known) if cue else None)
        writer.write(img)
        progress((i + 1) / len(a.indices), f"產生疊圖影片 {i + 1}/{len(a.indices)} 格")
    return writer.close()


def run(opts: Options, progress: ProgressFn = _noop) -> Analysis:
    """progress 回報 0~1 的進度。有疊圖影片時：姿勢偵測 0~70%、報告 70~75%、影片 75~100%；
    沒有影片時姿勢偵測佔 0~95%。"""
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    span = 0.7 if opts.video_out else 0.95
    a = analyze(opts, lambda f, m: progress(f * span, m))
    progress(span, "計算指標與產生報告")
    from snowcoach.report import write_report
    write_metrics_csv(a, out / "metrics.csv")
    write_keypoints_csv(a, out / "keypoints.csv")
    write_summary_json(a, out / "summary.json")
    write_report(a, out / "report.md")
    from snowcoach.html_report import write_html_report
    write_html_report(a, out / "report.html")
    log(f"已寫出 {out / 'metrics.csv'}、{out / 'keypoints.csv'}、{out / 'report.md'}、{out / 'report.html'}")
    if opts.video_out:
        log("  產生疊圖影片 ...")
        codec = render_video(a, out / "overlay.mp4", lambda f, m: progress(0.75 + 0.25 * f, m))
        log(f"已寫出 {out / 'overlay.mp4'}（{codec}）")
    return a
