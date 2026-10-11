"""中文摘要報告（Markdown）。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from snowcoach.coach.coach import BAD, NA, OK, enough_data, turn_peak
from snowcoach.disclaimer import LONG as DISCLAIMER
from snowcoach.coach.targets import CATEGORY_ZH, DISPLAY_ONLY, LEVELS, METRICS, TARGETS, cue_text, relabel

if TYPE_CHECKING:
    from snowcoach.pipeline import Analysis

REPORT_METRICS = ["fore_aft", "front_load", "knee_front", "knee_back", "knee_diff",
                  "inclination", "angulation", "shoulder"]

DEFINITIONS = {
    "fore_aft": "重心沿板子方向的位置，以站距為單位：0 = 兩腳正中間，+ = 偏板頭，− = 偏板尾。",
    "front_load": "由前後重心換算的前腳負重比例（50% = 兩腳平均）。",
    "knee_front": "前腳膝蓋彎曲角度（0° = 完全打直）。",
    "knee_back": "後腳膝蓋彎曲角度（0° = 完全打直）。",
    "knee_diff": "前膝減後膝的彎曲差；評分看絕對值，差太多代表兩腳負重不均。",
    "inclination": "重心相對兩腳中點、偏離鉛直線的角度（往腳尖側或腳跟側倒）。評分用每個轉彎的代表值。",
    "angulation": "下半身傾倒減上半身傾倒；正值 = 上半身比腿直（有折髖），負值 = 整個人往彎內倒。只看轉彎中。",
    "shoulder": "肩線與板子方向的夾角（水平面上）；正值 = 開肩（胸口轉向板頭），負值 = 反向扭轉。",
}


def _name(k: str, stance_known: bool = True) -> str:
    name = METRICS[k][0] if k in METRICS else DISPLAY_ONLY[k][0]
    return relabel(name, stance_known) if k.startswith("knee") else name


def _fmt(k: str, v: float) -> str:
    if v is None or np.isnan(v):
        return "—"
    if k == "fore_aft":
        return f"{v:+.2f}"
    if k == "front_load":
        return f"{v:.0f}%"
    return f"{v:.0f}°"


def _band(level: str, k: str) -> str:
    if k not in METRICS:
        return "—"
    lo, hi = TARGETS[level][k]["ok"]
    if k == "fore_aft":
        return f"{lo:+.2f} ~ {hi:+.2f}"
    if k == "inclination":
        return f"{lo:.0f}° ~ {hi:.0f}°（每個彎）"
    if k == "knee_diff":
        return f"≤ {hi:.0f}°"
    return f"{lo:.0f}° ~ {hi:.0f}°"


def metric_stats(a: "Analysis") -> dict:
    out = {}
    n = len(a.indices)
    for k in REPORT_METRICS:
        v = a.series[k]
        valid = ~np.isnan(v)
        reasons = Counter(r for r, ok in zip(a.reasons[k], valid) if not ok and r)
        d = {
            "available": float(valid.mean()) if n else 0.0,
            "median": float(np.median(v[valid])) if valid.any() else None,
            "p10": float(np.percentile(v[valid], 10)) if valid.any() else None,
            "p90": float(np.percentile(v[valid], 90)) if valid.any() else None,
            "top_reason": reasons.most_common(1)[0][0] if reasons else None,
        }
        if k in a.fj.status:
            st = a.fj.status[k]
            if k == "inclination":
                st = np.array([st[t.start] for t in a.turns]) if a.turns else np.array([])
            judged = st[st != NA]
            d["judged"] = int(len(judged))
            d["in_band"] = float((judged == OK).mean()) if len(judged) else None
            d["bad"] = float((judged == BAD).mean()) if len(judged) else None
        out[k] = d
    return out


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def build_report(a: "Analysis") -> str:
    lv = LEVELS[a.opts.level]
    info = a.info
    detected = sum(p is not None for p in a.poses)
    src_zh = {"selfie": "自拍", "follow": "他人拍攝（跟拍/定點）"}[a.source.source]
    how = {"user": "使用者指定", "auto": f"自動判斷，信心 {a.source.confidence:.0%}",
           "auto-uncertain": "自動判斷但不確定"}[a.source.decided_by]
    if a.stance.stance is None:
        stance_txt = "無法確定"
    else:
        stance_txt = a.stance.stance.capitalize()
        stance_txt += "（使用者指定）" if a.stance.source == "user" else \
            f"（自動判斷，方向明確的片段中 {a.stance.confidence:.0%} 一致）"
    n_switch = sum(bool(a.stance.is_switch(i)) for i in range(len(a.indices)))
    if a.opts.switch and a.stance.stance is not None:
        stance_txt += f"；switch 滑行約 {n_switch / a.fps:.1f} 秒" if n_switch else "；沒有偵測到 switch 滑行"
    span = f"{a.times[0]:.1f}–{a.times[-1]:.1f} 秒"

    L: list[str] = []
    L += ["# 單板滑雪動作分析報告", "", f"> ⚠️ {DISCLAIMER}", ""]
    L += [f"- 影片：`{Path(info.path).name}`（{info.width}×{info.height}，{info.fps:.0f} fps）",
          f"- 分析範圍：{span}，共 {len(a.indices)} 格，偵測到騎手 {detected} 格（{detected / len(a.indices):.0%}）",
          f"- 拍攝方式：{src_zh}（{how}）" + (f"，持桿手：{'左手' if a.source.stick_arm == 'left' else '右手'}"
                                             if a.source.source == 'selfie' and a.source.stick_arm else ""),
          f"- 站姿：{stance_txt}",
          f"- 程度：{lv['zh']} — {lv['desc']}",
          f"- 姿勢模型：{a.opts.backend}（找人：{a.opts.detector}）；鏡頭校正：{a.opts.lens}"
          + (f"；相機側傾修正：{a.opts.roll:+.1f}°" if a.opts.roll else ""),
          ""]

    known = a.stance.known
    L += ["## 最優先的建議", ""]
    f = a.top
    enough = {m: ok for m, ok in enough_data(a.fj, a.turns, a.fps).items() if m not in a.skip}
    judged = [m for m, ok in enough.items() if ok]
    lacking = [m for m, ok in enough.items() if not ok]
    if f is None:
        if judged:
            L += [f"有足夠資料的指標（{'、'.join(_name(m, known) for m in judged)}）"
                  f"大致都在「{lv['zh']}」的目標區間內，沒有需要優先修正的問題。"]
            if lacking:
                L += ["", f"但{'、'.join(_name(m, known) for m in lacking)}的資料不足，無法下結論，"
                      "這不代表它們沒有問題。"]
            L += ["", "可以試著用更高一級的標準檢視（`--level`），或拍一段更長、轉彎更多、全身入鏡的影片。", ""]
        else:
            L += ["這段影片可判斷的資料太少，無法給出建議。請參考下方「無法判斷的主因」，"
                  "盡量讓全身（包含雙腳與板子）入鏡。", ""]
    else:
        key = (f.metric, f.direction)
        cue, why, drill = (cue_text(key, known, i) for i in range(3))
        cat = CATEGORY_ZH[METRICS[f.metric][2]]
        L += [f"### {cue}", "",
              f"**類別：{cat}**（依「平衡 → 壓力 → 立刃 → 旋轉」的順序，先處理最前面的問題）", "",
              f"- 觀察：{why}",
              f"- 依據：可判斷的 {f.samples} {f.unit}中，{f.share:.0%} 超出目標區間，其中 {f.bad_share:.0%} 明顯超出。",
              f"- 練習：{drill}", ""]
        others = [m for m, g in a.findings.items() if g is not None and m != f.metric and m not in a.skip]
        if others:
            L += ["其他也超出目標的指標（先把上面這項練好，這些常會跟著改善）："
                  + "、".join(_name(m, known) for m in others), ""]
        if lacking:
            L += [f"資料不足、無法評估的指標：{'、'.join(_name(m, known) for m in lacking)}。", ""]

    L += ["## 各項指標", "",
          "| 指標 | 中位數 | 常見範圍 (P10–P90) | 目標區間 | 在目標內 | 可判斷的比例 | 無法判斷的主因 |",
          "|---|---|---|---|---|---|---|"]
    for k, d in metric_stats(a).items():
        rng = "—" if d["p10"] is None else f"{_fmt(k, d['p10'])} ~ {_fmt(k, d['p90'])}"
        med = "—" if d["median"] is None else _fmt(k, d["median"])
        inb = _pct(d.get("in_band")) if k in METRICS else "（僅顯示）"
        if k in a.skip:
            inb += "（不評分）"
        L.append(f"| {_name(k, known)} | {med} | {rng} | {_band(a.opts.level, k)} | {inb} | "
                 f"{d['available']:.0%} | {d['top_reason'] or '—'} |")
    L.append("")

    L += ["## 轉彎", ""]
    if not a.turns:
        L += ["沒有偵測到明確的轉彎（身體傾倒資料不足，或一直直滑）。", ""]
    else:
        toe = [t for t in a.turns if t.side == "toe"]
        heel = [t for t in a.turns if t.side == "heel"]
        L += [f"共 {len(a.turns)} 個轉彎：腳尖側（toe side）{len(toe)} 個、腳跟側（heel side）{len(heel)} 個。", "",
              relabel("| # | 邊 | 時間 | 長度 | 身體傾倒 | 前膝 | 後膝 | 折髖 |", known),
              "|---|---|---|---|---|---|---|---|"]
        for t in a.turns:
            seg = slice(t.start, t.end)

            def med(k: str) -> str:
                v = a.series[k][seg]
                v = v[~np.isnan(v)]
                return _fmt(k, float(np.median(v))) if len(v) else "—"
            L.append(f"| {t.index + 1} | {'腳尖側' if t.side == 'toe' else '腳跟側'} | "
                     f"{t.start_time:.1f}–{t.end_time:.1f} 秒 | {t.duration:.1f} 秒 | "
                     f"{_fmt('inclination', turn_peak(a.series['inclination'], t))} | "
                     f"{med('knee_front')} | {med('knee_back')} | {med('angulation')} |")
        L.append("")
        if len(toe) >= 2 and len(heel) >= 2:
            tp = np.nanmean([turn_peak(a.series["inclination"], t) for t in toe])
            hp = np.nanmean([turn_peak(a.series["inclination"], t) for t in heel])
            L += [f"平均傾倒：腳尖側 {tp:.0f}°、腳跟側 {hp:.0f}°。"
                  + ("兩側差距明顯，可以多練較弱的一側。" if abs(tp - hp) > 10 else ""), ""]

    L += ["## 說明與限制", ""]
    for n in a.notes:
        L.append(f"- {n}")
    L += ["- 「無法判斷」代表該部位沒入鏡、被遮擋或估計不合理，工具不會用猜的補上數值。",
          "- 3D 姿勢由單一鏡頭影像推估，角度為近似值，適合看趨勢與前後比較，不宜當成精確量測。",
          "- 身體傾倒與折髖以畫面垂直方向當作重力方向；相機明顯傾斜時數值會偏差，可用 `--roll` 修正。",
          "- 前後腳依每一段「頭部與胸口朝向」判斷（會偵測 switch）；換邊過渡期間標示為無法判斷。",
          "- 目標區間為一般教學原則的起始值，可在 `snowcoach/coach/targets.py` 調整。", ""]
    L += ["### 指標定義", ""]
    for k in REPORT_METRICS:
        d = relabel(DEFINITIONS[k], known) if k.startswith("knee") else DEFINITIONS[k]
        L.append(f"- **{_name(k, known)}**：{d}")
    L.append("")
    return "\n".join(L)


def write_report(a: "Analysis", path: Path) -> None:
    path.write_text(build_report(a), encoding="utf-8")

