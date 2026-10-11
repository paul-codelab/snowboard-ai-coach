"""疊圖：骨架、板子軸線、指標面板（含目標區間）、立刃軌跡、中文字幕。

中文字一律用 PIL 搭配 TrueType 字型繪製（OpenCV 內建字型沒有中文字）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from snowcoach.coach.coach import BAD, NA, OK, WARN
from snowcoach.coach.targets import TARGETS, relabel
from snowcoach.disclaimer import SHORT as DISCLAIMER
from snowcoach.joints import BONES, IDX
from snowcoach.pose.base import PoseFrame

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msjhl.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
    "C:/Windows/Fonts/msyh.ttc",
]


def find_font() -> str | None:
    env = os.environ.get("SNOWCOACH_FONT")
    if env and Path(env).exists():
        return env
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


# 顏色（RGB）
C_OK = (60, 200, 90)
C_WARN = (245, 190, 40)
C_BAD = (235, 70, 60)
C_NA = (150, 150, 150)
C_TOE = (57, 135, 229)     # 與 HTML 報告一致：藍 = 腳尖側
C_HEEL = (235, 104, 52)    # 橘 = 腳跟側
C_TEXT = (255, 255, 255)
STATUS_COLOR = {OK: C_OK, WARN: C_WARN, BAD: C_BAD, NA: C_NA}

# 面板上顯示的指標：(key, 顯示範圍, 評分用 key 或 None)
PANEL = [
    ("fore_aft", (-0.5, 0.5), "fore_aft"),
    ("front_load", (0, 100), None),
    ("knee_front", (0, 120), "knee_front"),
    ("knee_back", (0, 120), "knee_back"),
    ("inclination", (-70, 70), "inclination"),
    ("angulation", (-30, 60), "angulation"),
    ("shoulder", (-60, 60), "shoulder"),
]
LABELS = {
    "fore_aft": "前後重心", "front_load": "前腳負重", "knee_front": "前膝彎曲", "knee_back": "後膝彎曲",
    "inclination": "身體傾倒", "angulation": "折髖", "shoulder": "肩膀對齊",
}


def fmt_value(key: str, v: float) -> str:
    if key == "fore_aft":
        return f"{v:+.2f}"
    if key == "front_load":
        return f"{v:.0f}%"
    if key == "inclination":
        if abs(v) < 3:
            return "0° 平"
        side = "腳尖側" if v > 0 else "腳跟側"
        return f"{abs(v):.0f}° {side}"
    return f"{v:+.0f}°" if key in ("angulation", "shoulder") else f"{v:.0f}°"


class OverlayRenderer:
    def __init__(self, width: int, height: int, level: str, header: str, fps: float,
                 stance_known: bool = True):
        self.w, self.h = width, height
        self.stance_known = stance_known
        self.level = level
        self.header = header
        self.fps = fps
        self.u = min(width, height) / 1080.0  # 以 1080 寬為基準的縮放
        path = find_font()
        if path is None:
            print("警告：找不到中文字型，影片中的文字將無法顯示。"
                  "請安裝 Noto Sans CJK 或文泉驛正黑，或設定 SNOWCOACH_FONT。", file=sys.stderr)
        self.font_path = path
        self.f_small = self._font(26)
        self.f_mid = self._font(32)
        self.f_big = self._font(44)

    def _font(self, size: int):
        if self.font_path is None:
            return ImageFont.load_default()
        return ImageFont.truetype(self.font_path, max(10, int(size * self.u)))

    def S(self, v: float) -> int:
        return int(round(v * self.u))

    # ------------------------------------------------------------------
    def draw_skeleton(self, img: np.ndarray, pose: PoseFrame | None, edge: float) -> None:
        if pose is None:
            return
        ok = pose.conf >= 0.5
        th = max(2, self.S(5))
        for a, b in BONES:
            ia, ib = IDX[a], IDX[b]
            if ok[ia] and ok[ib]:
                col = (80, 220, 255) if "left" in a and "left" in b else (255, 170, 60) if "right" in a and "right" in b else (240, 240, 240)
                cv2.line(img, tuple(np.int32(pose.xy[ia])), tuple(np.int32(pose.xy[ib])), col, th, cv2.LINE_AA)
        for i in np.where(ok)[0]:
            cv2.circle(img, tuple(np.int32(pose.xy[i])), th + 1, (255, 255, 255), -1, cv2.LINE_AA)
        # 板子軸線（由兩腳推得）
        feet = []
        for s in ("left", "right"):
            pts = [pose.xy[IDX[f"{s}_{j}"]] for j in ("ankle", "heel", "toe") if ok[IDX[f"{s}_{j}"]]]
            if pts:
                feet.append(np.mean(pts, axis=0))
        if len(feet) == 2:
            a, b = feet
            d = b - a
            p0, p1 = a - 0.45 * d, b + 0.45 * d
            col_rgb = C_TOE if edge == 1 else C_HEEL if edge == -1 else (230, 230, 230)
            cv2.line(img, tuple(np.int32(p0)), tuple(np.int32(p1)), col_rgb[::-1], th + 3, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def draw_panel(self, draw: ImageDraw.ImageDraw, values: dict[str, float], status: dict[str, int],
                   reasons: dict[str, str | None], edge_hist: np.ndarray) -> None:
        S = self.S
        x0, y0 = S(20), S(20)
        pw = S(520)
        row = S(62)
        ph = S(60) + row * len(PANEL) + S(120)
        draw.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=S(18), fill=(0, 0, 0, 150))
        draw.text((x0 + S(18), y0 + S(12)), self.header, font=self.f_small, fill=C_TEXT)
        y = y0 + S(56)
        for key, (lo, hi), judge_key in PANEL:
            v = values.get(key, np.nan)
            st = status.get(judge_key, NA) if judge_key else (OK if not np.isnan(v) else NA)
            draw.text((x0 + S(18), y), relabel(LABELS[key], self.stance_known) if key.startswith("knee") else LABELS[key], font=self.f_small, fill=C_TEXT)
            if np.isnan(v):
                draw.text((x0 + S(160), y), "無法判斷", font=self.f_small, fill=C_NA)
            else:
                draw.text((x0 + S(160), y), fmt_value(key, v), font=self.f_small, fill=STATUS_COLOR[st] if judge_key else C_TEXT)
            # 目標區間條
            bx0, bx1 = x0 + S(330), x0 + pw - S(20)
            by = y + S(16)
            draw.rounded_rectangle([bx0, by - S(6), bx1, by + S(6)], radius=S(6), fill=(90, 90, 90, 200))

            def X(val: float) -> float:
                return bx0 + (np.clip(val, lo, hi) - lo) / (hi - lo) * (bx1 - bx0)

            if judge_key:
                t = TARGETS[self.level][judge_key]
                bands = [t["ok"]]
                if key == "inclination":  # 兩側對稱
                    bands = [t["ok"], (-t["ok"][1], -t["ok"][0])]
                for blo, bhi in bands:
                    draw.rectangle([X(blo), by - S(6), X(bhi), by + S(6)], fill=C_OK + (170,))
            if not np.isnan(v):
                mx = X(v)
                col = STATUS_COLOR[st] if judge_key else C_TEXT
                draw.ellipse([mx - S(10), by - S(10), mx + S(10), by + S(10)], fill=col, outline=(255, 255, 255))
            y += row
        # 立刃軌跡：最近 3 秒的身體傾倒（上 = 腳尖側、下 = 腳跟側）
        ty0, ty1 = y + S(10), y + S(100)
        tx0, tx1 = x0 + S(18), x0 + pw - S(20)
        mid = (ty0 + ty1) / 2
        draw.line([tx0, mid, tx1, mid], fill=(200, 200, 200, 160), width=1)
        draw.text((tx0, ty0 - S(4)), "腳尖側", font=self.f_small, fill=C_TOE)
        draw.text((tx0, ty1 - S(30)), "腳跟側", font=self.f_small, fill=C_HEEL)
        n = len(edge_hist)
        if n > 1:
            xs = np.linspace(tx0 + S(100), tx1, n)
            pts = None
            for xv, v in zip(xs, edge_hist):
                if np.isnan(v):
                    pts = None
                    continue
                yv = mid - np.clip(v / 60.0, -1, 1) * (ty1 - ty0) / 2
                if pts is not None:
                    draw.line([pts, (xv, yv)], fill=C_TOE if v > 0 else C_HEEL, width=max(2, S(4)))
                pts = (xv, yv)

    def draw_caption(self, draw: ImageDraw.ImageDraw, text: str) -> None:
        S = self.S
        bbox = draw.textbbox((0, 0), text, font=self.f_big)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = self.w / 2, self.h - S(160)
        pad = S(22)
        draw.rounded_rectangle([cx - tw / 2 - pad, cy - th / 2 - pad, cx + tw / 2 + pad, cy + th / 2 + pad],
                               radius=S(20), fill=(200, 40, 40, 200))
        draw.text((cx - tw / 2, cy - th / 2 - bbox[1]), text, font=self.f_big, fill=C_TEXT)

    def draw_disclaimer(self, draw: ImageDraw.ImageDraw) -> None:
        """畫面底部固定顯示一行小字警語。"""
        S = self.S
        bbox = draw.textbbox((0, 0), DISCLAIMER, font=self.f_small)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (self.w - tw) / 2, self.h - S(24) - th
        draw.rounded_rectangle([x - S(10), y - S(6), x + tw + S(10), y + th + S(6) + bbox[1]],
                               radius=S(10), fill=(0, 0, 0, 120))
        draw.text((x, y), DISCLAIMER, font=self.f_small, fill=(235, 235, 235, 230))

    # ------------------------------------------------------------------
    def render(self, frame: np.ndarray, pose: PoseFrame | None, edge: float,
               values: dict[str, float], status: dict[str, int], reasons: dict[str, str | None],
               edge_hist: np.ndarray, caption: str | None) -> np.ndarray:
        img = frame.copy()
        self.draw_skeleton(img, pose, edge)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        self.draw_panel(draw, values, status, reasons, edge_hist)
        if caption:
            self.draw_caption(draw, caption)
        self.draw_disclaimer(draw)
        out = Image.alpha_composite(pil, layer).convert("RGB")
        return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
