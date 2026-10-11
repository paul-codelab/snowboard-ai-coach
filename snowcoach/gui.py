"""圖形介面（本機網頁）：snowcoach gui

拖放影片、選程度與站姿、按「開始分析」，完成後直接看疊圖影片、摘要，並下載 HTML 報告與所有結果。
預設只在本機（127.0.0.1）開啟；加 --lan 才讓同一個網路的其他裝置（例如手機）連進來。
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from snowcoach.coach.targets import LEVELS, cue_text
from snowcoach.disclaimer import LONG as DISCLAIMER

LEVEL_CHOICES = [(f"{v['zh']}：{v['desc']}", k) for k, v in LEVELS.items()]
STANCE_CHOICES = [("自動判斷", "auto"), ("Regular（左腳在前）", "regular"), ("Goofy（右腳在前）", "goofy")]
SOURCE_CHOICES = [("自動判斷", "auto"), ("自拍（自拍桿）", "selfie"), ("別人拍（跟拍、定點、手機）", "follow")]
LENS_CHOICES = [("不校正（Insta360 重新構圖、手機）", "none"), ("GoPro 廣角", "gopro-wide"),
                ("GoPro SuperView", "gopro-superview")]


def yolo_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


@dataclass
class Result:
    out_dir: Path
    summary_md: str
    video: Path | None
    report_html: Path
    zip_path: Path


def _out_dir(video: str, base: str | Path) -> Path:
    # 只留安全字元（中英文、數字、- _），避免 # 和空白在網址或路徑中出問題
    stem = re.sub(r"[^\w\-]+", "_", Path(video).stem).strip("_")[:40] or "video"
    return Path(base) / f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}"


def _summary(a, report_md: str) -> str:
    """介面上顯示的摘要：最優先的建議放最前面，接著是完整的 Markdown 報告。"""
    head = ""
    if a.top is not None:
        head = f"## 👉 {cue_text((a.top.metric, a.top.direction), a.stance.known)}\n\n"
    return head + report_md


def run_analysis(video: str, level: str = "intermediate", stance: str = "auto", source: str = "auto",
                 switch: bool = True, roll: float = 0.0, lens: str = "none", detector: str = "mediapipe",
                 start: float | None = None, end: float | None = None, make_video: bool = True,
                 out_base: str | Path = "out/gui", progress=None) -> Result:
    """實際的分析流程（和 Gradio 無關，可單獨測試）。"""
    from snowcoach.pipeline import Options, run

    if not video:
        raise ValueError("請先選擇影片")
    if start is not None and end is not None and end <= start:
        raise ValueError("結束時間要大於開始時間")
    out = _out_dir(video, out_base)
    opts = Options(video=video, out=str(out), source=source, stance=stance, level=level, lens=lens,
                   detector=detector, start=start, end=end, switch=switch, roll=roll or 0.0,
                   video_out=make_video)
    a = run(opts, progress or (lambda f, m: None))
    zip_path = out / "snowcoach_結果.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f != zip_path:
                z.write(f, f.name)
    md = (out / "report.md").read_text(encoding="utf-8")
    return Result(out, _summary(a, md), out / "overlay.mp4" if make_video else None,
                  out / "report.html", zip_path)


def _no_phone_home() -> None:
    """關掉 Gradio／Hugging Face 的使用統計與版本檢查，介面只在本機運作、不對外連線。

    必須在 import gradio 之前設定。
    """
    for key, value in (("GRADIO_ANALYTICS_ENABLED", "False"), ("HF_HUB_DISABLE_TELEMETRY", "1"),
                       ("HF_HUB_OFFLINE", "1"), ("DO_NOT_TRACK", "1")):
        os.environ.setdefault(key, value)


def build_app():
    _no_phone_home()
    import gradio as gr

    detectors = [("MediaPipe（預設）", "mediapipe")]
    if yolo_available():
        detectors.append(("YOLO（騎手很小時較準）", "yolo"))

    def on_run(video, level, stance, source, switch, roll, lens, detector, start, end, make_video,
               progress=gr.Progress()):
        try:
            r = run_analysis(video, level, stance, source, switch, roll, lens, detector,
                             start or None, end or None, make_video,
                             progress=lambda f, m: progress(f, desc=m))
        except ValueError as e:
            raise gr.Error(str(e))
        except Exception as e:  # 給使用者看得懂的訊息，細節留在終端機
            print(f"分析失敗：{e!r}", file=sys.stderr)
            raise gr.Error(f"分析失敗：{e}")
        return (r.summary_md, str(r.video) if r.video else None,
                [str(r.report_html), str(r.zip_path)], f"結果也存在：`{r.out_dir}`")

    with gr.Blocks(title="snowcoach 單板滑雪動作分析", analytics_enabled=False) as app:
        gr.Markdown("# 🏂 單板滑雪動作分析\n上傳一段滑雪影片，產生疊上骨架與建議的影片，以及一份中文報告。")
        gr.Markdown(f"> ⚠️ **非專業建議，僅供參考。** {DISCLAIMER}")
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="影片（拖放或點選上傳）", sources=["upload"])
                level = gr.Radio(LEVEL_CHOICES, value="intermediate", label="程度")
                stance = gr.Radio(STANCE_CHOICES, value="auto", label="慣用站姿（知道的話直接選，最準）")
                source = gr.Radio(SOURCE_CHOICES, value="auto", label="拍攝方式")
                with gr.Accordion("進階設定", open=False):
                    switch = gr.Checkbox(value=True, label="偵測 switch（逆腳）滑行")
                    roll = gr.Slider(-30, 30, value=0, step=0.5,
                                     label="相機側傾修正（度）：畫面水平線右低左高為正，相機沒歪就保持 0")
                    lens = gr.Dropdown(LENS_CHOICES, value="none", label="鏡頭校正")
                    detector = gr.Dropdown(detectors, value="mediapipe", label="找人用的偵測器")
                    with gr.Row():
                        start = gr.Number(value=None, label="從第幾秒開始（空白 = 開頭）", minimum=0)
                        end = gr.Number(value=None, label="到第幾秒結束（空白 = 結尾）", minimum=0)
                    make_video = gr.Checkbox(value=True, label="產生疊圖影片（關掉會快很多）")
                go = gr.Button("開始分析", variant="primary")
                gr.Markdown("分析在你的電腦上進行，影片不會上傳到網路。30 秒的影片大約需要 2～4 分鐘。")
            with gr.Column(scale=2):
                out_video = gr.Video(label="疊圖影片", interactive=False)
                files = gr.File(label="下載：HTML 報告（用瀏覽器打開）、全部結果（zip）", file_count="multiple",
                                interactive=False)
                where = gr.Markdown()
                summary = gr.Markdown("分析完成後，建議與報告會顯示在這裡。")
        go.click(on_run, [video, level, stance, source, switch, roll, lens, detector, start, end, make_video],
                 [summary, out_video, files, where])
    return app


def main(port: int = 7860, lan: bool = False, open_browser: bool = True) -> int:
    if importlib.util.find_spec("gradio") is None:
        print("需要安裝圖形介面套件：pip install \"snowcoach[gui]\"", file=sys.stderr)
        return 1
    host = "0.0.0.0" if lan else "127.0.0.1"
    if lan:
        print("注意：同一個網路中的其他裝置都能連進來使用。", file=sys.stderr)
    app = build_app()
    app.queue(default_concurrency_limit=1)  # CPU 一次只跑一支影片
    app.launch(server_name=host, server_port=port, inbrowser=open_browser,
               allowed_paths=[str(Path("out").resolve())], footer_links=[])
    return 0
