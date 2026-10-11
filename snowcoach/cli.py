"""命令列介面：snowcoach analyze VIDEO [選項]"""

from __future__ import annotations

import argparse
import os
import sys

from snowcoach.coach.targets import LEVEL_ALIASES, LEVELS


def _level(v: str) -> str:
    key = LEVEL_ALIASES.get(v.strip().lower(), LEVEL_ALIASES.get(v.strip()))
    if key is None:
        raise argparse.ArgumentTypeError("程度請用 beginner/intermediate/advanced（或 初學/中級/進階）")
    return key


def build_parser() -> argparse.ArgumentParser:
    from snowcoach.disclaimer import LONG
    p = argparse.ArgumentParser(prog="snowcoach", description="單板滑雪動作影片分析。" + LONG)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="分析一支影片",
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       epilog="程度說明：\n" + "\n".join(
                           f"  {k:<13}{v['zh']}：{v['desc']}" for k, v in LEVELS.items()))
    a.add_argument("video", help="影片檔路徑")
    a.add_argument("-o", "--out", default=None, help="輸出資料夾（預設 out/<影片檔名>）")
    a.add_argument("--source", choices=["auto", "selfie", "follow"], default="auto",
                   help="拍攝方式：selfie = 自拍桿，follow = 他人跟拍或定點拍（預設自動判斷）")
    a.add_argument("--stance", choices=["auto", "regular", "goofy"], default="auto",
                   help="慣用站姿：regular = 左腳在前，goofy = 右腳在前（預設自動判斷）")
    a.add_argument("--no-switch", action="store_true",
                   help="不偵測 switch（逆腳）滑行，整段都用慣用站姿計算")
    a.add_argument("--roll", type=float, default=0.0,
                   help="相機側傾角度（度）：畫面中水平線右邊低、左邊高為正。"
                        "相機明顯歪斜時用來修正身體傾倒與折髖（預設 0）")
    a.add_argument("--level", type=_level, default="intermediate",
                   help="程度：beginner/intermediate/advanced 或 初學/中級/進階（預設中級）")
    a.add_argument("--lens", default="none",
                   help="鏡頭校正：none（Insta360 重新構圖、手機）、gopro-wide、gopro-superview、"
                        "fisheye:<對角視角>、calib:<json>（預設 none）")
    a.add_argument("--backend", choices=["mediapipe", "yolo-pose", "sam3d"], default="mediapipe",
                   help="姿勢模型（預設 mediapipe，含 3D 與腳跟/腳尖）")
    a.add_argument("--detector", choices=["mediapipe", "yolo", "none"], default="mediapipe",
                   help="找人用的偵測器；騎手很小時可試 yolo（需安裝 ultralytics）")
    a.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                   help="MediaPipe 模型大小（heavy 較準但較慢）")
    a.add_argument("--start", type=float, default=None, help="開始時間（秒）")
    a.add_argument("--end", type=float, default=None, help="結束時間（秒）")
    a.add_argument("--stride", type=int, default=1, help="每 N 格分析一格（加快速度）")
    a.add_argument("--no-video", action="store_true", help="不產生疊圖影片")

    g = sub.add_parser("gui", help="開啟圖形介面（本機網頁，需要 pip install \"snowcoach[gui]\"）")
    g.add_argument("--port", type=int, default=7860, help="網頁埠號（預設 7860）")
    g.add_argument("--lan", action="store_true", help="讓同一個網路的其他裝置（例如手機）也能連進來")
    g.add_argument("--no-browser", action="store_true", help="不要自動打開瀏覽器")

    sub.add_parser("download-models", help="預先下載預設模型（之後可離線使用；位置由 SNOWCOACH_MODELS 決定）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "analyze":
        from snowcoach.pipeline import Options, run
        out = args.out or os.path.join("out", os.path.splitext(os.path.basename(args.video))[0][:60].strip("# ") or "video")
        opts = Options(video=args.video, out=out, source=args.source, stance=args.stance,
                       level=args.level, lens=args.lens, backend=args.backend, detector=args.detector,
                       model=args.model, start=args.start, end=args.end, stride=args.stride,
                       switch=not args.no_switch, roll=args.roll,
                       video_out=not args.no_video)
        a = run(opts)
        if a.top is not None:
            from snowcoach.coach.targets import cue_text
            print(f"最優先的建議：{cue_text((a.top.metric, a.top.direction), a.stance.known)}")
        from snowcoach.disclaimer import SHORT
        print(f"（{SHORT}）")
        print(f"結果在 {out}/")
        return 0
    if args.cmd == "gui":
        from snowcoach.gui import main as gui_main
        return gui_main(port=args.port, lan=args.lan, open_browser=not args.no_browser)
    if args.cmd == "download-models":
        from snowcoach.models import download_defaults, model_dir
        download_defaults()
        print(f"模型已就緒：{model_dir()}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
