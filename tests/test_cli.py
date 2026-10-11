import pytest

from snowcoach.cli import build_parser


def test_level_aliases():
    p = build_parser()
    for raw, want in [("初學", "beginner"), ("中級", "intermediate"), ("進階", "advanced"), ("advanced", "advanced")]:
        assert p.parse_args(["analyze", "x.mp4", "--level", raw]).level == want


def test_bad_level():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["analyze", "x.mp4", "--level", "pro"])


def test_defaults():
    a = build_parser().parse_args(["analyze", "x.mp4"])
    assert (a.source, a.stance, a.level, a.lens, a.backend) == ("auto", "auto", "intermediate", "none", "mediapipe")


def test_switch_and_roll_flags():
    a = build_parser().parse_args(["analyze", "x.mp4", "--no-switch", "--roll", "-7.5"])
    assert a.no_switch and a.roll == -7.5


def test_help_carries_disclaimer():
    assert "僅供參考" in build_parser().description


def test_download_models_command():
    assert build_parser().parse_args(["download-models"]).cmd == "download-models"


def test_gui_does_not_phone_home(monkeypatch):
    from snowcoach.gui import _no_phone_home
    monkeypatch.delenv("GRADIO_ANALYTICS_ENABLED", raising=False)
    _no_phone_home()
    import os
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
