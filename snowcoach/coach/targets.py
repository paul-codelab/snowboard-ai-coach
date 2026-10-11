"""各程度的目標區間與建議文字。

這些區間是依單板教學常見原則自行訂定的起始值（不是量測標準），寫在這裡方便調整：
- ok：目標區間（綠色）
- bad_lo / bad_hi：超過就是明顯問題（紅色）；介於 ok 與 bad 之間為黃色
- None 代表該方向不設限
"""

from __future__ import annotations

LEVELS = {
    "beginner": {
        "zh": "初學",
        "desc": "能用落葉飄或滑行式（會側滑）轉彎下緩坡，正在練習連續換刃。",
    },
    "intermediate": {
        "zh": "中級",
        "desc": "能在中級道連續轉彎、換刃順暢，開始用立刃減少側滑。",
    },
    "advanced": {
        "zh": "進階",
        "desc": "能在各種雪道刻滑（carving），轉彎時板刃切雪，身體有明顯的傾倒與折髖。",
    },
}

LEVEL_ALIASES = {
    "beginner": "beginner", "初學": "beginner", "初级": "beginner", "初階": "beginner",
    "intermediate": "intermediate", "中級": "intermediate", "中级": "intermediate", "中階": "intermediate",
    "advanced": "advanced", "進階": "advanced", "进阶": "advanced", "高級": "advanced",
}

# 優先順序：平衡 → 壓力 → 立刃 → 旋轉（先解決前面的，後面的問題常會跟著改善）
CATEGORIES = ["balance", "pressure", "edge", "rotation"]
CATEGORY_ZH = {"balance": "平衡", "pressure": "壓力", "edge": "立刃", "rotation": "旋轉"}

METRICS = {
    # key: (中文名稱, 單位, 類別)
    "fore_aft": ("前後重心", "", "balance"),
    "knee_front": ("前膝彎曲", "°", "pressure"),
    "knee_back": ("後膝彎曲", "°", "pressure"),
    "knee_diff": ("兩膝差", "°", "pressure"),
    "inclination": ("身體傾倒", "°", "edge"),
    "angulation": ("折髖", "°", "edge"),
    "shoulder": ("肩膀對齊", "°", "rotation"),
}
# 只顯示、不評分的指標
DISPLAY_ONLY = {
    "front_load": ("前腳負重", "%"),
}

TARGETS: dict[str, dict[str, dict[str, float | None | tuple]]] = {
    "beginner": {
        "fore_aft":    {"ok": (-0.08, 0.12), "bad_lo": -0.20, "bad_hi": 0.30},
        "knee_front":  {"ok": (20, 60), "bad_lo": 10, "bad_hi": 90},
        "knee_back":   {"ok": (20, 60), "bad_lo": 10, "bad_hi": 90},
        "knee_diff":   {"ok": (0, 20), "bad_lo": None, "bad_hi": 35},
        "inclination": {"ok": (5, 25), "bad_lo": None, "bad_hi": None},
        "angulation":  {"ok": (-5, 25), "bad_lo": -15, "bad_hi": None},
        "shoulder":    {"ok": (-10, 30), "bad_lo": -25, "bad_hi": 50},
    },
    "intermediate": {
        "fore_aft":    {"ok": (-0.10, 0.15), "bad_lo": -0.22, "bad_hi": 0.32},
        "knee_front":  {"ok": (25, 70), "bad_lo": 12, "bad_hi": 100},
        "knee_back":   {"ok": (25, 70), "bad_lo": 12, "bad_hi": 100},
        "knee_diff":   {"ok": (0, 20), "bad_lo": None, "bad_hi": 35},
        "inclination": {"ok": (12, 40), "bad_lo": 6, "bad_hi": None},
        "angulation":  {"ok": (0, 30), "bad_lo": -10, "bad_hi": None},
        "shoulder":    {"ok": (-10, 25), "bad_lo": -25, "bad_hi": 45},
    },
    "advanced": {
        "fore_aft":    {"ok": (-0.15, 0.20), "bad_lo": -0.28, "bad_hi": 0.35},
        "knee_front":  {"ok": (30, 90), "bad_lo": 15, "bad_hi": 115},
        "knee_back":   {"ok": (30, 90), "bad_lo": 15, "bad_hi": 115},
        "knee_diff":   {"ok": (0, 25), "bad_lo": None, "bad_hi": 40},
        "inclination": {"ok": (25, 60), "bad_lo": 15, "bad_hi": None},
        "angulation":  {"ok": (5, 35), "bad_lo": -5, "bad_hi": None},
        "shoulder":    {"ok": (-15, 20), "bad_lo": -30, "bad_hi": 40},
    },
}

# (指標, 方向) → (影片字幕, 報告說明, 練習方法)
CUES: dict[tuple[str, str], tuple[str, str, str]] = {
    ("fore_aft", "low"): (
        "重心太後面：前膝往板頭壓",
        "重心常落在後腳（坐後座），板頭會變輕、轉彎難以啟動，也容易被板子帶走。",
        "直滑時雙手輕碰前膝，感覺重量在兩腳中間略偏前；每個彎一開始主動把重心移向前腳。",
    ),
    ("fore_aft", "high"): (
        "重心太前面：回到兩腳中間",
        "重心常壓在前腳，板尾會甩出、容易側滑失控。",
        "直滑時前後腳輪流輕壓（前後搖擺），找到兩腳平均受力的位置後保持住。",
    ),
    ("knee_front", "low"): (
        "前膝太直：屈膝降低重心",
        "前腳膝蓋常打直，沒有緩衝空間，遇到不平的雪面容易被彈起。",
        "邊滑邊做「上下起伏」：換刃時站高、轉彎中蹲低，每個彎做一次。",
    ),
    ("knee_front", "high"): (
        "前膝蹲太低：稍微站高",
        "前腳蹲得太深，腿部容易疲勞，也少了往下吸收的空間。",
        "在緩坡用較高的姿勢滑行，膝蓋保持自然微彎即可。",
    ),
    ("knee_back", "low"): (
        "後膝太直：後腳也要彎",
        "後腳膝蓋常打直，重心容易往前、板尾缺乏控制。",
        "滑行時想像坐在高腳椅上，兩膝同時彎曲，後膝不要鎖死。",
    ),
    ("knee_back", "high"): (
        "後膝蹲太低：稍微站高",
        "後腳蹲得太深，重心容易落到板尾。",
        "在緩坡用較高的姿勢滑行，兩膝彎曲程度保持一致。",
    ),
    ("knee_diff", "high"): (
        "兩膝彎曲不一致：一起彎",
        "前後兩膝的彎曲程度差很多，重量分配不均，板子容易往一邊扭。",
        "停止狀態下練習兩膝同時下蹲、同時伸直，再到緩坡邊滑邊做。",
    ),
    ("inclination", "low"): (
        "傾倒不夠：往彎內倒，信任板刃",
        "轉彎時身體幾乎沒有往彎內倒，板子立不起來，只能靠側滑轉彎。",
        "在寬的緩坡做大半徑轉彎，轉彎中讓身體慢慢往彎內倒，感受板刃咬雪。",
    ),
    ("angulation", "low"): (
        "整個人往內倒：從髖部折",
        "轉彎時上半身跟著一起往內倒（而不是從髖部折），重量壓不到板刃，容易打滑或內倒摔。",
        "Toe side 時前手摸前膝外側、heel side 時後手往後腳跟方向伸，讓上半身保持直立、從髖部折。",
    ),
    ("shoulder", "high"): (
        "肩膀轉太開：與板子平行",
        "肩膀常轉向板頭，用上半身甩彎，下半身跟不上時容易打滑。",
        "雙手放在前膝上方、與板子平行滑行，用膝蓋和腳踝帶動轉彎。",
    ),
    ("shoulder", "low"): (
        "肩膀反向扭轉：放鬆上半身",
        "肩膀常往板尾方向扭（反向旋轉），身體扭成麻花，轉彎不流暢。",
        "視線看向前進方向，前手輕鬆指向板頭，上半身放鬆跟著板子走。",
    ),
}


_SIDE_WORDS = [("前後兩膝", "左右兩膝"), ("前膝", "左膝"), ("後膝", "右膝"), ("前腳", "左腳"), ("後腳", "右腳")]


def relabel(text: str, stance_known: bool) -> str:
    """站姿未確定時，膝蓋相關文字的「前/後」改成「左/右」。"""
    if stance_known:
        return text
    for a, b in _SIDE_WORDS:
        text = text.replace(a, b)
    return text


def cue_text(key: tuple[str, str], stance_known: bool, part: int = 0) -> str:
    text = CUES[key][part]
    return relabel(text, stance_known) if key[0].startswith("knee") else text
