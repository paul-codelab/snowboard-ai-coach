"""HTML 報告：單一檔案、不需網路，手機也能看，可以直接傳給朋友。

內容：最優先的建議（附問題最明顯的畫面）、各指標狀態、指標隨時間變化的圖表
（標出轉彎與目標區間，滑鼠/手指移動可看數值）、每個轉彎的代表畫面、說明與限制。
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from snowcoach.coach.coach import enough_data, turn_peak
from snowcoach.coach.targets import CATEGORY_ZH, LEVELS, METRICS, TARGETS, cue_text, relabel
from snowcoach.report import DEFINITIONS, REPORT_METRICS, _band, _fmt, _name, metric_stats
from snowcoach.disclaimer import LONG as DISCLAIMER
from snowcoach.video import read_frames

if TYPE_CHECKING:
    from snowcoach.pipeline import Analysis

# 圖表：(標題, 指標, 單位, y 範圍)；膝蓋兩條線畫在同一張
CHARTS = [
    ("身體傾倒", ["inclination"], "°", (-70, 70)),
    ("膝蓋彎曲", ["knee_front", "knee_back"], "°", (0, 120)),
    ("前後重心", ["fore_aft"], "", (-0.6, 0.6)),
    ("折髖", ["angulation"], "°", (-30, 60)),
    ("肩膀對齊", ["shoulder"], "°", (-60, 60)),
]


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------------------
# 代表畫面
# ---------------------------------------------------------------------------

def _crop_jpeg(frame: np.ndarray, pose, width: int = 360) -> str:
    """以騎手為中心切出直式畫面，回傳 base64 JPEG。"""
    h, w = frame.shape[:2]
    b = pose.bbox(0.3) if pose is not None else None
    if b is None:
        crop = frame
    else:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        ch = min(h, max(b[3] - b[1], (b[2] - b[0]) * 4 / 3) * 1.5)
        cw = min(w, ch * 3 / 4)
        x0 = int(np.clip(cx - cw / 2, 0, w - cw))
        y0 = int(np.clip(cy - ch / 2, 0, h - ch))
        crop = frame[y0:y0 + int(ch), x0:x0 + int(cw)]
    scale = width / crop.shape[1]
    crop = cv2.resize(crop, (width, int(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _worst_frame(a: "Analysis", metric: str, direction: str) -> int | None:
    """最優先問題最明顯的那一格（評分用的數值最偏離目標的方向）。"""
    v = a.judged.get(metric)
    if v is None or np.all(np.isnan(v)):
        return None
    idx = np.nanargmin(v) if direction == "low" else np.nanargmax(v)
    return int(idx)


def grab_images(a: "Analysis", positions: list[int]) -> dict[int, str]:
    """讀出指定位置（分析序列中的索引）的影格，畫上骨架後切圖。"""
    from snowcoach.camera.lens import parse_lens
    from snowcoach.overlay import OverlayRenderer

    positions = sorted({p for p in positions if 0 <= p < len(a.indices)})
    if not positions:
        return {}
    lens = parse_lens(a.opts.lens)
    lens.prepare(a.info.width, a.info.height)
    r = OverlayRenderer(a.info.width, a.info.height, a.opts.level, "", a.fps)
    by_frame = {a.indices[p]: p for p in positions}
    out = {}
    for fidx, frame in read_frames(a.info.path, sorted(by_frame), a.info):
        p = by_frame[fidx]
        img = lens.undistort(frame).copy()
        r.draw_skeleton(img, a.poses[p], a.edge[p])
        out[p] = _crop_jpeg(img, a.poses[p])
    return out


# ---------------------------------------------------------------------------
# 資料整理
# ---------------------------------------------------------------------------

def _status_label(a: "Analysis", m: str, enough: dict[str, bool]) -> tuple[str, str]:
    """(css class, 文字)；狀態一律附文字，不只靠顏色。"""
    if m in a.skip:
        return "info", "僅供參考"
    if m not in enough or not enough[m]:
        return "na", "資料不足"
    f = a.findings.get(m)
    if f is None:
        return "good", "良好"
    return ("bad", "需要改善") if f.bad_share >= 0.15 else ("warn", "注意")


def _series_json(a: "Analysis") -> dict:
    known = a.stance.known
    t = [round(float(x), 3) for x in a.times]

    def clean(v: np.ndarray, nd: int) -> list:
        return [None if np.isnan(x) else round(float(x), nd) for x in v]

    charts = []
    for title, keys, unit, (lo, hi) in CHARTS:
        series = [{"name": _name(k, known), "values": clean(a.series[k], 3 if k == "fore_aft" else 1)}
                  for k in keys]
        tgt = TARGETS[a.opts.level][keys[0]]["ok"]
        bands = [list(tgt)]
        if keys[0] == "inclination":
            bands.append([-tgt[1], -tgt[0]])
        vals = [x for s in series for x in s["values"] if x is not None]
        if vals:  # 範圍至少涵蓋資料
            lo, hi = min(lo, min(vals)), max(hi, max(vals))
        charts.append({"title": title + ("（只顯示，不評分）" if keys[0] in a.skip else ""),
                       "unit": unit, "y": [lo, hi], "bands": bands, "series": series,
                       "turns": keys[0] == "inclination"})
    turns = [{"s": t[x.start], "e": t[min(x.end, len(t)) - 1], "side": x.side} for x in a.turns]
    return {"t": t, "charts": charts, "turns": turns}


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;
--good:#0ca30c;--warn:#fab219;--bad:#d03b3b;--band:rgba(12,163,12,.10);--toe:rgba(42,120,214,.10);--heel:rgba(235,104,52,.12)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;
--band:rgba(12,163,12,.16);--toe:rgba(57,135,229,.16);--heel:rgba(217,89,38,.18)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--band:rgba(12,163,12,.16);
--toe:rgba(57,135,229,.16);--heel:rgba(217,89,38,.18)}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Segoe UI","Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif}
main{max-width:920px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 12px}
.sub{color:var(--ink2);font-size:13px;word-break:break-all}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 0}
.chip{background:var(--surface);border:1px solid var(--border);border-radius:999px;padding:4px 12px;font-size:13px;color:var(--ink2)}
.chip b{color:var(--ink);font-weight:600}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px}
.hero{display:grid;grid-template-columns:1fr auto;gap:16px;align-items:start}
.hero h3{font-size:22px;margin:4px 0 8px}
.hero img{width:180px;border-radius:10px;display:block}
.hero figcaption{font-size:12px;color:var(--muted);margin-top:4px;max-width:180px}
.tag{display:inline-block;font-size:12px;color:var(--ink2);border:1px solid var(--border);border-radius:6px;padding:0 6px}
.hero p{margin:6px 0}.hero .k{color:var(--muted)}
@media (max-width:560px){.hero{grid-template-columns:1fr}.hero img{width:100%;max-width:300px}}
table{width:100%;min-width:560px;border-collapse:collapse;font-size:14px}
td.name{min-width:8.5em}.top{display:inline-block;font-size:11px;font-weight:600;color:var(--ink);border:1px solid var(--ink2);border-radius:6px;padding:0 5px;margin-left:4px}
th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--muted);font-weight:500;font-size:12px}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.st{white-space:nowrap;font-weight:600}
.st::before{display:inline-block;width:1.2em;text-align:center;margin-right:2px}
.st.good::before{content:"✓";color:var(--good)}.st.warn::before{content:"!";color:var(--warn)}
.st.bad::before{content:"✕";color:var(--bad)}.st.na::before{content:"–";color:var(--muted)}
.st.info::before{content:"i";color:var(--muted)}
.st.na,.st.info{color:var(--ink2);font-weight:400}
.reason{color:var(--muted);font-size:12px}
.tablewrap{overflow-x:auto}
.chart{margin:0 0 14px}.chart h3{font-size:14px;margin:0 0 2px;font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--ink2);margin:0 0 4px}
.legend i{display:inline-block;vertical-align:middle;margin-right:4px}
.legend .line{width:14px;height:2px}.legend .box{width:12px;height:10px;border-radius:2px}
svg.plot{width:100%;height:170px;display:block;touch-action:pan-y}
svg text{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;
padding:6px 10px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;z-index:9;min-width:120px}
.tip .v{font-weight:700;color:var(--ink);font-variant-numeric:tabular-nums}.tip .n{color:var(--ink2)}
.tip .key{display:inline-block;width:12px;height:2px;vertical-align:middle;margin-right:6px}
.turns{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
.turn img{width:100%;border-radius:8px;display:block;background:var(--grid)}
.turn .t{font-weight:600;margin:6px 0 0}.turn .d{font-size:12px;color:var(--ink2)}
.side{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:baseline}
details{margin-top:8px}summary{cursor:pointer;color:var(--ink2)}
ul{padding-left:20px}li{margin:4px 0}
.foot{color:var(--muted);font-size:12px;margin-top:32px}
.notice{margin:14px 0 0;padding:10px 12px;border:1px solid var(--border);border-left:4px solid var(--warn);
border-radius:8px;background:var(--surface);font-size:13px;color:var(--ink2)}
.notice b{color:var(--ink)}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const NS = 'http://www.w3.org/2000/svg';
const tip = document.getElementById('tip');
const COLORS = ['var(--s1)', 'var(--s2)'];
function el(tag, attrs, parent){ const e=document.createElementNS(NS,tag); for(const k in attrs) e.setAttribute(k,attrs[k]); if(parent) parent.appendChild(e); return e; }
function niceTicks(lo, hi, n){ const span=hi-lo, raw=span/n, mag=Math.pow(10,Math.floor(Math.log10(raw)));
  const step=[1,2,2.5,5,10].map(m=>m*mag).find(s=>span/s<=n)||raw; const out=[];
  for(let v=Math.ceil(lo/step)*step; v<=hi+1e-9; v+=step) out.push(+v.toFixed(6)); return out; }
function fmt(v, unit){ if(v===null||v===undefined) return '無法判斷'; return unit==='°' ? Math.round(v)+'°' : (v>=0?'+':'')+v.toFixed(2); }
function draw(box, c){
  const svg = box.querySelector('svg'); while(svg.firstChild) svg.removeChild(svg.firstChild);
  const W = svg.clientWidth || 600, H = 170, m = {l:40, r:10, t:8, b:22};
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const t = D.t, t0 = t[0], t1 = t[t.length-1] || 1;
  const X = v => m.l + (v - t0) / Math.max(t1 - t0, 1e-6) * (W - m.l - m.r);
  const [y0, y1] = c.y; const Y = v => m.t + (y1 - v) / (y1 - y0) * (H - m.t - m.b);
  if (c.turns) for (const tr of D.turns) el('rect', {x:X(tr.s), y:m.t, width:Math.max(1,X(tr.e)-X(tr.s)), height:H-m.t-m.b, fill: tr.side==='toe'?'var(--toe)':'var(--heel)'}, svg);
  for (const [lo, hi] of c.bands) { const a=Math.max(lo,y0), b=Math.min(hi,y1); if (b>a) el('rect', {x:m.l, y:Y(b), width:W-m.l-m.r, height:Y(a)-Y(b), fill:'var(--band)'}, svg); }
  for (const v of niceTicks(y0, y1, 4)) { el('line', {x1:m.l, x2:W-m.r, y1:Y(v), y2:Y(v), stroke: v===0?'var(--axis)':'var(--grid)', 'stroke-width':1}, svg);
    const tx = el('text', {x:m.l-6, y:Y(v)+4, 'text-anchor':'end'}, svg); tx.textContent = c.unit==='°' ? v+'°' : v; }
  for (const v of niceTicks(t0, t1, Math.max(2, Math.floor(W/80)))) { const tx = el('text', {x:X(v), y:H-6, 'text-anchor':'middle'}, svg); tx.textContent = v+'s'; }
  c.series.forEach((s, k) => { let d='', pen=false;
    s.values.forEach((v, i) => { if (v===null) { pen=false; return; } const yv=Math.max(y0, Math.min(y1, v)); d += (pen?'L':'M') + X(t[i]).toFixed(1) + ' ' + Y(yv).toFixed(1); pen=true; });
    el('path', {d, fill:'none', stroke:COLORS[k], 'stroke-width':2, 'stroke-linejoin':'round', 'stroke-linecap':'round'}, svg); });
  const cross = el('line', {y1:m.t, y2:H-m.b, stroke:'var(--axis)', 'stroke-width':1, visibility:'hidden'}, svg);
  const dots = c.series.map((s,k) => el('circle', {r:4, fill:COLORS[k], stroke:'var(--surface)', 'stroke-width':2, visibility:'hidden'}, svg));
  const hit = el('rect', {x:m.l, y:0, width:W-m.l-m.r, height:H, fill:'transparent'}, svg);
  function move(ev){ const r = svg.getBoundingClientRect(); const px = (ev.clientX - r.left) * W / r.width;
    const tv = t0 + (px - m.l) / (W - m.l - m.r) * (t1 - t0); let i = 0, best = Infinity;
    for (let j=0;j<t.length;j++){ const dd=Math.abs(t[j]-tv); if(dd<best){best=dd;i=j;} }
    cross.setAttribute('x1', X(t[i])); cross.setAttribute('x2', X(t[i])); cross.setAttribute('visibility','visible');
    while (tip.firstChild) tip.removeChild(tip.firstChild);
    const hd = document.createElement('div'); hd.className='n'; hd.textContent = t[i].toFixed(1)+' 秒'; tip.appendChild(hd);
    c.series.forEach((s,k) => { const v = s.values[i]; const row=document.createElement('div');
      const key=document.createElement('span'); key.className='key'; key.style.background=COLORS[k]; row.appendChild(key);
      const vv=document.createElement('span'); vv.className='v'; vv.textContent=fmt(v, c.unit); row.appendChild(vv);
      if (c.series.length>1) { const nn=document.createElement('span'); nn.className='n'; nn.textContent=' '+s.name; row.appendChild(nn); }
      tip.appendChild(row);
      if (v===null) dots[k].setAttribute('visibility','hidden'); else { dots[k].setAttribute('cx', X(t[i])); dots[k].setAttribute('cy', Y(Math.max(y0,Math.min(y1,v)))); dots[k].setAttribute('visibility','visible'); } });
    tip.style.display='block'; const tw=tip.offsetWidth; let lx=ev.clientX+14; if(lx+tw>window.innerWidth-8) lx=ev.clientX-tw-14;
    tip.style.left=lx+'px'; tip.style.top=(ev.clientY-10)+'px'; }
  function leave(){ tip.style.display='none'; cross.setAttribute('visibility','hidden'); dots.forEach(d=>d.setAttribute('visibility','hidden')); }
  hit.addEventListener('pointermove', move); hit.addEventListener('pointerdown', move); hit.addEventListener('pointerleave', leave);
}
const boxes = [...document.querySelectorAll('.chart')];
function drawAll(){ boxes.forEach((b,i) => draw(b, D.charts[i])); }
drawAll(); let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(drawAll, 120); });
"""


def build_html(a: "Analysis", images: dict[int, str]) -> str:
    known = a.stance.known
    lv = LEVELS[a.opts.level]
    detected = sum(p is not None for p in a.poses)
    src = {"selfie": "自拍", "follow": "他人拍攝"}[a.source.source]
    st = a.stance.stance.capitalize() if a.stance.stance else "無法確定"
    n_switch = sum(bool(a.stance.is_switch(i)) for i in range(len(a.indices)))
    enough = {m: ok for m, ok in enough_data(a.fj, a.turns, a.fps).items()}
    stats = metric_stats(a)
    E = _esc
    P: list[str] = []

    P.append(f"<h1>單板滑雪動作分析</h1><div class='sub'>{E(Path(a.info.path).name)} · "
             f"{a.times[0]:.1f}–{a.times[-1]:.1f} 秒</div>")
    chips = [("拍攝", src), ("站姿", st + (f"（switch {n_switch / a.fps:.1f} 秒）" if n_switch else "")),
             ("程度", lv["zh"]), ("偵測到騎手", f"{detected / len(a.indices):.0%}"), ("轉彎", f"{len(a.turns)} 個")]
    P.append("<div class='chips'>" + "".join(f"<span class='chip'>{E(k)} <b>{E(v)}</b></span>" for k, v in chips)
             + "</div>")
    P.append(f"<div class='notice' role='note'><b>⚠️ 非專業建議，僅供參考。</b>{E(DISCLAIMER)}</div>")

    # 最優先的建議
    P.append("<h2>最優先的建議</h2>")
    f = a.top
    if f is None:
        judged = [m for m, ok in enough.items() if ok and m not in a.skip]
        lacking = [m for m, ok in enough.items() if not ok and m not in a.skip]
        if judged:
            msg = (f"有足夠資料的指標（{'、'.join(_name(m, known) for m in judged)}）大致都在"
                   f"「{lv['zh']}」的目標區間內，沒有需要優先修正的問題。")
            if lacking:
                msg += f" 但{'、'.join(_name(m, known) for m in lacking)}的資料不足，無法下結論。"
        else:
            msg = "這段影片可判斷的資料太少，無法給出建議。請盡量讓全身（包含雙腳與板子）入鏡。"
        P.append(f"<div class='card'><p>{E(msg)}</p></div>")
    else:
        key = (f.metric, f.direction)
        cue, why, drill = (cue_text(key, known, i) for i in range(3))
        pos = _worst_frame(a, f.metric, f.direction)
        img = ""
        if pos is not None and images.get(pos):
            img = (f"<figure style='margin:0'><img alt='問題最明顯的畫面' src='data:image/jpeg;base64,{images[pos]}'>"
                   f"<figcaption>問題最明顯的畫面（{a.times[pos]:.1f} 秒）</figcaption></figure>")
        P.append(
            "<div class='card hero'><div>"
            f"<span class='tag'>{E(CATEGORY_ZH[METRICS[f.metric][2]])}</span>"
            f"<h3>{E(cue)}</h3>"
            f"<p><span class='k'>觀察　</span>{E(why)}</p>"
            f"<p><span class='k'>依據　</span>可判斷的 {f.samples} {E(f.unit)}中，{f.share:.0%} 超出目標區間，"
            f"其中 {f.bad_share:.0%} 明顯超出。</p>"
            f"<p><span class='k'>練習　</span>{E(drill)}</p>"
            "<p class='reason'>依「平衡 → 壓力 → 立刃 → 旋轉」的順序挑出排在最前面的問題，一次只練一件事；"
            "即使後面的指標偏離更多，也先處理這一項，後面的問題常會跟著改善。</p>"
            f"</div>{img}</div>")

    # 各項指標
    P.append("<h2>各項指標</h2><div class='card tablewrap'><table><thead><tr>"
             "<th>指標</th><th>狀態</th><th>中位數</th><th>目標</th><th>在目標內</th><th>可判斷</th></tr></thead><tbody>")
    for k in REPORT_METRICS:
        d = stats[k]
        if k in METRICS:
            cls, label = _status_label(a, k, enough)
        else:
            cls, label = "info", "僅供參考"
        reason = f"<div class='reason'>{E(d['top_reason'])}</div>" if d["top_reason"] and d["available"] < 0.5 else ""
        inb = "—" if d.get("in_band") is None else f"{d['in_band']:.0%}"
        med = "—" if d["median"] is None else _fmt(k, d["median"])
        top = "<span class='top'>最優先</span>" if a.top is not None and a.top.metric == k else ""
        P.append(f"<tr><td class='name'>{E(_name(k, known))}{top}{reason}</td><td><span class='st {cls}'>{E(label)}</span></td>"
                 f"<td class='num'>{E(med)}</td><td class='num'>{E(_band(a.opts.level, k))}</td>"
                 f"<td class='num'>{E(inb)}</td><td class='num'>{d['available']:.0%}</td></tr>")
    P.append("</tbody></table></div>")

    # 圖表
    P.append("<h2>隨時間變化</h2><p class='reason'>綠色底 = 目標區間；身體傾倒圖中藍色底 = 腳尖側轉彎、橘色底 = 腳跟側轉彎。"
             "線中斷處是無法判斷的片段。移動滑鼠或手指可看數值，逐格數據見 metrics.csv。</p><div class='card'>")
    for i, (title, keys, _, _) in enumerate(CHARTS):
        legend = []
        if len(keys) > 1:
            legend += [f"<span><i class='line' style='background:var(--s{j + 1})'></i>{E(_name(k, known))}</span>"
                       for j, k in enumerate(keys)]
        legend.append("<span><i class='box' style='background:var(--band);border:1px solid var(--good)'></i>"
                      f"目標區間（{E(lv['zh'])}）</span>")
        if keys[0] == "inclination":
            legend += ["<span><i class='box' style='background:var(--toe)'></i>腳尖側轉彎（上方為腳尖側）</span>",
                       "<span><i class='box' style='background:var(--heel)'></i>腳跟側轉彎</span>"]
        P.append(f"<div class='chart'><h3>{E(title)}</h3><div class='legend'>{''.join(legend)}</div>"
                 f"<svg class='plot' role='img' aria-label='{E(title)}隨時間變化'></svg></div>")
    P.append("</div>")

    # 轉彎
    P.append("<h2>每個轉彎</h2>")
    if not a.turns:
        P.append("<div class='card'><p>沒有偵測到明確的轉彎（身體傾倒資料不足，或一直直滑）。</p></div>")
    else:
        P.append("<div class='turns'>")
        for t in a.turns:
            pk = turn_peak(a.series["inclination"], t)
            seg = slice(t.start, t.end)

            def med(k: str) -> str:
                v = a.series[k][seg]
                v = v[~np.isnan(v)]
                return _fmt(k, float(np.median(v))) if len(v) else "—"
            pos = _turn_frame(a, t)
            img = f"<img alt='第 {t.index + 1} 個轉彎' src='data:image/jpeg;base64,{images[pos]}'>" \
                if pos is not None and images.get(pos) else ""
            side = "腳尖側" if t.side == "toe" else "腳跟側"
            color = "var(--s1)" if t.side == "toe" else "var(--s2)"
            P.append(f"<div class='turn card' style='padding:8px'>{img}"
                     f"<div class='t'><span class='side' style='background:{color}'></span>#{t.index + 1} {side}</div>"
                     f"<div class='d'>{t.start_time:.1f}–{t.end_time:.1f} 秒 · 傾倒 {E(_fmt('inclination', pk))}<br>"
                     f"{E(relabel('前膝', known))} {E(med('knee_front'))} · {E(relabel('後膝', known))} {E(med('knee_back'))}"
                     f" · 折髖 {E(med('angulation'))}</div></div>")
        P.append("</div>")

    # 說明
    P.append("<h2>說明與限制</h2><div class='card'><ul>")
    for n in a.notes:
        P.append(f"<li>{E(n)}</li>")
    P.append("<li>「無法判斷」代表該部位沒入鏡、被遮擋或估計不合理，工具不會用猜的補上數值。</li>"
             "<li>3D 姿勢由單一鏡頭影像推估，角度為近似值，適合看趨勢與前後比較。</li>"
             "<li>身體傾倒與折髖以畫面垂直方向當作重力方向；相機明顯傾斜時可用 <code>--roll</code> 修正。</li>"
             "<li>目標區間為一般教學原則的起始值。</li></ul>"
             "<details><summary>程度定義</summary><ul>"
             + "".join(f"<li><b>{E(v['zh'])}</b>：{E(v['desc'])}</li>" for v in LEVELS.values())
             + "</ul></details><details><summary>指標定義</summary><ul>")
    for k in REPORT_METRICS:
        dd = relabel(DEFINITIONS[k], known) if k.startswith("knee") else DEFINITIONS[k]
        P.append(f"<li><b>{E(_name(k, known))}</b>：{E(dd)}</li>")
    P.append("</ul></details></div>")
    P.append(f"<p class='foot'>姿勢模型：{E(a.opts.backend)}（找人：{E(a.opts.detector)}）· 鏡頭校正：{E(a.opts.lens)}"
             f"{' · 相機側傾修正 ' + format(a.opts.roll, '+.1f') + '°' if a.opts.roll else ''} · 由 snowcoach 產生"
             f"<br>{E(DISCLAIMER)}</p>")

    data = json.dumps(_series_json(a), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return ("<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>滑雪動作分析</title><style>{CSS}</style></head><body><main>{''.join(P)}</main>"
            f"<div class='tip' id='tip' role='status'></div>"
            f"<script type='application/json' id='data'>{data}</script><script>{JS}</script></body></html>")


def _turn_frame(a: "Analysis", t) -> int | None:
    """轉彎的代表畫面：傾倒最大的那一格。"""
    seg = np.abs(a.series["inclination"][t.start:t.end])
    if np.all(np.isnan(seg)):
        return None
    return int(t.start + np.nanargmax(seg))


def write_html_report(a: "Analysis", path: Path) -> None:
    positions = [p for p in (_turn_frame(a, t) for t in a.turns) if p is not None]
    if a.top is not None:
        w = _worst_frame(a, a.top.metric, a.top.direction)
        if w is not None:
            positions.append(w)
    images = grab_images(a, positions)
    path.write_text(build_html(a, images), encoding="utf-8")

