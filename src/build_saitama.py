#!/usr/bin/env python3
"""埼玉県 土地利用シミュレーション 単一HTML生成。

  python3 build_saitama.py          # data/saitama/{meta.json,fields.npz} を使う（実データ）
  python3 build_saitama.py --dummy  # 合成データ（UI検証用）

契約: docs/saitama_schema.md。時間場(uint8)を gzip→base64 で埋め込み、区分はブラウザで判定。
出力: site-saitama/index.html（外部通信ゼロ・単一ファイル）
"""
import argparse, base64, gzip, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(".")
DATA = ROOT / "data" / "saitama"
OUT = ROOT / "site-saitama" / "index.html"
NX, NP = 11, 3

PRESETS = [
    {"id": "real", "label": "現実的", "speeds": {"motorway": 80, "trunk": 50, "primary": 40, "secondary": 30, "tertiary": 22, "none": 12}},
    {"id": "congested", "label": "渋滞寄り", "speeds": {"motorway": 60, "trunk": 35, "primary": 28, "secondary": 20, "tertiary": 15, "none": 8}},
    {"id": "fast", "label": "高速寄り", "speeds": {"motorway": 100, "trunk": 60, "primary": 50, "secondary": 40, "tertiary": 30, "none": 15}},
]


def load_real():
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    f = np.load(DATA / "fields.npz")
    return meta, {k: np.asarray(f[k], dtype=np.uint8) for k in ("valid", "water", "tS", "tI", "core", "retreat")}


def make_dummy():
    """小さな楕円の県。契約と同じ形（11X段×3速度）。"""
    nc, nr, x0, y0 = 64, 44, 0.0, 0.0
    N = nc * nr
    cc, rr = np.meshgrid(np.arange(nc), np.arange(nr))
    cx, cy = (x0 + (cc + .5) * 500).ravel(), (y0 + (rr + .5) * 500).ravel()
    valid = ((((cc - 31.5) / 30) ** 2 + ((rr - 21.5) / 20) ** 2) <= 1).ravel().astype(np.uint8)
    wat = ((np.abs(rr - (12 + 0.2 * cc)) < 0.6) | (((cc - 44) ** 2 + (rr - 30) ** 2) < 10)).ravel()
    water = (wat & (valid == 1)).astype(np.uint8)
    X = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150]
    # (name, x_m, y_m, 生き残る最大X段)
    st = [("北駅", 12000, 15000, 10), ("中央駅", 16000, 11000, 10), ("東駅", 24000, 12000, 8),
          ("南駅", 15000, 5000, 6), ("西駅", 7000, 10000, 5), ("港駅", 22000, 18000, 3),
          ("丘駅", 10000, 19000, 2), ("端駅", 27000, 6000, 0), ("県外A", -3000, 11000, 10), ("県外B", 35000, 20000, 6)]
    stations = [{"name": n, "x": float(x), "y": float(y), "alive": [1 if i <= lim else 0 for i in range(NX)], "core_m": 1200} for n, x, y, lim in st]
    ics = [("北IC", 12500, 17000, "IC"), ("東IC", 25000, 14000, "IC"), ("南SIC", 14000, 3500, "SIC"), ("中JCT", 18000, 11000, "JCT"), ("県外IC", 36000, 8000, "IC")]
    ic = [{"name": n, "x": float(x), "y": float(y), "kind": k} for n, x, y, k in ics]
    avg = [28.0, 18.0, 38.0]
    tS = np.full((NX, NP, N), 255, np.uint8)
    core = np.zeros((NX, N), np.uint8)
    retreat = np.zeros((NX, N), np.uint8)
    d_all = np.stack([np.hypot(cx - s["x"], cy - s["y"]) for s in stations])
    for i in range(NX):
        al = np.array([s["alive"][i] for s in stations], bool)
        if al.any():
            dm = d_all[al].min(0)
            for p in range(NP):
                tS[i, p] = np.clip(dm / 1000 * 1.3 / avg[p] * 60, 0, 255).astype(np.uint8)
            core[i] = (dm <= 1200) & (valid == 1)
        near = d_all.argmin(0)
        dead = ~al[near]
        retreat[i] = (dead & (valid == 1) & (core[i] == 0)).astype(np.uint8)
    di = np.stack([np.hypot(cx - s["x"], cy - s["y"]) for s in ic]).min(0)
    tI = np.stack([np.clip(di / 1000 * 1.3 / (a * .7) * 60, 0, 255).astype(np.uint8) for a in avg])
    meta = {"grid": {"x0": x0, "y0": y0, "ncols": nc, "nrows": nr, "cell": 500, "crs": "EPSG:6677"},
            "X": X, "Xlabel": [f"{x}人/ha" for x in X], "r": 1200, "presets": PRESETS,
            "stations": stations, "ic": ic,
            "sources": ["【ダミーデータ】UI検証用の合成データです。実在の地形・駅・ICとは無関係です。"],
            "limits": ["【ダミーデータ】数値に意味はありません。"], "built": "dummy", "dummy": True}
    return meta, {"valid": valid, "water": water, "tS": tS.reshape(NX, NP, N), "tI": tI, "core": core, "retreat": retreat}


def check(meta, F):
    g = meta["grid"]; N = g["ncols"] * g["nrows"]
    exp = {"valid": (N,), "water": (N,), "tS": (NX, NP, N), "tI": (NP, N), "core": (NX, N), "retreat": (NX, N)}
    for k, s in exp.items():
        if F[k].shape != s:
            sys.exit(f"契約違反: {k} の形が {F[k].shape}（期待 {s}）")
    if len(meta["X"]) != NX or len(meta["presets"]) != NP:
        sys.exit("契約違反: X は11個・presets は3個")


def pack(F):
    raw = b"".join(np.ascontiguousarray(F[k]).tobytes() for k in ("valid", "water", "tS", "tI", "core", "retreat"))
    return base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    a = ap.parse_args()
    if a.dummy:
        meta, F = make_dummy()
    else:
        if not (DATA / "meta.json").exists() or not (DATA / "fields.npz").exists():
            sys.exit("data/saitama/meta.json / fields.npz がありません（--dummy で合成確認可）")
        meta, F = load_real()
    check(meta, F)
    html = TEMPLATE.replace("__META__", json.dumps(meta, ensure_ascii=False).replace("</", "<\\/")).replace("__DATA__", pack(F))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB, dummy={a.dummy})")


TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>埼玉 土地利用シミュレーション</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--mut:#5b6577;--line:#d9dee7;--acc:#0b5cad;--accsoft:#e3eefa;
--c0:#d55e00;--c1:#0072b2;--c2:#f0e442;--c3:#cc79a7;--c4:#4f7f5d;--c5:#c4cad3;--mapbg:#eef1f5;--ov:#1d2330;--warn:#b42318;--warnbg:#fde8e6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#10141b;--card:#181e28;--ink:#e8ecf3;--mut:#9aa5b8;--line:#2c3544;--acc:#6db1ff;--accsoft:#1c2c42;
--c0:#e8731c;--c1:#3a9bd8;--c2:#e6d93c;--c3:#d98ab6;--c4:#5f9470;--c5:#3a4352;--mapbg:#0b0e13;--ov:#f3f5f9;--warn:#ff8a7d;--warnbg:#3a1a17}}
:root[data-theme="dark"]{--bg:#10141b;--card:#181e28;--ink:#e8ecf3;--mut:#9aa5b8;--line:#2c3544;--acc:#6db1ff;--accsoft:#1c2c42;
--c0:#e8731c;--c1:#3a9bd8;--c2:#e6d93c;--c3:#d98ab6;--c4:#5f9470;--c5:#3a4352;--mapbg:#0b0e13;--ov:#f3f5f9;--warn:#ff8a7d;--warnbg:#3a1a17}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;overflow-x:hidden}
.wrap{max-width:1800px;margin:0 auto;padding:16px}
h1{font-size:1.35rem;margin:0 0 4px}
h2{font-size:1rem;margin:0 0 8px}
.sub{color:var(--mut);margin:0 0 12px;font-size:.9rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:12px;min-width:0}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:12px;align-items:start}
@media(max-width:760px){.grid{grid-template-columns:minmax(0,1fr)}.mapbox{min-height:0}.wrap{padding:10px 6px}.mapcard{padding:6px!important}}
.ctl{margin-bottom:14px}
.ctl label.t{display:flex;justify-content:space-between;gap:8px;font-weight:600;font-size:.92rem}
.ctl output{color:var(--acc);font-variant-numeric:tabular-nums;text-align:right}
input[type=range]{width:100%;margin:6px 0 2px;accent-color:var(--acc)}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip,.seg label{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:999px;padding:5px 12px;font-size:.85rem;cursor:pointer;min-height:34px;display:inline-flex;align-items:center}
.chip:hover{border-color:var(--acc)}
.seg{display:flex;flex-wrap:wrap;gap:6px}
.seg input{position:absolute;opacity:0;pointer-events:none}
.seg input:checked+label{background:var(--accsoft);border-color:var(--acc);color:var(--acc);font-weight:600}
.seg input:focus-visible+label,.chip:focus-visible,input[type=range]:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.seg{position:relative}
.mapbox{position:relative;background:var(--mapbg);border-radius:8px;overflow:hidden;line-height:0;height:78vh;min-height:460px}
canvas{display:block;width:100%;height:100%;touch-action:none;cursor:grab;user-select:none}
canvas.drag{cursor:grabbing}
.zb{position:absolute;top:8px;right:8px;display:flex;flex-direction:column;gap:6px;line-height:1}
.zb button{width:40px;height:40px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;font:600 18px system-ui;cursor:pointer;opacity:.94}
.zb button:hover{border-color:var(--acc)}
#zlv{position:absolute;left:8px;bottom:8px;background:var(--card);color:var(--mut);border:1px solid var(--line);border-radius:6px;padding:1px 8px;font-size:.78rem;line-height:1.5}
.legend{display:flex;flex-wrap:wrap;gap:4px 12px;margin-top:8px;font-size:.82rem}
.legend span{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.sw{width:14px;height:14px;border-radius:3px;border:1px solid rgba(0,0,0,.25);display:inline-block}
.layers{display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:8px;font-size:.88rem}
.layers label{display:inline-flex;align-items:center;gap:5px;min-height:30px;cursor:pointer}
table{width:100%;border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums}
th,td{padding:5px 6px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;font-size:.8rem}
.badge{display:inline-block;background:var(--warnbg);color:var(--warn);border:1px solid var(--warn);border-radius:999px;padding:0 9px;font-size:.78rem;font-weight:700;margin-left:6px;vertical-align:1px}
.cov{display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;padding:5px 0;border-bottom:1px solid var(--line);font-size:.9rem}
.mut{color:var(--mut);font-size:.82rem}
.q{display:grid;gap:8px}
.q button{text-align:left;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:10px 12px;font:inherit;cursor:pointer;min-height:44px}
.q button:hover{border-color:var(--acc);background:var(--accsoft)}
ul{margin:4px 0;padding-left:1.2em}
li{margin:2px 0}
.tbl{overflow-x:auto}
#probe{min-height:1.4em;font-size:.82rem;color:var(--mut);line-height:1.4;margin-top:6px}
.banner{background:var(--warnbg);color:var(--warn);border:1px solid var(--warn);border-radius:8px;padding:8px 12px;margin-bottom:12px;font-weight:600;display:none}
</style>
</head>
<body>
<div class="wrap">
<h1>埼玉 土地利用シミュレーション</h1>
<p class="sub">コンパクトシティ極限（生き残った駅のコア円に全員が住む）にしたとき、残りの土地は農地・工業・緑地にどう配分されるか。</p>
<div class="banner" id="dummyBanner">ダミーデータ表示中：数値に意味はありません（UI検証用）</div>
<div class="grid">
 <div>
  <div class="card mapcard">
   <div class="mapbox" id="stage"><canvas id="map" aria-label="土地区分の地図"></canvas>
    <div class="zb"><button id="zin" aria-label="拡大">+</button><button id="zout" aria-label="縮小">&minus;</button><button id="zrs" aria-label="リセット" style="font-size:13px">1x</button></div>
    <div id="zlv">ホイール/ピンチで拡大・ドラッグで移動</div></div>
   <div id="probe" aria-live="polite">地図をタップ／ホバーでセルの値を表示</div>
   <div class="legend" id="legend"></div>
   <div class="layers" id="layers">
    <label><input type="checkbox" id="lyS" checked>生存駅</label>
    <label><input type="checkbox" id="lyC" checked>コア円</label>
    <label><input type="checkbox" id="lyI">IC</label>
    <label><input type="checkbox" id="lyR">撤退地域</label>
   </div>
  </div>
  <div class="card" id="qcard">
   <h2>この画面が答える3つの問い</h2>
   <div class="q">
    <button id="q1">1. Xを上げると、居住・農地・工業・緑地の面積比はどう変わるか<br><span class="mut">X を最小にして動かし始める</span></button>
    <button id="q2">2. どのY・Zで全域を覆う（区分が意味を失う）か<br><span class="mut">Y=90・Z=30 にして飽和境目を表示</span></button>
    <button id="q3">3. 農地と工業が奪い合う土地（競合）はどこに、どれだけあるか<br><span class="mut">競合のまま表示し、競合を強調</span></button>
   </div>
  </div>
 </div>
 <div>
  <div class="card">
   <div class="ctl"><label class="t" for="sx">X　残す条件（コア円内の人口密度）<output id="ox"></output></label>
    <input type="range" id="sx" min="0" max="10" step="1" value="5"><div class="mut" id="xnote"></div></div>
   <div class="ctl"><label class="t" for="sy">Y　農地：駅から車で（分）<output id="oy"></output></label>
    <input type="range" id="sy" min="1" max="120" step="1" value="60">
    <div class="chips" data-for="sy"><button class="chip" data-v="30">30分</button><button class="chip" data-v="60">60分</button><button class="chip" data-v="90">90分</button></div></div>
   <div class="ctl"><label class="t" for="sz">Z　工業：ICから一般道で（分）<output id="oz"></output></label>
    <input type="range" id="sz" min="1" max="60" step="1" value="15">
    <div class="chips" data-for="sz"><button class="chip" data-v="10">10分</button><button class="chip" data-v="15">15分</button><button class="chip" data-v="30">30分</button></div></div>
   <div class="ctl"><div class="t" style="font-weight:600;font-size:.92rem;margin-bottom:6px">速度の仮定</div><div class="seg" id="segP"></div></div>
   <div class="ctl" style="margin-bottom:0"><div class="t" style="font-weight:600;font-size:.92rem;margin-bottom:6px">競合の扱い</div><div class="seg" id="segR">
    <input type="radio" name="prio" id="pr0" value="conflict" checked><label for="pr0">競合のまま</label>
    <input type="radio" name="prio" id="pr1" value="industry"><label for="pr1">工業へ</label>
    <input type="radio" name="prio" id="pr2" value="farm"><label for="pr2">農地へ</label></div></div>
  </div>
  <div class="card">
   <h2>指標</h2>
   <table><thead><tr><th>区分</th><th>km²</th><th>%</th></tr></thead><tbody id="tb"></tbody></table>
   <div class="cov"><span>農地側の到達圏被覆率（居住を除く）<span id="bF"></span></span><b id="vF"></b></div>
   <div class="cov"><span>工業側の到達圏被覆率（居住を除く）<span id="bI"></span></span><b id="vI"></b></div>
   <div class="cov"><span>生存駅数（県内）</span><b id="vS"></b></div>
   <div class="cov"><span>撤退地域面積</span><b id="vR"></b></div>
   <div class="mut" id="sat" style="margin-top:6px"></div>
   <div class="mut" id="ms"></div>
  </div>
 </div>
</div>
<div class="card">
 <h2>仮定：速度表（km/h）</h2>
 <div class="tbl"><table id="spd"></table></div>
 <p class="mut">所要時間は上表の速度で道路種別ごとに計算した最短経路。道路外（none）は徒歩に近い低速で接続。</p>
</div>
<div class="grid">
 <div class="card"><h2>出典</h2><ul id="src"></ul></div>
 <div class="card"><h2>限界</h2><ul id="lim"></ul></div>
</div>
<p class="mut" id="built"></p>
</div>
<script type="application/json" id="meta">__META__</script>
<script type="text/plain" id="blob">__DATA__</script>
<script>
"use strict";
const NX=11,NP=3,NAMES=["居住","工業","農地","競合","緑地・森林","区分外"];
const META=JSON.parse(document.getElementById("meta").textContent);
const G=META.grid,NC=G.ncols,NR=G.nrows,N=NC*NR,CELL=G.cell||500,KM2=CELL*CELL/1e6;
let valid,water,tS,tI,core,retreat; // tS[x][p] は subarray
const PRIO={conflict:0,industry:1,farm:2};
/* 契約の区分判定。x:段0..10, p:速度0..2(またはid), prio:"conflict"|"industry"|"farm"。有効外セルは 255。 */
function classify(x,y,z,preset,prio){
  const p=typeof preset==="string"?META.presets.findIndex(q=>q.id===preset):preset;
  const out=new Uint8Array(N), ts=tS[x][p], ti=tI[p], co=core[x];
  const cf= prio==="industry"?1:(prio==="farm"?2:3);
  for(let i=0;i<N;i++){
    if(!valid[i]){out[i]=255;continue}
    if(water[i]){out[i]=5;continue}
    let c=4;
    if(co[i]) c=0;
    else{const f=ts[i]<=y,d=ti[i]<=z; if(f&&d)c=cf; else if(d)c=1; else if(f)c=2;}
    out[i]=c;
  }
  return out;
}
window.classify=classify;
/* 被覆率（優先に依存しない）: 分母=有効かつ非水域かつ非コア。分子=そのうち farm / ind の数 */
function coverage(x,y,z,p){
  const ts=tS[x][p],ti=tI[p],co=core[x];let den=0,nf=0,ni=0;
  for(let i=0;i<N;i++){ if(!valid[i]||water[i]||co[i])continue; den++; if(ts[i]<=y)nf++; if(ti[i]<=z)ni++; }
  return {den,nf,ni};
}
/* 被覆率が95%を超える最小のY/Z（無ければ null） */
function satPoint(arr,co,den){
  const h=new Float64Array(256);
  for(let i=0;i<N;i++){ if(!valid[i]||water[i]||co[i])continue; h[arr[i]]++; }
  let s=0; for(let t=0;t<256;t++){ s+=h[t]; if(den&&s/den>0.95) return t; } return null;
}
const st={x:5,y:60,z:15,p:0,prio:"conflict",focus:-1,cls:null};
const $=id=>document.getElementById(id);
function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
function hex(c){const m=c.replace("#","");return [parseInt(m.slice(0,2),16),parseInt(m.slice(2,4),16),parseInt(m.slice(4,6),16)]}
const cv=$("map"),ctx=cv.getContext("2d");
const off=document.createElement("canvas");off.width=NC;off.height=NR;const octx=off.getContext("2d");
const roff=document.createElement("canvas");roff.width=NC;roff.height=NR;const rctx=roff.getContext("2d");
let pal=[];
function readPal(){pal=[0,1,2,3,4,5].map(i=>hex(css("--c"+i)));}
const V={k:1,tx:0,ty:0,w:0,h:0,s0:1};
function sc(){return V.s0*V.k}
function clampV(){const s=sc(),mw=NC*s,mh=NR*s;
  V.tx=mw<=V.w?(V.w-mw)/2:Math.min(0,Math.max(V.w-mw,V.tx));
  V.ty=mh<=V.h?(V.h-mh)/2:Math.min(0,Math.max(V.h-mh,V.ty));}
function px(wx,wy){const s=sc();return [(wx-G.x0)/CELL*s+V.tx,(NR-(wy-G.y0)/CELL)*s+V.ty]}
function sizeStage(){
  const stg=$("stage"),W=stg.parentElement.clientWidth;
  const narrow=window.innerWidth<=760;
  const hh=narrow?Math.min(Math.round(window.innerHeight*.66),Math.round(W*NR/NC)+16):Math.max(460,Math.round(window.innerHeight*.78));
  stg.style.height=hh+"px";
  const nw=cv.clientWidth,nh=hh;
  V.w=nw;V.h=nh;V.s0=Math.min(nw/NC,nh/NR);
}
function zoomAt(f,mx,my){
  const k0=V.k,k1=Math.min(16,Math.max(1,k0*f));if(k1===k0)return;
  const s0=sc();const cx=(mx-V.tx)/s0,cy=(my-V.ty)/s0;
  V.k=k1;const s1=sc();V.tx=mx-cx*s1;V.ty=my-cy*s1;clampV();viewChanged();
}
function viewChanged(){$("zlv").textContent=V.k===1?"ホイール/ピンチで拡大・ドラッグで移動":"拡大 "+V.k.toFixed(1)+"倍";draw()}
function drawBase(){
  const img=octx.createImageData(NC,NR),d=img.data,cls=st.cls,mb=hex(css("--mapbg")),f=st.focus;
  for(let i=0;i<N;i++){
    const c=cls[i]; if(c===255)continue;
    const col=(i/NC)|0,row=i-col*NC; // row=col変数名の入替: col→行, row→列
    const o=((NR-1-col)*NC+row)*4; let r=pal[c];
    let R=r[0],Gc=r[1],B=r[2];
    if(f>=0&&c!==f){R=(R+mb[0]*3)>>2;Gc=(Gc+mb[1]*3)>>2;B=(B+mb[2]*3)>>2}
    d[o]=R;d[o+1]=Gc;d[o+2]=B;d[o+3]=255;
  }
  octx.putImageData(img,0,0);
}
function hatch(){const h=document.createElement("canvas");h.width=h.height=8;const g=h.getContext("2d");g.strokeStyle=css("--ov");g.lineWidth=1.6;g.beginPath();g.moveTo(0,8);g.lineTo(8,0);g.moveTo(-2,2);g.lineTo(2,-2);g.moveTo(6,10);g.lineTo(10,6);g.stroke();return h}
const tmpC=document.createElement("canvas");
function draw(){
  const dpr=window.devicePixelRatio||1;
  sizeStage();clampV();const w=V.w,h=V.h;
  if(cv.width!==Math.round(w*dpr)||cv.height!==Math.round(h*dpr)){cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr)}
  const s=sc();
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
  ctx.imageSmoothingEnabled=false;ctx.drawImage(off,V.tx,V.ty,NC*s,NR*s);
  const S=META.stations,X=st.x;
  if($("lyR").checked){
    const rm=rctx.createImageData(NC,NR),rd=rm.data,re=retreat[X];
    for(let i=0;i<N;i++){ if(re[i]&&valid[i]){const col=(i/NC)|0,row=i-col*NC,o=((NR-1-col)*NC+row)*4;rd[o+3]=255} }
    rctx.putImageData(rm,0,0);
    const t=tmpC;t.width=Math.max(1,Math.round(w));t.height=Math.round(h);const tc=t.getContext("2d");
    tc.clearRect(0,0,t.width,t.height);tc.fillStyle=tc.createPattern(hatch(),"repeat");tc.fillRect(0,0,t.width,t.height);
    tc.globalCompositeOperation="destination-in";tc.imageSmoothingEnabled=false;tc.drawImage(roff,V.tx,V.ty,NC*s,NR*s);tc.globalCompositeOperation="source-over";
    ctx.globalAlpha=.55;ctx.drawImage(t,0,0,w,h);ctx.globalAlpha=1;
  }
  const ov=css("--ov"),bg=css("--card");
  if($("lyC").checked){
    ctx.lineWidth=1.5;ctx.strokeStyle=ov;
    for(const s of S){ if(!s.alive[X])continue; const [a,b]=px(s.x,s.y),r=Math.max(5,(s.core_m||META.r)/CELL*sc()); ctx.beginPath();ctx.arc(a,b,r,0,6.2832);ctx.stroke(); }
  }
  if($("lyS").checked){
    for(const s of S){ if(!s.alive[X])continue; const [a,b]=px(s.x,s.y); if(a<-20||b<-20||a>w+20||b>h+20)continue; ctx.beginPath();ctx.arc(a,b,3.6,0,6.2832);ctx.fillStyle=bg;ctx.fill();ctx.lineWidth=2;ctx.strokeStyle=ov;ctx.stroke();
      if(V.k>=4){ctx.font="12px system-ui,'Noto Sans JP',sans-serif";ctx.lineWidth=3;ctx.strokeStyle=bg;ctx.fillStyle=ov;ctx.strokeText(s.name,a+7,b+4);ctx.fillText(s.name,a+7,b+4);} }
  }
  if($("lyI").checked){
    for(const s of META.ic){ const [a,b]=px(s.x,s.y),q=4.5; ctx.beginPath();ctx.moveTo(a,b-q);ctx.lineTo(a+q,b);ctx.lineTo(a,b+q);ctx.lineTo(a-q,b);ctx.closePath();ctx.fillStyle=s.kind==="JCT"?bg:ov;ctx.fill();ctx.lineWidth=1.5;ctx.strokeStyle=bg;ctx.stroke(); }
  }
}
function fmt(v,d=1){return v.toLocaleString("ja-JP",{minimumFractionDigits:d,maximumFractionDigits:d})}
function metrics(){
  const cnt=new Float64Array(6);let vn=0;const cls=st.cls;
  for(let i=0;i<N;i++){const c=cls[i];if(c!==255){cnt[c]++;vn++}}
  const rows=[0,1,2,3,4,5].map(c=>`<tr><td><span class="sw" style="background:var(--c${c})"></span> ${NAMES[c]}</td><td>${fmt(cnt[c]*KM2)}</td><td>${vn?fmt(cnt[c]/vn*100):"-"}</td></tr>`);
  rows.push(`<tr><td><b>県内計</b></td><td><b>${fmt(vn*KM2)}</b></td><td>100.0</td></tr>`);
  $("tb").innerHTML=rows.join("");
  const cv_=coverage(st.x,st.y,st.z,st.p),pf=cv_.nf/cv_.den*100,pi=cv_.ni/cv_.den*100;
  $("vF").textContent=fmt(pf)+"%";$("vI").textContent=fmt(pi)+"%";
  $("bF").innerHTML=pf>95?'<span class="badge">飽和</span>':"";$("bI").innerHTML=pi>95?'<span class="badge">飽和</span>':"";
  let ns=0;for(const s of META.stations){ if(!s.alive[st.x])continue; const c=Math.floor((s.x-G.x0)/CELL),r=Math.floor((s.y-G.y0)/CELL); if(c>=0&&c<NC&&r>=0&&r<NR&&valid[r*NC+c])ns++; }
  $("vS").textContent=ns+" 駅";
  let rn=0;const re=retreat[st.x];for(let i=0;i<N;i++)if(re[i]&&valid[i])rn++;
  $("vR").textContent=fmt(rn*KM2)+" km²";
  const ys=satPoint(tS[st.x][st.p],core[st.x],cv_.den),zs=satPoint(tI[st.p],core[st.x],cv_.den);
  $("sat").textContent=`飽和境目（この X・速度）：農地 ${ys===null?"到達せず":"Y≥"+ys+"分"} ／ 工業 ${zs===null?"到達せず":"Z≥"+zs+"分"}（被覆率95%超）`;
  return {cnt,pf,pi,ns,rn};
}
let pend=false;
function update(){
  const t0=performance.now();
  st.cls=classify(st.x,st.y,st.z,st.p,st.prio);
  readPal();drawBase();draw();const m=metrics();
  const ms=performance.now()-t0;window.__lastMs=ms;window.__metrics=m;
  $("ms").textContent=`再計算 ${ms.toFixed(1)} ms`;
}
function syncLabels(){
  $("ox").textContent=(META.Xlabel&&META.Xlabel[st.x])||("X="+META.X[st.x]);
  $("xnote").textContent=`段 ${st.x+1}/${NX}　X=${META.X[st.x]} 人/ha`;
  $("oy").textContent=st.y+" 分";$("oz").textContent=st.z+" 分";
}
function set(k,v){st[k]=v;syncLabels();update()}
$("sx").oninput=e=>set("x",+e.target.value);
$("sy").oninput=e=>set("y",+e.target.value);
$("sz").oninput=e=>set("z",+e.target.value);
document.querySelectorAll(".chips").forEach(c=>c.onclick=e=>{const b=e.target.closest("button");if(!b)return;const s=$(c.dataset.for);s.value=b.dataset.v;s.dispatchEvent(new Event("input"))});
document.querySelectorAll('input[name=prio]').forEach(r=>r.onchange=e=>{set("prio",e.target.value)});
["lyS","lyC","lyI","lyR"].forEach(id=>$(id).onchange=()=>draw());
const segP=$("segP");
META.presets.forEach((p,i)=>{segP.insertAdjacentHTML("beforeend",`<input type="radio" name="sp" id="sp${i}" value="${i}"${i===0?" checked":""}><label for="sp${i}">${p.label}</label>`)});
segP.onchange=e=>set("p",+e.target.value);
$("legend").innerHTML=[0,1,2,3,4,5].map(c=>`<span><i class="sw" style="background:var(--c${c})"></i>${NAMES[c]}</span>`).join("")+
 '<span>○ 生存駅</span><span>◯ コア円</span><span>◆ IC</span><span>斜線 撤退地域</span>';
$("q1").onclick=()=>{$("sx").value=0;$("sx").dispatchEvent(new Event("input"));st.focus=-1;$("sx").scrollIntoView({block:"center",behavior:"smooth"});$("sx").focus({preventScroll:true})};
$("q2").onclick=()=>{st.focus=-1;$("sy").value=90;$("sz").value=30;st.y=90;st.z=30;syncLabels();update();$("sat").scrollIntoView({block:"center",behavior:"smooth"})};
$("q3").onclick=()=>{st.focus=3;document.getElementById("pr0").checked=true;st.prio="conflict";update();cv.scrollIntoView({block:"center",behavior:"smooth"})};
$("legend").onclick=e=>{};
$("legend").querySelectorAll("span").forEach((s,i)=>{if(i<6){s.style.cursor="pointer";s.title="クリックで強調／解除";s.onclick=()=>{st.focus=st.focus===i?-1:i;update()}}});
function probe(ev){
  const r=cv.getBoundingClientRect(),mx=ev.clientX-r.left,my=ev.clientY-r.top,s=sc();
  const c=Math.floor((mx-V.tx)/s),row=NR-1-Math.floor((my-V.ty)/s);if(c<0||c>=NC||row<0||row>=NR)return;
  const i=row*NC+c;if(!valid[i]){$("probe").textContent="県外";return}
  $("probe").textContent=`${NAMES[st.cls[i]]}　駅から ${tS[st.x][st.p][i]}分 ／ ICから ${tI[st.p][i]}分${core[st.x][i]?" ／ コア円内":""}${retreat[st.x][i]?" ／ 撤退地域":""}`;
}
window.__view=V;
const ptr=new Map();let pinch=null,moved=0;
cv.addEventListener("wheel",e=>{e.preventDefault();const r=cv.getBoundingClientRect();zoomAt(Math.exp(-e.deltaY*(e.deltaMode?0.05:0.0015)),e.clientX-r.left,e.clientY-r.top)},{passive:false});
cv.addEventListener("pointerdown",e=>{cv.setPointerCapture(e.pointerId);ptr.set(e.pointerId,{x:e.clientX,y:e.clientY});moved=0;cv.classList.add("drag");
  if(ptr.size===2){const [a,b]=[...ptr.values()];pinch={d:Math.hypot(a.x-b.x,a.y-b.y)}}else probe(e)});
cv.addEventListener("pointermove",e=>{
  const p=ptr.get(e.pointerId);if(!p){probe(e);return}
  const dx=e.clientX-p.x,dy=e.clientY-p.y;p.x=e.clientX;p.y=e.clientY;
  if(ptr.size===2&&pinch){const [a,b]=[...ptr.values()],d=Math.hypot(a.x-b.x,a.y-b.y),r=cv.getBoundingClientRect();
    zoomAt(d/pinch.d,(a.x+b.x)/2-r.left,(a.y+b.y)/2-r.top);pinch.d=d;}
  else if(ptr.size===1){moved+=Math.abs(dx)+Math.abs(dy);V.tx+=dx;V.ty+=dy;clampV();draw();if(moved<4)probe(e)}
});
const up=e=>{ptr.delete(e.pointerId);pinch=null;if(!ptr.size)cv.classList.remove("drag")};
cv.addEventListener("pointerup",up);cv.addEventListener("pointercancel",up);
function zc(f){zoomAt(f,V.w/2,V.h/2)}
$("zin").onclick=()=>zc(1.6);$("zout").onclick=()=>zc(1/1.6);
$("zrs").onclick=()=>{V.k=1;V.tx=V.ty=0;clampV();viewChanged()};
new ResizeObserver(()=>{if(st.cls)draw()}).observe(cv);
matchMedia("(prefers-color-scheme:dark)").addEventListener("change",()=>st.cls&&update());
function list(id,a){$(id).innerHTML=(a||[]).map(s=>`<li>${String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;")}</li>`).join("")}
list("src",META.sources);list("lim",META.limits);
$("built").textContent="データ作成日："+(META.built||"-");
if(META.dummy)$("dummyBanner").style.display="block";
(function(){
  const k=["道路種別",...META.presets.map(p=>p.label)];
  const keys=Object.keys(META.presets[0].speeds);
  $("spd").innerHTML="<thead><tr>"+k.map(h=>"<th>"+h+"</th>").join("")+"</tr></thead><tbody>"+
   keys.map(key=>"<tr><td>"+key+"</td>"+META.presets.map(p=>"<td>"+p.speeds[key]+"</td>").join("")+"</tr>").join("")+"</tbody>";
})();
async function gunzip(b64){
  const bin=atob(b64),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);
  const ds=new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream("gzip")));
  return new Uint8Array(await ds.arrayBuffer());
}
window.ready=(async()=>{
  const raw=await gunzip(document.getElementById("blob").textContent.trim());
  let o=0;const take=n=>{const a=raw.subarray(o,o+n);o+=n;return a};
  valid=take(N);water=take(N);
  tS=[];for(let x=0;x<NX;x++){tS.push([]);for(let p=0;p<NP;p++)tS[x].push(take(N))}
  tI=[];for(let p=0;p<NP;p++)tI.push(take(N));
  core=[];for(let x=0;x<NX;x++)core.push(take(N));
  retreat=[];for(let x=0;x<NX;x++)retreat.push(take(N));
  window.__data={valid,water,tS,tI,core,retreat,META};
  syncLabels();update();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
