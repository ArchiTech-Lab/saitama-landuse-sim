"""
本州スケールの収斂過程を再生する自己完結HTMLを書き出す。

  python3 src/build_replay.py           # 合成データ → docs/honshu_replay.html
  python3 src/build_replay.py --real    # 実データ   → docs/honshu_replay_real.html

画面でできること:
  * X（成立密度 人/ha）の切替。ここで全プリセットを解いて全部埋め込む
    （CAP＝コア円の実効面積×X ＝ X に厳密に線形なので、割当も隣接グラフも作り直さない）
  * 拡大・縮小・パン（ホイール／ドラッグ）
  * **コア円（最寄駅から半径1200m）の可視化**。CAPの分母はここであってボロノイ全体ではない
  * 駅名の表示とホバーでの照会・駅名検索

出力は単一HTML（PNGと配列を base64 で内包。外部通信ゼロ・file:// で開ける）。
重い計算はここ（Python）で済ませ、ブラウザは再生しかしない。
"""
import base64, json, math, os, struct, sys, time, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import honshu_core as hc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "honshu_replay.html")
STEP = 2          # 合成データの表示用の間引き（500mメッシュ → 1km/px）
FRAMES = 260      # 再生フレーム数
RCORE = 1.2       # コア円の半径 km（仕様書 §2：CAPの分母はこの円に固定）
REAL = "--real" in sys.argv

# ---- X プリセット（旧プロトタイプ station_consolidation_simulator.html の業態別Xを踏襲）----
#   X = 商圏人口 ÷ 商圏円の面積(ha)。500m円＝78.54ha、1200m円＝452.39ha。
PRESETS = [
    (10.0,  "参考：10人/ha（ほぼ制約なし）",
     "下限側の対照。この水準ではほとんどの駅が現況のまま残り、統合はわずかしか起きない。"),
    (19.1,  "喫茶店（個人）立地確率50%・1,500人/500m円",
     "人口1,500人規模の自治体の半数に個人喫茶がある、という所在確率から。密度制約としてはほぼ効かない。"),
    (31.8,  "コンビニ 実店舗平均・2,500人/500m円",
     "店舗数から逆算した1店あたり人口の平均。約7割の都市が2,000〜3,000人に収まる。"),
    (38.2,  "コンビニ 商圏3,000人/500m円",
     "立地適正化計画の40人/haの直接の根拠。3,000人 ÷ 78.54ha ≒ 38人/ha。"),
    (40.0,  "立地適正化計画 40人/ha（既定）",
     "都市計画運用指針の既成市街地基準。多くの自治体が居住誘導区域の目標密度に採用している。"),
    (50.0,  "50人/ha",
     "40と60.8のあいだ。刻みの効き方を見るための中間点。"),
    (60.8,  "ハンバーガー店 立地確率50%・27,500人/1200m円",
     "所在確率が半分を切るのが人口2〜3万人水準。チェーン飲食が成立するかどうかの分岐点。"),
    (82.8,  "喫茶店（個人）立地確率80%・6,500人/500m円",
     "80%の自治体に存在する人口規模。500m円の値を1200m円に持ち込むので上限側に振れた近似。"),
    (100.0, "100人/ha",
     "政令市の中心区の平均密度帯。"),
    (160.3, "大手コーヒーチェーン 立地確率50%・72,500人/1200m円",
     "1200m円452haに押し込むと160人/ha相当＝都心・副都心級。"),
    (182.4, "大手コーヒーチェーン 立地確率80%・82,500人/1200m円",
     "ほぼ確実に立地する人口規模。本州の大半が構造的に到達できない水準。"),
]
DEFAULT_X = 40.0
FULL_HA = math.pi * RCORE * RCORE * 100      # 452.39 ha（コア円 r=1200m の満円）
SUB = 4                                      # セルを SUB×SUB に細分して円との重なりを取る


def png(rgb):
    """numpy (h,w,3) uint8 → PNGバイト列（Pillow を使わない）"""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def subgrid():
    o = (np.arange(SUB) + .5) / SUB - .5
    gx, gy = np.meshgrid(o, o)
    return gx.ravel(), gy.ravel()


t0 = time.time()
if REAL:                       # ---- 本州の実データ（build_real.py が保存した npz）----
    import shpdbf
    from pyproj import Transformer
    OUT = os.path.join(HERE, "..", "docs", "honshu_replay_real.html")
    z = np.load(os.path.join(HERE, "..", "data", "honshu.npz"), allow_pickle=True)
    S, cellpop = z["S"], z["cellpop"]
    coreha = z["coreha"].astype(np.float64)          # コア円の実効面積(ha)。CAP = coreha × X
    row, col, owner = z["row"], z["col"], z["owner"]
    names = [str(v) for v in z["name"]]
    pr = z["edges"]
    N = len(S)
    adj = [[] for _ in range(N)]
    for a, b in pr:
        adj[a].append(int(b)); adj[b].append(int(a))

    # --- セルとコア円の重なり（build_real.py と同じ式。coreha を再現できることで検算する）---
    TR = Transformer.from_crs("EPSG:6668",
        "+proj=laea +lat_0=37 +lon_0=138 +ellps=GRS80 +units=m +no_defs", always_xy=True)
    la, lo = shpdbf.mesh500_center(row, col)
    MX, MY = TR.transform(lo, la)
    M = np.c_[MX, MY] / 1000.0
    cw_h = 6371.0088 * math.radians(1 / 240.0)                    # セルの高さ km（一定）
    cw_w = 6371.0088 * math.radians(1 / 160.0) * np.cos(np.radians(la))   # 幅 km（緯度で変わる）
    gx, gy = subgrid()
    dx = M[:, 0] - S[owner, 0]; dy = M[:, 1] - S[owner, 1]
    frac = (((dx[:, None] + gx[None, :] * cw_w[:, None]) ** 2 +
             (dy[:, None] + gy[None, :] * cw_h) ** 2) <= RCORE ** 2).mean(1)
    chk = np.bincount(owner, weights=z["area_ha"] * frac, minlength=N)
    chk = np.where(chk > 0, chk, FULL_HA)
    assert np.abs(chk - coreha).max() < 1e-3, "コア円の再現に失敗（投影かメッシュ中心の式が違う）"

    r0, c0 = int(row.min()), int(col.min())
    nr, nc = int(row.max()) - r0 + 1, int(col.max()) - c0 + 1
    og = np.full((nr, nc), -1, np.int32)
    fg = np.zeros((nr, nc), np.float32)
    og[row - r0, col - c0] = owner
    fg[row - r0, col - c0] = frac
    og = og[::-1]; fg = fg[::-1]                     # 行0を北端に
    # 駅の画素位置（所属セルの重心）と、コア円の画素半径（緯度で横が伸縮する）
    cpx = (col - c0) + .5
    rpx = (nr - 1 - (row - r0)) + .5
    sp = np.zeros((N, 2)); slat = np.zeros(N)
    for i in range(N):
        sel = np.nonzero(owner == i)[0]
        if len(sel):
            sp[i] = (cpx[sel].mean(), rpx[sel].mean()); slat[i] = la[sel].mean()
        else:
            sp[i] = (np.nan, np.nan); slat[i] = 36.0
    rx = RCORE / (6371.0088 * math.radians(1/160.0) * np.cos(np.radians(slat)))
    ry = RCORE / cw_h
    kmx, kmy = float(np.mean(cw_w)), cw_h
    starea = np.bincount(owner, weights=z["area_ha"], minlength=N)   # 駅ごとのボロノイ面積(ha)
    meta_extra = dict(real=1, cells=int((og >= 0).sum()))
else:
    G = hc.build_synthetic()
    A0 = hc.assign_and_graph(G)                     # X=40 基準。coreha = capn/40
    N, adj, cellpop = A0["N"], A0["adj"], A0["cellpop"]
    coreha = A0["capn"] / 40.0
    S = G["S"]; names = None
    og = A0["og"][::STEP, ::STEP][::-1]
    nr, nc = og.shape
    cell = G["cellkm"] * STEP                       # 1px の一辺 km
    xs = (np.arange(nc) + .5) * cell
    ys = (np.arange(nr) + .5) * cell                # 行0が北端なので y は反転して使う
    gx, gy = subgrid()
    ox = xs[None, :] - S[np.maximum(og, 0), 0]
    oy = (hc.H - ys[:, None]) - S[np.maximum(og, 0), 1]
    fg = np.zeros((nr, nc), np.float32)
    acc = np.zeros((nr, nc), np.float32)
    for a, b in zip(gx, gy):
        acc += ((ox + a*cell)**2 + (oy + b*cell)**2 <= RCORE**2)
    fg = np.where(og >= 0, acc / len(gx), 0).astype(np.float32)
    sp = np.c_[S[:, 0] / cell, nr - S[:, 1] / cell]
    rx = np.full(N, RCORE / cell); ry = RCORE / cell
    kmx = kmy = cell
    starea = np.bincount(A0["owner"], minlength=N) * (G["cellkm"]**2 * 100)
    meta_extra = dict(real=0, cells=int((og >= 0).sum()))
totalpop = float(cellpop.sum())
print(f"母点 {N:,} / 隣接辺 {sum(len(a) for a in adj)//2:,} / 人口 {totalpop:,.0f} "
      f"/ コア円にかかるセル {int((fg > 0).sum()):,}（うち完全内側 {int((fg >= .999).sum()):,}）")

# ---- 所属ラスタを PNG に（Xによらず不変。R=id>>8, G=id&255, B=コア円との重なり×255）----
ids = np.where(og >= 0, og, 65535).astype(np.uint16)
h, w = ids.shape
rgb = np.dstack([(ids >> 8).astype(np.uint8), (ids & 255).astype(np.uint8),
                 np.round(fg * 255).astype(np.uint8)])
png_b64 = base64.b64encode(png(rgb)).decode()
assert N < 65535, "駅数が uint16 を超えるとイベント列の型を変える必要がある"

b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
SHARE = {1: .22, 2: .16, 3: .02, 4: .57, 5: .03}   # 工程ごとのフレーム取り分

runs = []
for X, label, note in PRESETS:
    ta = time.time()
    capn = coreha * X
    A = dict(N=N, cellpop=cellpop, capn=capn, adj=adj)
    v2 = hc.run_v2(A, S, record=True)
    ev = v2["events"]
    ph = np.array([e[0] for e in ev])
    cnt = {k: int((ph == k).sum()) for k in SHARE}
    wgt = np.array([SHARE[p] / cnt[p] if cnt[p] else 0.0 for p in ph])
    cum = np.cumsum(wgt); cum /= cum[-1]
    fidx = np.minimum((cum * FRAMES).astype(int), FRAMES - 1)
    bounds = np.append(np.searchsorted(fidx, np.arange(FRAMES), side="left"),
                       len(ev)).astype(np.int32)
    nowmet = int((cellpop >= capn).sum())
    asg = v2["assign"]
    dead = np.array([i for i in range(N) if asg[i] >= 0 and asg[i] != i])
    darea = float(starea[dead].sum()) if len(dead) else 0.0
    runs.append(dict(
        X=X, label=label, note=note,
        cap=b64(capn.astype(np.float32)),
        evs=b64(np.array([[e[1], e[2]] for e in ev], np.uint16)),
        evp=b64(np.array([e[0] for e in ev], np.uint8)),
        bnd=b64(bounds), events=len(ev),
        alive=v2["alive"], short=v2["short"], repairs=v2["repairs"],
        moved=round(v2["moved"]*100, 1), dmed=round(v2["dmed"], 1),
        d95=round(v2["d95"], 1), dmax=round(v2["dmax"], 1),
        nowmet=nowmet, orphan=v2["orphan"], seas=v2["seas"],
        mass=round(v2["mass"]*100, 4),
        capmed=int(round(float(np.median(capn)))), capfull=int(round(FULL_HA*X)),
        dead=len(dead), deadarea=round(100*darea/starea.sum(), 1),
        deadkm2=int(round(darea/100))))
    print(f"  X={X:6.1f}  クラスタ {v2['alive']:5,}  未達 {v2['short']}  "
          f"現況達成 {nowmet:5,}  移動 {v2['moved']*100:5.1f}%  中位 {v2['dmed']:.1f}km  "
          f"解体 {v2['repairs']:5,}  淘汰 {len(dead):5,}駅・面積 {100*darea/starea.sum():4.1f}%  "
          f"({time.time()-ta:.1f}s)")

meta = dict(w=w, h=h, N=N, frames=FRAMES, totalpop=totalpop, rcore=RCORE,
            ry=float(ry), kmx=round(kmx, 4), kmy=round(kmy, 4),
            defi=[p[0] for p in PRESETS].index(DEFAULT_X), **meta_extra)
DATA = dict(meta=meta, png=png_b64, xy=b64(sp.astype(np.float32)),
            rx=b64(np.asarray(rx, np.float32)), coreha=b64(coreha.astype(np.float32)),
            pop=b64(cellpop.astype(np.float32)), names=names, runs=runs)

HTML = r"""<!doctype html><meta charset="utf-8">
<title>__TITLE__</title>
<style>
 :root{--bg:#0e1417;--fg:#e8eef1;--dim:#8fa3ad;--hi:#4ec9b0}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:13px/1.6 system-ui,"Noto Sans JP",sans-serif;overflow:hidden}
 #wrap{display:flex;gap:12px;padding:12px;align-items:stretch;height:100vh}
 #stage{position:relative;background:#070b0d;border:1px solid #26333a;border-radius:6px;
   flex:1 1 auto;min-width:0;overflow:hidden}
 canvas{display:block;cursor:grab} canvas.drag{cursor:grabbing}
 #side{width:340px;flex:0 0 340px;min-height:0;overflow-y:auto;padding-right:4px}
 h1{font-size:15px;margin:0 0 4px} .note{color:var(--dim);font-size:11.5px}
 .card{background:#131c21;border:1px solid #26333a;border-radius:6px;padding:9px 10px;margin:10px 0}
 .card h2{font:600 11px/1.4 inherit;color:var(--hi);margin:0 0 6px;letter-spacing:.04em}
 .kpi{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:10px 0}
 .kpi div{background:#162025;border:1px solid #26333a;border-radius:5px;padding:6px 8px}
 .kpi b{display:block;font-size:19px;color:var(--hi)} .kpi span{font-size:10.5px;color:var(--dim)}
 button{background:#1d2b32;color:var(--fg);border:1px solid #3a4f59;border-radius:5px;
   padding:6px 10px;cursor:pointer;font-size:12.5px} button:hover{border-color:var(--hi)}
 button.on{background:var(--hi);color:#06231d;border-color:var(--hi)}
 select,input[type=text]{width:100%;background:#0d1319;color:var(--fg);border:1px solid #3a4f59;
   border-radius:5px;padding:5px 6px;font:inherit}
 input[type=range]{width:100%} .row{display:flex;gap:6px;align-items:center;margin:8px 0}
 #xnow{font:700 20px/1.2 inherit;color:var(--hi);font-variant-numeric:tabular-nums;
   white-space:nowrap;flex:0 0 auto}
 #xcap{flex:1 1 0;line-height:1.35}
 #phase{font-weight:700;color:var(--hi)} table{border-collapse:collapse;font-size:11.5px;width:100%}
 td{padding:2px 4px;border-top:1px solid #26333a} td:last-child{text-align:right;color:var(--hi)}
 #cmp{font-size:11px} #cmp th{padding:2px 4px;color:var(--dim);font-weight:400;text-align:right;
   border-bottom:1px solid #26333a} #cmp th:first-child{text-align:left}
 #cmp td{text-align:right;color:var(--fg)} #cmp td:first-child{text-align:left;color:var(--dim)}
 #cmp tr.sel td{color:var(--hi);font-weight:700} #cmp tr{cursor:pointer}
 #cmp tr:hover td{background:#1a252b}
 #src a{color:var(--hi)} #src a:hover{text-decoration:none}
 #zoom{position:absolute;top:8px;right:8px;display:flex;gap:5px;z-index:3}
 #zoom button{padding:3px 9px;font-size:14px;background:rgba(19,28,33,.9)}
 #hud{position:absolute;left:8px;bottom:8px;z-index:3;font-size:11px;color:var(--dim);
   background:rgba(7,11,13,.72);border:1px solid #26333a;border-radius:5px;padding:4px 8px}
 #bar{display:inline-block;height:3px;background:var(--fg);vertical-align:2px;margin:0 5px}
 #tip{position:absolute;z-index:4;pointer-events:none;display:none;font-size:11.5px;
   background:rgba(7,11,13,.94);border:1px solid #3a4f59;border-radius:5px;padding:5px 8px;
   line-height:1.5;white-space:nowrap;max-width:280px}
 #tip b{color:var(--hi)} .sw{display:inline-block;width:11px;height:11px;border-radius:2px;
   vertical-align:-1px;margin-right:4px;border:1px solid #26333a}
</style>
<div id="wrap">
 <div id="stage">
  <canvas id="cv"></canvas>
  <div id="zoom"><button id="zi">＋</button><button id="zo">−</button><button id="zf">全体</button></div>
  <div id="hud"></div><div id="tip"></div>
 </div>
 <div id="side">
  <h1>__TITLE__</h1>
  <div class="note">__SUB__</div>

  <div class="card">
   <h2>X ＝ 成立密度（人/ha）</h2>
   <div class="row"><span id="xnow">–</span><span class="note" id="xcap"></span></div>
   <input type="range" id="xr" min="0" step="1" value="0">
   <select id="xs"></select>
   <div class="note" id="xnote" style="margin-top:6px"></div>
  </div>

  <div class="card">
   <h2>コア円（最寄駅から半径 1200m）</h2>
   <div class="row"><button id="core" class="on">濃淡で示す</button>
    <button id="ring" class="on">円を描く</button></div>
   <div class="note">
    <span class="sw" style="background:#4ec9b0"></span>濃い＝コア円の中（<b>CAPの分母はここだけ</b>・452.39ha）<br>
    <span class="sw" style="background:#235a4f"></span>薄い＝同じ駅のボロノイだがコア円の外（人口は数えるが面積は数えない）<br>
    円は緯度で横に伸びる（1px は緯度で <span id="pxkm"></span>）。拡大すると輪郭が出る。</div>
  </div>

  <div class="card">
   <h2>淘汰された駅圏</h2>
   <div class="row"><button id="dead" class="on">隠す</button>
    <span class="note" id="deadmsg"></span></div>
   <div class="note" id="deadfin"></div>
   <div class="note"><span class="sw" style="background:#363e45"></span>
    他の駅に吸収された駅のボロノイ。<b>この面積ぶんの駅が無くなる。</b>
    人口は吸収先に足されているので KPI の「収容済み人口」は減らない。
    駅の点・駅名・コア円も一緒に消える。<br>この灰は<b>人口メッシュが無い黒（もっと暗い）とは別</b>——灰＝人は住んでいるが駅が無くなる土地。</div>
  </div>

  <div class="card">
   <h2>駅を探す</h2>
   <input type="text" id="q" list="stlist" placeholder="駅名（Enterで次の候補へ）">
   <datalist id="stlist"></datalist>
   <div class="note" id="qmsg" style="margin-top:4px">駅名を入れて Enter。同名が複数あれば順に回る。</div>
   <div class="row"><button id="labs">駅名：存続駅のみ</button></div>
   <div class="note" id="labmsg">拡大すると駅名が出る。白＝存続する駅、灰＝吸収される駅。</div>
  </div>

  <div class="kpi">
   <div><b id="k1">–</b><span>クラスタ数</span></div>
   <div><b id="k2">–</b><span>達成（P≥CAP）</span></div>
   <div><b id="k3">–</b><span>未割当の駅</span></div>
   <div><b id="k4">–</b><span>収容済み人口</span></div>
  </div>
  <div class="row"><button id="play">▶ 再生</button><button id="rew">⟲ 先頭</button>
   <button id="mode">達成率で塗る</button></div>
  <div class="row"><button id="rec">⏺ 録画して保存（webm）</button><span class="note" id="recmsg"></span></div>
  <input type="range" id="fr" min="0" value="0">
  <div class="row"><span id="phase">–</span><span class="note" id="pos"></span></div>
  <table id="tbl"></table>

  <div class="card"><h2>X を変えたときの最終形（クリックで切替）</h2>
   <table id="cmp"></table>
   <div class="note" style="margin-top:6px">現況＝統合しないままで X を満たす駅。
    移動＝別の駅に付け替わる人口の割合。</div></div>

  <div class="card" id="src">
   <h2>出典とライセンス</h2>
   <div class="note">
    <a href="https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N02-2025.html"
       target="_blank" rel="noopener">「国土数値情報（鉄道データ N02-25・2025年）」（国土交通省）</a><br>
    <a href="https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-mesh500r6.html"
       target="_blank" rel="noopener">「国土数値情報（500mメッシュ別将来推計人口・令和6年推計）」（国土交通省）</a><br>
    いずれも <b>CC BY 4.0</b>。<b>本ページはこれらを加工して作成した</b>——乗換駅の統合（N02_005g）／
    最寄駅への割当／コア円の面積計算／集約アルゴリズム。<b>国土交通省が作成したものではない。</b>
    人口は PTN_2020（総務省「令和2年国勢調査」に基づく値）。<br>
    <b>⚠ これは仮定を置いた機械的な計算であって、鉄道事業者・自治体の計画でも、提案でもない。</b>
    特定の駅の存廃を主張するものではなく、「ある密度基準 X を全国一律に当てるとどうなるか」だけを示している。
   </div>
  </div>

  <div class="note" style="padding-bottom:20px">
   Phase1 種の設置／Phase2 容量つき成長／Phase3 余りの吸収／<b>Phase4 不足クラスタの解体</b>。
   Phase4 で色が消えて隣に吸われるのが「未達を残さない」仕掛け。
   <b>どのXでも未達は0</b>——Xを上げると残るクラスタが減るだけで、未達駅は残らない。</div>
 </div>
</div>
<script id="D" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById("D").textContent), M=D.meta, R=D.runs;
const un=s=>{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u.buffer};
const XY=new Float32Array(un(D.xy)), POP=new Float32Array(un(D.pop));
const RX=new Float32Array(un(D.rx)), CHA=new Float32Array(un(D.coreha));
const NAME=D.names||null, nm=i=>NAME?NAME[i]:("#"+i);
const $=id=>document.getElementById(id);
const cv=$("cv"), cx=cv.getContext("2d"), stage=$("stage"), tip=$("tip");
const DPR=Math.min(2,window.devicePixelRatio||1);
const full=document.createElement("canvas"); full.width=M.w; full.height=M.h;
const fx=full.getContext("2d");
const img=fx.createImageData(M.w,M.h), out=new Uint32Array(img.data.buffer);
let VW=0, VH=0, OWN=null, LV=null;

// ---- 所属ラスタ（PNG）を展開。B チャンネル＝コア円との重なり ----
const px=new Image(); px.onload=()=>{
  const c=document.createElement("canvas");c.width=M.w;c.height=M.h;
  const g=c.getContext("2d");g.drawImage(px,0,0);
  const d=g.getImageData(0,0,M.w,M.h).data;
  OWN=new Uint16Array(M.w*M.h); LV=new Uint8Array(M.w*M.h);
  for(let i=0,j=0;i<OWN.length;i++,j+=4){ OWN[i]=(d[j]<<8)|d[j+1];
    LV[i]= d[j+2]>=217 ? 2 : (d[j+2]>0 ? 1 : 0); }   // 2=円の中 1=かかる 0=外
  resize(); setX(M.defi); fit();
};
px.src="data:image/png;base64,"+D.png;

// ---- 選んだ X の状態（フレーム終端のスナップショットを先に作る）----
const F=M.frames;
let RUN=null, CAP=null, snapClu=[], snapPop=[], snapStat=[], cur=0, mode=0, timer=null;
function build(r){
  CAP=new Float32Array(un(r.cap));
  const EV=new Uint16Array(un(r.evs)), EP=new Uint8Array(un(r.evp)), bounds=new Int32Array(un(r.bnd));
  const cluOf=new Int32Array(M.N).fill(-1), cpop=new Float32Array(M.N);
  snapClu=[]; snapPop=[]; snapStat=[];
  let k=0;
  for(let f=0;f<F;f++){
    for(;k<bounds[f+1];k++){
      const s=EV[k*2], c=EV[k*2+1], old=cluOf[s];
      if(old>=0) cpop[old]-=POP[s];
      cluOf[s]=c; cpop[c]+=POP[s];
    }
    snapClu.push(Int32Array.from(cluOf)); snapPop.push(Float32Array.from(cpop));
    let live=0,met=0,una=0,hold=0;
    const seen=new Set();
    for(let i=0;i<M.N;i++){ if(cluOf[i]<0){una++;continue;} hold+=POP[i];
      if(!seen.has(cluOf[i])){seen.add(cluOf[i]); live++; if(cpop[cluOf[i]]>=CAP[cluOf[i]])met++;} }
    snapStat.push({live,met,una,hold,ph:EP[Math.max(0,bounds[f+1]-1)]});
  }
  $("fr").max=F-1;
  $("tbl").innerHTML=
   `<tr><td>駅（母点）</td><td>${M.N.toLocaleString()}</td></tr>
    <tr><td>現況で X を満たす駅</td><td>${r.nowmet.toLocaleString()}（${(100*r.nowmet/M.N).toFixed(1)}%）</td></tr>
    <tr><td>最終クラスタ数</td><td>${r.alive.toLocaleString()}</td></tr>
    <tr><td>未達クラスタ</td><td>${r.short}</td></tr>
    <tr><td>解体リペア</td><td>${r.repairs.toLocaleString()} 回</td></tr>
    <tr><td>移動対象人口</td><td>${r.moved}%</td></tr>
    <tr><td>移動距離 中位 / 95% / 最大</td><td>${r.dmed} / ${r.d95} / ${r.dmax} km</td></tr>
    <tr><td>人口の保存</td><td>${r.mass}%</td></tr>`;
}
// ---- X の切替 ----
const xr=$("xr"), xs=$("xs");
xr.max=R.length-1;
xs.innerHTML=R.map((r,i)=>`<option value="${i}">X=${r.X} ｜ ${r.label}</option>`).join("");
$("cmp").innerHTML=
 `<tr><th>X</th><th>クラスタ</th><th>現況達成</th><th>移動</th><th>中位km</th></tr>`+
 R.map((r,i)=>`<tr data-i="${i}"><td>${r.X}</td><td>${r.alive.toLocaleString()}</td>
   <td>${(100*r.nowmet/M.N).toFixed(0)}%</td><td>${r.moved}%</td><td>${r.dmed}</td></tr>`).join("");
$("cmp").onclick=e=>{const tr=e.target.closest("tr[data-i]"); if(tr)setX(+tr.dataset.i);};
let xi=-1;
function setX(i){
  if(i===xi) return;
  stop(); xi=i; RUN=R[i]; build(RUN);
  xr.value=i; xs.value=i;
  $("xnow").textContent=RUN.X+" 人/ha";
  $("xcap").textContent=
    `CAP＝コア円 452.39ha×X＝${RUN.capfull.toLocaleString()}人（実効の中央値 ${RUN.capmed.toLocaleString()}人）`;
  $("xnote").textContent=RUN.note;
  $("deadfin").innerHTML=`最終形では <b>${RUN.dead.toLocaleString()} 駅</b>が淘汰され、`+
    `その圏域は <b>${RUN.deadarea}%</b>（${RUN.deadkm2.toLocaleString()}km²）・人口の <b>${RUN.moved}%</b>。`;
  [...document.querySelectorAll("#cmp tr[data-i]")].forEach(tr=>
    tr.classList.toggle("sel",+tr.dataset.i===i));
  draw(Math.min(cur,F-1));
}
xr.oninput=e=>setX(+e.target.value);
xs.onchange=e=>setX(+e.target.value);

// ---- 色 ----
const hue=c=>{ let x=(c*2654435761)>>>0;
  const H=(x%360)/60, S=.55+((x>>9)&31)/124, L=.42+((x>>14)&31)/155;
  const C=(1-Math.abs(2*L-1))*S, X2=C*(1-Math.abs(H%2-1)), m=L-C/2;
  const t=[[C,X2,0],[X2,C,0],[0,C,X2],[0,X2,C],[X2,0,C],[C,0,X2]][H|0];
  return [Math.round((t[0]+m)*255),Math.round((t[1]+m)*255),Math.round((t[2]+m)*255)];};
function ratioRGB(t){ const a=[[215,48,39],[252,141,89],[254,224,144],[224,243,248],[69,117,180]];
  const x=Math.max(0,Math.min(.999,t))*(a.length-1),i=x|0,f=x-i;
  return [0,1,2].map(k=>Math.round(a[i][k]+(a[i+1][k]-a[i][k])*f)); }
const pack=(r,g,b)=>(255<<24)|(b<<16)|(g<<8)|r;
const shade=(c,k)=>pack(Math.min(255,c[0]*k|0),Math.min(255,c[1]*k|0),Math.min(255,c[2]*k|0));
let coreOn=true, ringOn=true, labAll=false, hideDead=true;
const DEAD=0xff453e36;                       // 淘汰された駅圏（rgb 54,62,69）。海 0xff110f0a より明るくして見分ける
const gone=(clu,i)=>hideDead && clu[i]>=0 && clu[i]!==i;   // 他の駅に吸収された＝この駅は残らない

// ---- 全体ラスタを塗る（フレーム or 表示条件が変わったときだけ）----
function paint(){
  if(!OWN||!snapClu.length) return;
  const clu=snapClu[cur], cp=snapPop[cur], N=M.N;
  const L=new Uint32Array(3*N);                       // [外, かかる, 中] × 駅
  let ndead=0;
  for(let i=0;i<N;i++){ const c=clu[i];
    if(gone(clu,i)){ L[i]=L[N+i]=L[2*N+i]=DEAD; ndead++; continue; }
    const rgb = c<0 ? [82,74,58] : (mode? ratioRGB(cp[c]/CAP[c]) : hue(c));
    if(coreOn){ L[i]=shade(rgb,.45); L[N+i]=shade(rgb,.72); L[2*N+i]=shade(rgb,1.0); }
    else      { const v=shade(rgb,1); L[i]=v; L[N+i]=v; L[2*N+i]=v; } }
  $("deadmsg").textContent = hideDead
    ? `いま ${ndead.toLocaleString()} 駅ぶんを隠している`
    : `淘汰された駅は ${snapStat[cur].live? (M.N-snapStat[cur].live-snapStat[cur].una).toLocaleString():0} 駅`;
  const SEA=0xff110f0a;
  for(let i=0;i<OWN.length;i++){ const o=OWN[i]; out[i]= o===65535 ? SEA : L[LV[i]*N+o]; }
  fx.putImageData(img,0,0);
}
// ---- 画面（拡大・移動）----
let sc=1, tx=0, ty=0, minSc=1;
function resize(){
  VW=Math.max(320,stage.clientWidth); VH=Math.max(240,stage.clientHeight);
  cv.style.width=VW+"px"; cv.style.height=VH+"px";
  cv.width=Math.round(VW*DPR); cv.height=Math.round(VH*DPR);
  minSc=Math.min(VW/M.w, VH/M.h);
}
function fit(){ sc=minSc; tx=(VW-M.w*sc)/2; ty=(VH-M.h*sc)/2; blit(); }
function clampView(){
  sc=Math.max(minSc*.98, Math.min(60, sc));
  const ww=M.w*sc, hh=M.h*sc;
  tx = ww<=VW ? (VW-ww)/2 : Math.min(0, Math.max(VW-ww, tx));
  ty = hh<=VH ? (VH-hh)/2 : Math.min(0, Math.max(VH-hh, ty));
}
function blit(){
  if(!OWN) return;
  clampView();
  cx.setTransform(DPR,0,0,DPR,0,0);
  cx.fillStyle="#070b0d"; cx.fillRect(0,0,VW,VH);
  cx.imageSmoothingEnabled=false;
  cx.drawImage(full, tx, ty, M.w*sc, M.h*sc);
  overlay();
  const mpp=M.kmy*1000/sc;                                     // 1画面px あたりの m（南北）
  const barkm=[1,2,5,10,20,50,100,200].reduce((a,b)=>
    Math.abs(b*1000/mpp-140)<Math.abs(a*1000/mpp-140)?b:a);
  $("hud").innerHTML=`1px ≒ ${mpp<1000?mpp.toFixed(0)+"m":(mpp/1000).toFixed(1)+"km"}`+
    ` ・ ×${(sc/minSc).toFixed(1)}<span id="bar" style="width:${(barkm*1000/mpp).toFixed(0)}px"></span>${barkm}km`;
}
function overlay(){
  const clu=snapClu[cur]; if(!clu) return;
  const vis=[];
  for(let i=0;i<M.N;i++){
    if(gone(clu,i)) continue;                       // 淘汰された駅は点も名前もコア円も出さない
    const X=XY[i*2]*sc+tx, Y=XY[i*2+1]*sc+ty;
    if(X<-30||Y<-30||X>VW+30||Y>VH+30||!isFinite(X)) continue;
    vis.push([i,X,Y]);
  }
  // コア円（緯度で横に伸びる楕円）
  if(ringOn && sc>=minSc*3 && vis.length<=500){   // 密すぎると輪郭が地図を覆うので出さない
    cx.lineWidth=Math.min(1.6,.5+sc*.06); cx.strokeStyle="rgba(255,255,255,.42)";
    cx.beginPath();
    for(const [i,X,Y] of vis){
      const a=RX[i]*sc, b=M.ry*sc; if(a<2) continue;
      cx.moveTo(X+a,Y); cx.ellipse(X,Y,a,b,0,0,6.2832);
    }
    cx.stroke();
  }
  // 駅の点
  for(const [i,X,Y] of vis){
    const seed = clu[i]===i;
    cx.fillStyle = clu[i]<0 ? "rgba(255,140,120,.85)" : (seed?"#fff":"rgba(255,255,255,.45)");
    const s = seed? Math.min(5,1.6+sc*.12) : Math.min(3,1.2+sc*.06);
    cx.fillRect(X-s/2, Y-s/2, s, s);
  }
  // 駅名
  const want = vis.filter(v=>labAll || clu[v[0]]===v[0]);
  if(sc>=minSc*4 && want.length<=140){
    cx.font="600 11px system-ui,'Noto Sans JP',sans-serif";
    cx.textBaseline="middle"; cx.lineWidth=3; cx.strokeStyle="rgba(7,11,13,.85)";
    for(const [i,X,Y] of want){
      const seed=clu[i]===i, t=nm(i);
      cx.strokeText(t,X+5,Y-1);
      cx.fillStyle=seed?"#fff":"rgba(210,222,228,.72)";
      cx.fillText(t,X+5,Y-1);
    }
    $("labmsg").textContent=`駅名 ${want.length} 件を表示中。`+
      (hideDead?"淘汰された駅は隠しているので、出ているのは残る駅だけ。"
               :"白＝存続する駅、灰＝吸収される駅。");
  }else{
    $("labmsg").textContent = sc<minSc*4 ? "もっと拡大すると駅名が出る（ホイール／＋）"
      : `この範囲に ${want.length} 駅。140駅以下まで拡大すると駅名が出る`;
  }
}
function draw(f){ cur=f; paint(); blit();
  const st=snapStat[f];
  $("k1").textContent=st.live.toLocaleString();
  $("k2").textContent=st.met.toLocaleString();
  $("k3").textContent=st.una.toLocaleString();
  $("k4").textContent=(st.hold/M.totalpop*100).toFixed(1)+"%";
  $("phase").textContent=
    ["","Phase1 種の設置","Phase2 容量つき成長","Phase3 余りの吸収","Phase4 解体リペア","Phase5 越境"][st.ph]||"";
  $("pos").textContent=` ${f+1}/${F}`;
  $("fr").value=f;
}
// ---- 操作 ----
cv.onwheel=e=>{ e.preventDefault();
  const k=Math.exp(-e.deltaY*(e.deltaMode?0.06:0.0015)), ns=Math.max(minSc*.98,Math.min(60,sc*k));
  const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  tx=mx-(mx-tx)*ns/sc; ty=my-(my-ty)*ns/sc; sc=ns; blit(); };
let drag=null;
cv.onmousedown=e=>{drag=[e.clientX,e.clientY,tx,ty];cv.classList.add("drag");tip.style.display="none";};
window.addEventListener("mouseup",()=>{drag=null;cv.classList.remove("drag");});
cv.onmousemove=e=>{
  const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  if(drag){ tx=drag[2]+(e.clientX-drag[0]); ty=drag[3]+(e.clientY-drag[1]); blit(); return; }
  if(!OWN||!snapClu.length){return;}
  const gx=Math.floor((mx-tx)/sc), gy=Math.floor((my-ty)/sc);
  if(gx<0||gy<0||gx>=M.w||gy>=M.h){tip.style.display="none";return;}
  const k=gy*M.w+gx, o=OWN[k];
  if(o===65535){tip.style.display="none";return;}
  const clu=snapClu[cur], cp=snapPop[cur], c=clu[o];
  const inside=["圏外","円にかかる","コア円の中"][LV[k]];
  tip.innerHTML = `<b>${nm(o)}</b>　この地点は<b>${inside}</b><br>`+
    `駅単独の人口 ${Math.round(POP[o]).toLocaleString()}人 ／ コア円 ${CHA[o].toFixed(0)}ha<br>`+
    (c<0 ? "まだどのクラスタにも属していない"
         : (c===o ? "<b>この駅は残る</b>（クラスタの中心）<br>" : `<b>この駅は淘汰される</b>（圏域は吸収される）<br>`)
           + `所属クラスタ <b>${nm(c)}</b>　${Math.round(cp[c]).toLocaleString()}人 / CAP ${Math.round(CAP[c]).toLocaleString()}人`+
           `　<b>×${(cp[c]/CAP[c]).toFixed(2)}</b>`);
  tip.style.display="block";
  tip.style.left=Math.min(mx+14, stage.clientWidth-tip.offsetWidth-6)+"px";
  tip.style.top=Math.min(my+14, stage.clientHeight-tip.offsetHeight-6)+"px";
};
cv.onmouseleave=()=>tip.style.display="none";
function zoomAt(k,mx,my){ const ns=Math.max(minSc*.98,Math.min(60,sc*k));
  tx=mx-(mx-tx)*ns/sc; ty=my-(my-ty)*ns/sc; sc=ns; blit(); }
$("zi").onclick=()=>zoomAt(1.6,VW/2,VH/2);
$("zo").onclick=()=>zoomAt(1/1.6,VW/2,VH/2);
$("zf").onclick=fit;
$("core").onclick=e=>{coreOn=!coreOn;e.target.classList.toggle("on",coreOn);
  e.target.textContent=coreOn?"濃淡で示す":"濃淡なし"; paint(); blit();};
$("ring").onclick=e=>{ringOn=!ringOn;e.target.classList.toggle("on",ringOn);
  e.target.textContent=ringOn?"円を描く":"円を消す"; blit();};
$("dead").onclick=e=>{hideDead=!hideDead;e.target.classList.toggle("on",hideDead);
  e.target.textContent=hideDead?"隠す":"表示する"; paint(); blit();};
$("labs").onclick=e=>{labAll=!labAll;e.target.classList.toggle("on",labAll);
  e.target.textContent=labAll?"駅名：全駅":"駅名：存続駅のみ"; blit();};
window.addEventListener("resize",()=>{            // 画面の中心と倍率を保ったまま追随する
  const px0=(VW/2-tx)/sc, py0=(VH/2-ty)/sc, k=sc/minSc;
  resize(); sc=minSc*k; tx=VW/2-px0*sc; ty=VH/2-py0*sc; blit(); });
// ---- 駅名検索 ----
if(NAME){
  const uniq=[...new Set(NAME)].sort();
  $("stlist").innerHTML=uniq.map(n=>`<option value="${n}">`).join("");
  let hits=[], hi=0, last="";
  $("q").addEventListener("change",jump); $("q").addEventListener("keydown",e=>{if(e.key==="Enter")jump();});
  function jump(){
    const q=$("q").value.trim(); if(!q) return;
    if(q!==last){ last=q; hi=0; hits=[]; for(let i=0;i<M.N;i++) if(NAME[i]===q) hits.push(i);
      if(!hits.length) for(let i=0;i<M.N;i++) if(NAME[i].includes(q)) hits.push(i); }
    if(!hits.length){ $("qmsg").textContent=`「${q}」は見つからない`; return; }
    const i=hits[hi%hits.length]; hi++;
    sc=Math.max(sc,Math.min(60,M.kmy*1000/14));       // 1px≒14m まで寄る（駅名が読める倍率）
    tx=VW/2-XY[i*2]*sc; ty=VH/2-XY[i*2+1]*sc; blit();
    $("qmsg").textContent=`${NAME[i]}（${hi}/${hits.length}件目）`;
  }
}
$("pxkm").textContent=(M.kmy*1000).toFixed(0)+"m × 経度で"+(M.kmx*1000).toFixed(0)+"m";
// ---- 再生 ----
$("fr").oninput=e=>{stop();draw(+e.target.value)};
function stop(){ if(timer){clearInterval(timer);timer=null;} $("play").classList.remove("on");
  $("play").textContent="▶ 再生"; }
$("play").onclick=()=>{
  if(timer){stop();return;}
  $("play").classList.add("on"); $("play").textContent="⏸ 停止";
  timer=setInterval(()=>{ if(cur>=F-1){stop();return;} draw(cur+1); },33);
};
$("rew").onclick=()=>{stop();draw(0)};
$("mode").onclick=e=>{mode=1-mode;e.target.classList.toggle("on",!!mode);
  e.target.textContent=mode?"クラスタ色に戻す":"達成率で塗る"; draw(cur)};
// ---- 録画（canvas.captureStream → webm。外部ライブラリ不要）----
const recBtn=$("rec"), recMsg=$("recmsg");
if(!window.MediaRecorder||!cv.captureStream){ recBtn.disabled=true; recMsg.textContent="この環境では不可"; }
else recBtn.onclick=()=>{
  stop(); draw(0);
  const st=cv.captureStream(30), mr=new MediaRecorder(st,{mimeType:"video/webm"}), ch=[];
  mr.ondataavailable=e=>ch.push(e.data);
  mr.onstop=()=>{ const a=document.createElement("a");
    a.href=URL.createObjectURL(new Blob(ch,{type:"video/webm"}));
    a.download="honshu_convergence_X"+RUN.X+".webm"; a.click();
    recMsg.textContent="保存しました"; recBtn.disabled=false; };
  recBtn.disabled=true; recMsg.textContent="録画中…";
  mr.start();
  let f=0; const iv=setInterval(()=>{ draw(f); if(++f>=F){ clearInterval(iv); setTimeout(()=>mr.stop(),300);} },33);
};
</script>
"""
TITLE = ("本州の駅勢圏の収斂（実データ・アルゴリズム v2）" if REAL
         else "駅勢圏の収斂（合成データ・アルゴリズム v2）")
SUB = ("国土数値情報 N02(2025) の駅を <b>N02_005g（同一駅グループ）で乗換駅を1駅に統合</b>、"
       "人口は 500mメッシュ PTN_2020（2020年国勢調査）。"
       "<b>色＝クラスタ</b>／<b>達成率</b>を切替。時間軸は暦年ではなく<b>処理の進行</b>。"
       "ホイールで拡大・ドラッグで移動・カーソルを置くと地点の所属が出る。<br>"
       "<b>黒は「海」ではなく「人口メッシュが無い」</b>＝無人の山地・湖・空港・埋立地・皇居まわり。"
       "塗ったのは 83,863km²で本州の陸の約37%（可住地率とほぼ一致）。" if REAL
       else "合成データ。1px＝1km。<b>色＝クラスタ</b>／<b>達成率</b>を切替。時間軸は<b>処理の進行</b>。"
            "ホイールで拡大・ドラッグで移動。")
html = HTML.replace("__DATA__", json.dumps(DATA, ensure_ascii=False)) \
           .replace("__TITLE__", TITLE).replace("__SUB__", SUB)
open(os.path.abspath(OUT), "w", encoding="utf-8").write(html)
print(f"→ {os.path.abspath(OUT)}  {os.path.getsize(os.path.abspath(OUT))/1024/1024:.2f} MB "
      f"（{time.time()-t0:.1f}s）")
