"""
本州の実データで駅勢圏ボロノイ × 人口集約を解く。

入力（data/raw/ に取得済み）:
  N02-25_GML.zip                    国土数値情報 鉄道 2025（駅は線分・路線ごとに重複）
  500m_mesh_2024_<02..35>_SHP.zip   500mメッシュ別将来推計人口（R6推計）。PTN_2020＝2020年国勢調査人口

出力: data/honshu.npz（駅点・所属・隣接・容量）＋ 標準出力のレポート
"""
import argparse, math, os, sys, time, zipfile
import numpy as np
from shapely import STRtree, points as shp_points
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shpdbf
import honshu_core as hc

RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(HERE, "..", "data", "honshu.npz")
PREFS = [f"{i:02d}" for i in range(2, 36)]          # 02 青森 〜 35 山口 ＝ 本州34県

# 本州ではない主な島（市区町村コード）。橋で繋がっていても地形的に本州ではないものを外す
ISLAND_SHICODE = {
    "13361","13362","13363","13364","13381","13382","13401","13402","13421",  # 伊豆諸島・小笠原
    "15224","15586",                                                          # 佐渡市・粟島浦村
    "28205","28224","28226",                                                  # 淡路島
    "32525","32526","32527","32528",                                          # 隠岐
    "34215","34431",                                                          # 江田島市・大崎上島町
    "35305",                                                                  # 周防大島町
}

ap = argparse.ArgumentParser()
ap.add_argument("--r", type=float, default=1.2)      # コア円半径 km
ap.add_argument("--X", type=float, default=40.0)     # 人/ha
ap.add_argument("--maxdist", type=float, default=30.0)  # 最寄駅がこれより遠いメッシュは対象外（km）
ap.add_argument("--group", choices=["n02g", "near200", "none"], default="near200",
                help="駅の名寄せ: n02g=N02の同一駅グループのみ / near200=さらに200m以内を統合 / none=無し")
A = ap.parse_args()
T = time.perf_counter()
def lap(m):
    global T; t = time.perf_counter(); print(f"  [{t-T:6.2f}s] {m}", flush=True); T = t

# ---------------- 1) 駅（N02） ----------------
zp = os.path.join(RAW, "N02-25_GML.zip")
with zipfile.ZipFile(zp) as z:
    for n in z.namelist():
        if "/UTF-8/N02-25_Station." in n and n.rsplit(".", 1)[-1] in ("shp", "dbf"):
            if not os.path.exists(os.path.join(RAW, n)):
                z.extract(n, RAW)
base = os.path.join(RAW, "N02-25_GML/UTF-8/N02-25_Station")
_, d, nrec = shpdbf.read_dbf(base + ".dbf", cols=["N02_003", "N02_004", "N02_005", "N02_005g"],
                             encoding="utf-8")
pt = shpdbf.read_shp_points(base + ".shp")           # 線分の重心（経度, 緯度）
gid = np.array(d["N02_005g"])
name = np.array(d["N02_005"])
lap(f"N02 駅レコード {nrec:,}（線分の重心を取得）")

# 同一駅グループ（N02_005g）で1点にまとめる ＝ 乗換駅を1駅として扱う
uniq, inv = np.unique(gid, return_inverse=True)
sx = np.bincount(inv, weights=pt[:, 0]) / np.bincount(inv)
sy = np.bincount(inv, weights=pt[:, 1]) / np.bincount(inv)
snm = np.array([name[np.nonzero(inv == i)[0][0]] for i in range(len(uniq))])
lap(f"N02_005g で統合 → {len(uniq):,} 駅")

# ---------------- 2) 投影（本州中心の正積方位図法） ----------------
TR = Transformer.from_crs("EPSG:6668",                       # JGD2011 緯度経度
    "+proj=laea +lat_0=37 +lon_0=138 +ellps=GRS80 +units=m +no_defs", always_xy=True)
SX, SY = TR.transform(sx, sy)
S_all = np.c_[SX, SY] / 1000.0                                # km

# ---------------- 3) 人口メッシュ（34県） ----------------
mid, shi, pop = [], [], []
for pp in PREFS:
    zf = os.path.join(RAW, f"500m_mesh_2024_{pp}_SHP.zip")
    with zipfile.ZipFile(zf) as z:
        nm = [x for x in z.namelist() if x.endswith(".dbf")][0]
        tmp = os.path.join("/tmp", os.path.basename(nm))
        open(tmp, "wb").write(z.read(nm))
    dd, n = shpdbf.read_dbf_cols(tmp, ["MESH_ID", "SHICODE", "PTN_2020"])
    os.remove(tmp)
    mid.append(dd["MESH_ID"]); shi.append(dd["SHICODE"]); pop.append(dd["PTN_2020"])
mid = np.concatenate(mid); shi = np.concatenate(shi); pop = np.concatenate(pop)
tot0 = pop.sum()
lap(f"500mメッシュ {len(mid):,} セル / 人口 {tot0:,.0f}（34県・PTN_2020）")

keep = ~np.isin(shi, list(ISLAND_SHICODE))
drop_isl = float(pop[~keep].sum())
mid, pop = mid[keep], pop[keep]
row, col = shpdbf.mesh500_rowcol(mid.astype(np.int64))
la, lo = shpdbf.mesh500_center(row, col)
MX, MY = TR.transform(lo, la)
M = np.c_[MX, MY] / 1000.0
area_ha = shpdbf.mesh500_area_km2(row) * 100.0
lap(f"島嶼を除外 {int((~keep).sum()):,}セル・{drop_isl:,.0f}人 → 残り {len(mid):,}セル")

# ---------------- 4) 最寄駅の割当（＝ボロノイ） ----------------
tree = STRtree(shp_points(S_all))
own_all = tree.nearest(shp_points(M)).astype(np.int64)
dist = np.hypot(M[:, 0] - S_all[own_all, 0], M[:, 1] - S_all[own_all, 1])
far = dist > A.maxdist
drop_far = float(pop[far].sum())
mid, pop, row, col, M, area_ha, own_all = (a[~far] for a in (mid, pop, row, col, M, area_ha, own_all))
lap(f"最寄駅まで{A.maxdist:.0f}km超を除外 {int(far.sum()):,}セル・{drop_far:,.0f}人")

# 本州のメッシュを1つも持たない駅（北海道・四国・九州の駅）を落とす
# ＋ 自分自身が本州のメッシュから5km以上離れている駅も落とす（海を渡って割り当てられた分）
used = np.unique(own_all)
mtree = STRtree(shp_points(M))
dnear = np.hypot(S_all[used, 0] - M[mtree.nearest(shp_points(S_all[used])), 0],
                 S_all[used, 1] - M[mtree.nearest(shp_points(S_all[used])), 1])
off = used[dnear > 5.0]
if len(off):
    print(f"    ※本州のメッシュから5km超の駅を除外: {len(off)}駅 例 {', '.join(snm[i] for i in off[:6])}")
    used = used[dnear <= 5.0]
    keep2 = np.isin(own_all, used)
    mid, pop, row, col, M, area_ha, own_all = (a[keep2] for a in (mid, pop, row, col, M, area_ha, own_all))
    own_all = np.unique(used)[np.searchsorted(np.unique(used), own_all)] if False else own_all
    tree2 = STRtree(shp_points(S_all[used]))
    own_all = used[tree2.nearest(shp_points(M))]      # 残った駅で割り当て直す
    used = np.unique(own_all)
remap = -np.ones(len(S_all), np.int64); remap[used] = np.arange(len(used))
S = S_all[used]; SNM = snm[used]; owner = remap[own_all].astype(np.int32)
N = len(S)
lap(f"本州の駅 {N:,}（対象メッシュ {len(pop):,} / 人口 {pop.sum():,.0f}）")

# 200m以内の別グループを統合（異名の乗換駅を拾う）
if A.group == "near200":
    par = np.arange(N)
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    tr2 = STRtree(shp_points(S))
    for a_, b_ in zip(*tr2.query(shp_points(S), predicate="dwithin", distance=0.2)):
        if a_ != b_:
            ra, rb = find(int(a_)), find(int(b_))
            if ra != rb: par[max(ra, rb)] = min(ra, rb)
    root = np.array([find(i) for i in range(N)])
    u2, inv2 = np.unique(root, return_inverse=True)
    merged = N - len(u2)
    S = np.c_[np.bincount(inv2, weights=S[:, 0]) / np.bincount(inv2),
              np.bincount(inv2, weights=S[:, 1]) / np.bincount(inv2)]
    SNM = SNM[u2]; owner = inv2[owner].astype(np.int32); N = len(S)
    lap(f"さらに200m以内を統合 {merged} 件 → {N:,} 駅")

cellpop = np.bincount(owner, weights=pop, minlength=N)

# ---- 同一メッシュが自治体境で分割されて2レコードある分を、面積で二重に数えない ----
# （人口は市区町村ごとの内訳なので合計して正しいが、面積は1メッシュ分しかない）
_k = row.astype(np.int64) * 100000 + col
_u, _inv, _cnt = np.unique(_k, return_inverse=True, return_counts=True)
if (_cnt > 1).any():
    area_ha = area_ha / _cnt[_inv]           # 生き残ったレコードで1メッシュの面積を分け合う
    lap(f"自治体境で分割されたメッシュ {int((_cnt > 1).sum()):,} 件（{int((_cnt[_inv] > 1).sum()):,}レコード）"
        f"の面積を案分 → 総面積 {area_ha.sum()/100:,.0f} km²")

# ---------------- 5) 隣接グラフ（メッシュの行列インデックスの4近傍） ----------------
r0, c0 = row.min(), col.min()
nr, nc = int(row.max() - r0 + 1), int(col.max() - c0 + 1)
og = np.full((nr, nc), -1, np.int32)
og[row - r0, col - c0] = owner
pairs = []
for a_, b_ in ((og[:, :-1], og[:, 1:]), (og[:-1, :], og[1:, :])):
    m = (a_ >= 0) & (b_ >= 0) & (a_ != b_)
    pairs.append(np.c_[a_[m], b_[m]])
pr = np.vstack(pairs); pr = np.unique(np.c_[pr.min(1), pr.max(1)], axis=0)
adj = [[] for _ in range(N)]
for a_, b_ in pr:
    adj[a_].append(int(b_)); adj[b_].append(int(a_))
lap(f"隣接グラフ 辺 {len(pr):,}（ラスタ {nr}×{nc}）")

# ---------------- 5b) 路線隣接（仕様書 §3.4 案B）----------------
# 人口メッシュは無人の山地で途切れるため、ラスタ隣接だけだと能登・紀伊などが分断される。
# 鉄道の線形そのものをグラフにし、その上で「駅のネットワーク・ボロノイ」を作って隣接を取る。
import heapq
with zipfile.ZipFile(zp) as z:
    for n in z.namelist():
        if "/UTF-8/N02-25_RailroadSection." in n and n.rsplit(".", 1)[-1] in ("shp",):
            if not os.path.exists(os.path.join(RAW, n)): z.extract(n, RAW)
rxy, rstarts, _, rm = shpdbf.read_shp_polylines(
    os.path.join(RAW, "N02-25_GML/UTF-8/N02-25_RailroadSection.shp"))
RX, RY = TR.transform(rxy[:, 0], rxy[:, 1])
R = np.c_[RX, RY] / 1000.0
key = np.round(R / 0.02).astype(np.int64)                      # 20m グリッドに丸めて節点化
_, nid = np.unique(key[:, 0] * 10_000_000 + key[:, 1], return_inverse=True)
nn = int(nid.max()) + 1
valid = np.ones(rm - 1, bool)
ps = rstarts[rstarts > 0]
valid[ps - 1] = False                       # 別パートの先頭へ飛ぶ辺は繋がない
ea, eb = nid[:-1][valid], nid[1:][valid]
ew = np.hypot(R[:-1, 0][valid] - R[1:, 0][valid], R[:-1, 1][valid] - R[1:, 1][valid])
assert ew.max() < 20, f"鉄道の1区間が {ew.max():.1f}km ＝ パート境界の処理が誤っている"
gadj = [[] for _ in range(nn)]
for a_, b_, w_ in zip(ea, eb, ew):
    if a_ != b_:
        gadj[a_].append((int(b_), float(w_))); gadj[b_].append((int(a_), float(w_)))
rtree = STRtree(shp_points(R))
snap = rtree.nearest(shp_points(S))
lab = np.full(nn, -1, np.int32); dd = np.full(nn, np.inf)
h = []
for i in range(N):
    nd = int(nid[snap[i]])
    if dd[nd] > 0: lab[nd] = i; dd[nd] = 0.0; heapq.heappush(h, (0.0, nd, i))
while h:
    d0, u, li = heapq.heappop(h)
    if d0 > dd[u] + 1e-12: continue
    for v, w_ in gadj[u]:
        nd2 = d0 + w_
        if nd2 < dd[v] - 1e-12:
            dd[v] = nd2; lab[v] = li; heapq.heappush(h, (nd2, v, li))
rp = set()
for a_, b_ in zip(ea, eb):
    la_, lb_ = lab[a_], lab[b_]
    if la_ >= 0 and lb_ >= 0 and la_ != lb_:
        rp.add((min(la_, lb_), max(la_, lb_)))
before = len(pr)
allp = set(map(tuple, pr)) | rp
pr = np.array(sorted(allp))
adj = [[] for _ in range(N)]
for a_, b_ in pr:
    adj[a_].append(int(b_)); adj[b_].append(int(a_))
lap(f"路線隣接を追加 辺 {before:,} → {len(pr):,}（鉄道節点 {nn:,}）")

# 残った分断は「最寄の他成分の駅」へ橋を架ける（山越え・海越えの実態を明示するため距離を記録）
def comps():
    c = np.full(N, -1); k = 0
    for i in range(N):
        if c[i] >= 0: continue
        st = [i]; c[i] = k
        while st:
            u = st.pop()
            for v in adj[u]:
                if c[v] < 0: c[v] = k; st.append(v)
        k += 1
    return c, k
cc, nk = comps(); bridges = []
while nk > 1:
    big = np.bincount(cc).argmax()
    tgt = np.nonzero(cc == big)[0]
    tt = STRtree(shp_points(S[tgt]))
    best = None
    for i in np.nonzero(cc != big)[0]:
        j = tgt[tt.nearest(shp_points(S[i:i+1]))[0]]
        d0 = float(np.hypot(*(S[i] - S[j])))
        if best is None or d0 < best[0]: best = (d0, int(i), int(j))
    d0, i, j = best
    adj[i].append(j); adj[j].append(i); bridges.append((SNM[i], SNM[j], round(d0, 1)))
    cc, nk = comps()
lap(f"分断の橋渡し {len(bridges)} 本（最長 {max([b[2] for b in bridges], default=0)}km）")
pr = np.array(sorted({(min(a_, b_), max(a_, b_)) for a_ in range(N) for b_ in adj[a_]}))

# ---------------- 6) コア円の実効容量（セルを4×4に細分して円との重なりを取る） ----------------
SUB = 4
oy = ((np.arange(SUB) + .5) / SUB - .5)
gx, gy = np.meshgrid(oy, oy)
capn = np.zeros(N)
cw = np.full(len(pop), np.nan)                     # セルの幅・高さ(km)
dlat_km = 6371.0088 * math.radians(1 / 240.0)
cw_h = dlat_km
cw_w = 6371.0088 * math.radians(1 / 160.0) * np.cos(np.radians((row + .5) / 240.0))
cellxy_tree = STRtree(shp_points(M))
for i in range(N):
    idx = cellxy_tree.query(shp_points(S[i:i+1]), predicate="dwithin", distance=A.r + 0.6)[1]
    if len(idx) == 0: continue
    sel = idx[owner[idx] == i]
    if len(sel) == 0: continue
    dx = M[sel, 0] - S[i, 0]; dy = M[sel, 1] - S[i, 1]
    px = dx[:, None] + gx.ravel()[None, :] * cw_w[sel][:, None]
    py = dy[:, None] + gy.ravel()[None, :] * cw_h
    frac = ((px**2 + py**2) <= A.r**2).mean(1)
    capn[i] = float((area_ha[sel] * frac).sum()) * A.X
full = math.pi * A.r * A.r * 100 * A.X
capn = np.where(capn > 0, capn, full)
lap(f"コア円の実効容量（4×4細分）: 中央値 {np.median(capn):,.0f}人 / 満円 {full:,.0f}人")

# 駅ごとの代表 SHICODE（自分のセルの最頻値）→ 都道府県
shi_keep = shi[keep][~far]
if "keep2" in dir(): shi_keep = shi_keep[keep2]
pref_of = np.full(N, "  ", dtype="<U2")
for i in range(N):
    sel = np.nonzero(owner == i)[0]
    if len(sel):
        v, c_ = np.unique(shi_keep[sel], return_counts=True)
        pref_of[i] = v[c_.argmax()][:2]
A_ = dict(N=N, owner=owner, og=og, cellpop=cellpop, adj=adj, capn=capn, edges=len(pr))
now_met = int((cellpop >= capn).sum())

print(f"\n本州: 駅 {N:,} / メッシュ {len(pop):,} / 人口 {pop.sum():,.0f} "
      f"/ r={A.r}km X={A.X}人/ha / 理論上限 n_max={pop.sum()/full:,.0f}駅")
print(f"現況（統合なし）で X を満たす駅: {now_met:,} / {N:,}（{100*now_met/N:.1f}%）")
print(f"除外: 島嶼 {drop_isl:,.0f}人 ／ 最寄駅{A.maxdist:.0f}km超 {drop_far:,.0f}人 "
      f"（元 {tot0:,.0f}人 の {100*(drop_isl+drop_far)/tot0:.2f}%）\n")

for dmax in (10.0, 1e9):
    v1 = hc.run_v1(A_, S, dmax)
    tag = "距離上限なし" if dmax > 1e8 else f"距離上限{dmax:.0f}km"
    print(f"  v1 {tag:>12}: 生存{v1['alive']:5,} 達成{v1['met']:5,} 未達{v1['short']:4,}"
          f"（{v1['shortpop']:,.0f}人） 移動{v1['moved']*100:4.1f}% "
          f"中位{v1['dmed']:.1f}km 95%{v1['d95']:.1f}km")
v2 = hc.run_v2(A_, S, record=True)
print(f"  v2 種成長+リペア : 生存{v2['alive']:5,} 達成{v2['met']:5,} 未達{v2['short']:4,}"
      f" 移動{v2['moved']*100:4.1f}% 中位{v2['dmed']:.1f}km 95%{v2['d95']:.1f}km 最大{v2['dmax']:.1f}km")
print(f"     解体リペア {v2['repairs']:,}回 / 到達不能な成分の駅 {v2['orphan']} "
      f"/ 海を渡した駅 {v2['seas']} / 人口の保存 {v2['mass']*100:.4f}%")
big = sorted(v2["cpop"].items(), key=lambda kv: -kv[1])[:12]
print("\n  最大クラスタ12（駅名・収容人口・CAP比）:")
for sd, p_ in big:
    print(f"    {SNM[sd]:<10} {p_:>12,.0f}人  ×{p_/capn[sd]:.1f}")
PREFN = {"02":"青森","03":"岩手","04":"宮城","05":"秋田","06":"山形","07":"福島","08":"茨城",
 "09":"栃木","10":"群馬","11":"埼玉","12":"千葉","13":"東京","14":"神奈川","15":"新潟","16":"富山",
 "17":"石川","18":"福井","19":"山梨","20":"長野","21":"岐阜","22":"静岡","23":"愛知","24":"三重",
 "25":"滋賀","26":"京都","27":"大阪","28":"兵庫","29":"奈良","30":"和歌山","31":"鳥取","32":"島根",
 "33":"岡山","34":"広島","35":"山口"}
asg = v2["assign"]
print("\n  都道府県別（駅数 → 存続クラスタ数・残存率・現況達成率）:")
rows = []
for pp, jp in PREFN.items():
    m = pref_of == pp
    if not m.any(): continue
    live = len({int(asg[i]) for i in np.nonzero(m)[0] if asg[i] >= 0 and pref_of[int(asg[i])] == pp})
    rows.append((jp, int(m.sum()), live, 100*live/int(m.sum()), 100*int((cellpop[m] >= capn[m]).sum())/int(m.sum())))
for jp, n_, l_, rt, nowr in sorted(rows, key=lambda r: -r[3])[:6]:
    print(f"    {jp:<4} {n_:4}駅 → {l_:4}  残存{rt:5.1f}%  現況達成{nowr:5.1f}%")
print("    …")
for jp, n_, l_, rt, nowr in sorted(rows, key=lambda r: r[3])[:6][::-1]:
    print(f"    {jp:<4} {n_:4}駅 → {l_:4}  残存{rt:5.1f}%  現況達成{nowr:5.1f}%")

np.savez_compressed(os.path.abspath(OUT), S=S, name=SNM, owner=owner, cellpop=cellpop,
                    capn=capn, row=row, col=col, pop=pop, area_ha=area_ha, pref=pref_of,
                    edges=pr.astype(np.int32),          # 隣接グラフ（Xを変えて解き直すのに要る）
                    coreha=(capn / A.X).astype(np.float32),   # コア円の実効面積(ha)＝capn/X
                    ev=np.array(v2["events"], np.int32), r=A.r, X=A.X,
                    stat=np.array([N, len(pop), pop.sum(), v2["alive"], v2["short"],
                                   v2["moved"], v2["dmed"], v2["d95"], v2["repairs"], now_met], float))
print(f"\n  保存: {os.path.abspath(OUT)}")
lap("v1 / v2 実行")
