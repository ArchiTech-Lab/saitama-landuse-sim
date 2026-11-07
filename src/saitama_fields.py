"""埼玉 時間場の計算。全距離・面積は EPSG:6677 (JGD2011 平面直角IX系, m)。
道路を 100m 格子に最大速度で焼き付け→8近傍 Dijkstra (scipy.sparse.csgraph)→500mセル中心の100mセル値を分に丸めて uint8。
出力: data/saitama/fields.npz, meta.json （キャッシュ: roads100.npz, alive11.npz）"""
import os, sys, re, json, time, datetime
import numpy as np
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import saitama_stations as sst

D = os.path.join(HERE, "..", "data", "saitama")
C = 500; F = 100; BUF = 30000; R_CORE = 1200.0
TR = Transformer.from_crs("EPSG:6668", "EPSG:6677", always_xy=True)   # JGD2011 経緯度 -> 平面直角IX系
CLS = {"tertiary": 1, "secondary": 2, "primary": 3, "trunk": 4, "motorway": 5}
PRESETS = [
    dict(id="real", label="現実的", mult=1.0),
    dict(id="congested", label="混雑時", mult=0.6),
    dict(id="fast", label="高速寄り", mult=1.2),
]
BASE = {"motorway": 80, "trunk": 50, "primary": 40, "secondary": 30, "tertiary": 22, "none": 12}

def preset_speeds(p):
    return {k: round(v * p["mult"], 1) for k, v in BASE.items()}

def pref():
    p = np.load(os.path.join(D, "pref.npz"))
    return dict(x0=float(p["x0"]), y0=float(p["y0"]), ncols=int(p["ncols"]), nrows=int(p["nrows"]),
                valid=p["valid"].astype(np.uint8), water=p["water"].astype(np.uint8))

def fgrid(P):
    gx0 = P["x0"] - BUF; gy0 = P["y0"] - BUF
    W = int(round((P["ncols"] * C + 2 * BUF) / F)); H = int(round((P["nrows"] * C + 2 * BUF) / F))
    return gx0, gy0, W, H

def road_raster(P):
    """道路クラス(0=なし,1..5)を100m格子に焼き付け。geojson は1行51MBなので正規表現で1地物ずつ処理。"""
    cache = os.path.join(D, "roads100.npz")
    gx0, gy0, W, H = fgrid(P)
    if os.path.exists(cache):
        z = np.load(cache)
        if z["cls"].shape == (H, W): return z["cls"]
    s = open(os.path.join(D, "roads.geojson"), encoding="utf-8").read()
    pat = re.compile(r'"highway":"([a-z_]+)".*?"coordinates":\[\[(.*?)\]\]\}', re.S)
    cls = np.zeros(H * W, np.uint8)
    A = []; B = []; K = []
    for m in pat.finditer(s):
        k = CLS.get(m.group(1).replace("_link", ""))
        if not k: continue
        c = np.array(re.findall(r"-?[\d.]+", m.group(2)), float).reshape(-1, 2)
        if len(c) < 2: continue
        A.append(c[:-1]); B.append(c[1:]); K.append(np.full(len(c) - 1, k, np.uint8))
    del s
    A = np.vstack(A); B = np.vstack(B); K = np.concatenate(K)
    ax, ay = TR.transform(A[:, 0], A[:, 1]); bx, by = TR.transform(B[:, 0], B[:, 1])
    L = np.hypot(bx - ax, by - ay)
    n = np.maximum(1, np.ceil(L / 40).astype(int)) + 1          # 40m 刻みでサンプル（100m格子を飛ばさない）
    seg = np.repeat(np.arange(len(L)), n)
    off = np.arange(len(seg)) - np.repeat(np.cumsum(n) - n, n)
    t = off / np.repeat(n - 1, n)
    px = ax[seg] + (bx - ax)[seg] * t; py = ay[seg] + (by - ay)[seg] * t
    ci = np.floor((px - gx0) / F).astype(int); ri = np.floor((py - gy0) / F).astype(int)
    ok = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    np.maximum.at(cls, ri[ok] * W + ci[ok], K[seg][ok])
    cls = cls.reshape(H, W)
    np.savez_compressed(cache, cls=cls)
    return cls

def graph(cls, speeds):
    """一般道のみ（motorway は使わない＝焼き付けでも none 扱い）。辺の所要時間[分]=距離/平均速度(調和)"""
    H, W = cls.shape
    lut = np.array([speeds["none"], speeds["tertiary"], speeds["secondary"], speeds["primary"],
                    speeds["trunk"], speeds["none"]], float)         # index 5=motorway→使わない
    v = lut[cls.ravel()]
    idx = np.arange(H * W).reshape(H, W)
    rows = []; cols = []; w = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0_, y1_ = 0, H - dy
        x0_, x1_ = (0, W - dx) if dx >= 0 else (-dx, W)
        a = idx[y0_:y1_, x0_:x1_].ravel(); b = idx[y0_ + dy:y1_ + dy, x0_ + dx:x1_ + dx].ravel()
        d = np.hypot(dx, dy) * F / 1000.0
        t = d * 30.0 * (1 / v[a] + 1 / v[b])                         # d/2*(1/v1+1/v2) h -> 分: ×60/2
        rows += [a, b]; cols += [b, a]; w += [t, t]
    return csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(H * W, H * W))

def src_cells(P, x, y):
    gx0, gy0, W, H = fgrid(P)
    ci = np.floor((np.asarray(x) - gx0) / F).astype(int); ri = np.floor((np.asarray(y) - gy0) / F).astype(int)
    ok = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    return np.unique(ri[ok] * W + ci[ok])

def center_idx(P):
    """500m セル i の中心を含む100mセルの 1次元添字"""
    gx0, gy0, W, H = fgrid(P)
    nc, nr = P["ncols"], P["nrows"]
    cc, rr = np.meshgrid(np.arange(nc), np.arange(nr))
    ci = ((P["x0"] + (cc + .5) * C - gx0) // F).astype(int); ri = ((P["y0"] + (rr + .5) * C - gy0) // F).astype(int)
    return (ri * W + ci).ravel()

def nh100(P):
    """国道沿線100m帯が各500mセルを覆う面積割合(%, uint8[N])。元の線(単純化前)を6677に投影→shapelyで100mバッファ。
    国道の選択条件は saitama_layers.py と同じ（trunk で ref 1〜507、または name/official_name に「国道」を含む trunk/primary/secondary）。"""
    cache = os.path.join(D, "nh100.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if z["nh"].size == P["ncols"] * P["nrows"]: return z["nh"]
    import glob, shapely
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    def refnum(t):
        for q in re.split(r"[;,]", t.get("ref", "")):
            q = q.strip()
            if q.isdigit() and 1 <= int(q) <= 507: return q
        return None
    lines = {}
    for fn in sorted(glob.glob(os.path.join(D, "raw", "osm", "roads_*.json"))):
        for e in json.load(open(fn, encoding="utf-8"))["elements"]:
            if e.get("type") != "way" or "geometry" not in e: continue
            t = e.get("tags", {}); hw = t.get("highway", "")
            if hw not in ("trunk", "primary", "trunk_link", "primary_link", "secondary"): continue
            nm = " ".join(t.get(k, "") for k in ("name", "official_name", "name:ja"))
            ref = refnum(t)
            if not (("国道" in nm) or (hw.startswith("trunk") and ref)): continue
            if not ref:
                mm = re.search(r"国道(\d+)号", nm); ref = mm.group(1) if mm else None
            if not ref: continue
            pts = e["geometry"]
            if len(pts) < 2: continue
            xs, ys = TR.transform([q["lon"] for q in pts], [q["lat"] for q in pts])
            lines[e["id"]] = LineString(list(zip(xs, ys)))
    band = unary_union([l.buffer(100.0, cap_style=2) for l in lines.values()])    # 6677 平面での100mバッファ
    nc, nr = P["ncols"], P["nrows"]
    out = np.zeros(nc * nr, np.uint8)
    cc, rr = np.meshgrid(np.arange(nc), np.arange(nr)); cc = cc.ravel(); rr = rr.ravel()
    boxes = shapely.box(P["x0"] + cc * C, P["y0"] + rr * C, P["x0"] + (cc + 1) * C, P["y0"] + (rr + 1) * C)
    shapely.prepare(band)
    hit = np.nonzero(shapely.intersects(boxes, band))[0]
    from shapely import area as area_m2      # 6677 平面座標なので m²
    a = area_m2(shapely.intersection(boxes[hit], band))
    out[hit] = np.clip(np.rint(a / (C * C) * 100), 0, 100).astype(np.uint8)
    np.savez_compressed(cache, nh=out)
    return out

def to_u8(t):
    t = np.where(np.isfinite(t), np.rint(t), 255)
    return np.minimum(t, 255).astype(np.uint8)

def route_len(G, src, W, tmax):
    dist, pred, srcs = dijkstra(G, directed=True, indices=src, min_only=True, return_predecessors=True, limit=tmax)
    n = G.shape[0]
    p = np.where(pred < 0, np.arange(n), pred)
    py, px = np.divmod(np.arange(n), W); qy, qx = np.divmod(p, W)
    own = np.hypot(px - qx, py - qy) * F                               # 親への辺長(m)、根は0
    acc = own.copy(); anc = p.copy()
    while True:                                                       # acc[v]=v→anc[v]の距離 を倍々に延長
        nxt = anc[anc]
        if np.array_equal(nxt, anc): break
        acc = acc + acc[anc] ; anc = nxt
    return dist, srcs, acc, anc

def ratio_median(P, G, src, W, valid_idx, tmax=30.0, dmin=2000.0):
    gx0, gy0, _, H = fgrid(P)
    dist, srcs, acc, anc = route_len(G, src, W, tmax)
    n = G.shape[0]; y, x = np.divmod(np.arange(n), W)
    sy, sx = np.divmod(np.where(srcs >= 0, srcs, 0), W)
    eu = np.hypot(x - sx, y - sy) * F
    m = np.zeros(n, bool); m[valid_idx] = True
    m &= np.isfinite(dist) & (dist <= tmax) & (srcs >= 0) & (eu >= dmin)
    r = acc[m] / eu[m]
    return float(np.median(r)), int(m.sum()), float(np.percentile(r, 10)), float(np.percentile(r, 90))

def ptn_X():
    src = open(os.path.join(HERE, "build_replay.py"), encoding="utf-8").read()
    blk = src[src.index("PRESETS = ["):src.index("DEFAULT_X")]
    return re.findall(r'^\s*\(([\d.]+),\s*"([^"]*)"', blk, re.M)

def main():
    t00 = time.time()
    P = pref(); nc, nr = P["ncols"], P["nrows"]; N = nc * nr
    gx0, gy0, W, H = fgrid(P)
    cls = road_raster(P); print(f"道路焼き付け {time.time()-t00:.0f}s  道路セル {(cls>0).sum():,} / {cls.size:,}", flush=True)
    st = sst.main()
    ic = json.load(open(os.path.join(D, "ic.json"), encoding="utf-8"))
    cidx = center_idx(P)
    # 駅: 格子(県bbox+30km)内
    sx, sy = st["S6677"][:, 0], st["S6677"][:, 1]
    inb = (sx >= gx0) & (sx < gx0 + W * F) & (sy >= gy0) & (sy < gy0 + H * F)
    ii = np.nonzero(inb)[0]
    X = st["X"]; XL = ptn_X()
    tS = np.full((11, 3, N), 255, np.uint8); tI = np.full((3, N), 255, np.uint8)
    core = np.zeros((11, N), np.uint8); retreat = np.zeros((11, N), np.uint8)
    # IC始点: IC/SIC のみ（県外バッファ内含む）
    icx, icy = TR.transform([d["lon"] for d in ic], [d["lat"] for d in ic])
    icx = np.array(icx); icy = np.array(icy)
    ickind = np.array([d["kind"] for d in ic])
    icsrc = src_cells(P, icx[np.isin(ickind, ["IC", "SIC"])], icy[np.isin(ickind, ["IC", "SIC"])])
    for pi, p in enumerate(PRESETS):
        G = graph(cls, preset_speeds(p))
        t = time.time()
        d = dijkstra(G, directed=True, indices=icsrc, min_only=True, limit=255.0)
        tI[pi] = to_u8(d[cidx])
        for k in range(11):
            a = ii[st["alive"][k, ii] == 1]
            d = dijkstra(G, directed=True, indices=src_cells(P, sx[a], sy[a]), min_only=True, limit=255.0)
            tS[k, pi] = to_u8(d[cidx])
        print(f"  preset {p['id']}: dijkstra 12回 {time.time()-t:.0f}s", flush=True)
    # コア円（セル中心が生存駅のコア円内）と撤退地域
    cc, rr = np.meshgrid(np.arange(nc), np.arange(nr))
    cx = (P["x0"] + (cc + .5) * C).ravel(); cy = (P["y0"] + (rr + .5) * C).ravel()
    cenxy = np.c_[cx, cy]; ctree = cKDTree(cenxy)
    corem = np.sqrt(np.minimum(st["coreha"], np.pi * R_CORE**2 / 1e4) * 1e4 / np.pi)   # 駅ごとのコア円半径(m)
    own_tree = None
    for k in range(11):
        a = np.nonzero((st["alive"][k] == 1) & (np.abs(sx - (P["x0"] + nc * C / 2)) < nc * C) & (np.abs(sy - (P["y0"] + nr * C / 2)) < nr * C))[0]
        stree = cKDTree(np.c_[sx[a], sy[a]])
        sm = stree.sparse_distance_matrix(ctree, R_CORE + 1, output_type="coo_matrix")
        ok = sm.data <= corem[a][sm.row]
        core[k, sm.col[ok]] = 1
        rs = np.nonzero(st["retreat_st"][k] == 1)[0]
        if len(rs):                                                   # 撤退駅のボロノイ(最寄り母点)セルを撤退地域に
            allp = np.nonzero(inb)[0]
            tr_ = cKDTree(np.c_[sx[allp], sy[allp]]); _, nn = tr_.query(cenxy)
            retreat[k] = np.isin(allp[nn], rs).astype(np.uint8)
    core[:, P["water"] == 1] = 0
    core[:, P["valid"] == 0] = 0; retreat[:, P["valid"] == 0] = 0
    np.savez_compressed(os.path.join(D, "fields.npz"), valid=P["valid"], water=P["water"], tS=tS, tI=tI, core=core, retreat=retreat, nh100=nh100(P))
    # meta
    stations = []
    for i in ii:
        if not st["alive"][:, i].any() and False: continue
        stations.append(dict(name=str(st["name"][i]), x=round(float(sx[i]), 1), y=round(float(sy[i]), 1),
                             alive=[int(v) for v in st["alive"][:, i]], core_m=round(float(corem[i]), 1)))
    # 全国の駅のうち格子内(県+30km)。死んだ駅も alive 全0で載せる（U が地図に描く場合のため）
    meta = dict(grid=dict(x0=P["x0"], y0=P["y0"], ncols=nc, nrows=nr, cell=C, crs="EPSG:6677"),
        X=[float(v) for v in X], Xlabel=[l for _, l in XL], r=int(R_CORE),
        presets=[dict(id=p["id"], label=p["label"], speeds=preset_speeds(p)) for p in PRESETS],
        stations=stations,
        ic=[dict(name=d["name"], x=round(float(icx[j]), 1), y=round(float(icy[j]), 1), kind=d["kind"]) for j, d in enumerate(ic)],
        sources=["国土数値情報 鉄道データ N02（駅）", "国土数値情報 行政区域データ N03-20250101（埼玉県境）",
                 "国土数値情報 土地利用細分メッシュ L03-b-16（河川地及び湖沼＝区分外）",
                 "国勢調査 地域メッシュ統計（500mメッシュ人口）および将来推計人口メッシュ",
                 "道路・IC：© OpenStreetMap contributors（ODbL）"],
        limits=["農地・工業は到達圏のみを塗る。標高・傾斜・現況の森林/市街は考慮しない（秩父山地も到達圏なら農地に塗られる）",
                "500m格子のため、工業セルは国道100m帯がかかるセル（面積を過大評価する）。工業の実面積は国道100m帯の実面積として別掲",
                "工業地帯＝ICから一般道でZ分以内かつ国道沿線100m以内（国道はOSM。ICはIC/SIC）",
                "道路は主要道（高速・国道・主要地方道・県道・市道の一部＝OSM の motorway/trunk/primary/secondary/tertiary）のみ。それ以外は一律12km/hの面として扱う",
                "所要時間は渋滞・信号・時間帯を反映しない。速度は仮定値で実測ではない",
                "駅からの時間Yも ICからの時間Zも一般道のみ（高速道路は使わない）の保守的な値。ICは Z の始点としてのみ使う",
                "食料自給率・収量・経済価値は算出していない（土地の配分のみ）",
                "水域（区分外）は L03-b 2016年版。セル中心がN03県境内のものを県内とする（500m格子の判定誤差あり）",
                "生存駅は本州全体の統合モデル（v2）の結果を県境の外側30kmまで含めて使う。撤退地域は v2 では全Xで0（未達クラスタなし）"],
        built=datetime.date.today().isoformat())
    json.dump(meta, open(os.path.join(D, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    # 迂回率（現実的・X=40・駅から30分以内）
    Gr = graph(cls, preset_speeds(PRESETS[0]))
    k40 = list(X).index(40.0); a = ii[st["alive"][k40, ii] == 1]
    vi = cidx[P["valid"] == 1]
    med, n_, p10, p90 = ratio_median(P, Gr, src_cells(P, sx[a], sy[a]), W, vi)
    json.dump(dict(median=med, n=n_, p10=p10, p90=p90), open(os.path.join(D, "detour.json"), "w"))
    print(f"迂回率中央値 {med:.3f} (n={n_}, p10={p10:.3f}, p90={p90:.3f})  総 {time.time()-t00:.0f}s")

if __name__ == "__main__":
    main()
