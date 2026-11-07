"""
本州スケールの共通コア（numpy + shapely.STRtree のみ）

bench_honshu.py（計量）と build_replay.py（HTML再生用の書き出し）から使う。
v2 は record=True で「いつどの駅がどのクラスタに入ったか」のイベント列を返す。
"""
import heapq, math
import numpy as np
from shapely import STRtree, points as shp_points

W, H = 1150.0, 900.0          # 本州の外接矩形 約1150km × 900km
CITIES = [(760, 640, 14e6, 42), (455, 430, 9e6, 34), (585, 505, 5e6, 30),
          (250, 300, 3.5e6, 28), (900, 760, 3.0e6, 26), (690, 590, 2.5e6, 22),
          (330, 360, 2.2e6, 22), (1020, 830, 1.6e6, 20)]
ISLANDS = [(300, 700, 22), (520, 250, 15), (980, 480, 12), (150, 560, 9)]
TINY    = [(230, 760, 3.0), (620, 205, 2.5), (1050, 430, 2.0), (95, 610, 2.5)]


def build_synthetic(cellkm=0.5, stations=7000, pop=103e6, seed=7, islands=False, tiny=False):
    """[REAL DATA] ここを e-Stat 500mメッシュ + N02 に差し替える"""
    rng = np.random.default_rng(seed)
    nx, ny = int(W / cellkm), int(H / cellkm)
    xs = (np.arange(nx) + .5) * cellkm
    ys = (np.arange(ny) + .5) * cellkm
    GX, GY = np.meshgrid(xs, ys)
    t = GX / W
    spine = 180 + 620 * t - 380 * (t - .45) ** 2 * 2.2
    width = 95 + 60 * np.exp(-((t - .62) ** 2) / .02)
    noise = 28 * np.sin(GX / 37) + 22 * np.cos(GY / 29) + 18 * np.sin((GX + GY) / 53)
    land = np.abs(GY - spine - noise) < width
    for group, on in ((ISLANDS, islands), (TINY, tiny)):
        if on:
            for cx, cy, rr in group:
                land |= ((GX - cx) ** 2 + (GY - cy) ** 2) < rr * rr
    del GX, GY

    popg = np.where(land, 8.0, 0.0)
    def bump(cx, cy, m, sg):
        i0, i1 = max(0, int((cx - 4*sg)/cellkm)), min(nx, int((cx + 4*sg)/cellkm) + 1)
        j0, j1 = max(0, int((cy - 4*sg)/cellkm)), min(ny, int((cy + 4*sg)/cellkm) + 1)
        if i0 >= i1 or j0 >= j1: return
        dx = xs[i0:i1][None, :] - cx; dy = ys[j0:j1][:, None] - cy
        popg[j0:j1, i0:i1] += m/(2*math.pi*sg*sg)*np.exp(-(dx*dx+dy*dy)/(2*sg*sg))*(cellkm**2)
    for cx, cy, m, sg in CITIES: bump(cx, cy, m, sg)
    for _ in range(400):
        bump(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(3e4, 4e5), 6.0)
    popg *= land
    popg *= pop / popg.sum()

    ly, lx = np.nonzero(land)
    mx, my, mpop = xs[lx], ys[ly], popg[ly, lx].astype(np.float64)
    w = mpop / mpop.sum()
    pick = rng.choice(len(mx), size=stations, replace=False, p=w)
    S = np.c_[mx[pick] + rng.normal(0, .2, stations), my[pick] + rng.normal(0, .2, stations)]
    if tiny:
        S = np.vstack([S, np.array([[c[0], c[1]] for c in TINY])])
    return dict(nx=nx, ny=ny, xs=xs, ys=ys, cellkm=cellkm, land=land,
                lx=lx, ly=ly, mx=mx, my=my, mpop=mpop, S=S)


def assign_and_graph(G, r=1.2, X=40.0):
    """最寄駅の割当（＝ボロノイ）／隣接グラフ／コア円の実効容量。多角形は作らない。"""
    S, nx, ny, cellkm = G["S"], G["nx"], G["ny"], G["cellkm"]
    N = len(S)
    owner = STRtree(shp_points(S)).nearest(shp_points(np.c_[G["mx"], G["my"]])).astype(np.int32)
    cellpop = np.bincount(owner, weights=G["mpop"], minlength=N)

    og = np.full((ny, nx), -1, np.int32)
    og[G["ly"], G["lx"]] = owner
    pairs = []
    for a, b in ((og[:, :-1], og[:, 1:]), (og[:-1, :], og[1:, :])):
        m = (a >= 0) & (b >= 0) & (a != b)
        pairs.append(np.c_[a[m], b[m]])
    pr = np.vstack(pairs)
    pr = np.unique(np.c_[pr.min(1), pr.max(1)], axis=0)
    adj = [[] for _ in range(N)]
    for a, b in pr:
        adj[a].append(int(b)); adj[b].append(int(a))

    rad = int(math.ceil(r / cellkm))
    capn = np.zeros(N)
    for i in range(N):
        cx, cy = S[i]
        i0, i1 = max(0, int(cx/cellkm) - rad), min(nx, int(cx/cellkm) + rad + 1)
        j0, j1 = max(0, int(cy/cellkm) - rad), min(ny, int(cy/cellkm) + rad + 1)
        dx = G["xs"][i0:i1][None, :] - cx; dy = G["ys"][j0:j1][:, None] - cy
        ok = (dx*dx + dy*dy <= r*r) & (og[j0:j1, i0:i1] == i)
        capn[i] = ok.sum() * (cellkm**2) * 100 * X
    capn = np.where(capn > 0, capn, math.pi*r*r*100*X)
    return dict(N=N, owner=owner, og=og, cellpop=cellpop, adj=adj, capn=capn, edges=len(pr))


def _dist(S, i, j):
    return float(math.hypot(S[i, 0] - S[j, 0], S[i, 1] - S[j, 1]))


def run_v1(A, S, dmax=10.0):
    """現行仕様 §3.1（ボトムアップ貪欲）。未達駅が残りうる。"""
    N, cellpop, capn, adj = A["N"], A["cellpop"], A["capn"], A["adj"]
    par = list(range(N))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    P = cellpop.copy(); nb = [set(a) for a in adj]
    alive, un, merges = set(range(N)), set(), 0
    h = [(P[i]/capn[i], i) for i in range(N) if P[i] < capn[i]]
    heapq.heapify(h)
    while h:
        r_, s = heapq.heappop(h)
        if s not in alive or s in un: continue
        if P[s] >= capn[s]: continue
        if abs(r_ - P[s]/capn[s]) > 1e-12:
            heapq.heappush(h, (P[s]/capn[s], s)); continue
        pool = [j for j in {find(q) for q in nb[s]} if j != s and j in alive and _dist(S, s, j) <= dmax]
        if not pool:
            un.add(s); continue
        t_ = max(pool, key=lambda j: P[j]/capn[j])
        P[t_] += P[s]; par[s] = t_; alive.discard(s); nb[t_] |= nb[s]; merges += 1
        if P[t_] < capn[t_]: heapq.heappush(h, (P[t_]/capn[t_], t_))
    root = np.array([find(i) for i in range(N)])
    live = sorted(alive)
    short = [i for i in live if P[i] < capn[i]]
    dist = sorted(_dist(S, i, root[i]) for i in range(N) if root[i] != i)
    moved = float(sum(cellpop[i] for i in range(N) if root[i] != i))
    return dict(alive=len(live), met=len(live)-len(short), short=len(short),
                shortpop=float(sum(P[i] for i in short)), merges=merges,
                moved=moved/cellpop.sum(), root=root,
                dmed=dist[len(dist)//2] if dist else 0,
                d95=dist[int(len(dist)*.95)] if dist else 0)


def run_v2(A, S, policy="relocate", record=False):
    """v2: 種成長 + 解体リペア。未達クラスタは構成上残らない。
    record=True で再生用イベント列 events=[(phase, station, cluster)] を返す。"""
    N, cellpop, capn, adj = A["N"], A["cellpop"], A["capn"], A["adj"]
    comp = np.full(N, -1); nc = 0
    for i in range(N):
        if comp[i] >= 0: continue
        st = [i]; comp[i] = nc
        while st:
            u = st.pop()
            for v in adj[u]:
                if comp[v] < 0: comp[v] = nc; st.append(v)
        nc += 1

    assign = np.full(N, -1); cpop = {}
    orphan, deadpop, seas, repairs = [], 0.0, 0, 0
    ev = []
    def rec(ph, u, s):
        if record: ev.append((ph, int(u), int(s)))

    def flood(freed, phase):
        h = []
        for u in freed:
            for v in adj[u]:
                if assign[v] >= 0: heapq.heappush(h, (_dist(S, u, v), u, int(assign[v])))
        while h:
            d, u, s = heapq.heappop(h)
            if assign[u] >= 0: continue
            assign[u] = s; cpop[s] += float(cellpop[u]); rec(phase, u, s)
            for v in adj[u]:
                if assign[v] < 0: heapq.heappush(h, (d + _dist(S, u, v), v, s))

    for c in range(nc):
        mem = np.nonzero(comp == c)[0]
        tot = float(cellpop[mem].sum()); need = float(capn[mem].mean())
        k = int(tot // need)
        if k < 1:
            orphan.extend(mem.tolist()); deadpop += tot; continue
        order = mem[np.argsort(-cellpop[mem]/capn[mem])]
        h = []
        for sd in order[:k]:
            sd = int(sd)
            assign[sd] = sd; cpop[sd] = float(cellpop[sd]); rec(1, sd, sd)
            for v in adj[sd]: heapq.heappush(h, (_dist(S, sd, v), v, sd))
        while h:
            d, u, sd = heapq.heappop(h)
            if assign[u] >= 0 or cpop[sd] >= capn[sd]: continue
            assign[u] = sd; cpop[sd] += float(cellpop[u]); rec(2, u, sd)
            if cpop[sd] < capn[sd]:
                for v in adj[u]:
                    if assign[v] < 0: heapq.heappush(h, (d + _dist(S, u, v), v, sd))
        flood([i for i in mem if assign[i] < 0], 3)
        while True:
            bad = [sd for sd in cpop if comp[sd] == c and cpop[sd] < capn[sd]]
            if not bad or len([sd for sd in cpop if comp[sd] == c]) <= 1: break
            s0 = min(bad, key=lambda sd: cpop[sd]/capn[sd])
            mems = np.nonzero(assign == s0)[0]
            del cpop[s0]
            for i in mems: assign[i] = -1
            flood(mems.tolist(), 4); repairs += 1
            left = [i for i in mems if assign[i] < 0]
            if left:
                cpop[s0] = 0.0
                for i in mems:
                    if assign[i] < 0:
                        assign[i] = s0; cpop[s0] += float(cellpop[i]); rec(4, i, s0)
                break

    # 成分に1つしか残らず、それでも CAP 未満＝その成分は構造的に不可能（未達として残さない）
    for c in range(nc):
        cl = [sd for sd in cpop if comp[sd] == c]
        if len(cl) == 1 and cpop[cl[0]] < capn[cl[0]]:
            mems = np.nonzero(assign == cl[0])[0]
            del cpop[cl[0]]
            for i in mems: assign[i] = -1
            orphan.extend(mems.tolist()); deadpop += float(cellpop[mems].sum())

    if orphan and policy == "relocate":
        live = np.array(sorted(cpop))
        for i in orphan:
            d = np.hypot(S[live, 0] - S[i, 0], S[live, 1] - S[i, 1])
            sd = int(live[int(np.argmin(d))])
            assign[i] = sd; cpop[sd] += float(cellpop[i]); seas += 1; rec(5, i, sd)
        deadpop = 0.0

    live = sorted(cpop)
    short = [sd for sd in live if cpop[sd] < capn[sd]]
    mv = [(i, int(assign[i])) for i in range(N) if assign[i] >= 0 and assign[i] != i]
    dist = sorted(_dist(S, i, sd) for i, sd in mv)
    return dict(alive=len(live), met=len(live)-len(short), short=len(short),
                shortpop=float(sum(cpop[sd] for sd in short)),
                orphan=len(orphan), deadpop=deadpop, seas=seas, repairs=repairs,
                moved=float(sum(cellpop[i] for i, sd in mv))/cellpop.sum(),
                dmed=dist[len(dist)//2] if dist else 0,
                d95=dist[int(len(dist)*.95)] if dist else 0,
                dmax=dist[-1] if dist else 0,
                mass=float(sum(cpop.values()))/cellpop.sum(),
                assign=assign, cpop=cpop, events=ev)
