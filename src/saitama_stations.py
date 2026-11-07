"""全国(本州)の駅生存を 11 個の X で再計算し、キャッシュする。座標は EPSG:6677 に変換。
出力: data/saitama/alive11.npz  (alive[11,N] 生存, retreat_st[11,N] 撤退(unreachable→relocate)駅, S6677[N,2], coreha, name, X, nalive[11])"""
import os, sys, time, re
import numpy as np
from pyproj import Transformer
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import honshu_core as hc

D = os.path.join(HERE, "..", "data")
OUT = os.path.join(D, "saitama", "alive11.npz")

def presets_X():
    src = open(os.path.join(HERE, "build_replay.py"), encoding="utf-8").read()
    blk = src[src.index("PRESETS = ["):src.index("DEFAULT_X")]
    return [float(m) for m in re.findall(r"^\s*\(([\d.]+),", blk, re.M)]

def main(force=False):
    if os.path.exists(OUT) and not force:
        return dict(np.load(OUT, allow_pickle=True))
    z = np.load(os.path.join(D, "honshu.npz"), allow_pickle=True)
    S, cellpop, coreha = z["S"], z["cellpop"], z["coreha"].astype(np.float64)
    N = len(S)
    adj = [[] for _ in range(N)]
    for a, b in z["edges"]:
        adj[a].append(int(b)); adj[b].append(int(a))
    Xs = presets_X(); assert len(Xs) == 11, Xs
    al = np.zeros((11, N), np.uint8); rt = np.zeros((11, N), np.uint8)
    for k, X in enumerate(Xs):
        t = time.time()
        A = dict(N=N, cellpop=cellpop, capn=coreha * X, adj=adj)
        v = hc.run_v2(A, S, record=True)
        for s in v["cpop"]: al[k, s] = 1
        capn = A["capn"]; asg = v["assign"]
        for sd, cp in v["cpop"].items():       # 未達(short)クラスタ＝密度Xに構造的に届かない圏域＝撤退地域
            if cp < capn[sd]: rt[k, np.nonzero(asg == sd)[0]] = 1
        for e in v["events"]:
            if e[0] == 5: rt[k, e[1]] = 1      # phase5 = 吸収できず最寄りへ再配置された孤立駅
        print(f"X={X} alive={v['alive']} short={v['short']} retreat_st={int(rt[k].sum())} orphan={v['orphan']} ({time.time()-t:.1f}s)", flush=True)
    TR = Transformer.from_crs("+proj=laea +lat_0=37 +lon_0=138 +ellps=GRS80 +units=m +no_defs", "EPSG:6668", always_xy=True)
    lo, la = TR.transform(S[:, 0] * 1000, S[:, 1] * 1000)
    T2 = Transformer.from_crs("EPSG:6668", "EPSG:6677", always_xy=True)
    x, y = T2.transform(lo, la)
    out = dict(alive=al, retreat_st=rt, S6677=np.c_[x, y], coreha=coreha, name=z["name"], X=np.array(Xs),
               nalive=al.sum(1), lon=lo, lat=la)
    np.savez_compressed(OUT, **out)
    return out

if __name__ == "__main__":
    main(force="--force" in sys.argv)
