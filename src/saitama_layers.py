#!/usr/bin/env python3
"""地図ベースレイヤー(鉄道・国道)を data/saitama/layers.json に保存する。
鉄道: 国土数値情報 N02-25 RailroadSection（N02_001=11 が新幹線）
国道: data/saitama/raw/osm/roads_*.json（OSM キャッシュ。ref/official_name タグが残っている）
座標は EPSG:6677（pyproj always_xy）、格子範囲+余白で切り、simplify(100m)。fields/meta には触らない。"""
import glob, json, re, sys
from collections import defaultdict
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, box
from shapely.ops import linemerge

sys.path.insert(0, "./src")
import shpdbf

D = "./"
meta = json.load(open(D + "data/saitama/meta.json", encoding="utf-8"))
g = meta["grid"]; M = 5000
BOX = box(g["x0"] - M, g["y0"] - M, g["x0"] + g["ncols"] * g["cell"] + M, g["y0"] + g["nrows"] * g["cell"] + M)
TR = Transformer.from_crs("EPSG:6668", "EPSG:6677", always_xy=True)


def emit(geoms, simp=100.0):
    out = []
    for gm in geoms:
        gm = gm.intersection(BOX)
        if gm.is_empty: continue
        gm = gm.simplify(simp)
        for ln in (gm.geoms if hasattr(gm, "geoms") else [gm]):
            if ln.geom_type == "LineString" and len(ln.coords) >= 2:
                out.append([[round(x), round(y)] for x, y in ln.coords])
    return out


def merged(lines):
    m = linemerge(MultiLineString(lines))
    return [m] if m.geom_type == "LineString" else list(m.geoms)


# ---- 鉄道 ----
base = D + "data/raw/N02-25_GML/UTF-8/N02-25_RailroadSection"
_, d, nrec = shpdbf.read_dbf(base + ".dbf", encoding="utf-8")
xy, starts, rec, m = shpdbf.read_shp_polylines(base + ".shp")
X, Y = TR.transform(xy[:, 0], xy[:, 1])
P = np.column_stack([X, Y])
ends = list(starts[1:]) + [m]
grp = defaultdict(list)
for k, (a, b) in enumerate(zip(starts, ends)):
    if b - a < 2: continue
    r = rec[k]
    kind = "shinkansen" if d["N02_001"][r] == "11" else "rail"
    key = (kind, d["N02_003"][r], d["N02_004"][r])
    ls = LineString(P[a:b])
    if ls.intersects(BOX): grp[key].append(ls)
rail = []
for (kind, name, op), ls in sorted(grp.items()):
    for pts in emit(merged(ls)):
        rail.append({"kind": kind, "name": name, "operator": op, "pts": pts})

# ---- 国道（OSM キャッシュ）----
def refnum(tags):
    for k in ("ref",):
        for t in re.split(r"[;,]", tags.get(k, "")):
            t = t.strip()
            if t.isdigit() and 1 <= int(t) <= 507: return t
    return None

ways = {}
for fn in sorted(glob.glob(D + "data/saitama/raw/osm/roads_*.json")):
    for e in json.load(open(fn, encoding="utf-8"))["elements"]:
        if e.get("type") == "way" and "geometry" in e: ways[e["id"]] = e
byref = defaultdict(list); names = {}
for e in ways.values():
    t = e.get("tags", {}); hw = t.get("highway", "")
    if hw not in ("trunk", "primary", "trunk_link", "primary_link", "secondary"): continue
    nm = " ".join(t.get(k, "") for k in ("name", "official_name", "name:ja"))
    ref = refnum(t)
    is_nh = ("国道" in nm) or (hw.startswith("trunk") and ref)
    if not is_nh: continue
    if not ref:
        mm = re.search(r"国道(\d+)号", nm)
        ref = mm.group(1) if mm else None
    if not ref: continue
    pts = [(p["lon"], p["lat"]) for p in e["geometry"]]
    if len(pts) < 2: continue
    xs, ys = TR.transform([p[0] for p in pts], [p[1] for p in pts])
    byref[ref].append(LineString(list(zip(xs, ys))))
nh = []
for ref in sorted(byref, key=int):
    for pts in emit(merged(byref[ref])):
        nh.append({"ref": ref, "name": f"国道{ref}号", "pts": pts})

json.dump({"rail": rail, "nh": nh}, open(D + "data/saitama/layers.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
sh = [r for r in rail if r["kind"] == "shinkansen"]
print("rail", len(rail), "shinkansen", len(sh), sorted({r["name"] for r in sh}))
print("nh lines", len(nh), "refs", sorted({r["ref"] for r in nh}, key=int))
