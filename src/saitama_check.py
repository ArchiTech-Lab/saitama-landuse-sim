"""M1 の 1〜7 を判定して PASS/FAIL を一覧出力。実行: python3 saitama_check.py"""
import os, sys, re, json, glob, base64, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import saitama_classify as sc
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data", "saitama")
res = []
def rec(n, name, ok, note): res.append((n, name, ok, note)); print(f"[{'PASS' if ok else 'FAIL'}] {n} {name}: {note}", flush=True)

F = sc.load(); meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
valid, water = F["valid"], F["water"]
NX, NP = F["core"].shape[0], F["tS"].shape[1]
PRIOS = ["industry"]
CELL_KM2 = 0.25

# 1 面積保存
from shapely.geometry import shape
from shapely.ops import unary_union, transform
from pyproj import Transformer
gj = json.load(open(os.path.join(DATA, "raw", "n03", "N03-20250101_11.geojson"), encoding="utf-8"))
tr = Transformer.from_crs("EPSG:6668", "EPSG:6677", always_xy=True)
u = unary_union([shape(f["geometry"]).buffer(0) for f in gj["features"] if f["properties"]["N03_001"] == "埼玉県"])
area = transform(lambda x, y, z=None: tr.transform(x, y), u).area / 1e6          # EPSG:6677 の平面面積 km²
va = valid.sum() * CELL_KM2
bad = 0
for x in (0, 4, 10):
    for p in range(NP):
        for pr in PRIOS:
            c = sc.classify(F, x, 30, 10, p, pr)
            if sum(int((c == k).sum()) for k in range(6)) != int(valid.sum()) or (c == 255).sum() != (valid == 0).sum(): bad += 1
rec(1, "面積保存", abs(va - area) / area <= 0.02 and bad == 0,
    f"valid {va:.1f}km² / N03(6677) {area:.1f}km² 差{100*(va-area)/area:+.2f}%、区分合計≠valid の組 {bad}")

# 2 入れ子
bad = []
for x in range(NX):
    for p in range(NP):
        for pr in ("conflict",):
            fm = [(lambda c: (c == 2) | (c == 1))(sc.classify(F, x, Y, 10, p, pr)) for Y in (15, 30, 45)]
            im = [(lambda c: c == 1)(sc.classify(F, x, 30, Z, p, pr)) for Z in (5, 10, 15)]
            if not (np.all(fm[0] <= fm[1]) and np.all(fm[1] <= fm[2]) and np.all(im[0] <= im[1]) and np.all(im[1] <= im[2])): bad.append((x, p))
rec(2, "入れ子 Y15⊂30⊂45, Z5⊂10⊂15", not bad, f"全{NX}X×{NP}速度で検査、違反 {len(bad)}")

# 3 単調性
inpref = []
core_a = F["core"].sum(1) * CELL_KM2
for k in range(NX):
    n = 0
    for s in meta["stations"]:
        pass
    inpref.append(0)
# 県内生存駅 = alive==1 かつ駅座標のセルが valid
g = meta["grid"]
nin = []
for k in range(NX):
    n = 0
    for s in meta["stations"]:
        if s["alive"][k] != 1: continue
        c = int((s["x"] - g["x0"]) // 500); r = int((s["y"] - g["y0"]) // 500)
        if 0 <= c < g["ncols"] and 0 <= r < g["nrows"] and valid[r * g["ncols"] + c]: n += 1
    nin.append(n)
mono_s = all(nin[i] >= nin[i + 1] for i in range(NX - 1)); mono_a = all(core_a[i] >= core_a[i + 1] - 1e-9 for i in range(NX - 1))
rec(3, "単調非増加", mono_s and mono_a, f"県内生存駅 {nin} / 居住面積km² {[round(float(v),1) for v in core_a]}")

# 4 全国の生存駅数 = 公開済みリプレイ
txt = open(os.path.join(ROOT, "docs", "honshu_replay_real.html"), encoding="utf-8").read()
pub = [int(v) for v in re.findall(r'"alive":\s*(\d+),\s*"short"', txt)]
del txt
st = np.load(os.path.join(DATA, "alive11.npz"), allow_pickle=True)
mine = [int(v) for v in st["nalive"]]
rec(4, "全国生存駅数=公開リプレイ", pub == mine and len(pub) == 11, f"リプレイ {pub} / 再計算 {mine}")

# 5 JS 一致
html_path = os.path.join(ROOT, "site-saitama", "index.html")
r = subprocess.run([sys.executable, os.path.join(HERE, "build_saitama.py")], capture_output=True, text=True)
if r.returncode != 0: rec(5, "Python/JS 一致", False, "build失敗 " + r.stderr[-200:])
else:
    from playwright.sync_api import sync_playwright
    exe = glob.glob("~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell")[0]
    XS, YS, ZS = (0, 4, 10), (15, 30, 45), (5, 10, 15)
    ncmp = 0; nbad = 0
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe); pg = b.new_page()
        pg.goto("file://" + os.path.abspath(html_path)); pg.evaluate("window.ready")
        js = """([x,y,z,p,pr])=>{const c=window.classify(x,y,z,p,pr);let s='';for(let i=0;i<c.length;i+=8192)s+=String.fromCharCode.apply(null,c.subarray(i,i+8192));return btoa(s)}"""
        for x in XS:
            for y in YS:
                for z in ZS:
                    for pi in range(NP):
                        for pr in PRIOS:
                            j = np.frombuffer(base64.b64decode(pg.evaluate(js, [x, y, z, pi, pr])), np.uint8)
                            py = sc.classify(F, x, y, z, pi, pr)
                            ncmp += 1
                            nbad += int((j != py).sum() > 0) if len(j) == len(py) else 1
        b.close()
    rec(5, "Python/JS 区分一致", nbad == 0 and ncmp == 81, f"{ncmp}組（X3×Y3×Z3×速度3）全セル比較、不一致 {nbad}")

# 6 投影
pat = re.compile(r"3857|4326|\.area\b|geod", re.I)
hits = []
for f in sorted(glob.glob(os.path.join(HERE, "saitama_*.py")) + [os.path.join(HERE, "build_saitama.py")]):
    if os.path.basename(f) == "saitama_check.py": continue
    for i, l in enumerate(open(f, encoding="utf-8"), 1):
        if pat.search(l): hits.append(f"{os.path.basename(f)}:{i}")
rec(6, "投影 6677 のみ", not hits, "3857/4326/生の.area/geod の出現なし（saitama_*.py, build_saitama.py）" if not hits else str(hits))
# 面積計算は check 自身のN03面積のみ（6677変換後）

# 7 迂回率
dj = json.load(open(os.path.join(DATA, "detour.json")))
rec(7, "道路経路長/直線距離の中央値", 1.05 <= dj["median"] <= 1.6,
    f"中央値 {dj['median']:.3f}（p10 {dj['p10']:.3f} / p90 {dj['p90']:.3f}、駅から30分以内・直線2km以上のvalidセル n={dj['n']}）。目安は1.05〜1.6")

print("\n== 要約 =="); print("ALL PASS" if all(r[2] for r in res) else "FAIL あり: " + ",".join(str(r[0]) for r in res if not r[2]))
