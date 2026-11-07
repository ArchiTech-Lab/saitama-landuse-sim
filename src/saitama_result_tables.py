"""結果文書の表と sensitivity.json を作る（新仕様：Y15/30/45・Z5/10/15、工業=IC圏∧国道100m帯）。"""
import json, numpy as np, sys
sys.path.insert(0, './src')
import saitama_fields as sf, saitama_classify as sc, saitama_stations as sst
from scipy.sparse.csgraph import dijkstra
D = './'
P = sf.pref(); cls = sf.road_raster(P); F = sc.load(); st = sst.main()
meta = json.load(open(D + 'data/saitama/meta.json', encoding='utf-8'))
cidx = sf.center_idx(P); gx0, gy0, W, H = sf.fgrid(P)
ic = json.load(open(D + 'data/saitama/ic.json', encoding='utf-8')); sel = [d for d in ic if d['kind'] in ('IC', 'SIC')]
ix, iy = sf.TR.transform([d['lon'] for d in sel], [d['lat'] for d in sel]); icsrc = sf.src_cells(P, ix, iy)
sx, sy = st['S6677'][:, 0], st['S6677'][:, 1]
ii = np.nonzero((sx >= gx0) & (sx < gx0 + W * 100) & (sy >= gy0) & (sy < gy0 + H * 100))[0]
V = (F['valid'] == 1) & (F['water'] == 0); nh = F['nh100']
nin = [215, 201, 185, 181, 181, 168, 160, 135, 122, 80, 69]   # saitama_check 3 の県内生存駅数
def cov(tS, tI, k, Y, Z):
    nc = V & (F['core'][k] == 0); n = nc.sum()
    return round(100 * float((nc & (tS <= Y)).sum() / n), 1), round(100 * float((nc & (tI <= Z) & (nh > 0)).sum() / n), 1)
def ind_area(tI, k, Z):
    m = V & (F['core'][k] == 0) & (tI <= Z) & (nh > 0)
    return int(m.sum()), round(float(m.sum() * .25), 1), round(float(nh[m].sum() / 100 * .25), 1)
def minsat(arr, k):
    nc = V & (F['core'][k] == 0); n = nc.sum(); h = np.bincount(arr[nc], minlength=256).cumsum() / n
    t = np.nonzero(h > 0.95)[0]; return int(t[0]) if len(t) else None
out = {'bg_speed': {}, 'by_X': {}, 'saturation': {}, 'ind_Z': {}, 'ind_dist': {}}
k = 4; a = ii[st['alive'][k, ii] == 1]
for bg in (6, 12, 20):
    sp = sf.preset_speeds(sf.PRESETS[0]); sp['none'] = bg; G = sf.graph(cls, sp)
    tS = sf.to_u8(dijkstra(G, directed=True, indices=sf.src_cells(P, sx[a], sy[a]), min_only=True, limit=255.)[cidx])
    tI = sf.to_u8(dijkstra(G, directed=True, indices=icsrc, min_only=True, limit=255.)[cidx])
    out['bg_speed'][bg] = {'farm_Y': {Y: cov(tS, tI, k, Y, 10)[0] for Y in (5, 10, 15, 20, 30)},
                           'ind_Z': {Z: dict(zip(('cells', 'cell_km2', 'real_km2'), ind_area(tI, k, Z)), cov=cov(tS, tI, k, 30, Z)[1]) for Z in (5, 10, 15)}}
for k in range(11):
    c = sc.classify(F, k, 30, 10, 0)
    out['by_X'][str(meta['X'][k])] = dict(stations_in=nin[k], **{f'farm_Y{Y}': cov(F['tS'][k, 0], F['tI'][0], k, Y, 10)[0] for Y in (5, 10, 15, 20, 30)},
        ind_cov=cov(F['tS'][k, 0], F['tI'][0], k, 30, 10)[1], ind_real_km2=round(sc.ind_real_km2(F, c), 1),
        **{f'km2_{n}': round(float((c == v).sum() * .25), 1) for n, v in (('core', 0), ('industry', 1), ('farm', 2), ('green', 4), ('water', 5))})
for k in (0, 4, 10):
    for p in range(3):
        out['saturation'][f"{meta['X'][k]}|{meta['presets'][p]['id']}"] = dict(
            minY=minsat(F['tS'][k, p], k), **{f'cov_Y{Y}': cov(F['tS'][k, p], F['tI'][p], k, Y, 10)[0] for Y in (5, 10, 15, 20, 30)})
for p in range(3):
    out['ind_Z'][meta['presets'][p]['id']] = {Z: dict(zip(('cells', 'cell_km2', 'real_km2'), ind_area(F['tI'][p], 4, Z)), cov=cov(F['tS'][4, p], F['tI'][p], 4, 30, Z)[1]) for Z in (5, 10, 15)}
# 分布（X=40・現実的・Z10）
k = 4; nc = V & (F['core'][k] == 0); band = V & (nh > 0)
cx = (P['x0'] + (np.arange(P['ncols']) + .5) * 500); cy = (P['y0'] + (np.arange(P['nrows']) + .5) * 500)
X_, Y_ = np.meshgrid(cx, cy); X_ = X_.ravel(); Y_ = Y_.ravel(); mx, my = X_[V].mean(), Y_[V].mean()
ind = nc & (F['tI'][0] <= 10) & band
q = lambda m, base: round(100 * float(m.sum() / max(base.sum(), 1)), 1)
quad = {'NE': (X_ >= mx) & (Y_ >= my), 'NW': (X_ < mx) & (Y_ >= my), 'SE': (X_ >= mx) & (Y_ < my), 'SW': (X_ < mx) & (Y_ < my)}
out['ind_dist'] = dict(band_cells=int(band.sum()), band_real_km2=round(float(nh[band].sum() / 100 * .25), 1),
    ind_cells=int(ind.sum()), ind_real_km2=round(float(nh[ind].sum() / 100 * .25), 1),
    share_of_band_reached=q(ind, band & (F['core'][k] == 0)),
    quad_share_of_ind={n: q(ind & m, ind) for n, m in quad.items()},
    quad_land_share={n: q(V & m, V) for n, m in quad.items()},
    quad_reach_rate={n: q(ind & m, band & (F['core'][k] == 0) & m) for n, m in quad.items()},
    medianZ_ind_min=float(np.median(F['tI'][0][ind])), medianY_ind_min=float(np.median(F['tS'][k, 0][ind])),
    ind_in_core_km2=round(float((band & (F['core'][k] == 1)).sum() * .25), 1))
json.dump(out, open(D + 'data/saitama/sensitivity.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out['ind_dist'], ensure_ascii=False)); print(out['saturation']); print(out['bg_speed']); print(out['ind_Z'])
for x, v in out['by_X'].items(): print(x, v)
