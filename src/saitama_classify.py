"""区分判定の Python 実装（docs/saitama_schema.md の仕様。JS の window.classify と同一）。
コード: 0 居住 / 1 工業 / 2 農地 / 3 競合 / 4 緑地・森林 / 5 区分外 / 255 県外(valid==0)"""
import os
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "saitama")

def load():
    f = np.load(os.path.join(D, "fields.npz"))
    return {k: f[k] for k in f.files}

def classify(F, x, Y, Z, p, prio=None):
    """prio は後方互換のため受けるが無視（農地∧工業は常に工業）。工業 = tI<=Z かつ 国道100m帯がかかる(nh100>0)。"""
    valid, water = F["valid"], F["water"]
    core = F["core"][x]
    farm = F["tS"][x, p] <= Y
    ind = (F["tI"][p] <= Z) & (F["nh100"] > 0)
    c = np.full(len(valid), 4, np.uint8)
    c[farm] = 2
    c[ind] = 1
    c[core == 1] = 0
    c[water == 1] = 5          # 区分外が最優先（core・閾値判定に勝つ）
    c[valid == 0] = 255
    return c

def coverage(F, x, Y, Z, p):
    """到達圏の被覆率（居住を除く）。分母=valid・非water・非core。農地側=tS<=Y、工業側=tI<=Z かつ nh100>0。"""
    nc = (F["valid"] == 1) & (F["water"] == 0) & (F["core"][x] == 0)
    n = nc.sum()
    return (nc & (F["tS"][x, p] <= Y)).sum() / n, (nc & (F["tI"][p] <= Z) & (F["nh100"] > 0)).sum() / n

def ind_real_km2(F, c):
    """工業の実面積km² = 工業セルの Σ nh100/100 × 0.25"""
    return float(F["nh100"][c == 1].sum() / 100 * 0.25)
