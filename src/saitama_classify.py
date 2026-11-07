"""区分判定の Python 実装（docs/saitama_schema.md の仕様。JS の window.classify と同一）。
コード: 0 居住 / 1 工業 / 2 農地 / 3 競合 / 4 緑地・森林 / 5 区分外 / 255 県外(valid==0)"""
import os
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "saitama")

def load():
    f = np.load(os.path.join(D, "fields.npz"))
    return {k: f[k] for k in f.files}

def classify(F, x, Y, Z, p, prio):
    valid, water = F["valid"], F["water"]
    core = F["core"][x]
    farm = F["tS"][x, p] <= Y
    ind = F["tI"][p] <= Z
    c = np.where(water == 1, 5, 4).astype(np.uint8)
    cf = {"conflict": 3, "industry": 1, "farm": 2}[prio]
    c[ind] = 1
    c[farm] = 2
    c[farm & ind] = cf
    c[core == 1] = 0
    c[water == 1] = 5          # 区分外が最優先（core・閾値判定に勝つ）
    c[valid == 0] = 255
    return c

def coverage(F, x, Y, Z, p):
    """被覆率: 分母=valid・非water・非core。分子=非コアで tS<=Y（農地側）/ tI<=Z（工業側）。競合は両方に数える。"""
    nc = (F["valid"] == 1) & (F["water"] == 0) & (F["core"][x] == 0)   # 分母=valid かつ非water かつ非core
    n = nc.sum()
    return (nc & (F["tS"][x, p] <= Y)).sum() / n, (nc & (F["tI"][p] <= Z)).sum() / n
