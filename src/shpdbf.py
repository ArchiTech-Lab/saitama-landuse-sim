"""
Shapefile / DBF の最小リーダ（geopandas・fiona を使わない）

必要なのは 2つだけ:
  read_dbf(path)            → (列名リスト, 行の辞書のリスト)
  read_shp_points(path)     → 各フィーチャの代表点（Point はそのまま / PolyLine は重心）
仕様: ESRI Shapefile（shp はビッグエンディアンのヘッダ + リトルエンディアンのレコード）
"""
import struct
import numpy as np


def read_dbf_cols(path, cols, encoding="utf-8"):
    """必要な列だけを numpy で切り出す高速版。数値列は float、文字列列は U で返す。
    343列 × 数万行の DBF を1列ずつ Python ループで回すと遅いので、
    レコードを (nrec, rlen) の uint8 行列として見て、列のバイト範囲だけを取る。"""
    with open(path, "rb") as f:
        buf = f.read()
    nrec, hlen, rlen = struct.unpack_from("<IHH", buf, 4)
    fields, off, pos = {}, 32, 1                    # pos: レコード先頭の削除フラグ1バイトを飛ばす
    while buf[off] != 0x0D:
        name = buf[off:off+11].split(b"\x00")[0].decode("ascii", "replace")
        typ, ln = chr(buf[off+11]), buf[off+16]
        fields[name] = (typ, pos, ln); pos += ln; off += 32
    M = np.frombuffer(buf, np.uint8, count=nrec * rlen, offset=hlen).reshape(nrec, rlen)
    out = {}
    for c in cols:
        typ, p0, ln = fields[c]
        raw = M[:, p0:p0+ln].tobytes()
        arr = np.frombuffer(raw, dtype=f"S{ln}")
        if typ in "NF":
            t = np.char.strip(arr)
            t = np.where(t == b"", b"0", t)
            out[c] = t.astype(float)
        else:
            out[c] = np.char.strip(arr).astype(str)
    return out, nrec


def read_dbf(path, cols=None, encoding="cp932"):
    with open(path, "rb") as f:
        buf = f.read()
    nrec, hlen, rlen = struct.unpack_from("<IHH", buf, 4)
    fields, off = [], 32
    while buf[off] != 0x0D:
        name = buf[off:off+11].split(b"\x00")[0].decode("ascii", "replace")
        typ = chr(buf[off+11]); ln = buf[off+16]
        fields.append((name, typ, ln)); off += 32
    keep = set(cols) if cols else None
    out = {name: [] for name, _, _ in fields if keep is None or name in keep}
    for i in range(nrec):
        p = hlen + i * rlen + 1                      # 先頭1バイトは削除フラグ
        for name, typ, ln in fields:
            if keep is None or name in keep:
                raw = buf[p:p+ln]
                if typ in "NF":
                    t = raw.strip()
                    try: v = float(t) if t else 0.0
                    except ValueError: v = 0.0
                else:
                    v = raw.decode(encoding, "replace").strip()
                out[name].append(v)
            p += ln
    return [n for n, _, _ in fields], out, nrec


def read_shp_points(path):
    """代表点を返す。1=Point, 3/13/23=PolyLine, 5/15/25=Polygon（重心）"""
    with open(path, "rb") as f:
        buf = f.read()
    pos, pts = 100, []
    n = len(buf)
    while pos + 8 <= n:
        _, clen = struct.unpack_from(">II", buf, pos)
        rec = pos + 8
        shp = struct.unpack_from("<I", buf, rec)[0]
        if shp == 0:
            pts.append((np.nan, np.nan))
        elif shp == 1:
            x, y = struct.unpack_from("<dd", buf, rec + 4)
            pts.append((x, y))
        elif shp in (3, 5, 13, 15, 23, 25):
            # [type 4][box 32][numParts 4][numPoints 4][parts 4*n][points 16*m]
            nparts = struct.unpack_from("<I", buf, rec + 36)[0]
            npo    = struct.unpack_from("<I", buf, rec + 40)[0]
            base = rec + 44 + 4 * nparts
            xy = np.frombuffer(buf, "<f8", count=2 * npo, offset=base).reshape(-1, 2)
            pts.append((float(xy[:, 0].mean()), float(xy[:, 1].mean())))
        else:
            pts.append((np.nan, np.nan))
        pos = rec + clen * 2
    return np.array(pts, float)


def mesh500_rowcol(codes):
    """500m（4次）メッシュコード9桁 → 全国一意の整数 (row, col)。JIS X 0410
       row: 緯度方向 1/240度 刻み ／ col: 経度方向 1/160度 刻み（原点 lon=100度）"""
    c = np.asarray(codes, dtype=np.int64)
    p  = c // 100_000            # 1次メッシュ4桁
    q  = (c // 1_000) % 100      # 2次 2桁
    r  = (c // 10) % 100         # 3次 2桁
    s  = c % 10                  # 4次 1桁（1=南西 2=南東 3=北西 4=北東）
    py, px = p // 100, p % 100
    row = (((py * 8 + q // 10) * 10 + r // 10) * 2) + np.where((s == 3) | (s == 4), 1, 0)
    col = (((px * 8 + q % 10) * 10 + r % 10) * 2) + np.where((s == 2) | (s == 4), 1, 0)
    return row, col


def mesh500_center(row, col):
    """(row, col) → 中心の緯度経度（度）"""
    return (row + .5) / 240.0, 100.0 + (col + .5) / 160.0


def mesh500_area_km2(row):
    """緯度帯ごとの正確なセル面積（球面）。1/240度 × 1/160度"""
    R = 6371.0088
    la1 = np.radians(row / 240.0); la2 = np.radians((row + 1) / 240.0)
    return R * R * np.radians(1 / 160.0) * (np.sin(la2) - np.sin(la1))


def read_shp_polylines(path):
    """PolyLine の全頂点を返す: (xy[M,2], part_start[K], part_of_record[K])"""
    with open(path, "rb") as f:
        buf = f.read()
    pos, n = 100, len(buf)
    XY, starts, recid = [], [], []
    m = 0
    rid = 0
    while pos + 8 <= n:
        _, clen = struct.unpack_from(">II", buf, pos)
        rec = pos + 8
        shp = struct.unpack_from("<I", buf, rec)[0]
        if shp in (3, 5, 13, 15, 23, 25):
            nparts = struct.unpack_from("<I", buf, rec + 36)[0]
            npo = struct.unpack_from("<I", buf, rec + 40)[0]
            parts = np.frombuffer(buf, "<i4", count=nparts, offset=rec + 44)
            xy = np.frombuffer(buf, "<f8", count=2*npo, offset=rec + 44 + 4*nparts).reshape(-1, 2)
            XY.append(xy)
            for k in range(nparts):
                starts.append(m + int(parts[k])); recid.append(rid)
            m += npo
        pos = rec + clen * 2
        rid += 1
    return np.vstack(XY), np.array(starts), np.array(recid), m
