# 埼玉プロジェクト 受け渡し契約（スキーマ）— 2026-11-07

ゴール＝`docs/saitama_goal_20261107.md`（v2）。本書はエージェント間の契約。食い違ったら本書が正。

## 格子
- CRS = **EPSG:6677**（JGD2011 平面直角IX系, m）。格子は 500m。原点＝県bbox（N03）を 500m に切り下げ/上げた左下。
- セル順 = 行優先（row=北→南 ではなく **row=南→北**、`idx = row*ncols + col`）。`x_c = x0 + (col+.5)*500`, `y_c = y0 + (row+.5)*500`。
- `valid[idx]` = セル中心が N03 の埼玉県内。`water[idx]` = valid かつ L03-b で河川・湖沼が過半（区分外）。
- 面積計算は valid のみ。1セル=0.25km²。

## ファイル（すべて `./data/saitama/`）
| ファイル | 作る人 | 内容 |
|---|---|---|
| `raw/` | A | 取得物。`manifest.json`に URL・取得日・件数 |
| `roads.geojson` | A | OSM 主要道（motorway,trunk,primary,secondary,tertiary + *_link）。属性 highway,name。bbox は県+30kmバッファ |
| `ic.json` | A | `[{name,lon,lat,kind:"IC"|"SIC"|"JCT"}]`（OSM motorway_junction。JCTは残して kind で区別） |
| `pref.npz` | A | 格子定義 x0,y0,ncols,nrows と `valid`(uint8), `water`(uint8)。N03・L03-b から作る（JGD2011→6677 は pyproj） |
| `meta.json` | P | 下記 |
| `fields.npz` | P | 下記 |

## meta.json（P が書く。U は読むだけ）
```
{ "grid": {"x0":..,"y0":..,"ncols":..,"nrows":..,"cell":500, "crs":"EPSG:6677"},
  "X": [10.0, 19.1, ...11個],  "Xlabel": ["…",..],
  "r": 1200,
  "presets": [{"id":"real","label":"現実的","speeds":{"motorway":80,"trunk":50,"primary":40,"secondary":30,"tertiary":22,"none":12}}, {"id":"congested",...}, {"id":"fast",...}],
  "stations": [{"name":..,"x":..,"y":..,"alive":[0/1 ×11], "core_m": 半径m}],   // 埼玉県内＋県境30km内の生存駅（県外の始点も含める）
  "ic": [{"name":..,"x":..,"y":..,"kind":..}],
  "sources": [...出典文字列], "limits": [...限界文字列], "built": "YYYY-MM-DD" }
```
## fields.npz（P が書く）
- `valid` uint8[N], `water` uint8[N]
- `tS` uint8[11, 3, N] : X段×速度プリセット×セル。**生存駅の中心から車の最短所要時間（分, 255で頭打ち）**
- `tI` uint8[3, N] : 速度プリセット×セル。**IC から一般道での最短所要時間（分）**
- `core` uint8[11, N] : セル中心が生存駅のコア円内なら 1（居住）
- `nh100` uint8[N] : 国道沿線100m帯（OSM国道の元の線を6677で100mバッファ）が500mセルを覆う面積割合(%)。工業の実面積＝工業セルのΣ nh100/100×0.25km²
- `retreat` uint8[11, N] : 撤退地域（生存駅に吸収されなかったセル）なら 1
- N = ncols*nrows。HTML埋め込みは base64（gzip 後）。

## 区分判定（Python と JS で**同一仕様**。U が JS、P が Python を実装し saitama_check.py で一致検証）
入力 Y(分), Z(分), x段, preset（prio 引数は後方互換のため残すが無視。UIには出さない）
```
if water: 区分外(5)（最優先・以降の判定に勝つ）
c = 緑地(4)
farm = tS[x,p,i] <= Y ;  ind = (tI[p,i] <= Z) and (nh100[i] > 0)   # 工業 = IC圏 かつ 国道100m帯がかかるセル
if core: 居住(0)
elif ind: 工業(1)          # farm∧ind も工業（工業＞農地で固定。競合区分は廃止）
elif farm: 農地(2) ; else 緑地(4)
(valid==0 は描画しない)
```
区分コード: 0 居住 / 1 工業 / 2 農地 / 3 欠番 / 4 緑地・森林 / 5 区分外
被覆率（到達圏の被覆率・居住を除く）= 分母は valid かつ非water かつ非core。分子は農地側＝非coreで tS<=Y、工業側＝非coreで tI<=Z かつ nh100>0。別々に出す。飽和バッジは農地側のみ。95%超で「飽和」。

## 出力物
- `src/build_saitama.py [--dummy]` → `./site-saitama/index.html`（単一HTML・外部通信ゼロ・≤3MB・幅375px対応）
- `--dummy` は fields/meta が無くても動く合成データ（小さな楕円の県・数個の駅）で UI を検証できる。
- `src/saitama_check.py` → PASS/FAIL 一覧（ゴール M1 の 1〜7 は P、8〜11 は U が担当部分を実装）。
