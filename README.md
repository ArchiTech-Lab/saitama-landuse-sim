# 埼玉県 土地利用シミュレーション

各Xで残った駅勢圏（コンパクトシティ極限モデル）をもとに、駅から車でY分以内を農地、ICから一般道でZ分以内を工業地帯、それ以外を緑地・森林として塗り分け、X・Y・Zを動かして比較する単一HTML。

- デモ：https://architech-lab.github.io/saitama-landuse-sim/
- 結果と読み方：`docs/saitama_result_20261107.md`
- 前提モデル：https://github.com/ArchiTech-Lab/honshu-station-density

## 読むときの注意
- 農地・工業は**到達圏のみ**を塗る。標高・傾斜・現況の森林/市街は考慮しない。
- 車速は仮定値（高速道路は使わない・渋滞なし）。道路のない土地の速度の仮定に結果が左右される。
- 撤退地域はアルゴリズムの構成上ゼロ。食料自給率は算出していない。

## 再現
`make_saitama.sh`（取得→格子→生存駅→時間場→HTML→検証）。元データは国土数値情報（N02・N03・L03-b）と OpenStreetMap。取得物は再配布しない。

## 出典・ライセンス
コードは MIT。道路・IC のデータは © OpenStreetMap contributors（ODbL）。行政区域・鉄道・土地利用・人口メッシュは国土数値情報・国勢調査（e-Stat）。
