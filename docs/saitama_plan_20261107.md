# 埼玉 土地利用シミュレーション — 実行計画（要約）
詳細＝会話の計画／ゴール＝`saitama_goal_20261107.md`／契約＝`saitama_schema.md`
- 構成：メイン（薄い）＋ A データ(haiku)・U UI(sonnet)・P 計算(sonnet)。第1波 A∥U → 第2波 P → 第3波 U を SendMessage で続行。
- 運用：返却は10〜15行／大きいデータは表示しない／検証はスクリプトのアサーション／失敗は2回自力再試行→診断を付けて返す／絶対パスで実行（cd を複合しない）。
- 既存資産：`src/honshu_core.py`（v2）, `src/build_real.py`（名寄せ・隣接）, `src/build_replay.py`（PRESETS=Xの11段階 L31〜、run_v2 は L173 のループ）。`data/honshu.npz` は単一X分のみ→11回再計算が必要。
- 環境：python3.14, numpy 2.5.2, pyproj 3.7.2 あり。**scipy 無し**（`pip install --user scipy` してよい）。Chromium は ~/.cache/ms-playwright にある。node あり。
- 禁止：EPSG:3857/緯度経度で面積・距離を測らない。`git push` 等の外部公開はしない（ゲート2）。取得は公開ページのGETのみ可。
