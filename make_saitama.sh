#!/bin/bash
# 埼玉プロジェクトの再現手順（取得→格子→生存駅→時間場→HTML→検証）。取得物は data/saitama/raw にキャッシュされ再取得しない。
set -eu
H="$(cd "$(dirname "$0")" && pwd)"
python3 "$H/src/build_saitama_osm.py"
python3 "$H/src/build_saitama_grid.py"
python3 "$H/src/saitama_stations.py"
python3 "$H/src/saitama_fields.py"
python3 "$H/src/saitama_layers.py"
python3 "$H/src/build_saitama.py"
python3 "$H/src/saitama_check.py"   # 5番のJS一致検証にヘッドレスChromium（playwright）が必要
