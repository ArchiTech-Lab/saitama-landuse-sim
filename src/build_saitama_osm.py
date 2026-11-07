import json,glob,datetime,os,math
D='./data/saitama'
feats={}
for p in sorted(glob.glob(f'{D}/raw/osm/roads_*.json')):
    for e in json.load(open(p))['elements']:
        if e.get('type')!='way' or 'geometry' not in e or e['id'] in feats: continue
        t=e.get('tags',{})
        feats[e['id']]={'type':'Feature','properties':{'highway':t.get('highway'),'name':t.get('name') or t.get('ref')},
          'geometry':{'type':'LineString','coordinates':[[round(g['lon'],6),round(g['lat'],6)] for g in e['geometry']]}}
fl=list(feats.values())
json.dump({'type':'FeatureCollection','features':fl},open(f'{D}/roads.geojson','w'),ensure_ascii=False,separators=(',',':'))
nodes=json.load(open(f'{D}/raw/osm/junction.json'))['elements']
def kind(t):
    s=' '.join(str(t.get(k,'')) for k in ('name','name:ja','name:en','ref','junction','official_name','description'))
    if 'ジャンクション' in s or 'JCT' in s.upper() or 'Junction' in s: return 'JCT'
    if 'スマート' in s or 'SIC' in s.upper() or 'Smart' in s: return 'SIC'
    return 'IC'
out=[]
for n in nodes:
    t=n.get('tags',{}); nm=t.get('name') or t.get('name:ja') or t.get('ref') or ''
    k=kind(t); lon,lat=n['lon'],n['lat']
    for o in out:  # 同名で1.5km以内は統合 (上下線・ランプ別ノード)
        if o['name']==nm and nm and math.hypot((o['lon']-lon)*91,(o['lat']-lat)*111)<1.5: break
    else: out.append({'name':nm,'lon':round(lon,6),'lat':round(lat,6),'kind':k})
json.dump(out,open(f'{D}/ic.json','w'),ensure_ascii=False)
from collections import Counter
print(len(fl),Counter(f['properties']['highway'] for f in fl),len(nodes),len(out),Counter(o['kind'] for o in out),sum(1 for o in out if not o['name']))
man={'built':str(datetime.date.today()),
 'sources':[
  {'name':'N03 行政区域 2025-01-01 埼玉県','url':'https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-2025/N03-20250101_11_GML.zip','fetched':'2026-11-07','count':84},
  {'name':'L03-b-16 土地利用細分メッシュ(100m) 1次メッシュ5338,5339,5438,5439','url':'https://nlftp.mlit.go.jp/ksj/gml/data/L03-b/L03-b-16/L03-b-16_{mesh}-jgd_GML.zip','fetched':'2026-11-07','count':'4 files; water code=1100 河川地及び湖沼'},
  {'name':'OSM Overpass roads (bbox 138.36,35.48,140.25,36.55; 3x2 tiles)','url':'https://overpass-api.de/api/interpreter','fetched':'2026-11-07','count':len(fl)},
  {'name':'OSM Overpass motorway_junction','url':'https://overpass-api.de/api/interpreter','fetched':'2026-11-07','count':len(nodes)}],
 'notes':['water: L03-b(2016年版, 最新21年版はURL不明のため16を使用)の1100を採用。OSM代替は不使用。','ic: 同名1.5km以内を統合']}
json.dump(man,open(f'{D}/raw/manifest.json','w'),ensure_ascii=False,indent=1)
