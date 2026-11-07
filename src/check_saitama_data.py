import json,numpy as np,shapely,sys
from shapely.geometry import shape
from shapely.ops import unary_union,transform
from pyproj import Transformer
D='./data/saitama'
z=np.load(f'{D}/pref.npz'); v=z['valid'];w=z['water']
area=v.sum()*.25
assert abs(area/3797-1)<.02,area
assert z['x0']%500==0 and v.size==int(z['ncols'])*int(z['nrows']) and w[v==0].sum()==0
assert -200000<z['x0']<0 and -100000<z['y0']<0   # 6677 IX系 埼玉の妥当範囲
ic=json.load(open(f'{D}/ic.json'))
tr=Transformer.from_crs('EPSG:6668','EPSG:6677',always_xy=True)
gj=json.load(open(f'{D}/raw/n03/N03-20250101_11.geojson'))
u=transform(lambda x,y,z=None:tr.transform(x,y),unary_union([shape(f['geometry']).buffer(0) for f in gj['features']]))
xy=np.array([tr.transform(o['lon'],o['lat']) for o in ic])
inside=shapely.contains_xy(u,xy[:,0],xy[:,1])
dist=np.array([u.distance(shapely.Point(*p)) for p in xy])
n_in=int(inside.sum());n_buf=int((dist<=30000).sum())
assert 20<=n_in<=200 and n_buf>=n_in
roads=json.load(open(f'{D}/roads.geojson'))['features']
nm=sum(f['properties']['highway']=='motorway' for f in roads)
assert nm>100
c=np.array(roads[0]['geometry']['coordinates'][0]);assert 138<c[0]<141 and 35<c[1]<37
print('PASS area',area,'IC in/buf30',n_in,n_buf,'motorway',nm,'water km2',w.sum()*.25)
