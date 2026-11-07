"""埼玉格子 pref.npz を作る (EPSG:6677, 500m)。N03で県境、L03-b(河川地及び湖沼=1100)で水域。"""
import json, numpy as np, shapely
from shapely.geometry import shape
from shapely.ops import unary_union, transform
from pyproj import Transformer
D='./data/saitama'
tr=Transformer.from_crs('EPSG:6668','EPSG:6677',always_xy=True)
gj=json.load(open(f'{D}/raw/n03/N03-20250101_11.geojson'))
geoms=[shape(f['geometry']) for f in gj['features'] if f['properties']['N03_001']=='埼玉県']
u=unary_union([g.buffer(0) for g in geoms])
up=transform(lambda x,y,z=None:tr.transform(x,y),u)
minx,miny,maxx,maxy=up.bounds
C=500
x0=np.floor(minx/C)*C; y0=np.floor(miny/C)*C
ncols=int(np.ceil((maxx-x0)/C)); nrows=int(np.ceil((maxy-y0)/C))
cx=x0+(np.arange(ncols)+.5)*C; cy=y0+(np.arange(nrows)+.5)*C
X,Y=np.meshgrid(cx,cy)  # [row,col], row=south->north
valid=shapely.contains_xy(up,X.ravel(),Y.ravel()).astype(np.uint8)
# 水域: L03-b 100mサブメッシュ中心を6677へ→500mセルに集計
tot=np.zeros(ncols*nrows,np.int32); wat=np.zeros(ncols*nrows,np.int32)
for m in ['5338','5339','5438','5439']:
    b=np.fromfile(f'{D}/raw/l03b/{m}/L03-b-16_{m}.shp',dtype=np.uint8)[100:].reshape(-1,136)
    bb=b[:,12:44].copy().view('<f8').reshape(-1,4)
    dbf=open(f'{D}/raw/l03b/{m}/L03-b-16_{m}.dbf','rb').read()
    n=len(bb); rec=np.frombuffer(dbf[129:129+23*n],dtype=np.uint8).reshape(-1,23)
    code=(rec[:,11:15].astype(np.int32)-48)@np.array([1000,100,10,1])
    lon=(bb[:,0]+bb[:,2])/2; lat=(bb[:,1]+bb[:,3])/2
    px,py=tr.transform(lon,lat)
    col=np.floor((px-x0)/C).astype(int); row=np.floor((py-y0)/C).astype(int)
    ok=(col>=0)&(col<ncols)&(row>=0)&(row<nrows)
    idx=row[ok]*ncols+col[ok]
    np.add.at(tot,idx,1); np.add.at(wat,idx,(code[ok]==1100))
water=((wat*2>tot)&(tot>0)&(valid==1)).astype(np.uint8)
np.savez_compressed(f'{D}/pref.npz',x0=x0,y0=y0,ncols=ncols,nrows=nrows,valid=valid,water=water)
print(x0,y0,ncols,nrows,int(valid.sum()),int(water.sum()),valid.sum()*.25, ((valid==1)&(tot==0)).sum())
