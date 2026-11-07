import json, numpy as np, sys
sys.path.insert(0,'./src')
import saitama_fields as sf, saitama_classify as sc, saitama_stations as sst
from scipy.sparse.csgraph import dijkstra
P=sf.pref(); cls=sf.road_raster(P); F=sc.load(); st=sst.main()
meta=json.load(open('./data/saitama/meta.json',encoding='utf-8'))
cidx=sf.center_idx(P); gx0,gy0,W,H=sf.fgrid(P)
ic=json.load(open('./data/saitama/ic.json',encoding='utf-8')); sel=[d for d in ic if d['kind'] in('IC','SIC')]
ix,iy=sf.TR.transform([d['lon'] for d in sel],[d['lat'] for d in sel]); icsrc=sf.src_cells(P,ix,iy)
sx,sy=st['S6677'][:,0],st['S6677'][:,1]
ii=np.nonzero((sx>=gx0)&(sx<gx0+W*100)&(sy>=gy0)&(sy<gy0+H*100))[0]
V=(F['valid']==1)&(F['water']==0)
def cov(tS,tI,k,Y,Z):
    nc=V&(F['core'][k]==0); n=nc.sum()
    return round(100*float((nc&(tS<=Y)).sum()/n),1), round(100*float((nc&(tI<=Z)).sum()/n),1)
def minsat(arr,k):
    nc=V&(F['core'][k]==0); n=nc.sum(); h=np.bincount(arr[nc],minlength=256).cumsum()/n
    t=np.nonzero(h>0.95)[0]; return int(t[0]) if len(t) else None
out={'bg_speed':{}, 'by_X':{}, 'saturation':{}, 'conflict':{}}
k=4; a=ii[st['alive'][k,ii]==1]
for bg in (6,12,20):
    sp=sf.preset_speeds(sf.PRESETS[0]); sp['none']=bg; G=sf.graph(cls,sp)
    tS=sf.to_u8(dijkstra(G,directed=True,indices=sf.src_cells(P,sx[a],sy[a]),min_only=True,limit=255.)[cidx])
    tI=sf.to_u8(dijkstra(G,directed=True,indices=icsrc,min_only=True,limit=255.)[cidx])
    out['bg_speed'][bg]={'farm_Y':{Y:cov(tS,tI,k,Y,10)[0] for Y in(30,60,90)},'ind_Z':{Z:cov(tS,tI,k,30,Z)[1] for Z in(10,15,30)}}
for k in range(11):
    c=sc.classify(F,k,60,15,0,'conflict'); f,i=cov(F['tS'][k,0],F['tI'][0],k,30,10)
    out['by_X'][str(meta['X'][k])]=dict(farm_Y30=f,ind_Z10=i,core_km2=round(float(F['core'][k].sum()*.25),1),
      stations_in=int(sum(1 for s in meta['stations'] if s['alive'][k])),
      **{f'km2_{n}':round(float((c==v).sum()*.25),1) for n,v in (('core',0),('industry',1),('farm',2),('conflict',3),('green',4),('water',5))})
for k in (0,4,10):
    for p in range(3):
        out['saturation'][f"{meta['X'][k]}|{meta['presets'][p]['id']}"]=dict(minY=minsat(F['tS'][k,p],k),minZ=minsat(F['tI'][p],k))
# 競合の場所（Y60/Z15・現実的・X=40）
k=4; c=sc.classify(F,k,60,15,0,'conflict'); nc=V&(F['core'][k]==0)
cx=(P['x0']+(np.arange(P['ncols'])+.5)*500); cy=(P['y0']+(np.arange(P['nrows'])+.5)*500)
X_,Y_=np.meshgrid(cx,cy); X_=X_.ravel(); Y_=Y_.ravel(); mx,my=X_[V].mean(),Y_[V].mean()
cf=c==3
q=lambda m:round(100*float(m.sum()/cf.sum()),1)
out['conflict']=dict(share_of_noncore=round(100*float(cf.sum()/nc.sum()),1),
  NE=q(cf&(X_>=mx)&(Y_>=my)),NW=q(cf&(X_<mx)&(Y_>=my)),SE=q(cf&(X_>=mx)&(Y_<my)),SW=q(cf&(X_<mx)&(Y_<my)),
  medianY_min=float(np.median(F['tS'][k,0][cf])),medianZ_min=float(np.median(F['tI'][0][cf])),
  noncore_nonconflict_medianY=float(np.median(F['tS'][k,0][nc&~cf])),
  west_of_139_0=None)
# 西半分(秩父側)の農地被覆
west=X_<mx-0; out['conflict']['west_half_share_of_land']=round(100*float((V&west).sum()/V.sum()),1)
out['conflict']['west_half_conflict_rate']=round(100*float((cf&west).sum()/(nc&west).sum()),1)
out['conflict']['east_half_conflict_rate']=round(100*float((cf&~west).sum()/(nc&~west).sum()),1)
json.dump(out,open('./data/saitama/sensitivity.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
print(json.dumps(out['conflict'],ensure_ascii=False)); print(out['saturation']); print(out['bg_speed'])
for x,v in out['by_X'].items(): print(x,v)
