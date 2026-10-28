"""Validate and pack the user's immutable SOMA K1-final CSVs at 50 Hz.

No filtering, lifting, clipping or smoothing. Mirrored variants share a split.
All source files receive a hash. Only unit conversion and linear/SLERP
resampling are performed; velocities use the upstream central differences.
"""
import argparse,csv,hashlib,io,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

ROOT=None; POSES=None; VELOCITIES=None; JOINTS=None

def initialize(root,out,joints):
    global ROOT,POSES,VELOCITIES,JOINTS
    ROOT=Path(root);JOINTS=joints
    POSES=np.load(Path(out)/'poses.npy',mmap_mode='r+')
    VELOCITIES=np.load(Path(out)/'velocities.npy',mmap_mode='r+')

def convert(item):
    rel,n,start,length=item
    raw=(ROOT/rel).read_bytes()
    header=raw.split(b'\n',1)[0].decode().strip().split(',')
    expected=['Frame','root_translateX','root_translateY','root_translateZ','root_rotateX','root_rotateY','root_rotateZ']+JOINTS
    if header!=expected: raise ValueError(f'{rel}: CSV schema differs from robot')
    a=np.loadtxt(io.BytesIO(raw),delimiter=',',skiprows=1)
    if a.shape!=(n,29) or not np.isfinite(a).all() or not np.array_equal(a[:,0],np.arange(n)):
        raise ValueError(f'{rel}: invalid shape, frame sequence or nonfinite data')
    times=np.arange(length,dtype=np.float64)/50
    source_times=np.arange(n,dtype=np.float64)/120
    u=times*120;i=np.floor(u).astype(np.int64);w=(u-i)[:,None]
    # Source uses extrinsic xyz, verified against soma_retargeter/io/csv.py.
    q=Slerp(source_times,Rotation.from_euler('xyz',a[:,4:7],degrees=True))(times)
    pos=(a[i,1:4]*(1-w)+a[i+1,1:4]*w)*.01
    joint=np.deg2rad(a[i,7:]*(1-w)+a[i+1,7:]*w)
    quat=q.as_quat(scalar_first=True)
    lin=np.gradient(pos,.02,axis=0);qd=np.gradient(joint,.02,axis=0)
    ang=(q[2:]*q[:-2].inv()).as_rotvec()/.04
    ang=np.concatenate([ang[:1],ang,ang[-1:]])
    POSES[start:start+length]=np.concatenate([pos,quat,joint],axis=-1).astype(np.float32)
    VELOCITIES[start:start+length]=np.concatenate([lin,ang,qd],axis=-1).astype(np.float32)
    return dict(name='bones/'+rel[:-4],source=str(ROOT/rel),source_frames=n,length=length,start=start,sha256=hashlib.sha256(raw).hexdigest())

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--qc',type=Path,required=True);p.add_argument('--xml',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=24);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    import xml.etree.ElementTree as ET
    joints=[j.attrib['name'] for j in ET.parse(a.xml).findall('.//joint') if j.attrib.get('type','hinge')=='hinge']
    rows=list(csv.DictReader(a.qc.open())); counts={f"{r['date']}/{r['motion']}.csv":int(r['frames']) for r in rows}
    paths=sorted(str(x.relative_to(a.source)) for x in a.source.rglob('*.csv'))
    if len(paths)!=142220 or len(set(paths))!=len(paths) or not set(paths)<=counts.keys():
        raise ValueError('Full K1-final inventory differs from documented 142220 clips / QC')
    if a.limit: paths=paths[:a.limit]
    items=[];offset=0
    for rel in paths:
        n=counts[rel];length=((n-1)*50+119)//120
        if length<3: raise ValueError(f'{rel}: fewer than 3 control frames')
        items.append((rel,n,offset,length));offset+=length
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'INCOMPLETE').touch()
    for name,dim in [('poses',29),('velocities',28)]:
        v=np.lib.format.open_memmap(a.out/f'{name}.npy',mode='w+',dtype='float32',shape=(offset,dim));v.flush();del v
    result=[]
    with mp.Pool(a.workers,initializer=initialize,initargs=(str(a.source),str(a.out),joints)) as pool:
        for r in pool.imap(convert,items,chunksize=32):
            group=r['name'].removesuffix('_M')
            r['split']='eval' if int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%100==0 else 'train'
            result.append(r)
            if len(result)%1000==0: print(json.dumps(dict(done=len(result),total=len(items))),flush=True)
    meta=dict(fps=50,joint_names=joints,clips=result,frames=offset,source_root=str(a.source.resolve()),source_fps=120,source_xml_sha256=hashlib.sha256(a.xml.read_bytes()).hexdigest(),split='sha256(date/name without mirror suffix) mod 100 == 0 held out',transforms=['cm to m','degrees to radians','extrinsic xyz to wxyz','linear and SLERP to 50 Hz'])
    (a.out/'manifest.json').write_text(json.dumps(meta));(a.out/'INCOMPLETE').unlink()
    print('BONES_POSES_VALIDATED',len(result),offset,flush=True)
if __name__=='__main__':main()
