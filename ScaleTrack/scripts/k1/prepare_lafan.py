"""Resample verified morphology-retargeted LAFAN1 into common K1 pose schema."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert not (a.source/'INCOMPLETE').exists()
report=json.loads((a.source/'report.json').read_text())
paths=sorted(a.source.glob('*.npz'));assert len(paths)==77
poses=[];velocities=[];clips=[];offset=0;continuity=[]
for path in paths:
    z=np.load(path);q=z['joint_pos'];pos=z['root_pos'];fps=float(z['fps']);names=z['joint_names'].tolist()
    n=len(q);length=int(np.ceil((n-1)*50/fps));t=np.arange(length)/50;st=np.arange(n)/fps
    u=t*fps;i=np.floor(u).astype(int);w=(u-i)[:,None]
    rot=Slerp(st,Rotation.from_quat(z['root_rot'],scalar_first=True))(t)
    jp=q[i]*(1-w)+q[i+1]*w;rp=pos[i]*(1-w)+pos[i+1]*w
    av=(rot[2:]*rot[:-2].inv()).as_rotvec()/.04;av=np.concatenate([av[:1],av,av[-1:]])
    pose=np.concatenate([rp,rot.as_quat(scalar_first=True),jp],axis=-1)
    vel=np.concatenate([np.gradient(rp,.02,axis=0),av,np.gradient(jp,.02,axis=0)],axis=-1)
    assert np.isfinite(pose).all() and np.isfinite(vel).all()
    poses.append(pose.astype('float32'));velocities.append(vel.astype('float32'))
    clips.append(dict(name='lafan/'+path.stem,start=offset,length=length,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),split='eval' if path.stem.endswith('subject5') else 'train'))
    offset+=length
    original=np.load(Path(report['source_root'])/path.name)
    delta=np.diff(q-original['joint_pos'],axis=0)
    continuity.append(dict(name=path.stem,max_retarget_delta_per_frame_rad=float(np.abs(delta).max()),max_joint_speed_rad_s=float(np.abs(np.diff(q,axis=0)*fps).max()),source_max_joint_speed_rad_s=float(np.abs(original['joint_vel']).max())))
a.out.mkdir(parents=True,exist_ok=False)
np.save(a.out/'poses.npy',np.concatenate(poses));np.save(a.out/'velocities.npy',np.concatenate(velocities))
(a.out/'manifest.json').write_text(json.dumps(dict(fps=50,joint_names=names,clips=clips,frames=offset,source_report=str(a.source/'report.json'),split='subject5 held out')))
(a.out/'continuity.json').write_text(json.dumps(continuity,indent=2))
print('LAFAN_POSES_VALIDATED',len(clips),offset,flush=True)
