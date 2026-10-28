"""Batched exact FK packaging into the upstream packed-library schema.

Run once with --initialize, then with torchrun --nproc_per_node=8. Writes
nonoverlapping ranges. A COMPLETE marker is published only after all ranks
validate finite FK outputs. Memory mapping avoids loading the full corpus.
"""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--poses',type=Path,required=True);p.add_argument('--robot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--initialize',action='store_true');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
meta=json.loads((a.poses/'manifest.json').read_text());robot=json.loads(a.robot.read_text())
assert not (a.poses/'INCOMPLETE').exists()
clips=meta['clips'][:a.limit] if a.limit else meta['clips'];j=len(robot['joint_names']);b=len(robot['body_names'])
shapes={'joint_pos':(j,),'joint_vel':(j,),'body_pos_w':(b,3),'body_quat_w':(b,4),'body_lin_vel_w':(b,3),'body_ang_vel_w':(b,3)}
if a.initialize:
    a.out.mkdir(parents=True,exist_ok=False)
    for split in ['train','eval']:
        selected=[c for c in clips if c['split']==split]
        if not selected: continue
        out=a.out/split;out.mkdir();(out/'INCOMPLETE').touch()
        frames=sum(c['length'] for c in selected)
        index=dict(fps=50,motion_names=[c['name'] for c in selected],lengths=[c['length'] for c in selected],joint_names=robot['joint_names'],body_names=robot['body_names'],source_starts=[c['start'] for c in selected],source_manifest=str(a.poses/'manifest.json'))
        (out/'index.json').write_text(json.dumps(index))
        for field,shape in shapes.items():
            arr=np.lib.format.open_memmap(out/f'{field}.npy',mode='w+',dtype='float32',shape=(frames,*shape));arr.flush();del arr
        print('INITIALIZED',split,len(selected),frames,flush=True)
    raise SystemExit
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'source/scaletrack/test'))
from sm_offline import load
rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));device=f'cuda:{int(os.environ.get("LOCAL_RANK","0"))}'
torch.cuda.set_device(device);torch.set_num_threads(1)
if world>1:dist.init_process_group('nccl')
fk=load('kinematics').HumanoidKinematics(robot['mjcf'],robot['joint_names'],robot['body_names'],device)
rot=load('rotations');order=[meta['joint_names'].index(n) for n in robot['joint_names']]
pose=np.load(a.poses/'poses.npy',mmap_mode='r');vel=np.load(a.poses/'velocities.npy',mmap_mode='r')
com=torch.tensor(robot['com_pos_b'],dtype=torch.float32,device=device)
for split in ['train','eval']:
    out=a.out/split
    if not out.exists():continue
    index=json.loads((out/'index.json').read_text());starts=np.r_[0,np.cumsum(index['lengths'])];source=np.array(index['source_starts'])
    arrays={field:np.load(out/f'{field}.npy',mmap_mode='r+') for field in shapes}
    begin=int(starts[-1])*rank//world;end=int(starts[-1])*(rank+1)//world
    for lo in range(begin,end,32768):
        hi=min(lo+32768,end);frames=np.arange(lo,hi);motion=np.searchsorted(starts[1:],frames,side='right');rows=source[motion]+frames-starts[motion]
        q=torch.from_numpy(np.array(pose[rows])).to(device);v=torch.from_numpy(np.array(vel[rows])).to(device)
        jp=q[:,7:][:,order];jv=v[:,6:][:,order]
        with torch.no_grad():
            result=fk.forward(q[:,:3],q[:,3:7],jp,v[:,:3],rot.quat_apply_inverse(q[:,3:7],v[:,3:6]),jv,com.expand(len(q),-1,-1))
        fields=dict(joint_pos=jp,joint_vel=jv,body_pos_w=result['pos'],body_quat_w=result['quat'],body_lin_vel_w=result['com_lin_vel'],body_ang_vel_w=result['ang_vel'])
        for key,value in fields.items():
            if not torch.isfinite(value).all():raise ValueError(f'{split} {key}: nonfinite FK at {lo}')
            arrays[key][lo:hi]=value.cpu().numpy()
    for arr in arrays.values():arr.flush()
    print('RANK_PACK_PASS',rank,split,begin,end,flush=True)
    if world>1:dist.barrier()
    if rank==0:
        (out/'INCOMPLETE').unlink();(out/'COMPLETE').write_text('All source poses validated; all packed FK fields finite.\n')
if world>1:dist.destroy_process_group()
