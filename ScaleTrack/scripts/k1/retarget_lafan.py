"""Exact fixed-orientation root translation retargeting for congruent leg chains.

Preserve EVERY joint angle and root orientation. Solve the least-squares root
translation from both foot positions, then independently verify both full foot
poses. This is exact only when the two leg chains differ by a common root-frame
translation; otherwise fail rather than altering joints or relaxing tolerance.
No filtering, clipping, floor correction, or discarded frames.
"""
import argparse,hashlib,json,multiprocessing as mp
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from audit_lafan import NAMES

def retarget(item):
    path,xml,out=item;path=Path(path);out=Path(out)
    model=mujoco.MjModel.from_xml_path(xml);data=mujoco.MjData(model)
    qadr=np.array([model.jnt_qposadr[model.joint(n).id] for n in NAMES])
    bids=[model.body(n).id for n in ['left_foot_link','right_foot_link']]
    refs=[NAMES.index(n)+1 for n in ['Left_Ankle_Roll','Right_Ankle_Roll']]
    z=np.load(path);joint=z['joint_pos'].astype('float64');pos=z['body_pos_w'][:,0].astype('float64');quat=z['body_quat_w'][:,0].astype('float64')
    roots=pos.copy();worst_p=worst_r=0.
    for t in range(len(joint)):
        data.qpos[:3]=pos[t];data.qpos[3:7]=quat[t];data.qpos[qadr]=joint[t];mujoco.mj_kinematics(model,data)
        target=z['body_pos_w'][t,refs].astype('float64')
        # Unique least-squares translation with all other generalized coordinates fixed.
        roots[t]=pos[t]+np.mean(target-data.xpos[bids],axis=0)
        data.qpos[:3]=roots[t];mujoco.mj_kinematics(model,data)
        pe=float(np.linalg.norm(data.xpos[bids]-target,axis=1).max())
        rot=Rotation.from_matrix(data.xmat[bids].reshape(-1,3,3))
        re=float((rot*Rotation.from_quat(z['body_quat_w'][t,refs],scalar_first=True).inv()).magnitude().max())
        worst_p=max(worst_p,pe);worst_r=max(worst_r,re)
    assert worst_p<2e-5 and worst_r<2e-5,(path,worst_p,worst_r)
    info=dict(name=path.stem,frames=len(joint),fps=float(z['fps'][0]),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),max_foot_position_error_m=worst_p,max_foot_rotation_error_rad=worst_r,max_joint_change_rad=0.,max_root_change_m=float(np.linalg.norm(roots-pos,axis=1).max()),failed_leg_frames=0,method='fixed joint angles and root orientation; exact two-foot translation least squares')
    np.savez(out/path.name,root_pos=roots,root_rot=quat,joint_pos=joint,fps=info['fps'],joint_names=np.array(NAMES))
    return info

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--xml',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    paths=sorted(a.source.glob('*.npz'));assert len(paths)==77
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'INCOMPLETE').touch()
    with mp.Pool(a.workers) as pool: records=list(pool.imap_unordered(retarget,[(str(path),str(a.xml),str(a.out)) for path in paths]))
    (a.out/'report.json').write_text(json.dumps(dict(source_root=str(a.source),target_xml_sha256=hashlib.sha256(a.xml.read_bytes()).hexdigest(),records=records),indent=2))
    (a.out/'INCOMPLETE').unlink();print('LAFAN_RETARGET_PASS',len(records),max(x['max_foot_position_error_m'] for x in records),flush=True)
