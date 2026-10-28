import argparse,hashlib,io,json,tarfile,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
NAMES=['AAHead_yaw','ALeft_Shoulder_Pitch','ARight_Shoulder_Pitch','Left_Hip_Pitch','Right_Hip_Pitch','Head_pitch','Left_Shoulder_Roll','Right_Shoulder_Roll','Left_Hip_Roll','Right_Hip_Roll','Left_Elbow_Pitch','Right_Elbow_Pitch','Left_Hip_Yaw','Right_Hip_Yaw','Left_Elbow_Yaw','Right_Elbow_Yaw','Left_Knee_Pitch','Right_Knee_Pitch','Left_Ankle_Pitch','Right_Ankle_Pitch','Left_Ankle_Roll','Right_Ankle_Roll']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--xml',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 digest=hashlib.sha256(a.archive.read_bytes()).hexdigest()
 assert digest=='92fafe9e6372bf08b26061cc7ed4356f9f5c4bbffce646a86a0574a331112e42'
 tree=ET.parse(a.xml);tree.find('compiler').set('meshdir',str(a.xml.resolve().parent/'meshes'))
 for side in ['Left','Right']:
  n=tree.find(f'.//body[@name="{side}_Hip_Pitch"]');v=n.get('pos').split();v[2]='-0.062';n.set('pos',' '.join(v))
 source=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'));target=mujoco.MjModel.from_xml_path(str(a.xml))
 m=source;d=mujoco.MjData(m);j=np.array([m.joint(n).id for n in NAMES]);qadr=m.jnt_qposadr[j];b=np.r_[m.body('Trunk').id,m.jnt_bodyid[j]]
 records=[];dest=a.out/'source';dest.mkdir(parents=True,exist_ok=False)
 with tarfile.open(a.archive) as t:
  for member in t:
   if not member.isfile() or not member.name.endswith('.npz'):continue
   raw=t.extractfile(member).read(); z=np.load(io.BytesIO(raw),allow_pickle=False);q=z['joint_pos']; pos=z['body_pos_w'];quat=z['body_quat_w']
   assert q.shape==(len(q),22) and pos.shape==(len(q),23,3) and quat.shape==(len(q),23,4)
   assert all(np.isfinite(z[k]).all() for k in z.files)
   maxpos=maxrot=0.
   for i in np.linspace(0,len(q)-1,min(len(q),128)).astype(int):
    d.qpos[:3]=pos[i,0];d.qpos[3:7]=quat[i,0];d.qpos[qadr]=q[i];mujoco.mj_forward(m,d)
    maxpos=max(maxpos,float(np.linalg.norm(d.xpos[b]-pos[i],axis=1).max()))
    maxrot=max(maxrot,float((1-np.abs((d.xquat[b]*quat[i]).sum(-1))).max()))
   (dest/Path(member.name).name).write_bytes(raw)
   rec=dict(name=Path(member.name).stem,frames=len(q),fps=float(z['fps'][0]),source_fk_pos_max=maxpos,source_fk_rot_max=maxrot,
            joint_limit_excess=float(np.maximum(m.jnt_range[j,0]-q,q-m.jnt_range[j,1]).max()),sha256=hashlib.sha256(raw).hexdigest())
   records.append(rec);print(json.dumps(rec),flush=True)
 assert len(records)==77
 report=dict(archive_sha256=digest,records=records,source_joint_names=NAMES,source_hip_z=-.062,target_hip_z=-.077)
 (a.out/'audit.json').write_text(json.dumps(report,indent=2))
 assert max(r['source_fk_pos_max'] for r in records)<1e-4
 assert max(r['source_fk_rot_max'] for r in records)<1e-5
 print('LAFAN_SOURCE_FK_PASS',flush=True)
