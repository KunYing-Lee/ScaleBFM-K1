"""Execute the portable K1 BFM ONNX in MuJoCo, with measured root localization.

Reference state initialization is applied once. Subsequent motion is produced
only by the policy and bounded PD torques. There is no reference root forcing,
action filtering, recovery controller, or pose correction.
"""
import argparse,hashlib,json,os,xml.etree.ElementTree as ET
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco,numpy as np,onnxruntime as ort
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--xml',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--motion',type=int,default=0);p.add_argument('--steps',type=int,default=500);p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args()
meta=json.loads(a.metadata.read_text());index=json.loads((a.library/'index.json').read_text())
assert meta['joint_names']==index['joint_names'] and meta['selected_body_names']==index['body_names']
start=sum(index['lengths'][:a.motion]);length=index['lengths'][a.motion]
ref={k:np.load(a.library/f'{k}.npy',mmap_mode='r')[start:start+length] for k in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']}
xml=ET.parse(a.xml);root=xml.getroot();root.find('compiler').set('meshdir',str(a.xml.resolve().parent/'meshes'))
root.find('option').set('timestep','0.005')
ET.SubElement(root.find('worldbody'),'geom',name='evaluation_ground',type='plane',size='0 0 .1',friction='1 .005 .0001',rgba='.25 .3 .35 1')
act=ET.SubElement(root,'actuator')
for name,limit in zip(meta['joint_names'],meta['torque_limit']):
    ET.SubElement(act,'motor',name=name,joint=name,ctrlrange=f'{-limit} {limit}',ctrllimited='true')
model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
jids=np.array([model.joint(n).id for n in meta['joint_names']]);qadr=model.jnt_qposadr[jids];vadr=model.jnt_dofadr[jids];bids=[model.body(n).id for n in meta['selected_body_names']]
data.qpos[:3]=ref['body_pos_w'][0,0];data.qpos[3:7]=ref['body_quat_w'][0,0];data.qpos[qadr]=ref['joint_pos'][0];data.qvel[vadr]=ref['joint_vel'][0]
r=Rotation.from_quat(data.qpos[3:7],scalar_first=True);omega=ref['body_ang_vel_w'][0,0]
data.qvel[:3]=ref['body_lin_vel_w'][0,0]-np.cross(omega,r.apply(model.body_ipos[bids[0]]));data.qvel[3:6]=r.inv().apply(omega)
mujoco.mj_forward(model,data)
session=ort.InferenceSession(str(a.model),providers=['CPUExecutionProvider']);session_inputs=session.get_inputs()
history=int(meta['history_buffer_size']);future=np.arange(6);action=np.zeros(22,dtype='float32')
buffers=[np.repeat(v[None,None],history,axis=1).astype('float32') for v in [data.qpos[3:7],data.qvel[3:6],data.qpos[qadr],data.qvel[vadr],action]]
a.out.mkdir(parents=True,exist_ok=True);positions=[];errors=[];targets=[];actions=[]
if a.video:
    import imageio.v2 as imageio
    renderer=mujoco.Renderer(model,height=480,width=640);camera=mujoco.MjvCamera();camera.distance=2.2;camera.azimuth=130;camera.elevation=-15
    writer=imageio.get_writer(a.out/'rollout.mp4',fps=50)
for frame in range(min(a.steps,length-6)):
    state=[data.qpos[3:7],data.qvel[3:6],data.qpos[qadr],data.qvel[vadr],action]
    for i,v in enumerate(state):buffers[i]=np.concatenate([buffers[i][:,1:],v[None,None].astype('float32')],axis=1)
    r=Rotation.from_quat(data.qpos[3:7],scalar_first=True);ids=frame+future
    target_pos=r.inv().apply((ref['body_pos_w'][ids]-data.qpos[:3]).reshape(-1,3)).reshape(1,6,-1,3).astype('float32')
    target_rot=(r.inv()*Rotation.from_quat(ref['body_quat_w'][ids].reshape(-1,4),scalar_first=True)).as_quat(scalar_first=True).reshape(1,6,-1,4).astype('float32')
    inputs=buffers+[target_pos,target_rot,np.array([7],dtype='int64'),future[None,:,None].astype('int64')]
    target,raw=session.run(None,{s.name:v for s,v in zip(session_inputs,inputs,strict=True)})
    assert np.isfinite(target).all() and np.isfinite(raw).all()
    action=raw[0];targets.append(target[0]);actions.append(action.copy())
    for _ in range(4):
        data.ctrl[:]=(target[0]-data.qpos[qadr])*meta['stiffness']-data.qvel[vadr]*meta['damping'];mujoco.mj_step(model,data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    positions.append(data.qpos[:3].copy());errors.append(float(np.linalg.norm(data.xpos[bids]-ref['body_pos_w'][frame+1],axis=1).mean()))
    if a.video:
        camera.lookat[:]=data.qpos[:3];renderer.update_scene(data,camera=camera);writer.append_data(renderer.render())
if a.video:writer.close();renderer.close()
np.savez(a.out/'rollout.npz',root_pos=positions,body_error=errors,joint_target=targets,action=actions)
report=dict(pipeline_pass=True,quality_pass=max(errors)<.5,mean_body_error_m=float(np.mean(errors)),max_body_error_m=max(errors),frames=len(errors),motion=index['motion_names'][a.motion],model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),reference_root_forcing=False,physics_dt=.005,control_dt=.02)
(a.out/'report.json').write_text(json.dumps(report,indent=2));print('SIM2SIM_PIPELINE_PASS',json.dumps(report))
