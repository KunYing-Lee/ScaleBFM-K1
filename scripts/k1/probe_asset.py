"""Instantiate the actual K1 USD and export simulator-order metadata and FK evidence."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
app=AppLauncher(a).app
import json
from pathlib import Path
import torch
import mujoco
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from scaletrack.robots.k1 import K1_CFG,K1_XML,K1_BODIES
from scaletrack.tasks.tracking.supermimic.kinematics import HumanoidKinematics
from scaletrack.tasks.tracking.supermimic.rotations import quat_angle_sq

@configclass
class Scene(InteractiveSceneCfg):
    robot=K1_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    ground=AssetBaseCfg(prim_path='/World/ground',spawn=sim_utils.GroundPlaneCfg())
sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(device=a.device,dt=.005))
scene=InteractiveScene(Scene(num_envs=32,env_spacing=2.5))
sim.reset()
r=scene['robot']
print('JOINTS',r.joint_names,flush=True)
print('BODIES',r.body_names,flush=True)
m=mujoco.MjModel.from_xml_path(K1_XML); d=mujoco.MjData(m)
fk=HumanoidKinematics(K1_XML,r.joint_names,K1_BODIES,a.device)
ids=[r.body_names.index(n) for n in K1_BODIES]
com=r.root_physx_view.get_coms().cpu().numpy()[0,ids,:3]
q=r.data.default_joint_pos.clone(); root=r.data.default_root_state.clone()
root[:,:3]+=scene.env_origins
r.write_root_state_to_sim(root);r.write_joint_state_to_sim(q,torch.zeros_like(q));scene.write_data_to_sim()
sim.forward();scene.update(0.005)
out=fk.forward(root[:,:3]-scene.env_origins,root[:,3:7],q)
pos_err=float((out['pos']-(r.data.body_pos_w[:,ids]-scene.env_origins[:,None])).norm(dim=-1).max())
rot_err=float(quat_angle_sq(out['quat'],r.data.body_quat_w[:,ids]).sqrt().max())
print('FK_ERRORS',pos_err,rot_err,flush=True)
assert pos_err<1e-4 and rot_err<1e-3
for _ in range(100):
    r.set_joint_position_target(q);scene.write_data_to_sim();sim.step();scene.update(.005)
assert torch.isfinite(r.data.root_state_w).all() and torch.isfinite(r.data.joint_pos).all()
meta=dict(joint_names=r.joint_names,body_names=K1_BODIES,com_pos_b=com.tolist(),mjcf=K1_XML,root_body='Trunk',fk_pos_error=pos_err,fk_rot_error=rot_err,
          native_body_names=r.body_names,nominal_joint_pos=q[0].tolist(),kp=r.data.joint_stiffness[0].tolist(),kd=r.data.joint_damping[0].tolist())
Path(a.output).write_text(json.dumps(meta,indent=2))
print('K1_ASSET_GPU_PHYSICS_PASS',flush=True)
app.close()
