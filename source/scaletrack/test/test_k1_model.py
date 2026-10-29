"""Independent MuJoCo checks for arbitrary-joint states and fixed-frame FK."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import mujoco
sys.path.insert(0,str(Path(__file__).parent))
from sm_offline import load,ASSET_XML

@pytest.mark.parametrize('nj',[22,29])
def test_state_integration_and_derivatives(nj):
    state=load('state');rot=load('rotations')
    spec=state.StateSpec(state.StateSpecCfg(num_joints=nj),'cpu',dtype=torch.float64)
    torch.manual_seed(nj)
    s=torch.randn(8,13+nj*2,dtype=torch.float64);s[:,3:7]=rot.quat_normalize(s[:,3:7]);s.requires_grad_()
    zero=torch.zeros(8,12+nj*2,dtype=torch.float64)
    out=spec.integrate(s,zero)
    p,q,j,v,w,jv=state.split_state(s)
    torch.testing.assert_close(out[:,:3],p+.02*v)
    torch.testing.assert_close(state.split_state(out)[2],j+.02*jv)
    assert spec.features(s).shape==(8,13+2*nj)
    torch.testing.assert_close(spec.error(s,s),torch.zeros_like(zero),atol=1e-12,rtol=0)
    assert torch.isfinite(torch.autograd.grad(out.square().sum(),s)[0]).all()

@pytest.mark.parametrize('xml',[ASSET_XML,ASSET_XML.parent.parent/'k1'/'K1_22dof.xml'])
def test_fk_against_mujoco_positions_rotations_and_velocities(xml):
    m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m)
    joints=[m.joint(i).name for i in range(m.njnt) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE]
    bodies=[m.body(i).name for i in range(1,m.nbody)]
    fk=load('kinematics').HumanoidKinematics(str(xml),joints,bodies,'cpu')
    rot=load('rotations');rng=np.random.default_rng(17)
    for _ in range(20):
        d.qpos[:3]=rng.normal(size=3);d.qpos[3:7]=rng.normal(size=4);d.qpos[3:7]/=np.linalg.norm(d.qpos[3:7]);d.qpos[7:]=rng.uniform(-.5,.5,len(joints));d.qvel[:]=rng.normal(size=m.nv)
        mujoco.mj_forward(m,d)
        values=[torch.tensor(x,dtype=torch.float32) for x in [d.qpos[:3],d.qpos[3:7],d.qpos[7:],d.qvel[:3],d.qvel[3:6],d.qvel[6:],m.body_ipos[1:]]]
        o=fk.forward(*values)
        np.testing.assert_allclose(o['pos'],d.xpos[1:],atol=2e-6,rtol=1e-6)
        np.testing.assert_allclose(np.abs((o['quat'].numpy()*d.xquat[1:]).sum(-1)),1,atol=1e-6)
        for i in range(1,m.nbody):
            vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,i,vel,0)
            np.testing.assert_allclose(o['ang_vel'][i-1],vel[:3],atol=4e-6,rtol=1e-5)
            np.testing.assert_allclose(o['com_lin_vel'][i-1],vel[3:],atol=4e-6,rtol=1e-5)
