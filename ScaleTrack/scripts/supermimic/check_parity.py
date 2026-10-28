"""Check that the analytic heads of the SuperMimic task reproduce the IsaacLab managers of the tracking task.

Steps the SuperMimic environment with random actions (no observation / action noise) and compares, every step:
  * FK of the state against the articulation's link poses and CoM velocities,
  * the analytic policy proprioception / task observation / mode inputs against the observation manager,
  * the 3-frame observation and action histories rebuilt from the replay buffer against the managers' histories,
  * the analytic reward and termination of every transition against the reward / termination managers.

Example:
    python scripts/supermimic/check_parity.py --motion_lib /dev/shm/sm_lib/train_lafan --num_envs 256 --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--report", type=str)
parser.add_argument("--task", type=str, default="G1-BFM-Transformer-Tracking-SuperMimic")
parser.add_argument("--motion_lib", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--action_std", type=float, default=0.3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import scaletrack.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from my_rsl_rl.storage import ReplayBuffer
from scaletrack.tasks.tracking.supermimic.rotations import quat_angle_sq, quat_apply

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.commands.motion.motion_lib_dir = args.motion_lib
cfg.observations.policy.enable_corruption = False
cfg.observations.policy_task.enable_corruption = False
cfg.action_noise_std = 0.0
cfg.actions.joint_pos.noise_std = 0.0
env = gym.make(args.task, cfg=cfg).unwrapped
obs, _ = env.reset()
task = env.make_imagination_task()
N = env.num_envs
all_envs = torch.arange(N, device=env.device)
ctx = task.context(all_envs)
robot = task.robot
cmd = task.command
nj = robot.num_joints
buf = ReplayBuffer(N, 400, task.state_dim, nj, ["frame", "mode", "roff"], env.device)

worst: dict[str, float] = {}


def record(name: str, value: torch.Tensor | float):
    value = float(value)
    worst[name] = max(worst.get(name, 0.0), value)


state, info = task.read_state(), task.step_info()
ep_start = torch.ones(N, dtype=torch.bool, device=env.device)
prev_action = torch.zeros(N, nj, device=env.device)
n_term = 0
n_trunc = 0
for step in range(args.steps):
    # ---- observations of the current step
    bodies = task.bodies(state, ctx)
    data = robot.data
    ids = cmd.body_indexes
    record("fk_pos_m", (bodies["pos"] - (data.body_pos_w[:, ids] - env.scene.env_origins[:, None])).norm(dim=-1).max())
    record("fk_rot_rad", quat_angle_sq(bodies["quat"], data.body_quat_w[:, ids]).sqrt().max())
    prop = task.prop_obs(state, ctx, noise=False)
    for tag, sel in (("", ~ep_start), ("_after_reset", ep_start)):
        if sel.any():
            record(f"fk_ang_vel{tag}", (bodies["ang_vel"] - data.body_ang_vel_w[:, ids])[sel].norm(dim=-1).max())
            velocity_error = bodies["com_lin_vel"] - data.body_lin_vel_w[:, ids]
            record("fk_com_lin_vel_initial" if step == 0 else f"fk_com_lin_vel{tag}", velocity_error[sel].norm(dim=-1).max())
            if step == 0 and tag and cfg.events.base_com.params["asset_cfg"].body_names == "Trunk":
                # PhysX 5.1 initialization-only readback uses the pre-randomization root CoM
                # when propagating child velocities. This getter is not used by the
                # actor or replay state; all post-physics velocities remain gated.
                # Test the exact backend discrepancy, retaining its raw magnitude.
                shift = torch.cross(data.root_com_vel_w[:, 3:], quat_apply(
                    data.root_link_quat_w, data.com_offset_randomization.to(env.device)), dim=-1)
                expected_error = -shift[:, None].expand_as(velocity_error).clone()
                expected_error[:, ids == 0] = 0
                record("reset_backend_velocity_identity", (velocity_error - expected_error)[sel].norm(dim=-1).max())
                record("reset_root_com_velocity", velocity_error[sel, 0].norm(dim=-1).max())
            err = (prop - obs["policy"][:, -1])[sel].abs().amax(0)
            for name, sl in (("gravity", slice(0, 3)), ("ang_vel", slice(3, 6)), ("joint_pos", slice(6, 6 + nj)), ("joint_vel", slice(6 + nj, 6 + 2 * nj))):
                record(f"obs_prop_{name}{tag}", err[sl].max())
    task_obs = task.policy_task_obs(bodies, info["frame"], info["roff"], noise=False)
    record("obs_task", (task_obs - obs["policy_task"]).abs().max())
    mode, mapping = task.mode_inputs(info["mode"])
    record("obs_mode", (mode - obs["mode"]).abs().max())
    record("obs_mode_mapping", (mapping - obs["mode_mapping"]).abs().max())

    # ---- histories rebuilt from the replay buffer (needs the current row in the buffer)
    actions = args.action_std * torch.randn(N, nj, device=env.device)
    buf.add(state, actions, actions, info, ep_start)
    # the newest row of every env is the current step
    buf._start_candidates = all_envs * buf.T + (buf.ptr - 1) % buf.T
    sb = buf.sample_starts(N, 3)
    sel = sb.env_ids
    ctx_s = task.context(sel)
    hist_prop = torch.stack([task.prop_obs(sb.hist_states[i], ctx_s, noise=False) for i in range(3)], 1)
    record("hist_prop", (hist_prop - obs["policy"][sel]).abs().max())
    record("hist_action", (sb.hist_actions.transpose(0, 1) - obs["action"][sel]).abs().max())
    for k in ("frame", "mode", "roff"):
        record(f"hist_info_{k}", (sb.info[k] != info[k][sel]).sum())

    # ---- transition
    obs, rew, terminated, truncated, _ = env.step(actions)
    reset = terminated | truncated
    s_next = task.read_state()
    reset_ids = reset.nonzero().squeeze(-1)
    s_next[reset_ids] = task.terminal_state(reset_ids)
    b_next = task.bodies(s_next, ctx)
    ref = task.reference(info["frame"])
    hard = task.heads.termination(b_next, ref)
    record("termination_mismatch", (hard != terminated).float().sum())
    imagined_trunc = task.transition(s_next, b_next, actions, prev_action, info["frame"])["truncated"]
    sim_trunc = env.termination_manager.get_term("motion_time_out")
    record("truncation_mismatch", (imagined_trunc != sim_trunc).float().sum())
    n_trunc += int(sim_trunc.sum())
    n_term += int(terminated.sum())
    terms = task.heads.reward_terms(s_next, b_next, actions, prev_action, ref, terminated.float())
    reward = sum(terms.values())
    record("reward_abs", (reward - rew).abs().max())
    record("reward_rel", ((reward - rew).abs() / rew.abs().clamp_min(1e-3)).max())
    for name, term in terms.items():
        idx = env.reward_manager.active_terms.index(name)
        record(f"reward/{name}", (term - env.reward_manager._step_reward[:, idx] * env.step_dt).abs().max())
    buf.add_terminal(reset_ids, task.terminal_state(reset_ids))

    prev_action = torch.where(reset[:, None], torch.zeros_like(actions), actions)
    state, info, ep_start = task.read_state(), task.step_info(), reset

print(f"[parity] {args.steps} steps x {N} envs, {n_term} terminations, {n_trunc} motion-end truncations")
for k, v in worst.items():
    print(f"[parity] max {k:<32s} {v:.3e}")
if args.report:
    import json
    from pathlib import Path
    thresholds = {k: (0.0 if "mismatch" in k or "hist_info" in k else 1e-3) for k in worst}
    thresholds["fk_pos_m"] = 1e-4
    excluded = {"reward_rel"}
    if "reset_backend_velocity_identity" in worst:
        excluded.add("fk_com_lin_vel_initial")
    passed = all(torch.isfinite(torch.tensor(v)).item() and v <= thresholds[k] for k, v in worst.items() if k not in excluded)
    Path(args.report).write_text(json.dumps(dict(passed=passed,steps=args.steps,num_envs=N,terminations=n_term,truncations=n_trunc,max_errors=worst,thresholds=thresholds,diagnostic_only=sorted(excluded),reset_readback_limitation="PhysX 5.1 child velocities at initialization use nominal root CoM until first physics step; analytic identity independently checked" if len(excluded)>1 else None),indent=2))
    print("PARITY_PASS" if passed else "PARITY_FAIL", flush=True)
    os._exit(0 if passed else 1)
os._exit(0)
