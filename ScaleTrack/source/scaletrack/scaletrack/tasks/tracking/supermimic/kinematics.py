"""Differentiable forward kinematics of a hinge-joint humanoid described by an MJCF file.

The kinematic tree (body offsets, joint axes) is read from the MJCF; joints are assumed to be hinges located at
the child body origin (true for g1_29dof.xml). Inputs and outputs follow the IsaacLab conventions so the results
can be compared one-to-one with ``Articulation.data``:

* joint positions / velocities are given in the simulator (PhysX) joint order,
* body positions and orientations are link frames (``body_link_pos_w`` / ``body_link_quat_w``),
* linear velocities are link-origin velocities, and ``com_lin_vel`` adds the centre-of-mass offset
  (``body_com_lin_vel_w``, which is what ``Articulation.data.body_lin_vel_w`` returns).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import torch

from .rotations import quat_apply, quat_from_axis_angle, quat_mul


def parse_mjcf_tree(xml_path: str):
    """Returns body names (DFS order), parent indices, local positions, local quaternions, joint names and axes.

    Index 0 is the floating root; body i > 0 is driven by joint i - 1.
    """
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.attrib.get("angle", "degree") != "radian":
        raise ValueError("only angle='radian' MJCF files are supported")
    world = root.find("worldbody")
    names, parents, pos, quat, joint_names, axes, body_joints = [], [], [], [], [], [], []

    def visit(node, parent):
        idx = len(names)
        names.append(node.attrib["name"])
        parents.append(parent)
        pos.append(np.fromstring(node.attrib.get("pos", "0 0 0"), sep=" "))
        quat.append(np.fromstring(node.attrib.get("quat", "1 0 0 0"), sep=" "))
        joints = node.findall("joint") if parent >= 0 else []
        if len(joints) > 1:
            raise ValueError(f"body {names[-1]} has more than one hinge joint")
        body_joints.append(len(joint_names) if joints else -1)
        if joints:
            joint = joints[0]
            if joint.attrib.get("type", "hinge") != "hinge":
                raise ValueError(f"joint of body {names[-1]} must be a hinge")
            if np.any(np.fromstring(joint.attrib.get("pos", "0 0 0"), sep=" ") != 0):
                raise ValueError(f"joint of body {names[-1]} is not at the body origin")
            joint_names.append(joint.attrib["name"])
            axis = np.fromstring(joint.attrib.get("axis", "0 0 1"), sep=" ")
            axes.append(axis / np.linalg.norm(axis))
        for child in node.findall("body"):
            visit(child, idx)

    visit(world.find("body"), -1)
    return names, parents, np.array(pos), np.array(quat), joint_names, np.array(axes), body_joints


class HumanoidKinematics:
    """Batched FK returning the poses and velocities of selected bodies."""

    def __init__(self, xml_path: str, sim_joint_names: list[str], output_body_names: list[str], device):
        names, parents, pos, quat, joint_names, axes, self.body_joints = parse_mjcf_tree(xml_path)
        if sorted(joint_names) != sorted(sim_joint_names):
            raise ValueError("MJCF joints do not match the simulator joints")
        self.device = torch.device(device)
        self.body_names = names
        self.parents = parents
        # Joint i of the tree (body i + 1) reads simulator joint sim_index[i].
        self.sim_index = torch.tensor([sim_joint_names.index(n) for n in joint_names], dtype=torch.long, device=device)
        self.local_pos = torch.tensor(pos, dtype=torch.float32, device=device)
        self.local_quat = torch.tensor(quat / np.linalg.norm(quat, axis=-1, keepdims=True), dtype=torch.float32, device=device)
        self.axes = torch.tensor(axes, dtype=torch.float32, device=device)
        self.output_index = [names.index(n) for n in output_body_names]
        self.num_bodies = len(names)

    def forward(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        root_lin_vel_w: torch.Tensor | None = None,
        root_ang_vel_b: torch.Tensor | None = None,
        joint_vel: torch.Tensor | None = None,
        com_pos_b: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Leading batch dims are arbitrary. ``com_pos_b`` (..., n_out, 3) is the CoM offset in each output link frame."""
        with_vel = root_lin_vel_w is not None
        q = joint_pos[..., self.sim_index]
        joint_quat = quat_from_axis_angle(self.axes.expand(*q.shape, 3), q)
        body_pos = [root_pos]
        body_quat = [root_quat]
        if with_vel:
            qd = joint_vel[..., self.sim_index]
            body_lin = [root_lin_vel_w]
            body_ang = [quat_apply(root_quat, root_ang_vel_b)]
        for i in range(1, self.num_bodies):
            p = self.parents[i]
            parent_pos, parent_quat = body_pos[p], body_quat[p]
            offset_w = quat_apply(parent_quat, self.local_pos[i].expand_as(parent_pos))
            j = self.body_joints[i]
            quat = quat_mul(parent_quat, self.local_quat[i].expand_as(parent_quat))
            if j >= 0:
                quat = quat_mul(quat, joint_quat[..., j, :])
            body_pos.append(parent_pos + offset_w)
            body_quat.append(quat)
            if with_vel:
                if j >= 0:
                    axis_w = quat_apply(quat, self.axes[j].expand_as(parent_pos))
                    body_ang.append(body_ang[p] + axis_w * qd[..., j : j + 1])
                else:
                    body_ang.append(body_ang[p])
                body_lin.append(body_lin[p] + torch.cross(body_ang[p], offset_w, dim=-1))
        idx = self.output_index
        out = {
            "pos": torch.stack([body_pos[i] for i in idx], dim=-2),
            "quat": torch.stack([body_quat[i] for i in idx], dim=-2),
        }
        if with_vel:
            out["lin_vel"] = torch.stack([body_lin[i] for i in idx], dim=-2)
            out["ang_vel"] = torch.stack([body_ang[i] for i in idx], dim=-2)
            if com_pos_b is not None:
                out["com_lin_vel"] = out["lin_vel"] + torch.cross(out["ang_vel"], quat_apply(out["quat"], com_pos_b), dim=-1)
        return out
