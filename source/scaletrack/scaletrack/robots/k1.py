"""K1 asset matching the explicitly selected SOMA K1-final reference data.

Actuator limits/gains: K1Lab 30f015a61f9583cba664b5ea65b0a1c73fba57e2.
Geometry and joint coordinates are preserved from the dataset's own MJCF.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from scaletrack.assets import ASSET_DIR

K1_XML = f"{ASSET_DIR}/robots/k1/K1_22dof.xml"
K1_JOINT_NAMES = [j.attrib["name"] for j in ET.parse(K1_XML).findall(".//joint") if j.attrib.get("type", "hinge") == "hinge"]
K1_BODIES = [
    "Trunk", "Left_Hip_Roll", "Left_Shank", "left_foot_link",
    "Right_Hip_Roll", "Right_Shank", "right_foot_link", "Head_2",
    "Left_Arm_2", "Left_Arm_3", "left_hand_link",
    "Right_Arm_2", "Right_Arm_3", "right_hand_link",
]
# effort, velocity, armature, stiffness, damping
K1_MOTORS = {
    ".*Knee_Pitch": (112.0, 12.57, 0.095625, 80.0, 4.0),
    ".*Hip_Yaw": (38.3, 17.59, 0.0282528, 80.0, 4.0),
    ".*Hip_Pitch": (68.0, 14.66, 0.0478125, 80.0, 4.0),
    ".*Hip_Roll": (76.0, 12.57, 0.0339552, 80.0, 4.0),
    ".*Ankle_(Pitch|Roll)": (38.3, 17.59, 0.0565056, 50.0, 2.0),
    ".*(Shoulder|Elbow).*": (14.0, 33.51, 0.001, 10.0, 1.0),
    ".*Head.*": (6.0, 7.85, 0.001, 4.0, 0.25),
}
K1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(K1_XML).with_suffix(".usd")),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "Left_Shoulder_Roll": -1.4, "Right_Shoulder_Roll": 1.4,
            "Left_Elbow_Yaw": -0.4, "Right_Elbow_Yaw": 0.4,
            ".*Hip_Pitch": -0.4, ".*Knee_Pitch": 0.8, ".*Ankle_Pitch": -0.4,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        f"motor_{i}": ImplicitActuatorCfg(
            joint_names_expr=[pattern], effort_limit_sim=v[0], velocity_limit_sim=v[1],
            armature=v[2], stiffness=v[3], damping=v[4],
        ) for i, (pattern, v) in enumerate(K1_MOTORS.items())
    },
)
K1_ACTION_SCALE = {pattern: 0.25 * v[0] / v[3] for pattern, v in K1_MOTORS.items()}
