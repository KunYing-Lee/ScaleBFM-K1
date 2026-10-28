from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from scaletrack.robots.k1 import K1_CFG, K1_ACTION_SCALE, K1_BODIES, K1_XML
from scaletrack.tasks.tracking.config.g1_29dof.flat_env_cfg import G1BFMTrackingEnvCfg
from scaletrack.tasks.tracking.supermimic.env_cfg import G1BFMTrackingSuperMimicEnvCfg


def configure_k1(cfg):
    cfg.robot_mjcf_path = K1_XML
    cfg.scene.robot = K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/Trunk/.*"
    cfg.scene.terrain.visual_material = None
    cfg.actions.joint_pos.scale = K1_ACTION_SCALE
    mapping = dict(zip(cfg.commands.motion.body_names, K1_BODIES, strict=True))
    cfg.commands.motion.anchor_body_name = "Trunk"
    cfg.commands.motion.body_names = K1_BODIES.copy()
    cfg.commands.motion.mode_candidates = {
        name: [mapping[body] for body in bodies]
        for name, bodies in cfg.commands.motion.mode_candidates.items()
    }
    cfg.context_foot_bodies = ("left_foot_link", "right_foot_link")
    cfg.context_mass_bodies = ("left_hand_link", "right_hand_link")
    cfg.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="Trunk")
    cfg.events.hand_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=list(cfg.context_mass_bodies))


@configclass
class K1BFMTrackingEnvCfg(G1BFMTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_k1(self)


@configclass
class K1BFMTrackingSuperMimicEnvCfg(G1BFMTrackingSuperMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_k1(self)

@configclass
class K1BFMTrackingPackedPPOEnvCfg(K1BFMTrackingSuperMimicEnvCfg):
    action_noise_std: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        original = K1BFMTrackingEnvCfg()
        self.observations.critic = original.observations.critic
        self.observations.critic_task = original.observations.critic_task
