from isaaclab.utils import configclass
from scaletrack.tasks.tracking.config.g1_29dof.agents.rsl_rl_ppo_cfg import G1BFMTransformerPPORunnerCfg
from scaletrack.tasks.tracking.config.g1_29dof.agents.rsl_rl_supermimic_cfg import G1BFMTransformerSuperMimicRunnerCfg

@configclass
class K1BFMTransformerPPORunnerCfg(G1BFMTransformerPPORunnerCfg):
    experiment_name = "k1_bfm_tracking"

@configclass
class K1BFMTransformerSuperMimicRunnerCfg(G1BFMTransformerSuperMimicRunnerCfg):
    experiment_name = "k1_bfm_supermimic"
    max_iterations = 320000
    eval_max_steps = None

    def __post_init__(self):
        self.supermimic = dict(self.supermimic)
        self.supermimic.update(
            budget_mode="updates", max_train_hours=None,
            eval_interval_updates=5000,
            ppo_critic_obs_shapes={"critic": [3, 1 + 14 * 15 - 3 + 22 * 2], "critic_task": [7, 14 * 24 + 1]},
        )
