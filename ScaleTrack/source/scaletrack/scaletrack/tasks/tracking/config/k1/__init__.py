import gymnasium as gym

gym.register(
    id="K1-BFM-Transformer-Tracking",
    entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:K1BFMTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cfg:K1BFMTransformerPPORunnerCfg",
    },
)
gym.register(
    id="K1-BFM-Transformer-Tracking-SuperMimic",
    entry_point="scaletrack.tasks.tracking.supermimic.env:SuperMimicTrackingEnv", disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:K1BFMTrackingSuperMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cfg:K1BFMTransformerSuperMimicRunnerCfg",
    },
)

gym.register(
    id="K1-BFM-Transformer-Tracking-Packed-PPO",
    entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:K1BFMTrackingPackedPPOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cfg:K1BFMTransformerPPORunnerCfg",
    },
)
