from dataclasses import dataclass, field

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@dataclass
class BpxJumpSymmetryCfg:
    use_data_augmentation: bool = True
    use_mirror_loss: bool = True
    data_augmentation_func: str = (
        "bpx_mjlab.symmetry:jump_mirror_data_augmentation"
    )
    mirror_loss_coeff: float = 0.2


@dataclass
class BpxJumpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: BpxJumpSymmetryCfg = field(default_factory=BpxJumpSymmetryCfg)


def bpx_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.01,
        ),
        experiment_name="bpx_velocity",
        max_iterations=10_000,
    )


def bpx_jump_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
        ),
        algorithm=BpxJumpPpoAlgorithmCfg(
            entropy_coef=0.01,
        ),
        experiment_name="bpx_jump_flat",
        max_iterations=12_000,
    )


def bpx_jump_directional_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = bpx_jump_ppo_runner_cfg()
    cfg.experiment_name = "bpx_jump_directional"
    cfg.max_iterations = 12_000
    return cfg
