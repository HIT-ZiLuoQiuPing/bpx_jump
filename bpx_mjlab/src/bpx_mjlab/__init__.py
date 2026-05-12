from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    bpx_flat_env_cfg,
    bpx_jump_directional_env_cfg,
    bpx_jump_flat_env_cfg,
)
from .rl_cfg import (
    bpx_jump_directional_ppo_runner_cfg,
    bpx_jump_ppo_runner_cfg,
    bpx_ppo_runner_cfg,
)


register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-BPX",
    env_cfg=bpx_flat_env_cfg(),
    play_env_cfg=bpx_flat_env_cfg(play=True),
    rl_cfg=bpx_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
    task_id="Mjlab-Jump-Flat-BPX",
    env_cfg=bpx_jump_flat_env_cfg(),
    play_env_cfg=bpx_jump_flat_env_cfg(play=True),
    rl_cfg=bpx_jump_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
    task_id="Mjlab-Jump-Directional-BPX",
    env_cfg=bpx_jump_directional_env_cfg(),
    play_env_cfg=bpx_jump_directional_env_cfg(play=True),
    rl_cfg=bpx_jump_directional_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
