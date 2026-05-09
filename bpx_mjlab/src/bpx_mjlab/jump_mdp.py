from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class InPlaceJumpCommand(CommandTerm):
    cfg: InPlaceJumpCommandCfg

    def __init__(self, cfg: InPlaceJumpCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.target_height_delta = torch.zeros(self.num_envs, device=self.device)
        self.target_dx = torch.zeros(self.num_envs, device=self.device)
        self.jump_command = torch.zeros(self.num_envs, 4, device=self.device)
        self.current_duration = torch.full(
            (self.num_envs,), cfg.resampling_time_range[1], device=self.device
        )

        self.cycle_start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._needs_cycle_start_update = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self.metrics["height_delta_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["xy_drift"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.jump_command

    def _capture_cycle_start(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.cycle_start_pos_w[env_ids] = self.robot.data.root_link_pos_w[env_ids]
        self._needs_cycle_start_update[env_ids] = False

    def _update_metrics(self) -> None:
        pending_ids = self._needs_cycle_start_update.nonzero(as_tuple=False).flatten()
        self._capture_cycle_start(pending_ids)

        duration_s = self.cfg.period_s if self.cfg.repeat else self.cfg.jump_duration_s
        max_command_step = max(duration_s / self._env.step_dt, 1.0)
        height_delta = self.robot.data.root_link_pos_w[:, 2] - self.cycle_start_pos_w[:, 2]
        xy_drift = torch.norm(
            self.robot.data.root_link_pos_w[:, :2] - self.cycle_start_pos_w[:, :2],
            dim=1,
        )

        self.metrics["height_delta_error"] += (
            torch.abs(height_delta - self.target_height_delta) / max_command_step
        )
        self.metrics["xy_drift"] += xy_drift / max_command_step

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        r = torch.empty(len(env_ids), device=self.device)
        self.target_height_delta[env_ids] = r.uniform_(
            *self.cfg.target_height_delta_range
        )
        self.target_dx[env_ids] = r.uniform_(*self.cfg.target_dx_range)
        self.current_duration[env_ids] = self.time_left[env_ids]
        self._needs_cycle_start_update[env_ids] = True

    def _update_command(self) -> None:
        reset_ids = (self._env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
        pending_ids = self._needs_cycle_start_update.nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0 or len(pending_ids) > 0:
            env_ids = torch.unique(torch.cat((reset_ids, pending_ids)))
            self._capture_cycle_start(env_ids)

        if self.cfg.repeat:
            period = max(self.cfg.period_s, self._env.step_dt)
            time_left = torch.clamp(self.time_left, min=0.0, max=period)
            self.phase = (period - time_left) / period
            angle = self.phase * (2.0 * math.pi)
        else:
            duration = max(self.cfg.jump_duration_s, self._env.step_dt)
            elapsed = torch.clamp(self.current_duration - self.time_left, min=0.0)
            self.phase = torch.clamp(elapsed / duration, min=0.0, max=1.0)
            angle = self.phase * math.pi

        self.jump_command[:, 0] = torch.sin(angle)
        self.jump_command[:, 1] = torch.cos(angle)
        self.jump_command[:, 2] = self.target_height_delta
        self.jump_command[:, 3] = self.target_dx


@dataclass(kw_only=True)
class InPlaceJumpCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    period_s: float = 2.0
    repeat: bool = True
    jump_duration_s: float = 2.0
    target_height_delta_range: tuple[float, float] = (0.12, 0.22)
    target_dx_range: tuple[float, float] = (0.0, 0.0)

    @dataclass
    class VizCfg:
        z_offset: float = 0.5
        scale: float = 0.5

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> InPlaceJumpCommand:
        return InPlaceJumpCommand(self, env)

    def __post_init__(self) -> None:
        if self.period_s <= 0.0:
            raise ValueError("period_s must be positive.")
        if self.jump_duration_s <= 0.0:
            raise ValueError("jump_duration_s must be positive.")
        if self.target_height_delta_range[0] < 0.0:
            raise ValueError("target_height_delta_range must be non-negative.")


def _jump_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    return command


def _jump_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    if hasattr(term, "phase"):
        return term.phase

    command = _jump_command(env, command_name)
    phase = torch.atan2(command[:, 0], command[:, 1]) / (2.0 * math.pi)
    return torch.remainder(phase, 1.0)


def _cycle_start_pos_w(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    if hasattr(term, "cycle_start_pos_w"):
        return term.cycle_start_pos_w

    asset: Entity = env.scene["robot"]
    start = asset.data.root_link_pos_w.clone()
    start[:, :2] = env.scene.env_origins[:, :2]
    return start


def _phase_mask(
    phase: torch.Tensor,
    start: float,
    end: float,
) -> torch.Tensor:
    if start <= end:
        if end >= 1.0:
            return ((phase >= start) & (phase <= end)).float()
        return ((phase >= start) & (phase < end)).float()
    return ((phase >= start) | (phase < end)).float()


def _foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return sensor.data.found > 0


def base_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2:3]


def base_height_delta(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    start_pos = _cycle_start_pos_w(env, command_name)
    return (asset.data.root_link_pos_w[:, 2:3] - start_pos[:, 2:3])


def horizontal_displacement(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    start_pos = _cycle_start_pos_w(env, command_name)
    return asset.data.root_link_pos_w[:, :2] - start_pos[:, :2]


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    return _foot_contact(env, sensor_name).float()


def crouch_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    crouch_depth: float = 0.05,
    std: float = 0.06,
    phase_range: tuple[float, float] = (0.0, 0.25),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    phase = _jump_phase(env, command_name)
    mask = _phase_mask(phase, *phase_range)
    height_delta = base_height_delta(env, command_name, asset_cfg).squeeze(1)
    contact = _foot_contact(env, sensor_name).all(dim=1).float()
    height_score = torch.exp(-torch.square(height_delta + crouch_depth) / (std**2))
    return mask * (0.7 * height_score + 0.3 * contact)


def takeoff_velocity_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_vz: float = 0.95,
    phase_range: tuple[float, float] = (0.25, 0.45),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    phase = _jump_phase(env, command_name)
    mask = _phase_mask(phase, *phase_range)
    asset: Entity = env.scene[asset_cfg.name]
    vz = asset.data.root_link_lin_vel_w[:, 2]
    return mask * torch.clamp(vz / target_vz, min=0.0, max=1.25)


def contact_release_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    phase_range: tuple[float, float] = (0.32, 0.55),
) -> torch.Tensor:
    phase = _jump_phase(env, command_name)
    mask = _phase_mask(phase, *phase_range)
    contact = _foot_contact(env, sensor_name).float()
    release = 1.0 - torch.mean(contact, dim=1)
    return mask * release


def flight_airborne_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    phase_range: tuple[float, float] = (0.45, 0.70),
) -> torch.Tensor:
    phase = _jump_phase(env, command_name)
    mask = _phase_mask(phase, *phase_range)
    all_airborne = (~_foot_contact(env, sensor_name)).all(dim=1).float()
    env.extras["log"]["Metrics/all_feet_airborne"] = torch.mean(all_airborne)
    return mask * all_airborne


def phase_gated_air_time_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    threshold_min: float = 0.08,
    threshold_max: float = 0.55,
    phase_range: tuple[float, float] = (0.38, 0.72),
) -> torch.Tensor:
    phase = _jump_phase(env, command_name)
    mask = _phase_mask(phase, *phase_range)

    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1) * mask
    in_air = current_air_time > 0
    num_in_air = torch.sum(in_air.float())
    mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
        num_in_air, min=1
    )
    env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
    return reward


class peak_height_reward:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.peak_height_delta = torch.zeros(env.num_envs, device=env.device)
        self.prev_phase = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.peak_height_delta[env_ids] = 0.0
        self.prev_phase[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float = 0.055,
        score_phase_range: tuple[float, float] = (0.60, 0.82),
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        phase = _jump_phase(env, command_name)
        height_delta = base_height_delta(env, command_name, asset_cfg).squeeze(1)
        new_cycle = phase < (self.prev_phase - 0.2)
        self.peak_height_delta = torch.where(
            new_cycle,
            height_delta,
            torch.maximum(self.peak_height_delta, height_delta),
        )

        target_height = _jump_command(env, command_name)[:, 2]
        score = torch.exp(
            -torch.square(self.peak_height_delta - target_height) / (std**2)
        )
        mask = _phase_mask(phase, *score_phase_range)

        env.extras["log"]["Metrics/peak_height_mean"] = torch.mean(
            self.peak_height_delta
        )
        env.extras["log"]["Metrics/peak_height_error"] = torch.mean(
            torch.abs(self.peak_height_delta - target_height)
        )

        self.prev_phase = phase.detach().clone()
        return mask * score


class landing_stability_reward:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.prev_phase = torch.zeros(env.num_envs, device=env.device)
        self.has_been_airborne = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.prev_phase[env_ids] = 0.0
        self.has_been_airborne[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        sensor_name: str,
        airborne_phase_start: float = 0.36,
        landing_phase_start: float = 0.70,
        xy_vel_std: float = 0.35,
        xy_drift_std: float = 0.12,
        upright_std: float = 0.35,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        phase = _jump_phase(env, command_name)
        new_cycle = phase < (self.prev_phase - 0.2)
        self.has_been_airborne = torch.where(
            new_cycle, torch.zeros_like(self.has_been_airborne), self.has_been_airborne
        )

        contact = _foot_contact(env, sensor_name)
        contact_fraction = torch.mean(contact.float(), dim=1)
        all_airborne = (~contact).all(dim=1)
        self.has_been_airborne |= all_airborne & (phase > airborne_phase_start)

        asset: Entity = env.scene[asset_cfg.name]
        xy_vel = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=1)
        xy_drift = torch.norm(horizontal_displacement(env, command_name, asset_cfg), dim=1)
        tilt = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

        xy_vel_score = torch.exp(-torch.square(xy_vel) / (xy_vel_std**2))
        xy_drift_score = torch.exp(-torch.square(xy_drift) / (xy_drift_std**2))
        upright_score = torch.exp(-tilt / (upright_std**2))
        active = self.has_been_airborne.float() * _phase_mask(
            phase, landing_phase_start, 1.0
        )
        score = contact_fraction * xy_vel_score * xy_drift_score * upright_score

        env.extras["log"]["Metrics/landing_stable"] = torch.mean(active * score)
        env.extras["log"]["Metrics/landing_contact_fraction"] = torch.mean(
            active * contact_fraction
        )
        self.prev_phase = phase.detach().clone()
        return active * score


class landing_rebound_penalty:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.prev_phase = torch.zeros(env.num_envs, device=env.device)
        self.has_been_airborne = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.has_landed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.landing_height_delta = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.prev_phase[env_ids] = 0.0
        self.has_been_airborne[env_ids] = False
        self.has_landed[env_ids] = False
        self.landing_height_delta[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        sensor_name: str,
        airborne_phase_start: float = 0.36,
        landing_phase_start: float = 0.70,
        height_tolerance: float = 0.015,
        airborne_weight: float = 1.0,
        upward_vel_weight: float = 2.5,
        height_weight: float = 20.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        phase = _jump_phase(env, command_name)
        new_cycle = phase < (self.prev_phase - 0.2)
        self.has_been_airborne = torch.where(
            new_cycle, torch.zeros_like(self.has_been_airborne), self.has_been_airborne
        )
        self.has_landed = torch.where(
            new_cycle, torch.zeros_like(self.has_landed), self.has_landed
        )
        self.landing_height_delta = torch.where(
            new_cycle, torch.zeros_like(self.landing_height_delta), self.landing_height_delta
        )

        contact = _foot_contact(env, sensor_name)
        all_airborne = (~contact).all(dim=1)
        self.has_been_airborne |= all_airborne & (phase > airborne_phase_start)

        height_delta = base_height_delta(env, command_name, asset_cfg).squeeze(1)
        landing_event = self.has_been_airborne & (phase > landing_phase_start) & (~self.has_landed)
        self.landing_height_delta = torch.where(
            landing_event,
            height_delta,
            self.landing_height_delta,
        )
        self.has_landed |= landing_event

        asset: Entity = env.scene[asset_cfg.name]
        any_airborne = (~contact).any(dim=1).float()
        upward_vz = torch.clamp(asset.data.root_link_lin_vel_w[:, 2], min=0.0)
        rebound_height = torch.clamp(
            height_delta - self.landing_height_delta - height_tolerance,
            min=0.0,
        )
        active = self.has_landed.float() * _phase_mask(phase, landing_phase_start, 1.0)
        penalty = (
            airborne_weight * any_airborne
            + upward_vel_weight * torch.square(upward_vz)
            + height_weight * torch.square(rebound_height)
        )

        env.extras["log"]["Metrics/landing_rebound_airborne"] = torch.mean(
            active * any_airborne
        )
        env.extras["log"]["Metrics/landing_rebound_height"] = torch.mean(
            active * rebound_height
        )
        env.extras["log"]["Metrics/landing_rebound_vz"] = torch.mean(
            active * upward_vz
        )

        self.prev_phase = phase.detach().clone()
        return active * penalty


class post_landing_hold_reward:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.prev_phase = torch.zeros(env.num_envs, device=env.device)
        self.has_been_airborne = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.prev_phase[env_ids] = 0.0
        self.has_been_airborne[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        sensor_name: str,
        airborne_phase_start: float = 0.36,
        landing_phase_start: float = 0.72,
        height_std: float = 0.06,
        xy_vel_std: float = 0.28,
        vz_std: float = 0.22,
        upright_std: float = 0.28,
        action_rate_std: float = 0.55,
        pose_std: float = 0.42,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        phase = _jump_phase(env, command_name)
        new_cycle = phase < (self.prev_phase - 0.2)
        self.has_been_airborne = torch.where(
            new_cycle, torch.zeros_like(self.has_been_airborne), self.has_been_airborne
        )

        contact = _foot_contact(env, sensor_name)
        all_airborne = (~contact).all(dim=1)
        contact_fraction = torch.mean(contact.float(), dim=1)
        self.has_been_airborne |= all_airborne & (phase > airborne_phase_start)

        asset: Entity = env.scene[asset_cfg.name]
        height_delta = base_height_delta(env, command_name, asset_cfg).squeeze(1)
        xy_vel = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=1)
        vz = torch.abs(asset.data.root_link_lin_vel_w[:, 2])
        tilt = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
        action_rate = torch.norm(
            env.action_manager.action - env.action_manager.prev_action,
            dim=1,
        )
        default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        pose_error = torch.mean(torch.square(joint_pos - default_joint_pos), dim=1)

        height_score = torch.exp(-torch.square(height_delta) / (height_std**2))
        xy_vel_score = torch.exp(-torch.square(xy_vel) / (xy_vel_std**2))
        vz_score = torch.exp(-torch.square(vz) / (vz_std**2))
        upright_score = torch.exp(-tilt / (upright_std**2))
        action_score = torch.exp(-torch.square(action_rate) / (action_rate_std**2))
        pose_score = torch.exp(-pose_error / (pose_std**2))

        active = self.has_been_airborne.float() * _phase_mask(
            phase, landing_phase_start, 1.0
        )
        score = contact_fraction * (
            0.20 * height_score
            + 0.20 * xy_vel_score
            + 0.20 * vz_score
            + 0.20 * upright_score
            + 0.10 * action_score
            + 0.10 * pose_score
        )

        env.extras["log"]["Metrics/post_landing_active"] = torch.mean(active)
        env.extras["log"]["Metrics/hold_contact"] = torch.mean(active * contact_fraction)
        env.extras["log"]["Metrics/post_landing_vz"] = torch.mean(active * vz)
        env.extras["log"]["Metrics/post_landing_airborne"] = torch.mean(
            active * (1.0 - contact_fraction)
        )
        env.extras["log"]["Metrics/landing_pose_error"] = torch.mean(active * pose_error)

        self.prev_phase = phase.detach().clone()
        return active * score


def xy_drift_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    drift = horizontal_displacement(env, command_name, asset_cfg)
    drift_norm = torch.norm(drift, dim=1)
    env.extras["log"]["Metrics/xy_drift"] = torch.mean(drift_norm)
    return torch.sum(torch.square(drift), dim=1)


def horizontal_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_lin_vel_w[:, :2]), dim=1)


def _mirror_leg_name(name: str) -> str:
    for src, dst in (
        ("fl_", "fr_"),
        ("fr_", "fl_"),
        ("hl_", "hr_"),
        ("hr_", "hl_"),
    ):
        if name.startswith(src):
            return dst + name[len(src):]
    return name


def _joint_mirror_sign(name: str) -> float:
    return -1.0 if "hip_roll" in name else 1.0


@lru_cache(maxsize=32)
def _joint_mirror_map(names: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    index_by_name = {name: index for index, name in enumerate(names)}
    indices: list[int] = []
    signs: list[float] = []
    for name in names:
        mirror_name = _mirror_leg_name(name)
        indices.append(index_by_name.get(mirror_name, index_by_name[name]))
        signs.append(_joint_mirror_sign(name))
    return tuple(indices), tuple(signs)


def _selected_joint_names(asset: Entity, joint_ids: list[int] | slice) -> tuple[str, ...]:
    if isinstance(joint_ids, slice):
        return tuple(asset.joint_names[joint_ids])
    return tuple(asset.joint_names[index] for index in joint_ids)


def _mirror_joint_values(values: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
    indices, signs = _joint_mirror_map(names)
    index_tensor = torch.tensor(indices, device=values.device, dtype=torch.long)
    sign_tensor = torch.tensor(signs, device=values.device, dtype=values.dtype)
    return values.index_select(-1, index_tensor) * sign_tensor


def _pair_error(
    values: torch.Tensor,
    names: tuple[str, ...],
    suffix: str,
    opposite_sign: bool,
) -> torch.Tensor:
    index_by_name = {name: index for index, name in enumerate(names)}
    errors = []
    for left, right in (("fl_", "fr_"), ("hl_", "hr_")):
        left_name = left + suffix
        right_name = right + suffix
        if left_name not in index_by_name or right_name not in index_by_name:
            continue
        left_value = values[:, index_by_name[left_name]]
        right_value = values[:, index_by_name[right_name]]
        if opposite_sign:
            errors.append(torch.abs(left_value + right_value))
        else:
            errors.append(torch.abs(left_value - right_value))
    if not errors:
        return torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
    return torch.stack(errors, dim=1).mean(dim=1)


def _left_right_symmetry_cost(
    values: torch.Tensor,
    names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mirrored_values = _mirror_joint_values(values, names)
    cost = torch.mean(torch.square(values - mirrored_values), dim=1)
    roll_error = _pair_error(values, names, "hip_roll_joint", opposite_sign=True)
    pitch_error = _pair_error(values, names, "hip_pitch_joint", opposite_sign=False)
    knee_error = _pair_error(values, names, "knee_joint", opposite_sign=False)
    return cost, roll_error, pitch_error, knee_error


def left_right_joint_symmetry_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    phase_range: tuple[float, float] = (0.20, 1.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    phase = _jump_phase(env, command_name)
    mask = _phase_mask(phase, *phase_range)
    asset: Entity = env.scene[asset_cfg.name]
    values = asset.data.joint_pos[:, asset_cfg.joint_ids]
    names = _selected_joint_names(asset, asset_cfg.joint_ids)
    cost, roll_error, pitch_error, knee_error = _left_right_symmetry_cost(values, names)

    env.extras["log"]["Metrics/lr_joint_asym"] = torch.mean(mask * cost)
    env.extras["log"]["Metrics/lr_roll_asym"] = torch.mean(mask * roll_error)
    env.extras["log"]["Metrics/lr_pitch_asym"] = torch.mean(mask * pitch_error)
    env.extras["log"]["Metrics/lr_knee_asym"] = torch.mean(mask * knee_error)
    return mask * cost


def left_right_action_symmetry_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    phase_range: tuple[float, float] = (0.20, 1.0),
) -> torch.Tensor:
    phase = _jump_phase(env, command_name)
    mask = _phase_mask(phase, *phase_range)
    action_term = env.action_manager.get_term("joint_pos")
    names = tuple(action_term.target_names)
    values = env.action_manager.action
    cost, _, pitch_error, knee_error = _left_right_symmetry_cost(values, names)

    env.extras["log"]["Metrics/lr_action_asym"] = torch.mean(mask * cost)
    env.extras["log"]["Metrics/lr_action_pitch_asym"] = torch.mean(mask * pitch_error)
    env.extras["log"]["Metrics/lr_action_knee_asym"] = torch.mean(mask * knee_error)
    return mask * cost
