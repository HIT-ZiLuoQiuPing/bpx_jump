from __future__ import annotations

import math
from dataclasses import dataclass, field
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

        max_command_step = max(self.cfg.period_s / self._env.step_dt, 1.0)
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
        self._needs_cycle_start_update[env_ids] = True

    def _update_command(self) -> None:
        reset_ids = (self._env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
        pending_ids = self._needs_cycle_start_update.nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0 or len(pending_ids) > 0:
            env_ids = torch.unique(torch.cat((reset_ids, pending_ids)))
            self._capture_cycle_start(env_ids)

        period = max(self.cfg.period_s, self._env.step_dt)
        time_left = torch.clamp(self.time_left, min=0.0, max=period)
        self.phase = (period - time_left) / period
        angle = self.phase * (2.0 * math.pi)

        self.jump_command[:, 0] = torch.sin(angle)
        self.jump_command[:, 1] = torch.cos(angle)
        self.jump_command[:, 2] = self.target_height_delta
        self.jump_command[:, 3] = self.target_dx


@dataclass(kw_only=True)
class InPlaceJumpCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    period_s: float = 2.0
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


def landing_stability_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    xy_vel_std: float = 0.35,
    xy_drift_std: float = 0.12,
    upright_std: float = 0.35,
    phase_range: tuple[float, float] = (0.70, 1.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    phase = _jump_phase(env, command_name)
    mask = _phase_mask(phase, *phase_range)
    asset: Entity = env.scene[asset_cfg.name]

    all_contact = _foot_contact(env, sensor_name).all(dim=1).float()
    xy_vel = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=1)
    xy_drift = torch.norm(horizontal_displacement(env, command_name, asset_cfg), dim=1)
    tilt = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    xy_vel_score = torch.exp(-torch.square(xy_vel) / (xy_vel_std**2))
    xy_drift_score = torch.exp(-torch.square(xy_drift) / (xy_drift_std**2))
    upright_score = torch.exp(-tilt / (upright_std**2))
    score = all_contact * xy_vel_score * xy_drift_score * upright_score

    env.extras["log"]["Metrics/landing_stable"] = torch.mean(score)
    return mask * score


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
