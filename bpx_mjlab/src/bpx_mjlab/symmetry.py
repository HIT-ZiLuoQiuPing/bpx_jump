from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict


_FOOT_PAIR_INDEX = torch.tensor((1, 0, 3, 2), dtype=torch.long)


def _base_env(env):
    return getattr(env, "unwrapped", env)


def _mirror_leg_name(name: str) -> str:
    pairs = (
        ("fl_", "fr_"),
        ("fr_", "fl_"),
        ("hl_", "hr_"),
        ("hr_", "hl_"),
    )
    for src, dst in pairs:
        if name.startswith(src):
            return dst + name[len(src):]
    return name


def _joint_sign(name: str) -> float:
    return -1.0 if "hip_roll" in name else 1.0


@lru_cache(maxsize=16)
def _joint_mirror_map(names: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    index_by_name = {name: i for i, name in enumerate(names)}
    indices: list[int] = []
    signs: list[float] = []
    for name in names:
        mirror_name = _mirror_leg_name(name)
        indices.append(index_by_name.get(mirror_name, index_by_name[name]))
        signs.append(_joint_sign(name))
    return tuple(indices), tuple(signs)


def _mirror_polar_vec3(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 1] *= -1.0
    return y


def _mirror_axial_vec3(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 0] *= -1.0
    y[..., 2] *= -1.0
    return y


def _mirror_feet(x: torch.Tensor) -> torch.Tensor:
    index = _FOOT_PAIR_INDEX.to(device=x.device)
    return x.index_select(-1, index)


def _mirror_foot_vecs(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    y = x.reshape(*shape[:-1], 4, 3)
    index = _FOOT_PAIR_INDEX.to(device=x.device)
    y = y.index_select(-2, index)
    y = _mirror_polar_vec3(y)
    return y.reshape(shape)


def _mirror_joint_like(x: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
    indices, signs = _joint_mirror_map(names)
    index = torch.tensor(indices, device=x.device, dtype=torch.long)
    sign = torch.tensor(signs, device=x.device, dtype=x.dtype)
    return x.index_select(-1, index) * sign


def _action_target_names(env) -> tuple[str, ...]:
    base_env = _base_env(env)
    action_term = base_env.action_manager.get_term("joint_pos")
    return tuple(action_term.target_names)


def _joint_names(env) -> tuple[str, ...]:
    base_env = _base_env(env)
    return tuple(base_env.scene["robot"].joint_names)


def _term_slices(env, group_name: str) -> list[tuple[str, slice]]:
    base_env = _base_env(env)
    obs_manager = base_env.observation_manager
    names = obs_manager.active_terms[group_name]
    dims = obs_manager.group_obs_term_dim[group_name]

    offset = 0
    slices: list[tuple[str, slice]] = []
    for name, dim in zip(names, dims, strict=False):
        length = 1
        for value in dim:
            length *= int(value)
        slices.append((name, slice(offset, offset + length)))
        offset += length
    return slices


def _mirror_obs_group(env, group_name: str, value: torch.Tensor) -> torch.Tensor:
    mirrored = value.clone()
    joint_names = _joint_names(env)
    action_names = _action_target_names(env)

    for term_name, term_slice in _term_slices(env, group_name):
        term = mirrored[..., term_slice]

        if term_name in ("base_lin_vel", "projected_gravity"):
            mirrored[..., term_slice] = _mirror_polar_vec3(term)
        elif term_name == "base_ang_vel":
            mirrored[..., term_slice] = _mirror_axial_vec3(term)
        elif term_name in ("joint_pos", "joint_vel"):
            mirrored[..., term_slice] = _mirror_joint_like(term, joint_names)
        elif term_name == "actions":
            mirrored[..., term_slice] = _mirror_joint_like(term, action_names)
        elif term_name in ("foot_contact", "foot_air_time"):
            mirrored[..., term_slice] = _mirror_feet(term)
        elif term_name == "foot_contact_forces":
            mirrored[..., term_slice] = _mirror_foot_vecs(term)
        elif term_name == "xy_displacement":
            xy = term.clone()
            xy[..., 1] *= -1.0
            mirrored[..., term_slice] = xy

    return mirrored


def _mirror_actions(env, actions: torch.Tensor) -> torch.Tensor:
    return _mirror_joint_like(actions, _action_target_names(env))


def jump_mirror_data_augmentation(
    obs: "TensorDict | None" = None,
    actions: torch.Tensor | None = None,
    env=None,
) -> tuple["TensorDict | None", torch.Tensor | None]:
    """Left-right mirror augmentation for the in-place BPX jump task."""
    if env is None:
        raise ValueError("env is required for BPX jump mirror augmentation.")

    obs_aug = None
    if obs is not None:
        mirrored_obs = obs.clone()
        for group_name in obs.keys():
            mirrored_obs[group_name] = _mirror_obs_group(env, group_name, obs[group_name])
        obs_aug = torch.cat((obs, mirrored_obs), dim=0)

    actions_aug = None
    if actions is not None:
        mirrored_actions = _mirror_actions(env, actions)
        actions_aug = torch.cat((actions, mirrored_actions), dim=0)

    return obs_aug, actions_aug
