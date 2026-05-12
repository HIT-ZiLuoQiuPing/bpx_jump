from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from bpx_mjlab import jump_mdp
from bpx_mjlab.bpx.bpx_constants import (
    BPX_ACTION_SCALE,
    FOOT_GEOMS,
    FOOT_SITES,
    get_bpx_robot_cfg,
)


RAYCAST_SENSOR_NAMES = ("terrain_scan", "foot_height_scan")
RAYCAST_TERM_NAMES = ("height_scan", "foot_height", "foot_height_scan")
RAYCAST_PARAM_KEYS = (
    "height_sensor_name",
    "sensor_name",
    "raycast_sensor_name",
    "terrain_sensor_name",
)


def _safe_set_asset_names(term, field_name: str, names: tuple[str, ...]) -> bool:
    """
    兼容不同 mjlab 版本：
    有的 term.params 里是 asset_cfg.body_names/site_names/geom_names；
    有的版本可能直接是 body_name/body_names 等。
    这里能设就设，不能设就跳过，避免导入阶段直接 KeyError。
    """
    params = getattr(term, "params", None)
    if not isinstance(params, dict):
        return False

    asset_cfg = params.get("asset_cfg", None)
    if asset_cfg is not None and hasattr(asset_cfg, field_name):
        setattr(asset_cfg, field_name, names)
        return True

    if field_name in params:
        params[field_name] = names
        return True

    singular_map = {
        "body_names": "body_name",
        "site_names": "site_name",
        "geom_names": "geom_name",
    }
    singular = singular_map.get(field_name)
    if singular is not None and singular in params:
        params[singular] = names[0] if len(names) == 1 else names
        return True

    return False


def _term_references_raycast(term_name: str, term_cfg) -> bool:
    params = getattr(term_cfg, "params", {})
    if term_name in RAYCAST_TERM_NAMES:
        return True
    if not isinstance(params, dict):
        return False
    if any(params.get(key) in RAYCAST_SENSOR_NAMES for key in RAYCAST_PARAM_KEYS):
        return True
    return any(value in RAYCAST_SENSOR_NAMES for value in params.values())


def _remove_raycast_dependencies(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.scene.sensors = tuple(
        sensor
        for sensor in (cfg.scene.sensors or ())
        if sensor.name not in RAYCAST_SENSOR_NAMES
    )

    for obs_group in cfg.observations.values():
        terms = getattr(obs_group, "terms", None)
        if not isinstance(terms, dict):
            continue
        for term_name, term_cfg in list(terms.items()):
            if _term_references_raycast(term_name, term_cfg):
                terms.pop(term_name, None)

    for reward_name, term_cfg in list(cfg.rewards.items()):
        if _term_references_raycast(reward_name, term_cfg):
            cfg.rewards.pop(reward_name, None)

    cfg.curriculum.pop("terrain_levels", None)


def bpx_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    # 仿真接触参数先保守一点，避免接触过多时报错。
    cfg.sim.njmax = 300
    cfg.sim.nconmax = 128
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.mujoco.ccd_iterations = 50

    # 换成 BPX 机器人。
    cfg.scene.entities = {
        "robot": get_bpx_robot_cfg(),
    }

    # Viewer 跟随 BPX 主躯干。
    cfg.viewer.body_name = "torso"
    cfg.viewer.distance = 2.0
    cfg.viewer.elevation = -10.0

    # 平地训练。
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # BPX 平地/跳跃任务不使用 raycast，统一静默剥离默认 velocity 配置里的 raycast 依赖。
    _remove_raycast_dependencies(cfg)

    # 四个脚尖接触传感器。
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=FOOT_GEOMS,
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # 非脚部碰地传感器：用于判断摔倒/非法接触。
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=r".*_collision_\d+$",
            exclude=FOOT_GEOMS,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        nonfoot_ground_cfg,
    )

    # 动作：12 个关节的位置控制。
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = BPX_ACTION_SCALE

    # 尝试把默认 Go1/G1 的 body/site/geom 名字替换成 BPX。
    # 这里全部 safe，不存在就跳过，避免任务注册失败。
    if "foot_height" in cfg.observations.get("critic", {}).terms:
        _safe_set_asset_names(
            cfg.observations["critic"].terms["foot_height"],
            "site_names",
            FOOT_SITES,
        )

    if "foot_friction" in cfg.events:
        _safe_set_asset_names(
            cfg.events["foot_friction"],
            "geom_names",
            FOOT_GEOMS,
        )

    if "base_com" in cfg.events:
        _safe_set_asset_names(
            cfg.events["base_com"],
            "body_names",
            ("torso",),
        )

    if "pose" in cfg.rewards:
        cfg.rewards["pose"].params["std_standing"] = {
            ".*_hip_roll_joint": 0.05,
            ".*_hip_pitch_joint": 0.10,
            ".*_knee_joint": 0.10,
        }
        cfg.rewards["pose"].params["std_walking"] = {
            ".*_hip_roll_joint": 0.30,
            ".*_hip_pitch_joint": 0.30,
            ".*_knee_joint": 0.60,
        }
        cfg.rewards["pose"].params["std_running"] = {
            ".*_hip_roll_joint": 0.30,
            ".*_hip_pitch_joint": 0.30,
            ".*_knee_joint": 0.60,
        }

    if "upright" in cfg.rewards:
        _safe_set_asset_names(
            cfg.rewards["upright"],
            "body_names",
            ("torso",),
        )

    if "body_ang_vel" in cfg.rewards:
        _safe_set_asset_names(
            cfg.rewards["body_ang_vel"],
            "body_names",
            ("torso",),
        )

    for reward_name in ("foot_clearance", "foot_swing_height", "foot_slip"):
        if reward_name in cfg.rewards:
            _safe_set_asset_names(
                cfg.rewards[reward_name],
                "site_names",
                FOOT_SITES,
            )

    # 初期先关掉一些容易干扰的奖励，等能跑起来再慢慢加。
    for reward_name in ("body_ang_vel", "angular_momentum", "air_time"):
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].weight = 0.0

    # 非脚部碰地就终止。
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": nonfoot_ground_cfg.name},
    )

    # 速度命令可视化高度。
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, UniformVelocityCommandCfg)
    cmd.viz.z_offset = 0.5

    if play:
        cfg.episode_length_s = int(1e9)

        if "actor" in cfg.observations:
            cfg.observations["actor"].enable_corruption = False

        cfg.events.pop("push_robot", None)

    return cfg


def bpx_jump_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = bpx_flat_env_cfg(play=play)

    cfg.episode_length_s = int(1e9) if play else 4.0
    cfg.scene.extent = 2.5
    cfg.sim.nconmax = max(cfg.sim.nconmax, 192)
    cfg.sim.njmax = max(cfg.sim.njmax, 800)
    cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch, 96)

    # Jump training starts from the cleanest possible flat-ground setting.
    cfg.curriculum.clear()
    for event_name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
        cfg.events.pop(event_name, None)

    cfg.commands = {
        "jump": jump_mdp.InPlaceJumpCommandCfg(
            entity_name="robot",
            resampling_time_range=(4.0, 4.0),
            period_s=2.0,
            repeat=False,
            jump_duration_s=2.0,
            target_height_delta_range=(0.20, 0.30),
            target_dx_range=(0.0, 0.0),
            debug_vis=True,
        )
    }

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    base_scale = BPX_ACTION_SCALE[".*_hip_pitch_joint"]
    joint_pos_action.scale = {
        ".*_hip_roll_joint": 0.55 * base_scale,
        ".*_hip_pitch_joint": 1.25 * base_scale,
        ".*_knee_joint": 1.45 * base_scale,
    }
    jump_joint_names = (
        "fl_hip_roll_joint",
        "fl_hip_pitch_joint",
        "fl_knee_joint",
        "fr_hip_roll_joint",
        "fr_hip_pitch_joint",
        "fr_knee_joint",
        "hl_hip_roll_joint",
        "hl_hip_pitch_joint",
        "hl_knee_joint",
        "hr_hip_roll_joint",
        "hr_hip_pitch_joint",
        "hr_knee_joint",
    )

    for obs_group_name, obs_group in cfg.observations.items():
        terms = getattr(obs_group, "terms", None)
        if not isinstance(terms, dict):
            continue

        if "command" in terms:
            terms["command"].params["command_name"] = "jump"
        else:
            terms["command"] = ObservationTermCfg(
                func=mdp.generated_commands,
                params={"command_name": "jump"},
            )

        terms["base_height"] = ObservationTermCfg(func=jump_mdp.base_height)
        terms["base_height_delta"] = ObservationTermCfg(
            func=jump_mdp.base_height_delta,
            params={"command_name": "jump"},
        )
        terms["foot_contact"] = ObservationTermCfg(
            func=jump_mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        )

        if obs_group_name == "critic":
            terms["xy_displacement"] = ObservationTermCfg(
                func=jump_mdp.horizontal_displacement,
                params={"command_name": "jump"},
            )

    cfg.rewards = {
        "crouch": RewardTermCfg(
            func=jump_mdp.crouch_reward,
            weight=1.0,
            params={
                "command_name": "jump",
                "sensor_name": "feet_ground_contact",
            },
        ),
        "takeoff_velocity": RewardTermCfg(
            func=jump_mdp.takeoff_velocity_reward,
            weight=2.2,
            params={"command_name": "jump", "target_vz": 1.05},
        ),
        "contact_release": RewardTermCfg(
            func=jump_mdp.contact_release_reward,
            weight=0.8,
            params={
                "command_name": "jump",
                "sensor_name": "feet_ground_contact",
            },
        ),
        "flight_airborne": RewardTermCfg(
            func=jump_mdp.flight_airborne_reward,
            weight=1.5,
            params={
                "command_name": "jump",
                "sensor_name": "feet_ground_contact",
                "phase_range": (0.45, 0.62),
            },
        ),
        "peak_height": RewardTermCfg(
            func=jump_mdp.peak_height_reward,
            weight=2.5,
            params={"command_name": "jump"},
        ),
        "peak_height_overshoot": RewardTermCfg(
            func=jump_mdp.peak_height_overshoot_penalty,
            weight=-0.35,
            params={"command_name": "jump"},
        ),
        "landing_stability": RewardTermCfg(
            func=jump_mdp.landing_stability_reward,
            weight=1.5,
            params={
                "command_name": "jump",
                "sensor_name": "feet_ground_contact",
            },
        ),
        "landing_rebound": RewardTermCfg(
            func=jump_mdp.landing_rebound_penalty,
            weight=-1.2,
            params={
                "command_name": "jump",
                "sensor_name": "feet_ground_contact",
            },
        ),
        "post_landing_airborne": RewardTermCfg(
            func=jump_mdp.post_landing_airborne_penalty,
            weight=-0.9,
            params={
                "command_name": "jump",
                "sensor_name": "feet_ground_contact",
            },
        ),
        "landing_absorption": RewardTermCfg(
            func=jump_mdp.landing_absorption_reward,
            weight=1.0,
            params={
                "command_name": "jump",
                "sensor_name": "feet_ground_contact",
            },
        ),
        "post_landing_hold": RewardTermCfg(
            func=jump_mdp.post_landing_hold_reward,
            weight=0.9,
            params={
                "command_name": "jump",
                "sensor_name": "feet_ground_contact",
                "landing_phase_start": 0.86,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=jump_joint_names,
                    preserve_order=True,
                ),
            },
        ),
        "upright": RewardTermCfg(
            func=mdp.upright,
            weight=0.8,
            params={
                "std": 0.45,
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)),
            },
        ),
        "pose": RewardTermCfg(
            func=mdp.posture,
            weight=0.25,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "std": {
                    ".*_hip_roll_joint": 0.35,
                    ".*_hip_pitch_joint": 0.95,
                    ".*_knee_joint": 1.15,
                },
            },
        ),
        "xy_drift": RewardTermCfg(
            func=jump_mdp.xy_drift_penalty,
            weight=-2.0,
            params={"command_name": "jump"},
        ),
        "left_right_joint_symmetry": RewardTermCfg(
            func=jump_mdp.left_right_joint_symmetry_penalty,
            weight=-0.8,
            params={
                "command_name": "jump",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=jump_joint_names,
                    preserve_order=True,
                ),
            },
        ),
        "left_right_action_symmetry": RewardTermCfg(
            func=jump_mdp.left_right_action_symmetry_penalty,
            weight=-0.25,
            params={"command_name": "jump"},
        ),
        "horizontal_velocity": RewardTermCfg(
            func=jump_mdp.horizontal_velocity_penalty,
            weight=-0.4,
        ),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=-0.02,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso",))},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.08),
        "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.015),
        "air_time": RewardTermCfg(
            func=jump_mdp.phase_gated_air_time_reward,
            weight=0.01,
            params={
                "command_name": "jump",
                "sensor_name": "feet_ground_contact",
                "threshold_min": 0.08,
                "threshold_max": 0.55,
                "phase_range": (0.38, 0.56),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-1.0e-4,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    if play:
        if "actor" in cfg.observations:
            cfg.observations["actor"].enable_corruption = False

    return cfg


def bpx_jump_directional_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = bpx_jump_flat_env_cfg(play=play)

    cfg.scene.extent = 3.5
    jump_cmd = cfg.commands["jump"]
    assert isinstance(jump_cmd, jump_mdp.InPlaceJumpCommandCfg)
    jump_cmd.target_height_delta_range = (0.18, 0.28)
    jump_cmd.target_dx_range = (0.0, 0.0)
    jump_cmd.target_dx_abs_range = (0.12, 0.28)

    cfg.rewards["directional_takeoff_velocity"] = RewardTermCfg(
        func=jump_mdp.directional_takeoff_velocity_reward,
        weight=1.0,
        params={
            "command_name": "jump",
            "phase_range": (0.24, 0.48),
            "target_time": 0.55,
            "velocity_std": 0.35,
            "max_target_vx": 0.75,
        },
    )
    cfg.rewards["target_dx_tracking"] = RewardTermCfg(
        func=jump_mdp.target_dx_tracking_reward,
        weight=2.0,
        params={
            "command_name": "jump",
            "phase_range": (0.55, 1.0),
            "x_std": 0.08,
            "y_std": 0.06,
        },
    )
    cfg.rewards["landing_stability"] = RewardTermCfg(
        func=jump_mdp.directional_landing_stability_reward,
        weight=1.5,
        params={
            "command_name": "jump",
            "sensor_name": "feet_ground_contact",
            "x_std": 0.10,
            "y_std": 0.07,
        },
    )
    cfg.rewards["xy_drift"] = RewardTermCfg(
        func=jump_mdp.lateral_drift_penalty,
        weight=-2.0,
        params={"command_name": "jump"},
    )
    cfg.rewards["horizontal_velocity"] = RewardTermCfg(
        func=jump_mdp.lateral_velocity_penalty,
        weight=-0.35,
    )

    return cfg
