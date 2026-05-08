from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from dataclasses import replace

from bpx_mjlab import jump_mdp
from bpx_mjlab.bpx.bpx_constants import (
    BPX_ACTION_SCALE,
    FOOT_GEOMS,
    FOOT_SITES,
    get_bpx_robot_cfg,
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


def _safe_pop_term(term_dict, key: str) -> None:
    if term_dict is not None and key in term_dict:
        term_dict.pop(key, None)


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

    # 平地训练不需要 raycast 扫描。
    # 默认 velocity task 里可能有 terrain_scan / foot_height_scan。
    # BPX 当前没有给这些 raycast sensor 配 frame，保留会导致：
    # RuntimeError: stack expects a non-empty TensorList
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ())
        if s.name not in ("terrain_scan", "foot_height_scan")
    )

    # 同时删除依赖这些 raycast sensor 的 observation term。
    for obs_group in cfg.observations.values():
        terms = getattr(obs_group, "terms", None)
        if not isinstance(terms, dict):
            continue

        for term_name, term_cfg in list(terms.items()):
            params = getattr(term_cfg, "params", {})
            sensor_name = params.get("sensor_name") if isinstance(params, dict) else None

            if (
                term_name in ("height_scan", "foot_height", "foot_height_scan")
                or sensor_name in ("terrain_scan", "foot_height_scan")
            ):
                terms.pop(term_name, None)

    cfg.curriculum.pop("terrain_levels", None)

    # 删除仍然引用 raycast sensor 的 reward term。
    # 只把 weight 设成 0 不够，因为 RewardManager 会先初始化 term。
    for reward_name, term_cfg in list(cfg.rewards.items()):
        params = getattr(term_cfg, "params", {})
        if not isinstance(params, dict):
            continue

        should_remove = False

        # 常见显式字段。
        for key in (
            "height_sensor_name",
            "sensor_name",
            "raycast_sensor_name",
            "terrain_sensor_name",
        ):
            if params.get(key) in ("terrain_scan", "foot_height_scan"):
                should_remove = True

        # 兜底：params 里任何值直接等于这两个名字，也删。
        for value in params.values():
            if value in ("terrain_scan", "foot_height_scan"):
                should_remove = True

        if should_remove:
            print(f"remove reward term using raycast sensor: {reward_name}")
            cfg.rewards.pop(reward_name, None)


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

    cfg.episode_length_s = int(1e9) if play else 8.0
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
            resampling_time_range=(2.0, 2.0),
            period_s=2.0,
            target_height_delta_range=(0.12, 0.22),
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
            weight=2.0,
            params={"command_name": "jump"},
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
            },
        ),
        "peak_height": RewardTermCfg(
            func=jump_mdp.peak_height_reward,
            weight=2.5,
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
        "air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.15,
            params={
                "sensor_name": "feet_ground_contact",
                "threshold_min": 0.08,
                "threshold_max": 0.65,
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-2.0e-5,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    if play:
        if "actor" in cfg.observations:
            cfg.observations["actor"].enable_corruption = False

    return cfg


def bpx_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = bpx_flat_env_cfg(play=play)

    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    from mjlab.tasks.velocity import mdp as vel_mdp

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = replace(ROUGH_TERRAINS_CFG)
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.scene.extent = 3.0

    cfg.sim.nconmax = 256
    cfg.sim.njmax = 2000
    cfg.sim.contact_sensor_maxmatch = 128

    bad_sensor_names = (
        "terrain_scan",
        "foot_height_scan",
    )
    bad_keywords = (
        "terrain_scan",
        "foot_height_scan",
        "height_scan",
        "foot_height",
    )

    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ())
        if s.name not in bad_sensor_names
    )

    for obs_group in cfg.observations.values():
        terms = getattr(obs_group, "terms", None)
        if not isinstance(terms, dict):
            continue

        for term_name, term_cfg in list(terms.items()):
            params = getattr(term_cfg, "params", {})
            func = getattr(term_cfg, "func", "")
            text = str(term_name) + " " + str(params) + " " + str(func)

            if any(keyword in text for keyword in bad_keywords):
                terms.pop(term_name, None)

    for reward_name, term_cfg in list(cfg.rewards.items()):
        params = getattr(term_cfg, "params", {})
        func = getattr(term_cfg, "func", "")
        text = str(reward_name) + " " + str(params) + " " + str(func)

        if any(keyword in text for keyword in bad_keywords):
            cfg.rewards.pop(reward_name, None)

    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=vel_mdp.terrain_levels_vel,
        params={"command_name": "twist"},
    )
    cfg.curriculum.pop("command_vel", None)

    cmd = cfg.commands["twist"]
    assert isinstance(cmd, UniformVelocityCommandCfg)
    cmd.ranges.lin_vel_x = (-0.35, 1.0)
    cmd.ranges.lin_vel_y = (-0.2, 0.2)
    cmd.ranges.ang_vel_z = (-0.45, 0.45)

    if hasattr(cmd, "rel_standing_envs"):
        cmd.rel_standing_envs = 0.02

    if hasattr(cmd, "rel_forward_envs"):
        cmd.rel_forward_envs = 0.5

    if hasattr(cmd, "resampling_time_range"):
        cmd.resampling_time_range = (4.0, 8.0)

    cfg.events.pop("push_robot", None)

    if play:
        cfg.episode_length_s = int(1e9)

        if "actor" in cfg.observations:
            cfg.observations["actor"].enable_corruption = False

        cfg.events.pop("push_robot", None)

    return cfg
