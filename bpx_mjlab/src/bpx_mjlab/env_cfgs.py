from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

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
