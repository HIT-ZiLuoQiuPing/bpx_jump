from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


_HERE = Path(__file__).parent
BPX_XML = _HERE / "xmls" / "bpx.xml"
assert BPX_XML.exists(), BPX_XML


def _load_assets(asset_dir: Path, meshdir: str | None = None) -> dict[str, bytes]:
    """Load mesh files into MuJoCo's in-memory asset dict.

    新版 mjlab 没有 update_assets，所以这里自己实现一个小版本。
    同时注册两种 key：
      - xxx.obj
      - assets/xxx.obj
    这样对 MuJoCo 的 meshdir 解析更稳。
    """
    if not asset_dir.exists():
        raise FileNotFoundError(f"Asset directory not found: {asset_dir}")

    assets: dict[str, bytes] = {}

    for path in asset_dir.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(asset_dir).as_posix()
        data = path.read_bytes()

        # key 1: bpx_body_001.obj
        assets[rel] = data

        # key 2: assets/bpx_body_001.obj
        if meshdir:
            meshdir_clean = str(meshdir).strip("/\\")
            assets[f"{meshdir_clean}/{rel}"] = data

    return assets


def get_assets(meshdir: str) -> dict[str, bytes]:
    asset_dir = BPX_XML.parent / "assets"
    return _load_assets(asset_dir, meshdir)


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(BPX_XML))

    # bpx.xml 里我们已经把 meshdir 改成 assets。
    # 这里把 mesh 文件读进 spec.assets，方便 mjlab 后续组合场景。
    spec.assets = get_assets(spec.meshdir)

    return spec


EFFORT_LIMIT = 30.0

ARMATURE = 0.005
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2.0 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ


BPX_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_hip_roll_joint",
        ".*_hip_pitch_joint",
        ".*_knee_joint",
    ),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
)


INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.42),
    joint_pos={
        ".*_hip_roll_joint": 0.0,
        ".*_hip_pitch_joint": 0.6,
        ".*_knee_joint": -1.2,
    },
    joint_vel={".*": 0.0},
)


FOOT_GEOMS = (
    "fl_toe_link_collision_0",
    "fr_toe_link_collision_0",
    "hl_toe_link_collision_0",
    "hr_toe_link_collision_0",
)

FOOT_SITES = (
    "fl_foot",
    "fr_foot",
    "hl_foot",
    "hr_foot",
)


FULL_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision_\d+$",),
    condim=3,
    priority=1,
    friction=(0.6,),
)


BPX_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(BPX_ACTUATOR_CFG,),
    soft_joint_pos_limit_factor=0.9,
)


def get_bpx_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=BPX_ARTICULATION,
    )


BPX_ACTION_SCALE: dict[str, float] = {}
for actuator in BPX_ARTICULATION.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    for name_expr in actuator.target_names_expr:
        BPX_ACTION_SCALE[name_expr] = 0.25 * EFFORT_LIMIT / STIFFNESS


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_bpx_robot_cfg())
    viewer.launch(robot.spec.compile())
