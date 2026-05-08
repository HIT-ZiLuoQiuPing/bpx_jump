import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from bpx_mjlab.env_cfgs import bpx_rough_env_cfg


def has_bad(x):
    if not torch.is_tensor(x):
        return False
    return torch.isnan(x).any().item() or torch.isinf(x).any().item()


def print_bad_tensor(name, x):
    print(f"\n[BAD] {name}")
    print("shape:", tuple(x.shape))
    print("nan:", torch.isnan(x).sum().item())
    print("inf:", torch.isinf(x).sum().item())
    finite = torch.isfinite(x)
    if finite.any():
        xf = x[finite]
        print("finite min:", xf.min().item())
        print("finite max:", xf.max().item())


cfg = bpx_rough_env_cfg(play=False)
cfg.scene.num_envs = 16

# 先尽量关掉训练随机扰动，排查环境本体。
cfg.events.pop("push_robot", None)
if "actor" in cfg.observations:
    cfg.observations["actor"].enable_corruption = False

env = ManagerBasedRlEnv(
    cfg=cfg,
    device="cuda:0",
    render_mode=None,
)

env.reset()

print("=== actor observation terms ===")
for name, term_cfg in cfg.observations["actor"].terms.items():
    print("-", name, getattr(term_cfg, "params", {}))

print("\n=== check actor terms at reset ===")
for name, term_cfg in cfg.observations["actor"].terms.items():
    try:
        out = term_cfg.func(env, **term_cfg.params)
        if has_bad(out):
            print_bad_tensor(f"actor.{name}", out)
        else:
            print(f"[OK] actor.{name}: {tuple(out.shape)}")
    except Exception as e:
        print(f"[ERR] actor.{name}: {e}")

# 再 step 几步，看是不是一步之后炸
print("\n=== step with zero actions ===")
action_dim = 12
actions = torch.zeros((cfg.scene.num_envs, action_dim), device=env.device)

for i in range(20):
    obs, reward, terminated, truncated, extras = env.step(actions)

    actor_obs = obs["actor"] if isinstance(obs, dict) and "actor" in obs else obs

    if has_bad(actor_obs):
        print_bad_tensor(f"full actor obs after step {i}", actor_obs)

        print("\n=== check each actor term after bad step ===")
        for name, term_cfg in cfg.observations["actor"].terms.items():
            try:
                out = term_cfg.func(env, **term_cfg.params)
                if has_bad(out):
                    print_bad_tensor(f"actor.{name}", out)
                else:
                    print(f"[OK] actor.{name}: {tuple(out.shape)}")
            except Exception as e:
                print(f"[ERR] actor.{name}: {e}")
        break
else:
    print("No NaN/Inf found in 20 zero-action steps.")
