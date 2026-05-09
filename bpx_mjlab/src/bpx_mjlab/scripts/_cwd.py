from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def find_workspace_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / ".git").exists() and (parent / "bpx_mjlab").exists():
            return parent
    return Path.cwd().resolve()


def run_from_workspace_root(main: Callable[[], object]) -> object:
    root = find_workspace_root()
    os.environ.setdefault("WANDB_DIR", str(root / "wandb"))
    os.environ.setdefault("WANDB_DATA_DIR", str(root / "wandb"))
    os.chdir(root)
    return main()
