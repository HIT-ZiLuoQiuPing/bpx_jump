from __future__ import annotations

from mjlab.scripts.train import main as mjlab_train_main

from bpx_mjlab.scripts._cwd import run_from_workspace_root


def main() -> object:
    return run_from_workspace_root(mjlab_train_main)
