# BPX Robot Model Assets

This repository contains open-source robot description assets for the `BPX` platform.

It is intended for developers who need model files for visualization, simulation setup, and integration research.

## Included Assets

- `bpx/urdf/bpx.urdf`: main URDF model entry
- `bpx/meshes/`: mesh assets referenced by the URDF model
- `mujoco/bpx.xml`: MuJoCo model entry
- `mujoco/meshes/`: mesh assets referenced by the MuJoCo model
- `usd/config.yaml`: USD conversion configuration used for asset generation workflows

## Directory Layout

```text
BPX/
├── bpx/
│   ├── meshes/
│   └── urdf/
├── mujoco/
│   ├── meshes/
│   └── bpx.xml
└── usd/
    └── config.yaml
```

## Quick Start

### URDF

Use `bpx/urdf/bpx.urdf` as the primary entry when loading the robot in URDF-compatible tools such as RViz, planning pipelines, or custom parsers.

Make sure the relative mesh paths remain unchanged when copying or packaging the asset directory.

### MuJoCo

Use `mujoco/bpx.xml` as the entry file for MuJoCo-based simulation and validation workflows.

The model references meshes from the local `mujoco/meshes/` directory.

## Notes

- Asset paths inside the model files are relative to their current directory layout.
- Joint limits, collision geometry, and inertial properties are already included in the published model files.
- Review compatibility and coordinate conventions in your target toolchain before integration.

## License And Usage

This repository is released under the `BSD-3-Clause` license. See `LICENSE`
for the full license text.
