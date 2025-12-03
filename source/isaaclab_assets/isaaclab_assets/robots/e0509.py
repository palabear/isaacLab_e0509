# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the e0509 robotic arm with rh_p12_rn gripper."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import pathlib

_current_file = pathlib.Path(__file__).resolve()
_workspace_root = _current_file.parents[4]
_meshes_dir = _workspace_root / "meshes"
_usd_path = str(_meshes_dir / "e0509_model.usda")

E0509_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": -0.5,
            "joint_3": 0.8,
            "joint_4": 0.0,
            "joint_5": -0.3,
            "joint_6": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-6]"],
            effort_limit=194.0,
            velocity_limit=1.5,
            stiffness=800.0,
            damping=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["rh_l1", "rh_r1_joint", "rh_l2", "rh_r2"],
            effort_limit=400.0,  # 200.0 -> 400.0: 2x stronger grip force!
            velocity_limit=2.0,
            stiffness=2000.0,  # 500.0 -> 2000.0: 4x stiffer (firm grasp, no slipping)
            damping=100.0,  # 50.0 -> 100.0: 2x more stable
        ),
    },
)
