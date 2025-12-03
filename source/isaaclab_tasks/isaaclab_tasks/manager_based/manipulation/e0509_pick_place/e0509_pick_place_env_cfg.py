# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
E0509 robot pick and place task configuration.

This environment implements a pick and place task where the e0509 robot arm must:
1. Reach and grasp an object
2. Lift the object
3. Move it to a goal location
4. Place it down
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg  # HIGH FRICTION!
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import pathlib

import isaaclab.envs.mdp as mdp
from isaaclab_assets.robots.e0509 import E0509_CFG

# Import local custom gripper action
from isaaclab_tasks.manager_based.manipulation.e0509_pick_place.mdp.gripper_action import MimicGripperActionCfg

from . import mdp as e0509_mdp

# 메시 파일 경로 설정
_current_file = pathlib.Path(__file__).resolve()
_workspace_root = _current_file.parents[6]  # isaaclab 워크스페이스 루트
_meshes_dir = _workspace_root / "meshes"

def _get_mesh_path(mesh_name: str) -> str:
    """메시 파일 경로를 반환합니다."""
    return str(_meshes_dir / mesh_name)

##
# Scene definition
##


@configclass
class E0509PickPlaceSceneCfg(InteractiveSceneCfg):
    """Configuration for the e0509 pick and place scene."""

    # Robot (e0509_model.usda) - Mounted on table
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_get_mesh_path("e0509_model.usda"),
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=True,  # ContactSensor 활성화
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.96, 0.095, -0.95), rot=(1.0, 0.0, 0.0, 0.0)),
        actuators=E0509_CFG.actuators,
    )

    # Table (table.usda) - On the ground
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05), rot=(0.7071, 0.7071, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=_get_mesh_path("table.usda"),
            scale=(0.1, 0.1, 0.1),
        ),
    )

    # # Object 1: Sanitizer (USD mesh)
    # sanitizer = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Sanitizer",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=_get_mesh_path("sanitizer_converted.usda"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             rigid_body_enabled=True,
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=4,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             collision_enabled=True,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.9, 0.0, -0.95),
    #         rot=(1.0, 1.0, 0.0, 0.0),
    #     ),
    # )

    # # Object 2: Water Bottle (USD mesh)
    # water_bottle = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/WaterBottle",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=_get_mesh_path("water_bottle_converted.usda"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             rigid_body_enabled=True,
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=4,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             collision_enabled=True,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.20),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.7, -0.2, -0.95),
    #         rot=(1.0, 1.0, 0.0, 0.0),
    #     ),
    # )

    # # Object 3: Syringe (USD mesh)
    # syringe = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Syringe",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=_get_mesh_path("syringe_converted.usda"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             rigid_body_enabled=True,
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=4,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             collision_enabled=True,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.85, 0.2, -0.95),
    #         rot=(1.0, 1.0, 0.0, 0.0),
    #     ),
    # )

    # Object 4: Medicine cabinet (USD mesh)
    # Curriculum Learning: Heavy mass to prevent pushing (gripper friction increased via actuator tuning)
    medicine_cabinet = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MedicineCabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_get_mesh_path("medicine_cabinet_converted.usda"),
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=16,  # High iteration for stable contact
                solver_velocity_iteration_count=4,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.5),  # 0.25kg → 2.5kg (10x heavier)
            # Note: USD file friction is handled in meshes/medicine_cabinet_converted.usda
            # Gripper friction comes from actuator stiffness (500→2000) + effort (200→400)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.2, -0.95),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # # Object 5: Tissue box (USD mesh)
    # tissue = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Tissue",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=_get_mesh_path("tissue_converted.usda"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             rigid_body_enabled=True,
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=4,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             collision_enabled=True,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.5, 0.2, -0.95),
    #         rot=(1.0, 1.57, 0.0, 0.0),
    #     ),
    # )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # End-effector frame transformer - gripper/gripper/base를 reference로!
    # offset: gripper base에서 end_effector_mesh까지 거리
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/e0509/gripper/gripper/base",  # gripper base가 reference
        debug_vis= False,  # 시각화 켜서 offset 확인
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/e0509/gripper/gripper/base",  # 자기자신
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, -0.2, 0.0)),  # base에서 end_effector_mesh까지 Z방향 13cm (추정)
            ),
        ],
    )
    
    # Contact sensor: 로봇팔-물체 충돌 감지 (그리퍼 제외)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/e0509/.*",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/MedicineCabinet",  # 물체와의 충돌만 감지
        ],
    )
##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Arm action: 6 DOF joint positions (continuous)
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_[1-6]"],
        scale=0.05,  # Very small - slow, precise movements to prevent trembling
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # Target object position relative to robot
        object_position = ObsTerm(func=e0509_mdp.object_position_in_robot_root_frame)

        # Previous actions
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # 1. 접근 및 자세 (가장 기본)
    # pregrasp_height=0.15 -> 물체 위 15cm 공중을 목표로 함 (충돌 방지)
    approach_and_orient = RewTerm(
        func=e0509_mdp.approach_and_orient_reward,
        weight=20.0,  # 10.0 → 20.0 (자세 학습 강화!)
        params={
            "pregrasp_height": 0.15,  # [중요] 0.0 아님! 공중부양 유도
            "orientation_strictness": 8.0,  # 4.0 → 8.0 (훨씬 더 엄격하게!)
            "object_cfg": SceneEntityCfg("medicine_cabinet"),
        },
    )
    
    # 2. 그리퍼 제어 (전략적)
    # 가까이 가면 닫고, 멀면 염
    grasp_encourage = RewTerm(
        func=e0509_mdp.gripper_close_encourage,
        weight=5.0,
        params={
            "switch_dist": 0.04,  # 4cm 앞에서 닫기 시작
        },
    )
    
    # 3. 최종 성공 (Jackpot)
    # 성공 시 아주 큰 점수를 줘서, 앞의 과정들이 의미 있음을 알려줌
    lift_success = RewTerm(
        func=e0509_mdp.lift_success_bonus,
        weight=25.0, # 성공 보상은 무조건 커야 함
        params={
            "lift_height": 0.05, 
            "gripper_threshold": 0.5,
        },
    )
    
    # 4. 페널티 (최소화)
    # 너무 세게 때리는 것만 방지
    object_stability = RewTerm(
        func=e0509_mdp.object_stability_penalty,
        weight=-2.0,
        params={
            "velocity_threshold": 0.15,
            "safe_distance": 0.10,
        },
    )
    
    # 5. 동작 안정화
    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.01,
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Object out of bounds (fell off table or flew away)
    object_out_of_bounds = DoneTerm(
        func=e0509_mdp.object_out_of_bounds,
        params={"x_limit": 1.5, "y_limit": 1.5, "z_min": -1.2, "z_max": 0.5},
    )
    
    # 로봇팔-물체 충돌 (그리퍼 제외)
    arm_collision = DoneTerm(
        func=e0509_mdp.arm_object_collision,
        params={
            "force_threshold": 5.0,  # 5N 이상의 힘 감지 시 충돌
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )


@configclass
class EventCfg:
    """Configuration for randomization events."""

    # Reset all entities to default state (fixed positions)
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    # Reset gripper to open position at start of each episode
    reset_gripper = EventTerm(func=e0509_mdp.reset_gripper_to_open, mode="reset")

    
    # Randomize target object selection (which of 5 objects to pick)
    randomize_target = EventTerm(
        func=e0509_mdp.randomize_target_object,
        mode="reset",
    )


##
# Environment configuration
##


@configclass
class E0509PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the e0509 pick and place environment."""

    # Scene settings
    scene: E0509PickPlaceSceneCfg = E0509PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4  # Action every 4 physics steps (0.04s at 100Hz)
        self.episode_length_s = 15.0  # Increased from 10s to give more time to learn
        
        # Simulation settings - increased precision for stability
        self.sim.dt = 1.0 / 120.0  # 120Hz physics simulation (increased from 100Hz)
        # Note: substeps is controlled by PhysX scene settings, not SimulationCfg
        self.sim.render_interval = self.decimation
        
        # Viewer settings - higher resolution for better visualization
        self.viewer.resolution = (1920, 1080)  # Full HD resolution
        self.viewer.eye = (2.5, 2.5, 2.5)  # Camera position
        self.viewer.lookat = (0.0, 0.0, 0.0)  # Look at center
        
        # PhysX settings for better stability
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
