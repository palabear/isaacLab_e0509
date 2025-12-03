#!/usr/bin/env python3
"""
Convert e0509 URDF to USD with proper physics setup for Isaac Sim.
"""

import argparse
from pathlib import Path

# SimulationApp을 가장 먼저 인스턴스화해야 함
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# 이제 다른 Omniverse/Isaac Sim 모듈 import 가능
import omni.isaac.core.utils.stage as stage_utils
from isaacsim.asset.importer.urdf import _urdf


def convert_urdf_to_usd(
    urdf_path: str,
    output_path: str,
    fix_base: bool = True,
    merge_fixed_joints: bool = False,
):
    """Convert URDF to USD with physics."""
    
    print(f"Converting URDF to USD:")
    print(f"  Input: {urdf_path}")
    print(f"  Output: {output_path}")
    print(f"  Fix base: {fix_base}")
    print(f"  Merge fixed joints: {merge_fixed_joints}")
    
    # Create new stage
    stage_utils.create_new_stage()
    
    # Import URDF configuration
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = merge_fixed_joints
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = fix_base
    import_config.make_default_prim = True
    import_config.create_physics_scene = True
    import_config.distance_scale = 1.0
    import_config.density = 0.0
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.default_drive_strength = 800.0  # Stiffness
    import_config.default_position_drive_damping = 40.0  # Damping
    
    # Get URDF interface and import
    urdf_interface = _urdf.acquire_urdf_interface()
    
    # Import URDF into stage - 새로운 API는 prim_path, urdf_path, config 순서
    success = urdf_interface.parse_urdf(
        "/World/e0509",  # USD stage에서 생성될 경로
        urdf_path,        # URDF 파일 경로
        import_config     # Import configuration
    )
    
    if not success:
        print("ERROR: Failed to import URDF!")
        return False
    
    # Save USD
    stage = stage_utils.get_current_stage()
    success = stage.GetRootLayer().Export(output_path)
    
    if success:
        print(f"✓ Successfully saved USD to: {output_path}")
        return True
    else:
        print("ERROR: Failed to save USD!")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf",
        type=str,
        default="/home/jiwoo/isaacsim_ws/doosan-robot2/dsr_description2/urdf/e0509old.urdf",
        help="Path to URDF file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/jiwoo/isaacsim_ws/e0509_converted.usd",
        help="Output USD path"
    )
    parser.add_argument("--fix-base", action="store_true", default=True)
    parser.add_argument("--merge-fixed", action="store_true", default=False)
    
    args = parser.parse_args()
    
    convert_urdf_to_usd(
        args.urdf,
        args.output,
        args.fix_base,
        args.merge_fixed,
    )
    
    # SimulationApp 종료
    simulation_app.close()
