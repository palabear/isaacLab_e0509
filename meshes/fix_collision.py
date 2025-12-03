#!/usr/bin/env python3
"""Fix USD files to properly set convex hull collision approximation."""

from pxr import Usd, UsdGeom, UsdPhysics
import os

def fix_usd_collision(usd_path):
    """Fix collision approximation in USD file."""
    print(f"Processing: {usd_path}")
    
    # Open the stage
    stage = Usd.Stage.Open(usd_path)
    
    # Get the default prim (should be the root object)
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        print(f"  ERROR: No default prim found")
        return False
    
    # Find the mesh prim
    mesh_prim = None
    for child in default_prim.GetChildren():
        if child.IsA(UsdGeom.Mesh):
            mesh_prim = child
            break
    
    if not mesh_prim:
        print(f"  ERROR: No mesh prim found")
        return False
    
    print(f"  Found mesh: {mesh_prim.GetPath()}")
    
    # Apply PhysicsMeshCollisionAPI if not already applied
    if not UsdPhysics.MeshCollisionAPI(mesh_prim):
        collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        print(f"  Applied PhysicsMeshCollisionAPI")
    else:
        collision_api = UsdPhysics.MeshCollisionAPI(mesh_prim)
        print(f"  PhysicsMeshCollisionAPI already applied")
    
    # Set approximation to convexHull
    approx_attr = collision_api.GetApproximationAttr()
    if not approx_attr:
        approx_attr = collision_api.CreateApproximationAttr()
    approx_attr.Set("convexHull")
    print(f"  Set approximation to convexHull")
    
    # Save the stage
    stage.Save()
    print(f"  Saved: {usd_path}")
    return True

def main():
    # List of USD files to fix
    usd_files = [
        "sanitizer_converted.usda",
        "water_bottle_converted.usda",
        "syringe_converted.usda",
        "medicine_cabinet_converted.usda",
        "tissue_converted.usda",
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="* 80)
    print("Fixing USD collision approximation")
    print("=" * 80)
    
    for usd_file in usd_files:
        usd_path = os.path.join(script_dir, usd_file)
        if os.path.exists(usd_path):
            fix_usd_collision(usd_path)
            print()
        else:
            print(f"WARNING: File not found: {usd_path}\n")
    
    print("=" * 80)
    print("Done!")
    print("=" * 80)

if __name__ == "__main__":
    main()
