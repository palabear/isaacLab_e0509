#!/usr/bin/env python3
"""Test script to check actual gripper joint values."""

import torch
from isaaclab.app import AppLauncher

# Create launcher
launcher = AppLauncher(headless=False)
simulation_app = launcher.app

from isaaclab_assets.robots.e0509 import E0509_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# Scene config
scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)

# Create scene
scene = InteractiveScene(scene_cfg)

# Spawn robot
robot_cfg = E0509_CFG.replace(prim_path="/World/envs/env_0/Robot")
robot = Articulation(robot_cfg)
scene.articulations["robot"] = robot

# Spawn ground
ground_cfg = sim_utils.GroundPlaneCfg()
ground_cfg.func("/World/ground", ground_cfg)

scene.reset()

print("\n" + "="*60)
print("E0509 Gripper Joint Range Test")
print("="*60)

# Get gripper joint indices
gripper_joint_names = ["rh_l1", "rh_l2", "rh_r1_joint", "rh_r2"]
all_joint_names = robot.data.joint_names

for i, name in enumerate(all_joint_names):
    if "rh_" in name:
        print(f"Joint {i}: {name}")

print("\n1. Initial State (Open):")
robot.write_joint_state_to_sim(
    torch.zeros((1, robot.num_joints), device=robot.device),
    torch.zeros((1, robot.num_joints), device=robot.device)
)
scene.write_data_to_sim()
for _ in range(10):
    simulation_app.update()

gripper_pos = robot.data.joint_pos[0, -4:]
print(f"Gripper positions: {gripper_pos.cpu().tolist()}")
print(f"Min: {gripper_pos.min().item():.4f}, Max: {gripper_pos.max().item():.4f}")

print("\n2. Setting to 1.1 radians:")
target_pos = torch.zeros((1, robot.num_joints), device=robot.device)
target_pos[0, -4:] = 1.1
robot.write_joint_state_to_sim(target_pos, torch.zeros_like(target_pos))
scene.write_data_to_sim()
for _ in range(50):
    simulation_app.update()

gripper_pos = robot.data.joint_pos[0, -4:]
print(f"Gripper positions: {gripper_pos.cpu().tolist()}")
print(f"Min: {gripper_pos.min().item():.4f}, Max: {gripper_pos.max().item():.4f}")

print("\n3. Setting to 63.0 (degrees?):")
target_pos[0, -4:] = 63.0
robot.write_joint_state_to_sim(target_pos, torch.zeros_like(target_pos))
scene.write_data_to_sim()
for _ in range(50):
    simulation_app.update()

gripper_pos = robot.data.joint_pos[0, -4:]
print(f"Gripper positions: {gripper_pos.cpu().tolist()}")
print(f"Min: {gripper_pos.min().item():.4f}, Max: {gripper_pos.max().item():.4f}")

print("\n4. Joint limits from USD:")
print("rh_r1_joint: lowerLimit=-0.0573, upperLimit=63.08265")
print("Expected: 0 rad = open, ~1.1 rad = closed")

print("\n" + "="*60)
print("Press Ctrl+C to exit")
print("="*60)

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
