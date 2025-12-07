# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL with policy transfer capabilities and ROS2 publishing."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from scripts.reinforcement_learning.rsl_rl import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL with policy transfer and ROS2 publishing.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# Joint ordering arguments
parser.add_argument(
    "--policy_transfer_file",
    type=str,
    default=None,
    help="Path to YAML file containing joint mapping configuration for policy transfer between physics engines.",
)
# ROS2 arguments
parser.add_argument("--publish_ros2", action="store_true", default=False, help="Publish robot state to ROS2.")
parser.add_argument("--ros2_env_id", type=int, default=0, help="Environment ID to publish (default: 0, first env)")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import yaml
import numpy as np

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# ROS2 imports (optional, only imported when --publish_ros2 is True)
if args_cli.publish_ros2:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
        from geometry_msgs.msg import PoseStamped
        from builtin_interfaces.msg import Time as RosTime
        ROS2_AVAILABLE = True
    except ImportError:
        print("[WARNING] ROS2 not available. Install rclpy to publish robot state to ROS2.")
        ROS2_AVAILABLE = False
else:
    ROS2_AVAILABLE = False

# PLACEHOLDER: Extension template (do not remove this comment)


def get_joint_mappings(args_cli, action_space_dim):
    """Get joint mappings based on command line arguments.

    Args:
            args_cli: Command line arguments
            action_space_dim: Dimension of the action space (number of joints)

    Returns:
            tuple: (source_to_target_list, target_to_source_list, source_to_target_obs_list)
    """
    num_joints = action_space_dim
    if args_cli.policy_transfer_file:
        # Load from YAML file
        try:
            with open(args_cli.policy_transfer_file) as file:
                config = yaml.safe_load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load joint mapping from {args_cli.policy_transfer_file}: {e}")

        source_joint_names = config["source_joint_names"]
        target_joint_names = config["target_joint_names"]
        # Find joint mapping
        source_to_target = []
        target_to_source = []

        # Create source to target mapping
        for joint_name in source_joint_names:
            if joint_name in target_joint_names:
                source_to_target.append(target_joint_names.index(joint_name))
            else:
                raise ValueError(f"Joint '{joint_name}' not found in target joint names")

        # Create target to source mapping
        for joint_name in target_joint_names:
            if joint_name in source_joint_names:
                target_to_source.append(source_joint_names.index(joint_name))
            else:
                raise ValueError(f"Joint '{joint_name}' not found in source joint names")
        print(f"[INFO] Loaded joint mapping for policy transfer from YAML: {args_cli.policy_transfer_file}")
        assert (
            len(source_to_target) == len(target_to_source) == num_joints
        ), "Number of source and target joints must match"
    else:
        # Use identity mapping (one-to-one)
        identity_map = list(range(num_joints))
        source_to_target, target_to_source = identity_map, identity_map

    # Create observation mapping (first 12 values stay the same for locomotion examples, then map joint-related values)
    obs_map = (
        [0, 1, 2]
        + [3, 4, 5]
        + [6, 7, 8]
        + [9, 10, 11]
        + [i + 12 + num_joints * 0 for i in source_to_target]
        + [i + 12 + num_joints * 1 for i in source_to_target]
        + [i + 12 + num_joints * 2 for i in source_to_target]
    )

    return source_to_target, target_to_source, obs_map


class RobotStatePublisher(Node):
    """ROS2 node for publishing robot state and actions."""

    def __init__(self, num_joints=6, num_gripper_joints=4):
        super().__init__('isaac_lab_robot_publisher')
        
        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, '/isaac_sim/joint_states', 10)
        self.action_pub = self.create_publisher(Float64MultiArray, '/isaac_sim/policy_actions', 10)
        self.observation_pub = self.create_publisher(Float64MultiArray, '/isaac_sim/observations', 10)
        self.object_pose_pub = self.create_publisher(PoseStamped, '/isaac_sim/object_pose', 10)
        
        # Robot configuration
        self.num_joints = num_joints
        self.num_gripper_joints = num_gripper_joints
        self.num_total_joints = num_joints + num_gripper_joints
        
        # Joint names (E0509 configuration)
        self.joint_names = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
            'rh_l1_joint', 'rh_l2_joint', 'rh_r1_joint', 'rh_r2_joint'
        ]
        
        self.get_logger().info(f'✅ ROS2 Robot State Publisher initialized')
        self.get_logger().info(f'   - Publishing joint states to: /isaac_sim/joint_states')
        self.get_logger().info(f'   - Publishing actions to: /isaac_sim/policy_actions')
        self.get_logger().info(f'   - Publishing observations to: /isaac_sim/observations')
        self.get_logger().info(f'   - Publishing object pose to: /isaac_sim/object_pose')
    
    def publish_joint_state(self, positions, velocities, efforts=None):
        """Publish current joint state."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.joint_names[:len(positions)]
        msg.position = positions.tolist() if isinstance(positions, np.ndarray) else positions
        msg.velocity = velocities.tolist() if isinstance(velocities, np.ndarray) else velocities
        if efforts is not None:
            msg.effort = efforts.tolist() if isinstance(efforts, np.ndarray) else efforts
        else:
            msg.effort = [0.0] * len(positions)
        
        self.joint_state_pub.publish(msg)
    
    def publish_action(self, actions):
        """Publish policy action."""
        msg = Float64MultiArray()
        msg.data = actions.tolist() if isinstance(actions, np.ndarray) else actions
        self.action_pub.publish(msg)
    
    def publish_observation(self, observations):
        """Publish full observation vector."""
        msg = Float64MultiArray()
        msg.data = observations.tolist() if isinstance(observations, np.ndarray) else observations
        self.observation_pub.publish(msg)
    
    def publish_object_pose(self, position, orientation):
        """Publish object pose (e.g., cube position)."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.w = float(orientation[0])
        msg.pose.orientation.x = float(orientation[1])
        msg.pose.orientation.y = float(orientation[2])
        msg.pose.orientation.z = float(orientation[3])
        self.object_pose_pub.publish(msg)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent with policy transfer capabilities and ROS2 publishing."""

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # Get joint mappings for policy transfer
    _, target_to_source, obs_map = get_joint_mappings(args_cli, env.action_space.shape[1])

    # Create torch tensors for mappings
    device = args_cli.device if args_cli.device else "cuda:0"
    target_to_source_tensor = torch.tensor(target_to_source, device=device) if target_to_source else None
    obs_map_tensor = torch.tensor(obs_map, device=device) if obs_map else None

    def remap_obs(obs):
        """Remap the observation to the target observation space."""
        if obs_map_tensor is not None:
            obs = obs[:, obs_map_tensor]
        return obs

    def remap_actions(actions):
        """Remap the actions to the target action space."""
        if target_to_source_tensor is not None:
            actions = actions[:, target_to_source_tensor]
        return actions

    # Initialize ROS2 publisher if requested
    ros2_publisher = None
    if args_cli.publish_ros2 and ROS2_AVAILABLE:
        rclpy.init()
        num_joints = env.action_space.shape[1]
        # Assume gripper has 4 joints (E0509 configuration)
        num_gripper_joints = 4
        ros2_publisher = RobotStatePublisher(num_joints=num_joints, num_gripper_joints=num_gripper_joints)
        print(f"[INFO] ROS2 publisher initialized. Publishing environment {args_cli.ros2_env_id}")
    elif args_cli.publish_ros2 and not ROS2_AVAILABLE:
        print("[WARNING] ROS2 not available. Skipping ROS2 publishing.")

    # Helper function to extract robot state from observations
    def extract_robot_state(obs_dict, env_id=0):
        """Extract robot joint positions, velocities, and object pose from observations.
        
        Observation structure (from E0509PickPlaceEnvCfg):
        - base_lin_vel (3)
        - base_ang_vel (3)
        - projected_gravity (3)
        - velocity_commands (3)
        - joint_pos (num_joints)
        - joint_vel (num_joints)
        - actions (num_joints)
        
        Object state is stored separately in env
        """
        # Get the full observation tensor for the specified environment
        if isinstance(obs_dict, dict):
            # If obs_dict is still a dictionary, concatenate all observations
            obs_tensor = torch.cat([v for v in obs_dict.values()], dim=-1)
        else:
            obs_tensor = obs_dict
        
        # Extract single environment data
        obs_single = obs_tensor[env_id].cpu().numpy()
        
        # Parse observation structure (assuming manipulation task)
        # Typical structure: [base_state(12), joint_pos(n), joint_vel(n), prev_actions(n)]
        num_actions = env.action_space.shape[1]
        
        # For E0509 pick-place task:
        # base state (12) + joint_pos (num_actions) + joint_vel (num_actions) + prev_actions (num_actions)
        base_state_dim = 12
        joint_pos_start = base_state_dim
        joint_vel_start = base_state_dim + num_actions
        prev_action_start = base_state_dim + 2 * num_actions
        
        joint_pos = obs_single[joint_pos_start:joint_vel_start]
        joint_vel = obs_single[joint_vel_start:prev_action_start]
        prev_actions = obs_single[prev_action_start:prev_action_start + num_actions]
        
        # Try to get object position from environment (if available)
        object_pos = np.array([0.0, 0.0, 0.0])
        object_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        try:
            # Access the environment's scene to get object state
            scene = env.unwrapped.scene
            if hasattr(scene, 'object') or hasattr(scene, 'cube'):
                obj = getattr(scene, 'object', getattr(scene, 'cube', None))
                if obj is not None and hasattr(obj, 'data'):
                    object_pos = obj.data.root_pos_w[env_id].cpu().numpy()
                    object_quat = obj.data.root_quat_w[env_id].cpu().numpy()
        except:
            pass  # Object position not available
        
        return {
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'prev_actions': prev_actions,
            'object_pos': object_pos,
            'object_quat': object_quat,
            'full_obs': obs_single
        }

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(remap_obs(obs))
            # env stepping
            obs, _, _, _ = env.step(remap_actions(actions))
        
        # Publish to ROS2 if enabled
        if ros2_publisher is not None:
            try:
                # Extract robot state from the selected environment
                robot_state = extract_robot_state(obs, env_id=args_cli.ros2_env_id)
                
                # Publish joint state (positions + velocities)
                ros2_publisher.publish_joint_state(
                    positions=robot_state['joint_pos'],
                    velocities=robot_state['joint_vel']
                )
                
                # Publish policy action (remapped actions for the selected environment)
                policy_action = remap_actions(actions)[args_cli.ros2_env_id].cpu().numpy()
                ros2_publisher.publish_action(policy_action)
                
                # Publish full observation vector
                ros2_publisher.publish_observation(robot_state['full_obs'])
                
                # Publish object pose
                ros2_publisher.publish_object_pose(
                    position=robot_state['object_pos'],
                    orientation=robot_state['object_quat']
                )
                
                # Spin ROS2 node to process callbacks
                rclpy.spin_once(ros2_publisher, timeout_sec=0)
                
            except Exception as e:
                print(f"[WARNING] ROS2 publishing error: {e}")
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # cleanup
    if ros2_publisher is not None:
        ros2_publisher.destroy_node()
        rclpy.shutdown()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
