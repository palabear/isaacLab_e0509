#!/usr/bin/env python3
"""
Sim2Real E0509 Policy Controller
---------------------------------

Isaac Lab에서 학습된 PPO 정책을 실제 Doosan E0509 로봇에서 실행하는 ROS2 노드

Environment:
    - OS: Ubuntu 24.04
    - ROS2: Jazzy
    - Robot: Doosan E0509 (6-DOF)
    - Isaac Sim: 5.1.0
    - Training: RSL-RL PPO

Data Flow:
    Real Robot Sensors → Observation (25-dim) → Policy Network → Action (6-dim) → Real Robot Control

Subscribed Topics:
    - /joint_states (sensor_msgs/JointState) - Real robot sensor data
    - /object_detection/pose (geometry_msgs/PoseStamped) - Medicine cabinet position (optional)

Published Topics:
    - /joint_trajectory_controller/joint_trajectory (trajectory_msgs/JointTrajectory) - Robot commands

Author: IsaacLab E0509 Deployment Team
Date: 2025-12-06
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

# Doosan RT control messages
try:
    from dsr_msgs2.msg import ServojRtStream
except ImportError:
    print("Warning: dsr_msgs2 not found. Install Doosan ROS2 packages or use Mock mode.")
    ServojRtStream = None

import torch
import numpy as np
from scipy.spatial.transform import Rotation
import time
from collections import deque


class Sim2RealE0509Controller(Node):
    """
    Sim2Real Policy Controller for Doosan E0509 Robot
    
    Loads trained policy from Isaac Lab and executes on real hardware with safety checks.
    """
    
    # =========================================================================
    # Robot Specifications (Doosan E0509)
    # =========================================================================
    JOINT_NAMES = [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',  # Arm
        'rh_l1', 'rh_r1_joint', 'rh_l2', 'rh_r2'  # Gripper
    ]
    
    # Joint limits (radians) - from E0509 specification
    JOINT_LIMITS_LOWER = np.array([
        -6.2832, -6.2832, -2.7053, -6.2832, -2.7053, -6.2832,  # Arm
        0.0, 0.0, 0.0, 0.0  # Gripper (adjust based on actual limits)
    ])
    JOINT_LIMITS_UPPER = np.array([
        6.2832, 6.2832, 2.7053, 6.2832, 2.7053, 6.2832,  # Arm  
        0.087, 0.087, 0.087, 0.087  # Gripper (adjust based on actual limits)
    ])
    
    # Velocity limits (rad/s) - conservative for safety
    # For real robot, start very slow to avoid oscillation:
    #   - Testing: 0.3 rad/s = 17.2 deg/s (very safe, stable)
    #   - Normal: 0.5 rad/s = 28.6 deg/s (safe)
    #   - Fast: 1.0 rad/s = 57.3 deg/s (moderate)
    MAX_JOINT_VELOCITY = 0.5  # rad/s - Safe speed
    
    # Action limits for clipping (prevents sudden movements)
    MAX_ACTION_CHANGE = 1.0  # Maximum change in action between steps
    MAX_ACTION_VALUE = 2.0  # Maximum absolute action value (clip policy output)
    
    # Control frequency (must match Isaac Lab training)
    CONTROL_FREQUENCY = 50.0  # Hz (100Hz physics / 2 decimation)
    
    def __init__(self):
        super().__init__('sim2real_e0509_controller')
        
        # =====================================================================
        # 1. Load Trained Policy Model
        # =====================================================================
        self._load_policy_model()
        
        # =====================================================================
        # 2. Initialize Robot State Variables
        # =====================================================================
        self._init_robot_state()
        
        # =====================================================================
        # 3. Setup ROS2 Communication
        # =====================================================================
        self._setup_ros2_interfaces()
        
        # =====================================================================
        # 4. Safety & Statistics
        # =====================================================================
        self._init_safety_monitoring()
        
        self.get_logger().info('=' * 70)
        self.get_logger().info('🚀 Sim2Real E0509 Policy Controller Initialized')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'   Model: {self.model_path}')
        self.get_logger().info(f'   Device: {self.device}')
        self.get_logger().info(f'   Control Rate: {self.CONTROL_FREQUENCY} Hz')
        self.get_logger().info(f'   Observation Dim: {self.obs_dim}')
        self.get_logger().info(f'   Action Dim: {self.action_dim}')
        self.get_logger().info('=' * 70)
        self.get_logger().warn('⚠️  Make sure robot is in REMOTE/ROS MODE!')
        self.get_logger().warn('⚠️  Emergency stop button should be accessible!')
        self.get_logger().info('Waiting for sensor data...')
    
    def _load_policy_model(self):
        """Load trained policy model."""
        # Model path (no ONNX needed - actor_obs_normalization=False)
        self.model_path = '/home/jiwoo/IsaacLab/logs/rsl_rl/e0509_lift/2025-12-06_14-21-07/exported/policy.pt'
        
        # Device selection (CPU for real robot, GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load JIT model
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            self.get_logger().info(f'✅ Policy model loaded: {self.model_path}')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to load policy model: {e}')
            raise
        
        # No normalization needed (actor_obs_normalization=False in training config)
        self.get_logger().info('ℹ️  No observation normalization (actor_obs_normalization=False)')
        
        # Observation and action dimensions
        self.obs_dim = 34  # E0509: joint_pos(10) + joint_vel(10) + target_pose(7) + prev_actions(7)
        self.action_dim = 7  # E0509: 6-DOF arm + 1 gripper
    
    def _init_robot_state(self):
        """Initialize robot state variables."""
        # Joint names: 6 arm joints + 4 gripper joints
        self.JOINT_NAMES = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',  # Arm
            'rh_l1', 'rh_r1_joint', 'rh_l2', 'rh_r2'  # Gripper
        ]
        
        # Home position - Robot starts here every time
        # Matches training default position for consistency
        self.home_joint_pos = np.array([
            0.0,      # joint_1
            0.0,      # joint_2
            1.5708,   # joint_3 (90 degrees)
            0.0,      # joint_4
            1.5708,   # joint_5 (90 degrees)
            0.0,      # joint_6
            0.0,      # rh_l1 (gripper)
            0.0,      # rh_r1_joint
            0.0,      # rh_l2
            0.0,      # rh_r2
        ], dtype=np.float32)
        
        # Default joint positions (from e0509.py)
        # ⚠️ CRITICAL: These must match Isaac Lab training environment!
        self.default_joint_pos = np.array([
            0.0,      # joint_1
            0.0,      # joint_2
            1.5708,   # joint_3 (90 degrees)
            0.0,      # joint_4
            1.5708,   # joint_5 (90 degrees)
            0.0,      # joint_6
            0.0,      # rh_l1 (gripper)
            0.0,      # rh_r1_joint
            0.0,      # rh_l2
            0.0,      # rh_r2
        ], dtype=np.float32)
        
        # Current robot state (updated from /joint_states)
        self.current_joint_pos = self.default_joint_pos.copy()
        self.current_joint_vel = np.zeros(10, dtype=np.float32)
        
        # Default joint velocities (same as training - all zeros)
        self.default_joint_vel = np.zeros(10, dtype=np.float32)
        
        # Previous action (for observation and safety check)
        self.previous_action = np.zeros(7, dtype=np.float32)
        
        # Target pose (medicine cabinet position)
        # ⚠️ CRITICAL: Training uses ROBOT BASE FRAME coordinates!
        # ⚠️ Training range: x=[0.4, 0.6], y=[-0.25, 0.25], z=[0.25, 0.5]
        # 
        # Home EE position is approximately [0.5, 0.0, 0.4] in base frame
        # Medicine cabinet target should be within robot's reach
        #
        # For testing, use a reachable target in robot base frame (MUST be in training range!)
        self.target_position = np.array([0.50, 0.0, 0.35], dtype=np.float32)  # Base frame coordinates
        # Top-down grasp quaternion (w, x, y, z)
        self.target_orientation = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        
        # Action scale - Controls robot movement speed
        # Training used 0.2, but real robot needs careful tuning:
        #   - 0.05: Very slow and stable (too slow for convergence)
        #   - 0.1: Slow but steady
        #   - 0.2: Training speed (original)
        # Lower = slower and safer movements
        self.action_scale = 0.2  # Match training speed
        
        # State flags
        self.has_joint_data = False
        self.has_target_data = False
        self.emergency_stop = False
        
        # Initialization phase
        self.is_initialized = False
        self.initialization_step = 0
        self.max_init_steps = 300  # 6 seconds at 50Hz to reach home position
        
        # Initialization target (always start at home position)
        self.init_target_pos = self.home_joint_pos.copy()
    
    def _setup_ros2_interfaces(self):
        """Setup ROS2 publishers and subscribers."""
        # Robot namespace (change if using different namespace)
        robot_ns = 'dsr01'  # Doosan robot namespace
        
        # Subscriber: Real robot joint states
        # Doosan: /dsr01/joint_states
        self.joint_state_sub = self.create_subscription(
            JointState,
            f'/{robot_ns}/joint_states',  # Doosan ROS2 driver topic
            self.joint_state_callback,
            10
        )
        
        # Subscriber: Target pose (medicine cabinet detection)
        self.target_pose_sub = self.create_subscription(
            PoseStamped,
            '/object_detection/pose',
            self.target_pose_callback,
            10
        )
        
        # Publisher: Doosan Real-time Servo Control
        # servoj_rt_stream: Real-time joint position control
        if ServojRtStream is not None:
            self.servoj_pub = self.create_publisher(
                ServojRtStream,
                f'/{robot_ns}/servoj_rt_stream',
                10
            )
        else:
            self.get_logger().warn('ServojRtStream not available - using mock mode')
            self.servoj_pub = None
        
        # Control loop timer (50 Hz)
        self.control_timer = self.create_timer(
            1.0 / self.CONTROL_FREQUENCY,
            self.control_loop
        )
    
    def _init_safety_monitoring(self):
        """Initialize safety monitoring and statistics."""
        # Action history for smoothing and safety check
        self.action_history = deque(maxlen=10)
        
        # Statistics
        self.control_step = 0
        self.start_time = time.time()
        
        # Emergency stop conditions
        self.max_consecutive_warnings = 999999  # Disable emergency stop for debugging
        self.warning_count = 0
    
    # =========================================================================
    # ROS2 Callbacks
    # =========================================================================
    
    def joint_state_callback(self, msg: JointState):
        """
        Process real robot joint state messages.
        
        ⚠️ IMPORTANT: Joint order in msg.name must match JOINT_NAMES!
        """
        try:
            # Extract joint positions and velocities
            for i, joint_name in enumerate(self.JOINT_NAMES):
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    self.current_joint_pos[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        self.current_joint_vel[i] = msg.velocity[idx]
            
            self.has_joint_data = True
            
            # DEBUG: Log received joint states
            if self.control_step % 20 == 0:
                self.get_logger().info(f'📥 ROS joint_states: {[f"{x:.3f}" for x in self.current_joint_pos[:6]]}')
            
        except Exception as e:
            self.get_logger().error(f'Joint state callback error: {e}')
    
    def target_pose_callback(self, msg: PoseStamped):
        """Process target pose (medicine cabinet position) messages."""
        self.target_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ], dtype=np.float32)
        
        self.target_orientation = np.array([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ], dtype=np.float32)
        
        self.has_target_data = True
    
    # =========================================================================
    # Forward Kinematics (for debugging)
    # =========================================================================
    
    def compute_forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute end-effector position using forward kinematics.
        Simple approximation for E0509 robot.
        
        Args:
            joint_positions: Joint angles in radians [6]
        
        Returns:
            ee_position: End-effector position [x, y, z] in base frame
        """
        # E0509 approximate link lengths (in meters)
        # These are rough estimates - adjust based on actual robot specs
        L1 = 0.0  # Base height
        L2 = 0.409  # Link 2 length
        L3 = 0.367  # Link 3 length  
        L4 = 0.124  # Link 4+5+6 combined length to EE
        
        q1, q2, q3, q4, q5, q6 = joint_positions[:6]
        
        # Simplified FK (ignoring wrist orientation)
        # This is an approximation - real FK would use DH parameters
        x = (L2 * np.cos(q2) + L3 * np.cos(q2 + q3) + L4 * np.cos(q2 + q3 + q4)) * np.cos(q1)
        y = (L2 * np.cos(q2) + L3 * np.cos(q2 + q3) + L4 * np.cos(q2 + q3 + q4)) * np.sin(q1)
        z = L1 + L2 * np.sin(q2) + L3 * np.sin(q2 + q3) + L4 * np.sin(q2 + q3 + q4)
        
        return np.array([x, y, z], dtype=np.float32)
    
    # =========================================================================
    # Observation Processing (Sim2Real Critical Section)
    # =========================================================================
    
    def get_observation(self) -> np.ndarray:
        """
        Build observation vector from real robot sensors.
        
        ⚠️ CRITICAL: This must exactly match Isaac Lab training environment!
        
        Observation Structure (34-dim):
            - joint_pos_rel (10): current_pos - default_pos (6 arm + 4 gripper)
            - joint_vel (10): current velocity (6 arm + 4 gripper)
            - target_pose (7): target position (3) + quaternion (4)
            - previous_actions (7): last action (6 arm + 1 gripper)
        
        Returns:
            obs (np.ndarray): Observation vector (34,)
        """
        obs = np.zeros(34, dtype=np.float32)
        
        # Joint positions (relative to default) - 10 joints
        # ⚠️ Isaac Lab uses relative positions!
        joint_pos_rel = self.current_joint_pos - self.default_joint_pos
        obs[0:10] = joint_pos_rel
        
        # Joint velocities (relative to default) - 10 joints
        # ⚠️ Isaac Lab uses relative velocities (joint_vel_rel)!
        joint_vel_rel = self.current_joint_vel - self.default_joint_vel
        obs[10:20] = joint_vel_rel
        
        # Target pose (position + orientation) - 7 values
        obs[20:23] = self.target_position      # x, y, z
        obs[23:27] = self.target_orientation   # w, x, y, z (quaternion)
        
        # Previous actions - 7 values
        obs[27:34] = self.previous_action
        
        return obs
    
    def normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observation using mean and std from training.
        
        ⚠️ CRITICAL: RSL-RL applies normalization during training!
        
        Args:
            obs (np.ndarray): Raw observation (25,)
        
        Returns:
            obs_normalized (np.ndarray): Normalized observation (25,)
        """
        if self.obs_mean is not None and self.obs_std is not None:
            obs_normalized = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        else:
            obs_normalized = obs
        
        return obs_normalized
    
    # =========================================================================
    # Policy Inference
    # =========================================================================
    
    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute action using trained policy network.
        
        Args:
            obs (np.ndarray): Raw observation (34,)
        
        Returns:
            action (np.ndarray): Policy action (7,)
        """
        # No normalization needed (actor_obs_normalization=False)
        
        # Convert to torch tensor
        obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # Policy inference (no gradient computation)
        with torch.no_grad():
            action_tensor = self.model(obs_tensor)
        
        # Convert to numpy
        action = action_tensor.cpu().numpy().flatten()
        
        # Clip action to prevent extreme values
        action = np.clip(action, -self.MAX_ACTION_VALUE, self.MAX_ACTION_VALUE)
        
        return action
    
    # =========================================================================
    # Action Processing & Safety Checks
    # =========================================================================
    
    def process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Process policy action and apply safety checks.
        
        Safety Checks:
            1. Action change rate limiting (prevent sudden movements)
            2. Joint limit checking
            3. Velocity limit checking
        
        Args:
            action (np.ndarray): Raw policy action (7,) - 6 arm + 1 gripper
        
        Returns:
            safe_action (np.ndarray): Safe action after checks (7,)
        """
        # 1. Action smoothing (limit sudden changes)
        if len(self.action_history) > 0:
            action_change = np.abs(action - self.previous_action)
            max_change = np.max(action_change)
            
            if max_change > self.MAX_ACTION_CHANGE:
                # Scale down action change
                scale = self.MAX_ACTION_CHANGE / max_change
                action = self.previous_action + (action - self.previous_action) * scale
                
                self.warning_count += 1
                if self.control_step % 10 == 0:  # Only log every 10 steps to reduce spam
                    self.get_logger().warn(
                        f'⚠️  Action change too large! max_change={max_change:.3f}, scaled by {scale:.3f} '
                        f'(Total warnings: {self.warning_count})'
                    )
                
                # Emergency stop if too many warnings
                if self.warning_count >= self.max_consecutive_warnings:
                    self.get_logger().error('🛑 EMERGENCY STOP: Too many action warnings!')
                    self.emergency_stop = True
                    return np.zeros(7)
            else:
                self.warning_count = max(0, self.warning_count - 1)  # Decay warnings
        
        # 2. Compute target joint positions (10 joints: 6 arm + 4 gripper)
        # ⚠️ CRITICAL: Training uses (action * scale) + default_pos, NOT current_pos + delta!
        # Action is 7-dim (6 arm + 1 gripper binary), expand gripper action to 4 joints
        expanded_action = np.zeros(10, dtype=np.float32)
        expanded_action[0:6] = action[0:6]  # Arm actions
        expanded_action[6:10] = action[6]   # Gripper action (broadcast to all 4 gripper joints)
        
        # Isaac Lab: target = (action * scale) + default_joint_pos
        target_joint_pos = (expanded_action * self.action_scale) + self.default_joint_pos
        
        # 3. Joint limit check
        target_joint_pos = np.clip(
            target_joint_pos,
            self.JOINT_LIMITS_LOWER,
            self.JOINT_LIMITS_UPPER
        )
        
        # 4. Velocity limit check
        dt = 1.0 / self.CONTROL_FREQUENCY
        velocity = (target_joint_pos - self.current_joint_pos) / dt
        
        if np.any(np.abs(velocity) > self.MAX_JOINT_VELOCITY):
            # Scale down if too fast
            max_vel_rad = np.max(np.abs(velocity))
            max_vel_deg = max_vel_rad * 57.2958  # Convert to deg/s
            scale = self.MAX_JOINT_VELOCITY / max_vel_rad
            target_joint_pos = self.current_joint_pos + (target_joint_pos - self.current_joint_pos) * scale
            
            if self.control_step % 10 == 0:  # Only log every 10 steps
                self.get_logger().warn(
                    f'⚠️  Velocity limit applied | '
                    f'Requested: {max_vel_deg:.1f} deg/s ({max_vel_rad:.2f} rad/s) | '
                    f'Limited to: {self.MAX_JOINT_VELOCITY * 57.2958:.1f} deg/s ({self.MAX_JOINT_VELOCITY:.2f} rad/s) | '
                    f'Scale: {scale:.3f}'
                )
        
        # Compute safe action from safe target (collapse gripper joints back to single action)
        safe_expanded_action = (target_joint_pos - self.current_joint_pos) / self.action_scale
        safe_action = np.zeros(7, dtype=np.float32)
        safe_action[0:6] = safe_expanded_action[0:6]  # Arm actions
        safe_action[6] = np.mean(safe_expanded_action[6:10])  # Average gripper joints
        
        return safe_action
    
    def publish_servoj_command(self, target_joint_pos: np.ndarray):
        """
        Publish Doosan servoj real-time command.
        
        Args:
            target_joint_pos (np.ndarray): Target joint positions in RADIANS (10,) - 6 arm + 4 gripper
        """
        if self.servoj_pub is None or ServojRtStream is None:
            return
        
        msg = ServojRtStream()
        # Doosan servoj_rt_stream expects DEGREES, not radians!
        # Convert radians to degrees
        target_deg = target_joint_pos[:6] * 57.2958  # rad to deg
        
        msg.pos = target_deg.tolist()
        msg.vel = [0.0] * 6  # Use default velocity
        msg.acc = [0.0] * 6  # Use default acceleration
        msg.time = 0.0  # Use control loop time
        
        self.servoj_pub.publish(msg)
    
    # =========================================================================
    # Initialization Phase
    # =========================================================================
    
    def initialize_robot_position(self):
        """
        Move robot to HOME position before starting policy inference.
        
        Home position: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        This ensures the robot always starts from the same position every time
        the controller is launched, regardless of where it was when stopped.
        """
        # Check if already at home position (within tolerance)
        position_error = np.abs(self.current_joint_pos - self.init_target_pos)
        max_error = np.max(position_error[:6])  # Only check arm joints
        
        if max_error < 0.02:  # 0.02 rad ≈ 1.1 degrees
            self.is_initialized = True
            self.get_logger().info('✅ Robot at HOME position - Starting policy inference')
            self.get_logger().info(f'   Home position: [{self.home_joint_pos[0]:.3f}, {self.home_joint_pos[1]:.3f}, '
                                 f'{self.home_joint_pos[2]:.3f}, {self.home_joint_pos[3]:.3f}, '
                                 f'{self.home_joint_pos[4]:.3f}, {self.home_joint_pos[5]:.3f}]')
            return
        
        # Gradually move towards home position
        # Use simple proportional control with very slow speed for safety
        alpha = 0.02  # Interpolation factor (2% per step = very slow movement)
        target_pos = self.current_joint_pos + alpha * (self.init_target_pos - self.current_joint_pos)
        
        # Publish command
        self.publish_servoj_command(target_pos)
        
        self.initialization_step += 1
        
        # Log progress every 1 second
        if self.initialization_step % 50 == 0:
            self.get_logger().info(
                f'🔄 Moving to HOME position... Step {self.initialization_step}/{self.max_init_steps} | '
                f'Max error: {max_error:.3f} rad ({max_error * 57.3:.1f} deg) | '
                f'Current: [{self.current_joint_pos[0]:.3f}, {self.current_joint_pos[1]:.3f}, '
                f'{self.current_joint_pos[2]:.3f}, {self.current_joint_pos[3]:.3f}, '
                f'{self.current_joint_pos[4]:.3f}, {self.current_joint_pos[5]:.3f}]'
            )
        
        # Safety timeout
        if self.initialization_step > self.max_init_steps:
            self.get_logger().warn(
                f'⚠️  Initialization timeout! Proceeding with current position. '
                f'Max error: {max_error:.3f} rad ({max_error * 57.3:.1f} deg)'
            )
            self.is_initialized = True
    
    # =========================================================================
    # Main Control Loop
    # =========================================================================
    
    def control_loop(self):
        """
        Main control loop (50 Hz).
        
        Flow:
            0. Initialize robot to training default position (first time)
            1. Check if sensor data is ready
            2. Build observation from sensors
            3. Compute action using policy
            4. Apply safety checks
            5. Send command to robot
            6. Update statistics
        """
        # Wait for sensor data
        if not self.has_joint_data:
            return
        
        # Initialization phase: Move to training default position first
        if not self.is_initialized:
            self.initialize_robot_position()
            return
        
        # Emergency stop check
        if self.emergency_stop:
            self.get_logger().error('🛑 EMERGENCY STOP ACTIVE - Control loop halted')
            return
        
        try:
            # Step 1: Get observation from real robot sensors
            obs = self.get_observation()
            
            # DEBUG: Log first observation
            if self.control_step == 0:
                self.get_logger().info(f'🔍 First observation (34-dim):')
                self.get_logger().info(f'   joint_pos_rel[0:10]: {obs[0:10]}')
                self.get_logger().info(f'   joint_vel[10:20]: {obs[10:20]}')
                self.get_logger().info(f'   target_pose[20:27]: {obs[20:27]}')
                self.get_logger().info(f'   prev_actions[27:34]: {obs[27:34]}')
            
            # Step 2: Compute action using trained policy
            action = self.compute_action(obs)
            
            # DEBUG: Log first action
            if self.control_step == 0:
                self.get_logger().info(f'🔍 Policy output (7-dim): {action}')
            
            # Step 3: Apply safety checks and process action
            safe_action = self.process_action(action)
            
            # Step 4: Compute target joint positions (expand 7-dim action to 10 joints)
            expanded_action = np.zeros(10, dtype=np.float32)
            expanded_action[0:6] = safe_action[0:6]  # Arm actions
            expanded_action[6:10] = safe_action[6]   # Gripper action (broadcast)
            target_joint_pos = self.current_joint_pos + expanded_action * self.action_scale
            
            # Step 5: Publish servoj command to robot
            self.publish_servoj_command(target_joint_pos)
            
            # Step 6: Update state
            self.previous_action = safe_action.copy()
            self.action_history.append(safe_action.copy())
            self.control_step += 1
            
            # Log every 0.2 second (10 steps) for debugging
            if self.control_step % 10 == 0:
                elapsed = time.time() - self.start_time
                freq = self.control_step / elapsed if elapsed > 0 else 0
                
                # Compute current EE position and distance to target
                current_ee_pos = self.compute_forward_kinematics(self.current_joint_pos)
                ee_to_target_dist = np.linalg.norm(current_ee_pos - self.target_position)
                
                self.get_logger().info(
                    f'📊 Step {self.control_step:5d} | Freq: {freq:5.1f} Hz | '
                    f'Raw Action: [{action[0]:7.4f}, {action[1]:7.4f}, {action[2]:7.4f}, {action[3]:7.4f}, {action[4]:7.4f}, {action[5]:7.4f}, {action[6]:7.4f}] | '
                    f'Warnings: {self.warning_count}'
                )
                self.get_logger().info(
                    f'    Joint pos: [{self.current_joint_pos[0]:.3f}, {self.current_joint_pos[1]:.3f}, {self.current_joint_pos[2]:.3f}, '
                    f'{self.current_joint_pos[3]:.3f}, {self.current_joint_pos[4]:.3f}, {self.current_joint_pos[5]:.3f}]'
                )
                self.get_logger().info(
                    f'    Current EE: [{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}] | '
                    f'Target: [{self.target_position[0]:.3f}, {self.target_position[1]:.3f}, {self.target_position[2]:.3f}] | '
                    f'Distance: {ee_to_target_dist:.3f}m'
                )
                self.get_logger().info(
                    f'    Obs joint_pos_rel[0:6]: [{obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}, {obs[3]:.3f}, {obs[4]:.3f}, {obs[5]:.3f}]'
                )
        
        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')
            self.emergency_stop = True


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    controller = Sim2RealE0509Controller()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('🛑 Shutting down by user request...')
    except Exception as e:
        controller.get_logger().error(f'❌ Fatal error: {e}')
    finally:
        # Cleanup
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
