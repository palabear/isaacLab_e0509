# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom gripper action for mimic joints (1 action -> 4 joints)."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class MimicGripperActionCfg(ActionTermCfg):
    """Configuration for mimic gripper action (1 action controls all gripper joints)."""
    
    asset_name: str = "robot"
    """Name of the articulation asset."""
    
    joint_names: list[str] | str = ["rh_.*"]
    """Gripper joint names (regex pattern or list)."""
    
    scale: float = 0.55
    """Scaling factor for action. Default maps [-1,+1] -> [-0.55, +0.55]"""
    
    offset: float = 0.55
    """Offset applied after scaling. Default shifts to [0, 1.1] range."""
    
    def __post_init__(self):
        """Set class_type after class is defined."""
        self.class_type = MimicGripperAction


class MimicGripperAction(ActionTerm):
    """Gripper action for mimic joints where 1 action controls all gripper joints.
    
    This action term receives a single scalar action and applies it to all gripper joints.
    Useful for mimic joints where multiple joints are mechanically coupled.
    
    Action mapping:
    - Input: 1 value in [-1, +1] (from PPO)
    - Output: Same value to all 4 gripper joints in [0, 1.1] range
      - -1.0 -> 0.0 (fully open)
      - +1.0 -> 1.1 (fully closed)
    """

    cfg: MimicGripperActionCfg
    _asset: Articulation

    def __init__(self, cfg: MimicGripperActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Resolve gripper joints
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        
        # Create action buffers
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)  # 1 action
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)  # 4 joints
        
        print(f"[MimicGripperAction] Controlling {self._num_joints} joints: {self._joint_names}")
        print(f"[MimicGripperAction] Action mapping: [-1,+1] -> [0, 1.1] (scale={cfg.scale}, offset={cfg.offset})")

    @property
    def action_dim(self) -> int:
        """Return 1 (single gripper action)."""
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Process single action and broadcast to all gripper joints."""
        # Store raw action (1 value per env)
        self._raw_actions[:] = actions
        
        # Apply scale and offset: action_out = offset + scale * action_in
        # Mapping: action=-1 (open) -> 0.0, action=+1 (close) -> 1.1
        processed = self.cfg.offset + self.cfg.scale * actions
        
        # Broadcast to all gripper joints
        self._processed_actions[:] = processed.repeat(1, self._num_joints)
        
        # Clamp to valid range [0, 1.1]
        self._processed_actions.clamp_(0.0, 1.1)

    def apply_actions(self):
        """Apply processed actions to gripper joints."""
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset action to neutral (open gripper)."""
        if env_ids is None:
            self._raw_actions[:] = -1.0  # Open gripper
        else:
            self._raw_actions[env_ids] = -1.0


# Fix forward reference
MimicGripperActionCfg.class_type = MimicGripperAction
