#!/usr/bin/env python3
"""Analyze observation structure from trained environment."""

import torch
import numpy as np
import sys
import os

# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from omni.isaac.lab.envs import ManagerBasedRLEnv
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

def main():
    """Analyze observation dimensions."""
    
    # Parse environment config
    env_cfg = parse_env_cfg("Isaac-E0509-PickPlace-v0", device="cpu", num_envs=1)
    
    # Create environment
    print("[INFO] Creating environment...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Reset and get observation
    print("[INFO] Resetting environment...")
    obs_dict, _ = env.reset()
    
    # Analyze observation structure
    print("\n" + "="*60)
    print("OBSERVATION ANALYSIS")
    print("="*60)
    
    if "policy" in obs_dict:
        policy_obs = obs_dict["policy"]
        print(f"\n[Policy Observation Shape]: {policy_obs.shape}")
        print(f"[Total Dimensions]: {policy_obs.shape[-1]}")
        print(f"\n[Sample Values]:\n{policy_obs[0].cpu().numpy()}")
        
        # Try to get observation manager info
        obs_manager = env.observation_manager
        print(f"\n[Observation Manager Groups]: {list(obs_manager._group_obs_term_names.keys())}")
        
        if "policy" in obs_manager._group_obs_term_names:
            terms = obs_manager._group_obs_term_names["policy"]
            print(f"\n[Policy Observation Terms]: {terms}")
            
            # Get dimensions for each term
            print("\n[Term Dimensions]:")
            total_dims = 0
            for term_name in terms:
                term_cfg = obs_manager._group_obs_term_cfgs["policy"][term_name]
                # Get the observation to see its shape
                term_data = obs_manager._obs_terms["policy"][term_name].func(env, term_cfg)
                dims = term_data.shape[-1]
                total_dims += dims
                print(f"  {term_name:30s}: {dims:3d} dims | {term_data[0].cpu().numpy()}")
            
            print(f"\n[Calculated Total]: {total_dims} dimensions")
    
    print("\n" + "="*60)
    
    # Cleanup
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
