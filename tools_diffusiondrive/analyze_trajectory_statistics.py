"""
Analyze trajectory statistics from nuScenes dataset pickle files.
This script reads the processed data and computes min/max statistics for all trajectory dimensions.

Usage:
    python tools/analyze_trajectory_statistics.py --data-path ./data/nuscenes/nuscenes_infos_train.pkl
    python tools/analyze_trajectory_statistics.py --data-path ./data/nuscenes/nuscenes_infos_val.pkl
    python tools/analyze_trajectory_statistics.py --data-path ./data/nuscenes/nuscenes_infos_train.pkl --save-json ./trajectory_stats.json
"""

import argparse
import pickle
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def analyze_trajectories(data_path, verbose=True):
    """
    Analyze trajectory statistics from a nuScenes data pickle file.
    
    Args:
        data_path (str): Path to the pickle file containing nuScenes infos
        verbose (bool): Whether to print detailed progress
    
    Returns:
        dict: Dictionary containing statistics for all trajectory types
    """
    # Load data
    if verbose:
        print(f"Loading data from: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    infos = data['infos']
    if verbose:
        print(f"Total samples: {len(infos)}")
    
    # Initialize statistics dictionaries
    stats = {
        'agent_6dof': {
            'x_min': float('inf'), 'x_max': float('-inf'),
            'y_min': float('inf'), 'y_max': float('-inf'),
            'z_min': float('inf'), 'z_max': float('-inf'),
            'yaw_min': float('inf'), 'yaw_max': float('-inf'),
            'pitch_min': float('inf'), 'pitch_max': float('-inf'),
            'roll_min': float('inf'), 'roll_max': float('-inf'),
            'count': 0
        },
        'agent_6dof_agent_first': {
            'x_min': float('inf'), 'x_max': float('-inf'),
            'y_min': float('inf'), 'y_max': float('-inf'),
            'z_min': float('inf'), 'z_max': float('-inf'),
            'yaw_min': float('inf'), 'yaw_max': float('-inf'),
            'pitch_min': float('inf'), 'pitch_max': float('-inf'),
            'roll_min': float('inf'), 'roll_max': float('-inf'),
            'count': 0
        },
        'ego_6dof': {
            'x_min': float('inf'), 'x_max': float('-inf'),
            'y_min': float('inf'), 'y_max': float('-inf'),
            'z_min': float('inf'), 'z_max': float('-inf'),
            'yaw_min': float('inf'), 'yaw_max': float('-inf'),
            'pitch_min': float('inf'), 'pitch_max': float('-inf'),
            'roll_min': float('inf'), 'roll_max': float('-inf'),
            'count': 0
        },
        'ego_6dof_ego': {
            'x_min': float('inf'), 'x_max': float('-inf'),
            'y_min': float('inf'), 'y_max': float('-inf'),
            'z_min': float('inf'), 'z_max': float('-inf'),
            'yaw_min': float('inf'), 'yaw_max': float('-inf'),
            'pitch_min': float('inf'), 'pitch_max': float('-inf'),
            'roll_min': float('inf'), 'roll_max': float('-inf'),
            'count': 0
        },
        'agent_2d': {
            'x_min': float('inf'), 'x_max': float('-inf'),
            'y_min': float('inf'), 'y_max': float('-inf'),
            'count': 0
        },
        'ego_2d': {
            'x_min': float('inf'), 'x_max': float('-inf'),
            'y_min': float('inf'), 'y_max': float('-inf'),
            'count': 0
        }
    }
    
    # Process each sample
    for idx, info in enumerate(infos):
        if verbose and (idx + 1) % 1000 == 0:
            print(f"Processing sample {idx + 1}/{len(infos)}")
        
        # Process agent 6DOF trajectories
        if 'gt_agent_fut_trajs_6dof' in info and 'gt_agent_fut_masks' in info:
            trajs_6dof = info['gt_agent_fut_trajs_6dof']  # (num_agents, fut_ts, 6)
            masks = info['gt_agent_fut_masks']  # (num_agents, fut_ts)
            
            if trajs_6dof.size > 0:
                valid_mask = masks > 0
                valid_data = trajs_6dof[valid_mask]  # (num_valid_points, 6)
                
                if valid_data.size > 0:
                    stats['agent_6dof']['x_min'] = min(stats['agent_6dof']['x_min'], valid_data[:, 0].min())
                    stats['agent_6dof']['x_max'] = max(stats['agent_6dof']['x_max'], valid_data[:, 0].max())
                    stats['agent_6dof']['y_min'] = min(stats['agent_6dof']['y_min'], valid_data[:, 1].min())
                    stats['agent_6dof']['y_max'] = max(stats['agent_6dof']['y_max'], valid_data[:, 1].max())
                    stats['agent_6dof']['z_min'] = min(stats['agent_6dof']['z_min'], valid_data[:, 2].min())
                    stats['agent_6dof']['z_max'] = max(stats['agent_6dof']['z_max'], valid_data[:, 2].max())
                    stats['agent_6dof']['yaw_min'] = min(stats['agent_6dof']['yaw_min'], valid_data[:, 3].min())
                    stats['agent_6dof']['yaw_max'] = max(stats['agent_6dof']['yaw_max'], valid_data[:, 3].max())
                    stats['agent_6dof']['pitch_min'] = min(stats['agent_6dof']['pitch_min'], valid_data[:, 4].min())
                    stats['agent_6dof']['pitch_max'] = max(stats['agent_6dof']['pitch_max'], valid_data[:, 4].max())
                    stats['agent_6dof']['roll_min'] = min(stats['agent_6dof']['roll_min'], valid_data[:, 5].min())
                    stats['agent_6dof']['roll_max'] = max(stats['agent_6dof']['roll_max'], valid_data[:, 5].max())
                    stats['agent_6dof']['count'] += valid_data.shape[0]
        
        # Process agent 6DOF poses in agent_first frame
        if 'gt_agent_fut_poses_6dof_agent_first' in info and 'gt_agent_fut_masks' in info:
            poses_6dof = info['gt_agent_fut_poses_6dof_agent_first']  # (num_agents, fut_ts, 6)
            masks = info['gt_agent_fut_masks']  # (num_agents, fut_ts)
            
            if poses_6dof.size > 0:
                valid_mask = masks > 0
                valid_data = poses_6dof[valid_mask]  # (num_valid_points, 6)
                
                if valid_data.size > 0:
                    stats['agent_6dof_agent_first']['x_min'] = min(stats['agent_6dof_agent_first']['x_min'], valid_data[:, 0].min())
                    stats['agent_6dof_agent_first']['x_max'] = max(stats['agent_6dof_agent_first']['x_max'], valid_data[:, 0].max())
                    stats['agent_6dof_agent_first']['y_min'] = min(stats['agent_6dof_agent_first']['y_min'], valid_data[:, 1].min())
                    stats['agent_6dof_agent_first']['y_max'] = max(stats['agent_6dof_agent_first']['y_max'], valid_data[:, 1].max())
                    stats['agent_6dof_agent_first']['z_min'] = min(stats['agent_6dof_agent_first']['z_min'], valid_data[:, 2].min())
                    stats['agent_6dof_agent_first']['z_max'] = max(stats['agent_6dof_agent_first']['z_max'], valid_data[:, 2].max())
                    stats['agent_6dof_agent_first']['yaw_min'] = min(stats['agent_6dof_agent_first']['yaw_min'], valid_data[:, 3].min())
                    stats['agent_6dof_agent_first']['yaw_max'] = max(stats['agent_6dof_agent_first']['yaw_max'], valid_data[:, 3].max())
                    stats['agent_6dof_agent_first']['pitch_min'] = min(stats['agent_6dof_agent_first']['pitch_min'], valid_data[:, 4].min())
                    stats['agent_6dof_agent_first']['pitch_max'] = max(stats['agent_6dof_agent_first']['pitch_max'], valid_data[:, 4].max())
                    stats['agent_6dof_agent_first']['roll_min'] = min(stats['agent_6dof_agent_first']['roll_min'], valid_data[:, 5].min())
                    stats['agent_6dof_agent_first']['roll_max'] = max(stats['agent_6dof_agent_first']['roll_max'], valid_data[:, 5].max())
                    stats['agent_6dof_agent_first']['count'] += valid_data.shape[0]
        
        # Process agent 2D trajectories
        if 'gt_agent_fut_trajs' in info and 'gt_agent_fut_masks' in info:
            trajs_2d = info['gt_agent_fut_trajs']  # (num_agents, fut_ts, 2)
            masks = info['gt_agent_fut_masks']  # (num_agents, fut_ts)
            
            if trajs_2d.size > 0:
                valid_mask = masks > 0
                valid_data = trajs_2d[valid_mask]  # (num_valid_points, 2)
                
                if valid_data.size > 0:
                    stats['agent_2d']['x_min'] = min(stats['agent_2d']['x_min'], valid_data[:, 0].min())
                    stats['agent_2d']['x_max'] = max(stats['agent_2d']['x_max'], valid_data[:, 0].max())
                    stats['agent_2d']['y_min'] = min(stats['agent_2d']['y_min'], valid_data[:, 1].min())
                    stats['agent_2d']['y_max'] = max(stats['agent_2d']['y_max'], valid_data[:, 1].max())
                    stats['agent_2d']['count'] += valid_data.shape[0]
        
        # Process ego 6DOF trajectories
        if 'gt_ego_fut_trajs_6dof' in info and 'gt_ego_fut_masks' in info:
            trajs_6dof = info['gt_ego_fut_trajs_6dof']  # (fut_ts, 6)
            masks = info['gt_ego_fut_masks']  # (fut_ts,)
            
            if trajs_6dof.size > 0:
                valid_mask = masks > 0
                valid_data = trajs_6dof[valid_mask]  # (num_valid_points, 6)
                
                if valid_data.size > 0:
                    stats['ego_6dof']['x_min'] = min(stats['ego_6dof']['x_min'], valid_data[:, 0].min())
                    stats['ego_6dof']['x_max'] = max(stats['ego_6dof']['x_max'], valid_data[:, 0].max())
                    stats['ego_6dof']['y_min'] = min(stats['ego_6dof']['y_min'], valid_data[:, 1].min())
                    stats['ego_6dof']['y_max'] = max(stats['ego_6dof']['y_max'], valid_data[:, 1].max())
                    stats['ego_6dof']['z_min'] = min(stats['ego_6dof']['z_min'], valid_data[:, 2].min())
                    stats['ego_6dof']['z_max'] = max(stats['ego_6dof']['z_max'], valid_data[:, 2].max())
                    stats['ego_6dof']['yaw_min'] = min(stats['ego_6dof']['yaw_min'], valid_data[:, 3].min())
                    stats['ego_6dof']['yaw_max'] = max(stats['ego_6dof']['yaw_max'], valid_data[:, 3].max())
                    stats['ego_6dof']['pitch_min'] = min(stats['ego_6dof']['pitch_min'], valid_data[:, 4].min())
                    stats['ego_6dof']['pitch_max'] = max(stats['ego_6dof']['pitch_max'], valid_data[:, 4].max())
                    stats['ego_6dof']['roll_min'] = min(stats['ego_6dof']['roll_min'], valid_data[:, 5].min())
                    stats['ego_6dof']['roll_max'] = max(stats['ego_6dof']['roll_max'], valid_data[:, 5].max())
                    stats['ego_6dof']['count'] += valid_data.shape[0]
        
        # Process ego 6DOF poses in ego frame
        if 'gt_ego_fut_poses_6dof_ego' in info and 'gt_ego_fut_masks' in info:
            poses_6dof = info['gt_ego_fut_poses_6dof_ego']  # (fut_ts, 6)
            masks = info['gt_ego_fut_masks']  # (fut_ts,)
            
            if poses_6dof.size > 0:
                valid_mask = masks > 0
                valid_data = poses_6dof[valid_mask]  # (num_valid_points, 6)
                
                if valid_data.size > 0:
                    stats['ego_6dof_ego']['x_min'] = min(stats['ego_6dof_ego']['x_min'], valid_data[:, 0].min())
                    stats['ego_6dof_ego']['x_max'] = max(stats['ego_6dof_ego']['x_max'], valid_data[:, 0].max())
                    stats['ego_6dof_ego']['y_min'] = min(stats['ego_6dof_ego']['y_min'], valid_data[:, 1].min())
                    stats['ego_6dof_ego']['y_max'] = max(stats['ego_6dof_ego']['y_max'], valid_data[:, 1].max())
                    stats['ego_6dof_ego']['z_min'] = min(stats['ego_6dof_ego']['z_min'], valid_data[:, 2].min())
                    stats['ego_6dof_ego']['z_max'] = max(stats['ego_6dof_ego']['z_max'], valid_data[:, 2].max())
                    stats['ego_6dof_ego']['yaw_min'] = min(stats['ego_6dof_ego']['yaw_min'], valid_data[:, 3].min())
                    stats['ego_6dof_ego']['yaw_max'] = max(stats['ego_6dof_ego']['yaw_max'], valid_data[:, 3].max())
                    stats['ego_6dof_ego']['pitch_min'] = min(stats['ego_6dof_ego']['pitch_min'], valid_data[:, 4].min())
                    stats['ego_6dof_ego']['pitch_max'] = max(stats['ego_6dof_ego']['pitch_max'], valid_data[:, 4].max())
                    stats['ego_6dof_ego']['roll_min'] = min(stats['ego_6dof_ego']['roll_min'], valid_data[:, 5].min())
                    stats['ego_6dof_ego']['roll_max'] = max(stats['ego_6dof_ego']['roll_max'], valid_data[:, 5].max())
                    stats['ego_6dof_ego']['count'] += valid_data.shape[0]
        
        # Process ego 2D trajectories
        if 'gt_ego_fut_trajs' in info and 'gt_ego_fut_masks' in info:
            trajs_2d = info['gt_ego_fut_trajs']  # (fut_ts, 2)
            masks = info['gt_ego_fut_masks']  # (fut_ts,)
            
            if trajs_2d.size > 0:
                valid_mask = masks > 0
                valid_data = trajs_2d[valid_mask]  # (num_valid_points, 2)
                
                if valid_data.size > 0:
                    stats['ego_2d']['x_min'] = min(stats['ego_2d']['x_min'], valid_data[:, 0].min())
                    stats['ego_2d']['x_max'] = max(stats['ego_2d']['x_max'], valid_data[:, 0].max())
                    stats['ego_2d']['y_min'] = min(stats['ego_2d']['y_min'], valid_data[:, 1].min())
                    stats['ego_2d']['y_max'] = max(stats['ego_2d']['y_max'], valid_data[:, 1].max())
                    stats['ego_2d']['count'] += valid_data.shape[0]
    
    return stats


def collect_trajectory_values(data_path, verbose=True):
    """
    Collect all trajectory values for histogram plotting.
    
    Args:
        data_path (str): Path to the pickle file
        verbose (bool): Whether to print progress
    
    Returns:
        dict: Dictionary containing all trajectory values
    """
    if verbose:
        print(f"Collecting trajectory values from: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    infos = data['infos']
    
    # Initialize lists to store all values
    values = {
        'agent_6dof': {'x': [], 'y': [], 'z': [], 'yaw': [], 'pitch': [], 'roll': []},
        'agent_6dof_agent_first': {'x': [], 'y': [], 'z': [], 'yaw': [], 'pitch': [], 'roll': []},
        'ego_6dof': {'x': [], 'y': [], 'z': [], 'yaw': [], 'pitch': [], 'roll': []},
        'ego_6dof_ego': {'x': [], 'y': [], 'z': [], 'yaw': [], 'pitch': [], 'roll': []},
        'agent_2d': {'x': [], 'y': []},
        'ego_2d': {'x': [], 'y': []}
    }
    
    # Collect all trajectory values
    for idx, info in enumerate(infos):
        if verbose and (idx + 1) % 1000 == 0:
            print(f"Collecting values from sample {idx + 1}/{len(infos)}")
        
        # Agent 6DOF
        if 'gt_agent_fut_trajs_6dof' in info and 'gt_agent_fut_masks' in info:
            trajs_6dof = info['gt_agent_fut_trajs_6dof']
            masks = info['gt_agent_fut_masks']
            if trajs_6dof.size > 0:
                valid_data = trajs_6dof[masks > 0]
                if valid_data.size > 0:
                    values['agent_6dof']['x'].extend(valid_data[:, 0].tolist())
                    values['agent_6dof']['y'].extend(valid_data[:, 1].tolist())
                    values['agent_6dof']['z'].extend(valid_data[:, 2].tolist())
                    values['agent_6dof']['yaw'].extend(valid_data[:, 3].tolist())
                    values['agent_6dof']['pitch'].extend(valid_data[:, 4].tolist())
                    values['agent_6dof']['roll'].extend(valid_data[:, 5].tolist())
        
        # Agent 6DOF in agent_first frame
        if 'gt_agent_fut_poses_6dof_agent_first' in info and 'gt_agent_fut_masks' in info:
            poses_6dof = info['gt_agent_fut_poses_6dof_agent_first']
            masks = info['gt_agent_fut_masks']
            if poses_6dof.size > 0:
                valid_data = poses_6dof[masks > 0]
                if valid_data.size > 0:
                    values['agent_6dof_agent_first']['x'].extend(valid_data[:, 0].tolist())
                    values['agent_6dof_agent_first']['y'].extend(valid_data[:, 1].tolist())
                    values['agent_6dof_agent_first']['z'].extend(valid_data[:, 2].tolist())
                    values['agent_6dof_agent_first']['yaw'].extend(valid_data[:, 3].tolist())
                    values['agent_6dof_agent_first']['pitch'].extend(valid_data[:, 4].tolist())
                    values['agent_6dof_agent_first']['roll'].extend(valid_data[:, 5].tolist())
        
        # Agent 2D
        if 'gt_agent_fut_trajs' in info and 'gt_agent_fut_masks' in info:
            trajs_2d = info['gt_agent_fut_trajs']
            masks = info['gt_agent_fut_masks']
            if trajs_2d.size > 0:
                valid_data = trajs_2d[masks > 0]
                if valid_data.size > 0:
                    values['agent_2d']['x'].extend(valid_data[:, 0].tolist())
                    values['agent_2d']['y'].extend(valid_data[:, 1].tolist())
        
        # Ego 6DOF
        if 'gt_ego_fut_trajs_6dof' in info and 'gt_ego_fut_masks' in info:
            trajs_6dof = info['gt_ego_fut_trajs_6dof']
            masks = info['gt_ego_fut_masks']
            if trajs_6dof.size > 0:
                valid_data = trajs_6dof[masks > 0]
                if valid_data.size > 0:
                    values['ego_6dof']['x'].extend(valid_data[:, 0].tolist())
                    values['ego_6dof']['y'].extend(valid_data[:, 1].tolist())
                    values['ego_6dof']['z'].extend(valid_data[:, 2].tolist())
                    values['ego_6dof']['yaw'].extend(valid_data[:, 3].tolist())
                    values['ego_6dof']['pitch'].extend(valid_data[:, 4].tolist())
                    values['ego_6dof']['roll'].extend(valid_data[:, 5].tolist())
        
        # Ego 6DOF in ego frame
        if 'gt_ego_fut_poses_6dof_ego' in info and 'gt_ego_fut_masks' in info:
            poses_6dof = info['gt_ego_fut_poses_6dof_ego']
            masks = info['gt_ego_fut_masks']
            if poses_6dof.size > 0:
                valid_data = poses_6dof[masks > 0]
                if valid_data.size > 0:
                    values['ego_6dof_ego']['x'].extend(valid_data[:, 0].tolist())
                    values['ego_6dof_ego']['y'].extend(valid_data[:, 1].tolist())
                    values['ego_6dof_ego']['z'].extend(valid_data[:, 2].tolist())
                    values['ego_6dof_ego']['yaw'].extend(valid_data[:, 3].tolist())
                    values['ego_6dof_ego']['pitch'].extend(valid_data[:, 4].tolist())
                    values['ego_6dof_ego']['roll'].extend(valid_data[:, 5].tolist())
        
        # Ego 2D
        if 'gt_ego_fut_trajs' in info and 'gt_ego_fut_masks' in info:
            trajs_2d = info['gt_ego_fut_trajs']
            masks = info['gt_ego_fut_masks']
            if trajs_2d.size > 0:
                valid_data = trajs_2d[masks > 0]
                if valid_data.size > 0:
                    values['ego_2d']['x'].extend(valid_data[:, 0].tolist())
                    values['ego_2d']['y'].extend(valid_data[:, 1].tolist())
    
    # Convert to numpy arrays
    for traj_type in values:
        for dim in values[traj_type]:
            values[traj_type][dim] = np.array(values[traj_type][dim])
    
    return values


def plot_histograms(values, output_dir='./trajectory_histograms'):
    """
    Plot histograms for all trajectory dimensions.
    
    Args:
        values (dict): Dictionary containing trajectory values
        output_dir (str): Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nPlotting histograms to: {output_dir}")
    
    # Plot Agent 6DOF - Always plot if any dimension has data
    agent_6dof_has_data = any(len(values['agent_6dof'][dim]) > 0 for dim in values['agent_6dof'])
    if agent_6dof_has_data:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Agent 6DOF Trajectory Distribution (gt_agent_fut_trajs_6dof)', fontsize=16)
        
        dims = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
        labels = ['X (m)', 'Y (m)', 'Z (m)', 'Yaw (rad)', 'Pitch (rad)', 'Roll (rad)']
        
        for idx, (dim, label) in enumerate(zip(dims, labels)):
            ax = axes[idx // 3, idx % 3]
            data = values['agent_6dof'][dim]
            
            if len(data) > 0:
                # Adaptive bins: use fewer bins for small datasets
                num_bins = min(100, max(10, len(data) // 10))
                ax.hist(data, bins=num_bins, alpha=0.7, edgecolor='black')
                mean_val = data.mean()
                std_val = data.std()
                ax.set_title(f'{label}: n={len(data)}, mean={mean_val:.4f}, std={std_val:.4f}')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, color='red')
                ax.set_title(f'{label}: No Data')
            
            ax.set_xlabel(label)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'agent_6dof_distribution.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: agent_6dof_distribution.png")
    
    # Plot Agent 6DOF in agent_first frame - Always plot if any dimension has data
    agent_6dof_agent_first_has_data = any(len(values['agent_6dof_agent_first'][dim]) > 0 for dim in values['agent_6dof_agent_first'])
    if agent_6dof_agent_first_has_data:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Agent 6DOF Pose Offsets in Agent_First Frame (gt_agent_fut_poses_6dof_agent_first)', fontsize=16)
        
        dims = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
        labels = ['X (m)', 'Y (m)', 'Z (m)', 'Yaw (rad)', 'Pitch (rad)', 'Roll (rad)']
        
        for idx, (dim, label) in enumerate(zip(dims, labels)):
            ax = axes[idx // 3, idx % 3]
            data = values['agent_6dof_agent_first'][dim]
            
            if len(data) > 0:
                # Adaptive bins: use fewer bins for small datasets
                num_bins = min(100, max(10, len(data) // 10))
                ax.hist(data, bins=num_bins, alpha=0.7, edgecolor='black')
                mean_val = data.mean()
                std_val = data.std()
                ax.set_title(f'{label}: n={len(data)}, mean={mean_val:.4f}, std={std_val:.4f}')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, color='red')
                ax.set_title(f'{label}: No Data')
            
            ax.set_xlabel(label)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'agent_6dof_agent_first_distribution.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: agent_6dof_agent_first_distribution.png")
    
    # Plot Ego 6DOF - Always plot if any dimension has data
    ego_6dof_has_data = any(len(values['ego_6dof'][dim]) > 0 for dim in values['ego_6dof'])
    if ego_6dof_has_data:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Ego 6DOF Trajectory Distribution (gt_ego_fut_trajs_6dof)', fontsize=16)
        
        dims = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
        labels = ['X (m)', 'Y (m)', 'Z (m)', 'Yaw (rad)', 'Pitch (rad)', 'Roll (rad)']
        
        for idx, (dim, label) in enumerate(zip(dims, labels)):
            ax = axes[idx // 3, idx % 3]
            data = values['ego_6dof'][dim]
            
            if len(data) > 0:
                # Adaptive bins: use fewer bins for small datasets
                num_bins = min(100, max(10, len(data) // 10))
                ax.hist(data, bins=num_bins, alpha=0.7, edgecolor='black')
                mean_val = data.mean()
                std_val = data.std()
                ax.set_title(f'{label}: n={len(data)}, mean={mean_val:.4f}, std={std_val:.4f}')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, color='red')
                ax.set_title(f'{label}: No Data')
            
            ax.set_xlabel(label)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ego_6dof_distribution.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: ego_6dof_distribution.png")
    
    # Plot Ego 6DOF in ego frame - Always plot if any dimension has data
    ego_6dof_ego_has_data = any(len(values['ego_6dof_ego'][dim]) > 0 for dim in values['ego_6dof_ego'])
    if ego_6dof_ego_has_data:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Ego 6DOF Pose Offsets in Ego Frame (gt_ego_fut_poses_6dof_ego)', fontsize=16)
        
        dims = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
        labels = ['X (m)', 'Y (m)', 'Z (m)', 'Yaw (rad)', 'Pitch (rad)', 'Roll (rad)']
        
        for idx, (dim, label) in enumerate(zip(dims, labels)):
            ax = axes[idx // 3, idx % 3]
            data = values['ego_6dof_ego'][dim]
            
            if len(data) > 0:
                # Adaptive bins: use fewer bins for small datasets
                num_bins = min(100, max(10, len(data) // 10))
                ax.hist(data, bins=num_bins, alpha=0.7, edgecolor='black')
                mean_val = data.mean()
                std_val = data.std()
                ax.set_title(f'{label}: n={len(data)}, mean={mean_val:.4f}, std={std_val:.4f}')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, color='red')
                ax.set_title(f'{label}: No Data')
            
            ax.set_xlabel(label)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ego_6dof_ego_distribution.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: ego_6dof_ego_distribution.png")
    
    # Plot Agent 2D - Always plot if any dimension has data
    agent_2d_has_data = any(len(values['agent_2d'][dim]) > 0 for dim in values['agent_2d'])
    if agent_2d_has_data:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Agent 2D Trajectory Distribution (gt_agent_fut_trajs)', fontsize=16)
        
        for idx, (dim, label) in enumerate([('x', 'X (m)'), ('y', 'Y (m)')]):
            ax = axes[idx]
            data = values['agent_2d'][dim]
            
            if len(data) > 0:
                # Adaptive bins: use fewer bins for small datasets
                num_bins = min(100, max(10, len(data) // 10))
                ax.hist(data, bins=num_bins, alpha=0.7, edgecolor='black')
                mean_val = data.mean()
                std_val = data.std()
                ax.set_title(f'{label}: n={len(data)}, mean={mean_val:.4f}, std={std_val:.4f}')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, color='red')
                ax.set_title(f'{label}: No Data')
            
            ax.set_xlabel(label)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'agent_2d_distribution.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: agent_2d_distribution.png")
    
    # Plot Ego 2D - Always plot if any dimension has data
    ego_2d_has_data = any(len(values['ego_2d'][dim]) > 0 for dim in values['ego_2d'])
    if ego_2d_has_data:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Ego 2D Trajectory Distribution (gt_ego_fut_trajs)', fontsize=16)
        
        for idx, (dim, label) in enumerate([('x', 'X (m)'), ('y', 'Y (m)')]):
            ax = axes[idx]
            data = values['ego_2d'][dim]
            
            if len(data) > 0:
                # Adaptive bins: use fewer bins for small datasets
                num_bins = min(100, max(10, len(data) // 10))
                ax.hist(data, bins=num_bins, alpha=0.7, edgecolor='black')
                mean_val = data.mean()
                std_val = data.std()
                ax.set_title(f'{label}: n={len(data)}, mean={mean_val:.4f}, std={std_val:.4f}')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, color='red')
                ax.set_title(f'{label}: No Data')
            
            ax.set_xlabel(label)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ego_2d_distribution.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: ego_2d_distribution.png")
    
    # Plot comparison: Agent vs Ego (2D)
    if len(values['agent_2d']['x']) > 0 and len(values['ego_2d']['x']) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Agent vs Ego 2D Trajectory Comparison', fontsize=16)
        
        for idx, (dim, label) in enumerate([('x', 'X (m)'), ('y', 'Y (m)')]):
            ax = axes[idx]
            ax.hist(values['agent_2d'][dim], bins=100, alpha=0.5, label='Agent', edgecolor='black')
            ax.hist(values['ego_2d'][dim], bins=100, alpha=0.5, label='Ego', edgecolor='black')
            ax.set_xlabel(label)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{label} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'agent_vs_ego_2d_comparison.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: agent_vs_ego_2d_comparison.png")
    
    print(f"\nAll histograms saved to: {output_dir}\n")


def print_statistics(stats):
    """Print statistics in a formatted way."""
    print("\n" + "="*80)
    print("TRAJECTORY STATISTICS")
    print("="*80)
    
    # 6DOF Statistics
    print("\n" + "-"*80)
    print("6DOF Trajectories (All dimensions: x, y, z, yaw, pitch, roll)")
    print("-"*80)
    
    print("\nAgent Trajectories in Lidar Frame (gt_agent_fut_trajs_6dof):")
    if stats['agent_6dof']['count'] > 0:
        print(f"  X dimension:     min={stats['agent_6dof']['x_min']:>10.4f}, max={stats['agent_6dof']['x_max']:>10.4f}")
        print(f"  Y dimension:     min={stats['agent_6dof']['y_min']:>10.4f}, max={stats['agent_6dof']['y_max']:>10.4f}")
        print(f"  Z dimension:     min={stats['agent_6dof']['z_min']:>10.4f}, max={stats['agent_6dof']['z_max']:>10.4f}")
        print(f"  Yaw (rad):       min={stats['agent_6dof']['yaw_min']:>10.4f}, max={stats['agent_6dof']['yaw_max']:>10.4f}")
        print(f"  Pitch (rad):     min={stats['agent_6dof']['pitch_min']:>10.4f}, max={stats['agent_6dof']['pitch_max']:>10.4f}")
        print(f"  Roll (rad):      min={stats['agent_6dof']['roll_min']:>10.4f}, max={stats['agent_6dof']['roll_max']:>10.4f}")
        print(f"  Total valid points: {stats['agent_6dof']['count']}")
    else:
        print("  No valid agent 6DOF trajectory points found")
    
    print("\nAgent Pose Offsets in Agent_First Frame (gt_agent_fut_poses_6dof_agent_first):")
    if stats['agent_6dof_agent_first']['count'] > 0:
        print(f"  X dimension:     min={stats['agent_6dof_agent_first']['x_min']:>10.4f}, max={stats['agent_6dof_agent_first']['x_max']:>10.4f}")
        print(f"  Y dimension:     min={stats['agent_6dof_agent_first']['y_min']:>10.4f}, max={stats['agent_6dof_agent_first']['y_max']:>10.4f}")
        print(f"  Z dimension:     min={stats['agent_6dof_agent_first']['z_min']:>10.4f}, max={stats['agent_6dof_agent_first']['z_max']:>10.4f}")
        print(f"  Yaw (rad):       min={stats['agent_6dof_agent_first']['yaw_min']:>10.4f}, max={stats['agent_6dof_agent_first']['yaw_max']:>10.4f}")
        print(f"  Pitch (rad):     min={stats['agent_6dof_agent_first']['pitch_min']:>10.4f}, max={stats['agent_6dof_agent_first']['pitch_max']:>10.4f}")
        print(f"  Roll (rad):      min={stats['agent_6dof_agent_first']['roll_min']:>10.4f}, max={stats['agent_6dof_agent_first']['roll_max']:>10.4f}")
        print(f"  Total valid points: {stats['agent_6dof_agent_first']['count']}")
    else:
        print("  No valid agent 6DOF pose offset points found")
    
    print("\nEgo Trajectories in Lidar Frame (gt_ego_fut_trajs_6dof):")
    if stats['ego_6dof']['count'] > 0:
        print(f"  X dimension:     min={stats['ego_6dof']['x_min']:>10.4f}, max={stats['ego_6dof']['x_max']:>10.4f}")
        print(f"  Y dimension:     min={stats['ego_6dof']['y_min']:>10.4f}, max={stats['ego_6dof']['y_max']:>10.4f}")
        print(f"  Z dimension:     min={stats['ego_6dof']['z_min']:>10.4f}, max={stats['ego_6dof']['z_max']:>10.4f}")
        print(f"  Yaw (rad):       min={stats['ego_6dof']['yaw_min']:>10.4f}, max={stats['ego_6dof']['yaw_max']:>10.4f}")
        print(f"  Pitch (rad):     min={stats['ego_6dof']['pitch_min']:>10.4f}, max={stats['ego_6dof']['pitch_max']:>10.4f}")
        print(f"  Roll (rad):      min={stats['ego_6dof']['roll_min']:>10.4f}, max={stats['ego_6dof']['roll_max']:>10.4f}")
        print(f"  Total valid points: {stats['ego_6dof']['count']}")
    else:
        print("  No valid ego 6DOF trajectory points found")
    
    print("\nEgo Pose Offsets in Ego Frame (gt_ego_fut_poses_6dof_ego):")
    if stats['ego_6dof_ego']['count'] > 0:
        print(f"  X dimension:     min={stats['ego_6dof_ego']['x_min']:>10.4f}, max={stats['ego_6dof_ego']['x_max']:>10.4f}")
        print(f"  Y dimension:     min={stats['ego_6dof_ego']['y_min']:>10.4f}, max={stats['ego_6dof_ego']['y_max']:>10.4f}")
        print(f"  Z dimension:     min={stats['ego_6dof_ego']['z_min']:>10.4f}, max={stats['ego_6dof_ego']['z_max']:>10.4f}")
        print(f"  Yaw (rad):       min={stats['ego_6dof_ego']['yaw_min']:>10.4f}, max={stats['ego_6dof_ego']['yaw_max']:>10.4f}")
        print(f"  Pitch (rad):     min={stats['ego_6dof_ego']['pitch_min']:>10.4f}, max={stats['ego_6dof_ego']['pitch_max']:>10.4f}")
        print(f"  Roll (rad):      min={stats['ego_6dof_ego']['roll_min']:>10.4f}, max={stats['ego_6dof_ego']['roll_max']:>10.4f}")
        print(f"  Total valid points: {stats['ego_6dof_ego']['count']}")
    else:
        print("  No valid ego 6DOF pose offset points found")
    
    # 2D Statistics
    print("\n" + "-"*80)
    print("2D Trajectories (Only x, y dimensions)")
    print("-"*80)
    
    print("\nAgent Trajectories (gt_agent_fut_trajs):")
    if stats['agent_2d']['count'] > 0:
        print(f"  X dimension:     min={stats['agent_2d']['x_min']:>10.4f}, max={stats['agent_2d']['x_max']:>10.4f}")
        print(f"  Y dimension:     min={stats['agent_2d']['y_min']:>10.4f}, max={stats['agent_2d']['y_max']:>10.4f}")
        print(f"  Total valid points: {stats['agent_2d']['count']}")
    else:
        print("  No valid agent 2D trajectory points found")
    
    print("\nEgo Trajectories (gt_ego_fut_trajs):")
    if stats['ego_2d']['count'] > 0:
        print(f"  X dimension:     min={stats['ego_2d']['x_min']:>10.4f}, max={stats['ego_2d']['x_max']:>10.4f}")
        print(f"  Y dimension:     min={stats['ego_2d']['y_min']:>10.4f}, max={stats['ego_2d']['y_max']:>10.4f}")
        print(f"  Total valid points: {stats['ego_2d']['count']}")
    else:
        print("  No valid ego 2D trajectory points found")
    
    print("="*80 + "\n")


def save_statistics_json(stats, output_path):
    """Save statistics to a JSON file."""
    # Convert inf values and numpy types to None/Python types for JSON serialization
    def convert_inf(obj):
        if isinstance(obj, dict):
            return {k: convert_inf(v) for k, v in obj.items()}
        elif obj == float('inf'):
            return None
        elif obj == float('-inf'):
            return None
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()  # convert to Python scalar
        else:
            return obj
    
    stats_json = convert_inf(stats)
    
    with open(output_path, 'w') as f:
        json.dump(stats_json, f, indent=2)
    
    print(f"Statistics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze trajectory statistics from nuScenes data')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the pickle file (e.g., nuscenes_infos_train.pkl)')
    parser.add_argument('--save-json', type=str, default=None,
                        help='Optional: Path to save statistics as JSON file')
    parser.add_argument('--plot-histograms', action='store_true',
                        help='Generate histogram plots for trajectory distributions')
    parser.add_argument('--output-dir', type=str, default='./trajectory_histograms',
                        help='Directory to save histogram plots (default: ./trajectory_histograms)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.data_path).exists():
        print(f"Error: File not found: {args.data_path}")
        return
    
    # Analyze trajectories
    stats = analyze_trajectories(args.data_path, verbose=not args.quiet)
    
    # Print statistics
    print_statistics(stats)
    
    # Save to JSON if requested

    # Plot histograms if requested
    if args.plot_histograms:
        values = collect_trajectory_values(args.data_path, verbose=not args.quiet)
        plot_histograms(values, output_dir=args.output_dir)

    if args.save_json:
        save_statistics_json(stats, args.save_json)
    
    

if __name__ == '__main__':
    main()
