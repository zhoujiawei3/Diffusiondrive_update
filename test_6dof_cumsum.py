#!/usr/bin/env python3
"""
Test script to verify 6DOF trajectory encoding and decoding.
This directly uses the nuscenes_converter.py encoding logic and tests cumsum_6dof decoding.
"""

import numpy as np
import torch
from pyquaternion import Quaternion


def nuscenes_style_encode(ego_fut_poses_global):
    """
    Encode absolute global poses to 6DOF offsets, exactly following nuscenes_converter.py logic.
    
    Args:
        ego_fut_poses_global: list of (ego_fut_ts+1) 4x4 transformation matrices in global frame
    
    Returns:
        ego_fut_trajs_6dof: numpy array of shape [ego_fut_ts, 6] containing [dx, dy, dz, dyaw, dpitch, droll]
        ego_fut_poses_lidar: list of (ego_fut_ts+1) 4x4 transformation matrices in lidar frame (for verification)
    """
    ego_fut_ts = len(ego_fut_poses_global) - 1
    
    # Simulate current pose (first pose is "current" pose in global frame)
    pose_current_global = ego_fut_poses_global[0]
    
    # Build transformation matrices for two-step conversion
    # Step 1: global to ego (current pose frame)
    ego_rot_mat = pose_current_global[:3, :3].T  # Inverse rotation
    ego_trans = pose_current_global[:3, 3]
    global2ego = np.eye(4)
    global2ego[:3, :3] = ego_rot_mat
    global2ego[:3, 3] = -ego_rot_mat @ ego_trans
    
    # Step 2: ego to lidar (identity for simplicity - assume lidar at ego origin)
    ego2lidar = np.eye(4)
    
    # Transform all future poses to current lidar frame
    ego_fut_poses_lidar = []
    ego_fut_poses_ego = []
    for pose_global in ego_fut_poses_global:
        # First: global to ego
        pose_ego = global2ego @ pose_global
        ego_fut_poses_ego.append(pose_ego.copy())
        # Second: ego to lidar
        pose_lidar = ego2lidar @ pose_ego
        ego_fut_poses_lidar.append(pose_lidar)
    
    # Extract positions in lidar frame
    ego_fut_positions_lidar = np.array([pose[:3, 3] for pose in ego_fut_poses_lidar])
    
    # Compute 6DOF offsets (this is the GT stored in data)
    ego_fut_trajs_6dof = np.zeros((ego_fut_ts, 6))
    
    for i in range(ego_fut_ts):
        # Translation offset: simple position difference in lidar frame
        ego_fut_trajs_6dof[i, :3] = ego_fut_positions_lidar[i+1] - ego_fut_positions_lidar[i]
        
        # Rotation offset: relative rotation from frame i to frame i+1
        pose_cur = ego_fut_poses_lidar[i]
        pose_next = ego_fut_poses_lidar[i+1]
        relative_rotation = pose_cur[:3, :3].T @ pose_next[:3, :3]
        
        # Extract rotation offset as euler angles
        rel_quat = Quaternion(matrix=relative_rotation)
        yaw, pitch, roll = rel_quat.yaw_pitch_roll
        ego_fut_trajs_6dof[i, 3:] = [yaw, pitch, roll]
    
    return ego_fut_trajs_6dof, ego_fut_poses_lidar


def cumsum_6dof(traj_offsets_6dof):
    """
    Accumulate 6DOF trajectory offsets using quaternion composition.
    (Copy of the decoder method)
    
    Args:
        traj_offsets_6dof: tensor of shape [..., ego_fut_ts, 6] containing [dx, dy, dz, dyaw, dpitch, droll]
    
    Returns:
        accumulated_6dof: tensor of shape [..., ego_fut_ts, 6] containing absolute [x, y, z, yaw, pitch, roll]
    
    Note:
        - First element (t=0) is kept as-is (no accumulation)
        - Starting from t=1, accumulate relative offsets
    """
    # Get shape info
    *batch_dims, ego_fut_ts, _ = traj_offsets_6dof.shape
    device = traj_offsets_6dof.device
    dtype = traj_offsets_6dof.dtype
    
    # Flatten batch dimensions for easier processing
    flat_offsets = traj_offsets_6dof.reshape(-1, ego_fut_ts, 6)
    batch_size = flat_offsets.shape[0]
    
    # Initialize output - copy input first
    accumulated = flat_offsets.clone()
    
    # Process each sample in batch
    for b in range(batch_size):
        # First element stays as-is (already copied in accumulated)
        # Start accumulation from t=1
        
        # Initialize with first element
        current_pos = flat_offsets[b, 0, :3].clone()
        current_yaw = flat_offsets[b, 0, 3].item()
        current_pitch = flat_offsets[b, 0, 4].item()
        current_roll = flat_offsets[b, 0, 5].item()
        
        # Create initial quaternion from euler angles
        current_quat = Quaternion(axis=[0, 0, 1], angle=current_yaw) * \
                      Quaternion(axis=[0, 1, 0], angle=current_pitch) * \
                      Quaternion(axis=[1, 0, 0], angle=current_roll)
        
        for t in range(1, ego_fut_ts):
            # Get current offset
            pos_offset = flat_offsets[b, t, :3]
            rot_offset = flat_offsets[b, t, 3:]
            
            # Position accumulation: offsets are in global frame, just add
            current_pos = current_pos + pos_offset
            
            # Rotation composition using quaternion multiplication
            dyaw = rot_offset[0].item()
            dpitch = rot_offset[1].item()
            droll = rot_offset[2].item()
            
            # Create quaternion from offset euler angles
            offset_quat = Quaternion(axis=[0, 0, 1], angle=dyaw) * \
                         Quaternion(axis=[0, 1, 0], angle=dpitch) * \
                         Quaternion(axis=[1, 0, 0], angle=droll)
            
            # Compose quaternions: q_new = q_current * q_offset
            current_quat = current_quat * offset_quat
            
            # Extract euler angles from composed quaternion
            yaw, pitch, roll = current_quat.yaw_pitch_roll
            
            # Store accumulated pose
            accumulated[b, t, :3] = current_pos
            accumulated[b, t, 3:] = torch.tensor([yaw, pitch, roll], device=device, dtype=dtype)
    
    # Reshape back to original batch dimensions
    accumulated = accumulated.reshape(*batch_dims, ego_fut_ts, 6)
    return accumulated


def extract_6dof_from_poses_in_lidar_frame(poses_lidar):
    """
    Extract 6DOF representation from transformation matrices in lidar frame.
    
    Args:
        poses_lidar: list of 4x4 transformation matrices in lidar frame
    
    Returns:
        poses_6dof: tensor of shape [num_timesteps, 6] containing [x, y, z, yaw, pitch, roll]
    """
    num_timesteps = len(poses_lidar)
    poses_6dof = np.zeros((num_timesteps, 6))
    
    for i, pose in enumerate(poses_lidar):
        # Position in lidar frame
        poses_6dof[i, :3] = pose[:3, 3]
        
        # Rotation as euler angles
        quat = Quaternion(matrix=pose[:3, :3])
        yaw, pitch, roll = quat.yaw_pitch_roll
        poses_6dof[i, 3:] = [yaw, pitch, roll]
    
    return torch.from_numpy(poses_6dof).float()


def generate_global_trajectory(num_timesteps=7):
    """
    Generate a test trajectory with realistic ego vehicle motion in global frame.
    
    Returns:
        poses_global: list of (num_timesteps) 4x4 transformation matrices in global frame
    """
    poses_global = []
    
    # Start at some global position
    current_pos = np.array([100.0, 200.0, 0.0])  # Global coordinates
    current_yaw = np.pi / 4  # Start facing northeast
    current_pitch = 0.0
    current_roll = 0.0
    
    for t in range(num_timesteps):
        # Create rotation matrix from euler angles (ZYX order)
        cy, sy = np.cos(current_yaw), np.sin(current_yaw)
        cp, sp = np.cos(current_pitch), np.sin(current_pitch)
        cr, sr = np.cos(current_roll), np.sin(current_roll)
        
        rot_mat = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        # Create 4x4 transformation matrix in global frame
        pose_global = np.eye(4)
        pose_global[:3, :3] = rot_mat
        pose_global[:3, 3] = current_pos
        poses_global.append(pose_global)
        
        # Update for next timestep (simulate driving motion in global frame)
        forward_distance = 2.0 + 0.5 * np.sin(t)  # varying speed
        lateral_distance = 0.1 * np.cos(t)  # slight lateral motion
        vertical_distance = 0.05 * np.sin(t)  # slight vertical motion
        
        # Transform motion by current rotation (move in vehicle's local frame)
        motion_local = np.array([forward_distance, lateral_distance, vertical_distance])
        motion_global = rot_mat @ motion_local
        current_pos += motion_global
        
        # Update rotation (turning + slight pitch/roll changes)
        current_yaw += 0.1 * (1 + 0.3 * np.sin(t))
        current_pitch += 0.02 * np.cos(t)
        current_roll += 0.01 * np.sin(t * 1.5)
    
    return poses_global


def test_encoding_decoding():
    """
    Main test function to verify encoding and decoding consistency.
    Uses the exact nuscenes_converter.py encoding flow.
    """
    print("=" * 80)
    print("Testing 6DOF Encoding/Decoding Consistency")
    print("(Using nuscenes_converter.py encoding logic)")
    print("=" * 80)
    
    # Test 1: Single trajectory - Full nuscenes_converter flow
    print("\n[Test 1] Single trajectory test with nuscenes_converter flow")
    print("-" * 80)
    
    ego_fut_ts = 6
    num_poses = ego_fut_ts + 1  # 7 poses -> 6 offsets
    
    # Step 1: Generate trajectory in global frame
    poses_global = generate_global_trajectory(num_poses)
    print(f"Generated {num_poses} poses in global frame")
    
    # Step 2: Encode using nuscenes_converter logic
    #   Input: poses in global frame
    #   Output: 6DOF offsets in lidar frame
    ego_fut_trajs_6dof, ego_fut_poses_lidar = nuscenes_style_encode(poses_global)
    print(f"Encoded offsets shape: {ego_fut_trajs_6dof.shape}")
    print(f"Encoded offsets (first 3 steps):\n{ego_fut_trajs_6dof[:3]}")
    
    # Step 3: Extract GT poses in lidar frame (for comparison)
    gt_poses_lidar_6dof = extract_6dof_from_poses_in_lidar_frame(ego_fut_poses_lidar)
    print(f"\nGT poses in lidar frame shape: {gt_poses_lidar_6dof.shape}")
    print(f"GT poses in lidar frame (first 4):\n{gt_poses_lidar_6dof[:4]}")
    
    # Step 4: Decode using cumsum_6dof
    #   Input: 6DOF offsets
    #   Output: accumulated 6DOF poses
    offsets_tensor = torch.from_numpy(ego_fut_trajs_6dof).float().unsqueeze(0)  # Add batch dim
    accumulated = cumsum_6dof(offsets_tensor).squeeze(0)
    print(f"\nDecoded (accumulated) poses shape: {accumulated.shape}")
    print(f"Decoded poses (first 3):\n{accumulated[:3]}")
    
    # Step 5: Compare decoded poses with GT poses in lidar frame
    #   Note: GT has (ego_fut_ts+1) poses, accumulated has ego_fut_ts poses
    #   accumulated[i] should match gt_poses_lidar_6dof[i+1]
    gt_relative = gt_poses_lidar_6dof[1:]  # Skip first pose (current pose at origin)
    error = torch.abs(accumulated - gt_relative)
    max_error = error.max()
    mean_error = error.mean()
    
    print(f"\n[Results]")
    print(f"Max error: {max_error:.6f}")
    print(f"Mean error: {mean_error:.6f}")
    print(f"Position error (mean): {error[:, :3].mean():.6f}")
    print(f"Rotation error (mean): {error[:, 3:].mean():.6f}")
    
    # Check if errors are within tolerance
    tolerance = 0.01  # 1cm for position, ~0.6 degree for rotation
    if max_error < tolerance:
        print(f"✓ PASS: Max error {max_error:.6f} < tolerance {tolerance}")
    else:
        print(f"✗ FAIL: Max error {max_error:.6f} >= tolerance {tolerance}")
        print(f"\nDetailed error (first 3 steps):")
        for i in range(min(3, len(error))):
            print(f"  Step {i}: pos_err={error[i, :3].numpy()}, rot_err={error[i, 3:].numpy()}")
    
    # Test 2: Batch of trajectories
    print("\n" + "=" * 80)
    print("[Test 2] Batch trajectory test (bs=4, cmd=3, mode=1)")
    print("-" * 80)
    
    batch_size = 4
    cmd_modes = 3
    modal_modes = 1
    ego_fut_ts = 6
    
    # Generate multiple trajectories
    all_offsets = []
    all_gt_poses = []
    
    for b in range(batch_size):
        for c in range(cmd_modes):
            for m in range(modal_modes):
                poses_global = generate_global_trajectory(ego_fut_ts + 1)
                ego_fut_trajs_6dof, ego_fut_poses_lidar = nuscenes_style_encode(poses_global)
                gt_poses_lidar_6dof = extract_6dof_from_poses_in_lidar_frame(ego_fut_poses_lidar)
                
                offsets = torch.from_numpy(ego_fut_trajs_6dof).float()
                gt_poses = gt_poses_lidar_6dof[1:]  # Skip first pose
                
                all_offsets.append(offsets)
                all_gt_poses.append(gt_poses)
    
    # Stack into batch tensor
    batch_offsets = torch.stack(all_offsets).reshape(batch_size, cmd_modes, modal_modes, ego_fut_ts, 6)
    batch_gt_poses = torch.stack(all_gt_poses).reshape(batch_size, cmd_modes, modal_modes, ego_fut_ts, 6)
    
    print(f"Batch offsets shape: {batch_offsets.shape}")
    print(f"Batch GT poses shape: {batch_gt_poses.shape}")
    
    # Decode batch
    batch_accumulated = cumsum_6dof(batch_offsets)
    
    # Compare
    batch_error = torch.abs(batch_accumulated - batch_gt_poses)
    max_error = batch_error.max()
    mean_error = batch_error.mean()
    
    print(f"\n[Results]")
    print(f"Max error: {max_error:.6f}")
    print(f"Mean error: {mean_error:.6f}")
    print(f"Position error (mean): {batch_error[..., :3].mean():.6f}")
    print(f"Rotation error (mean): {batch_error[..., 3:].mean():.6f}")
    
    if max_error < tolerance:
        print(f"✓ PASS: Max error {max_error:.6f} < tolerance {tolerance}")
    else:
        print(f"✗ FAIL: Max error {max_error:.6f} >= tolerance {tolerance}")
    
    # Test 3: Edge cases
    print("\n" + "=" * 80)
    print("[Test 3] Edge cases")
    print("-" * 80)
    
    # Test 3a: Zero motion
    print("\n[Test 3a] Zero motion test")
    zero_offsets = torch.zeros(1, 6, 6)
    zero_accumulated = cumsum_6dof(zero_offsets)
    print(f"Zero offsets accumulated: {zero_accumulated.squeeze()}")
    if torch.allclose(zero_accumulated, torch.zeros_like(zero_accumulated), atol=1e-6):
        print("✓ PASS: Zero offsets produce zero accumulated poses")
    else:
        print("✗ FAIL: Zero offsets should produce zero accumulated poses")
    
    # Test 3b: Pure translation
    print("\n[Test 3b] Pure translation test")
    trans_offsets = torch.zeros(1, 6, 6)
    trans_offsets[0, :, 0] = 1.0  # Move 1m in x direction each step
    trans_offsets[0, :, 1] = 0.5  # Move 0.5m in y direction each step
    trans_accumulated = cumsum_6dof(trans_offsets)
    expected_x = torch.arange(1, 7).float()
    expected_y = torch.arange(1, 7).float() * 0.5
    print(f"Translation accumulated x: {trans_accumulated[0, :, 0]}")
    print(f"Expected x: {expected_x}")
    print(f"Translation accumulated y: {trans_accumulated[0, :, 1]}")
    print(f"Expected y: {expected_y}")
    if torch.allclose(trans_accumulated[0, :, 0], expected_x, atol=1e-5) and \
       torch.allclose(trans_accumulated[0, :, 1], expected_y, atol=1e-5):
        print("✓ PASS: Pure translation accumulates correctly")
    else:
        print("✗ FAIL: Pure translation accumulation mismatch")
    
    # Test 3c: Pure rotation
    print("\n[Test 3c] Pure rotation test")
    rot_offsets = torch.zeros(1, 6, 6)
    rot_offsets[0, :, 3] = 0.1  # Rotate 0.1 rad in yaw each step
    rot_accumulated = cumsum_6dof(rot_offsets)
    expected_yaw = torch.arange(1, 7).float() * 0.1
    print(f"Rotation accumulated yaw: {rot_accumulated[0, :, 3]}")
    print(f"Expected yaw: {expected_yaw}")
    if torch.allclose(rot_accumulated[0, :, 3], expected_yaw, atol=1e-4):
        print("✓ PASS: Pure rotation accumulates correctly")
    else:
        print("✗ FAIL: Pure rotation accumulation mismatch")
        print(f"Difference: {rot_accumulated[0, :, 3] - expected_yaw}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_encoding_decoding()
