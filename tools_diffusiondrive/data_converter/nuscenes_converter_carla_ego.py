import os
import math
import copy
import argparse
from os import path as osp
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

import mmcv

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global

from projects.mmdet3d_plugin.datasets.map_utils.nuscmap_extractor import NuscMapExtractor

NameMapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def convert_local_coords_to_global_6dof(coordinates: np.ndarray,
                                        translation: Tuple[float, float, float],
                                        rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts local 6DOF coordinates to global 6DOF coordinates.
    :param coordinates: x,y,z,yaw,pitch,roll locations. array of shape [n_steps, 6]
    :param translation: Tuple of (x, y, z) location that is the center of the new frame
    :param rotation: Tuple representation of quaternion of new frame (w, x, y, z).
    :return: x,y,z,yaw,pitch,roll locations in global frame, array of shape [n_steps, 6].
    """
    if coordinates.size == 0:
        return coordinates
    
    # Extract positions and orientations
    local_positions = coordinates[:, :3]  # (n, 3)
    local_orientations = coordinates[:, 3:]  # (n, 3) - yaw, pitch, roll
    
    # Build transformation matrix from local to global
    base_quat = Quaternion(rotation)
    base_trans = np.array(translation)
    base_rot_mat = base_quat.rotation_matrix
    
    global_coords = np.zeros_like(coordinates)
    
    for i in range(len(coordinates)):
        # Transform position: rotate then translate
        global_pos = base_rot_mat @ local_positions[i] + base_trans
        global_coords[i, :3] = global_pos
        
        # Transform orientation: compose rotations
        # Local orientation as quaternion
        yaw, pitch, roll = local_orientations[i]
        # Create rotation matrix from euler angles (ZYX order for yaw-pitch-roll)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        
        # ZYX euler to rotation matrix
        local_rot_mat = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        # Compose rotations: global = base * local
        global_rot_mat = base_rot_mat @ local_rot_mat
        
        # Convert back to euler angles
        global_quat = Quaternion(matrix=global_rot_mat)
        global_yaw, global_pitch, global_roll = global_quat.yaw_pitch_roll
        global_coords[i, 3:] = [global_yaw, global_pitch, global_roll]
    
    return global_coords

def get_future_for_agent_6dof(predict_helper, instance_token: str, sample_token: str,
                               seconds: float, in_agent_frame: bool) -> np.ndarray:
    """
    Retrieves the agent's future 6DOF trajectory (x, y, z, yaw, pitch, roll).
    :param predict_helper: PredictHelper instance.
    :param instance_token: Instance token.
    :param sample_token: Sample token.
    :param seconds: How much future data to retrieve.
    :param in_agent_frame: If true, coordinates are in the agent's local frame.
    :return: np.ndarray of shape [n_timesteps, 6] with [x, y, z, yaw, pitch, roll].
    """
    # Use PredictHelper to get the list of future annotation records (full records)
    # predict_helper.get_future_for_agent(..., just_xy=False) returns list of records
    starting_annotation = predict_helper.get_sample_annotation(instance_token, sample_token)
    sequence = predict_helper.get_future_for_agent(instance_token, sample_token, seconds, in_agent_frame=False, just_xy=False)

    if len(sequence) == 0:
        return np.zeros((0, 6))

    # Extract full 6DOF pose from each annotation (in global frame)
    coords_6dof = np.zeros((len(sequence), 6))
    for i, record in enumerate(sequence):
        coords_6dof[i, :3] = record['translation']
        quat = Quaternion(record['rotation'])
        yaw, pitch, roll = quat.yaw_pitch_roll
        coords_6dof[i, 3:] = [yaw, pitch, roll]

    # If user requests agent-frame coordinates, convert global->local using the starting annotation
    if in_agent_frame:
        coords_6dof = convert_global_to_local_6dof(coords_6dof,
                                                   starting_annotation['translation'],
                                                   starting_annotation['rotation'])

    return coords_6dof

def convert_global_to_local_6dof(coordinates: np.ndarray,
                                 translation: Tuple[float, float, float],
                                 rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts global 6DOF coordinates to local 6DOF coordinates.
    :param coordinates: x,y,z,yaw,pitch,roll locations in global frame. array of shape [n_steps, 6]
    :param translation: Tuple of (x, y, z) location that is the center of the local frame
    :param rotation: Tuple representation of quaternion of local frame (w, x, y, z).
    :return: x,y,z,yaw,pitch,roll locations in local frame, array of shape [n_steps, 6].
    """
    if coordinates.size == 0:
        return coordinates
    
    # Extract positions and orientations
    global_positions = coordinates[:, :3]  # (n, 3)
    global_orientations = coordinates[:, 3:]  # (n, 3) - yaw, pitch, roll
    
    # Build inverse transformation matrix (global to local)
    base_quat = Quaternion(rotation)
    base_trans = np.array(translation)
    base_rot_mat_inv = base_quat.inverse.rotation_matrix
    
    local_coords = np.zeros_like(coordinates)
    
    for i in range(len(coordinates)):
        # Transform position: translate then rotate
        local_pos = base_rot_mat_inv @ (global_positions[i] - base_trans)
        local_coords[i, :3] = local_pos
        
        # Transform orientation: inverse composition
        # Global orientation as rotation matrix
        yaw, pitch, roll = global_orientations[i]
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        
        global_rot_mat = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        # Decompose: local = base^(-1) * global
        local_rot_mat = base_rot_mat_inv @ global_rot_mat
        
        # Convert back to euler angles
        local_quat = Quaternion(matrix=local_rot_mat)
        local_yaw, local_pitch, local_roll = local_quat.yaw_pitch_roll
        local_coords[i, 3:] = [local_yaw, local_pitch, local_roll]
    
    return local_coords

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

def geom2anno(map_geoms):
    MAP_CLASSES = (
        'ped_crossing',
        'divider',
        'boundary',
    )
    vectors = {}
    for cls, geom_list in map_geoms.items():
        if cls in MAP_CLASSES:
            label = MAP_CLASSES.index(cls)
            vectors[label] = []
            if geom_list!=None:
                for geom in geom_list:
                    line = np.array(geom.coords)
                    vectors[label].append(line)
    return vectors

def build_scene_validity_map(nusc):
    """
    Build a validity map for each scene based on special markers (offroad/collision/scene_change).
    
    For each scene, determines valid sample ranges based on timestamps where:
    - End point: timestamp of offroad or collision event
    - Start point: latest of (2 seconds before end, previous event timestamp, last scene_change<20m timestamp)
    
    Args:
        nusc: NuScenes dataset instance
        
    Returns:
        dict: Mapping from scene_token to set of valid sample_tokens
    """
    scene_validity_map = {}
    
    for scene in nusc.scene:
        scene_token = scene['token']
        
        # Collect all samples with their timestamps
        sample_records = []
        sample_cur = nusc.get('sample', scene['first_sample_token'])
        while sample_cur:
            sample_records.append({
                'token': sample_cur['token'],
                'timestamp': sample_cur['timestamp']
            })
            if sample_cur['next'] == '':
                break
            sample_cur = nusc.get('sample', sample_cur['next'])
        
        # Find all special markers in this scene by checking all sample_data
        markers = []  # List of (marker_type, distance_info, timestamp)
        for sample_rec_dict in sample_records:
            if sample_rec_dict['token']=='ds1e639c08727fe0c50925ac1176c920665':
                print("here")
            sample_rec = nusc.get('sample', sample_rec_dict['token'])
            lidar_token = sample_rec['data']['LIDAR_TOP']
            
            # Traverse all sample_data in this sample's timeline (current and all next)
            sd_rec = nusc.get('sample_data', lidar_token)
            while sd_rec:
                token_str = sd_rec['token']
                timestamp = sd_rec['timestamp']
                
                # Check scene_change first
                if 'scene_change' in token_str:
                    # Extract distance from scene_change (format: ds2scene_change_142.2m_xxx or similar)
                    distance = None
                    parts = token_str.split('_')
                    # Find the part right after 'scene' and 'change'
                    for i, part in enumerate(parts):
                        if part == 'change' and i + 1 < len(parts):
                            next_part = parts[i + 1]
                            if 'm' in next_part:
                                try:
                                    distance = float(next_part.replace('m', ''))
                                except:
                                    pass
                            break
                    markers.append(('scene_change', distance, timestamp))
                if 'ego_offroad' in token_str or 'ego_collision' in token_str:
                    marker_type = 'offroad' if 'offroad' in token_str else 'collision'
                    # offroad and collision events don't have distance info
                    markers.append((marker_type, None, timestamp))
                
                # Move to next sample_data, stop if we reach the next sample's keyframe
                if sd_rec['next'] == '':
                    break
                next_sd_rec = nusc.get('sample_data', sd_rec['next'])
                # Stop if next sample_data belongs to a different sample
                if next_sd_rec['is_key_frame']:
                    break
                sd_rec = next_sd_rec
        
        # Determine valid timestamp ranges based on offroad/collision endpoints
        valid_time_ranges = []  # List of (start_timestamp, end_timestamp)
        for marker_type, distance, event_timestamp in markers:
            if marker_type in ['offroad', 'collision']:
                # This is the endpoint
                end_timestamp = event_timestamp
                
                # Find the start point (latest of):
                # 1. Last scene_change before this with distance < 20m
                # 2. 4th sample before the event sample (for warmup)
                # 3. Previous offroad/collision event timestamp
                
                start_candidates = []
                
                # Candidate 1: The 4th sample from the beginning of the scene (index 3)
                if len(sample_records) >= 4:
                    fourth_sample_timestamp = sample_records[3]['timestamp']
                    start_candidates.append(fourth_sample_timestamp)
                elif len(sample_records) > 0:
                    # If less than 4 samples in total, use the first sample
                    start_candidates.append(sample_records[0]['timestamp'])
                
                # Candidate 2: Previous offroad/collision event
                prev_event_timestamp = None
                for m_type, m_dist, m_time in markers:
                    if m_time < event_timestamp and m_type in ['offroad', 'collision']:
                        if prev_event_timestamp is None or m_time > prev_event_timestamp:
                            prev_event_timestamp = m_time
                
                if prev_event_timestamp is not None:
                    start_candidates.append(prev_event_timestamp)  # Slightly before previous event
                
                # Additional candidate: 6 samples before the current event
                event_sample_idx = None
                for idx, sr in enumerate(sample_records):
                    if sr['timestamp'] >= event_timestamp:
                        event_sample_idx = idx
                        break
                
                if event_sample_idx is not None and event_sample_idx >= 6:
                    six_before_current_timestamp = sample_records[event_sample_idx - 6]['timestamp']
                    start_candidates.append(six_before_current_timestamp)
                
                # Candidate 3: Last scene_change with distance < 20m
                for m_type, m_dist, m_time in markers:
                    if m_time < event_timestamp and m_type == 'scene_change':
                        if m_dist is not None and m_dist < 20:
                            start_candidates.append(m_time)
                
                # Take the latest (maximum) start timestamp
                start_timestamp = max(start_candidates) if start_candidates else sample_records[0]['timestamp']
                
                # Add this valid time range (open interval)
                if start_timestamp < end_timestamp:
                    valid_time_ranges.append((start_timestamp, end_timestamp))
        
        # Filter samples based on timestamp ranges (open interval: start < timestamp < end)
        valid_sample_tokens = set()
        for sample_rec_dict in sample_records:
            sample_timestamp = sample_rec_dict['timestamp']
            # Check if this sample falls within any valid time range (open interval)
            for start_time, end_time in valid_time_ranges:
                if start_time < sample_timestamp < end_time:
                    valid_sample_tokens.add(sample_rec_dict['token'])
                    sample_rec = nusc.get('sample', sample_rec_dict['token'])
                    lidar_token = sample_rec['data']['LIDAR_TOP']
                    # assert 'scene_change' not in lidar_token
                    assert 'offroad' not in lidar_token
                    assert 'collision' not in lidar_token
                    break
        
        scene_validity_map[scene_token] = valid_sample_tokens
    
    return scene_validity_map

def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          roi_size=(30, 60),):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    print(version, root_path)
    import random
    rng = random.Random(0)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_map_extractor = NuscMapExtractor(root_path, roi_size)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    available_scenes = get_available_scenes(nusc)

    assert version in available_vers
    if version == 'v1.0-trainval':
        # train_scenes = splits.train
        # val_scenes = splits.val
        total_scenes_num = len(available_scenes)
        #训练集和验证集比例为14:3
        train_scenes_num = (total_scenes_num*7)//8
        train_scenes=rng.sample([s['name'] for s in available_scenes], train_scenes_num)
        val_scenes =[s['name'] for s in available_scenes if s['name'] not in train_scenes]
        print(f'train_scene_names: {train_scenes}',flush=True)
        print(f'val_scene_names: {val_scenes}',flush=True)
        # exit(0)
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
        out_path = osp.join(out_path, 'mini')
    else:
        raise ValueError('unknown')
    os.makedirs(out_path, exist_ok=True)

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_map_extractor, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(out_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)

def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

def _fill_trainval_infos(nusc,
                         nusc_map_extractor,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         fut_ts=12,
                         ego_fut_ts=6):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    cat2idx = {}
    for idx, dic in enumerate(nusc.category):
        cat2idx[dic['name']] = idx

    predict_helper = PredictHelper(nusc)
    
    # Build scene validity map: filter samples based on special markers
    scene_validity_map = build_scene_validity_map(nusc)
    #打印出train有多少个sample
    #打印出train有多少个sample
    train_sample=0
    for sample in mmcv.track_iter_progress(nusc.sample):
        scene_token = sample['scene_token']
        if scene_token in scene_validity_map:
            if sample['token'] not in scene_validity_map[scene_token]:
                continue  # Skip this sample as it's not in valid range
        if scene_token in train_scenes:
            train_sample+=1
    print(f'train sample num: {train_sample}',flush=True)
    
    for sample in mmcv.track_iter_progress(nusc.sample):
        # Check if this sample is in a valid range
        scene_token = sample['scene_token']
        if scene_token in scene_validity_map:
            if sample['token'] not in scene_validity_map[scene_token]:
                continue  # Skip this sample as it's not in valid range
        
        #
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', lidar_token)
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token) #the boxes are transformed into the current sensor's coordinate frame.
        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'map_location': map_location,
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # extract map annos
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        lidar2global = ego2global @ lidar2ego

        translation = list(lidar2global[:3, 3])
        rotation = list(Quaternion(matrix=lidar2global).q)
        map_geoms = nusc_map_extractor.get_map_geom_use_drivable_area(map_location, translation, rotation)
        map_annos = geom2anno(map_geoms)
        info['map_annos'] = map_annos

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            # object detection annos: boxes (locs, dims, yaw, velocity), names and valid flags
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]
            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NameMapping:
                    names[i] = NameMapping[names[i]]
            names = np.array(names)
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)  ## TODO update valid flag for tracking
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            
            # object tracking annos: instance_ids
            instance_inds = [nusc.getind('instance', anno['instance_token'])
                             for anno in annotations]

            # motion prediction annos: future trajectories offset in lidar frame and valid mask
            num_box = len(boxes)
            gt_fut_trajs = np.zeros((num_box, fut_ts, 2))
            gt_fut_trajs_6dof = np.zeros((num_box, fut_ts, 6))  # Add 6DOF trajectories for agents
            gt_fut_masks = np.zeros((num_box, fut_ts))
            # Store 6DOF offsets in agent_first coordinate frame: translation offset (3) + rotation offset (3) = 6
            gt_agent_fut_poses_6dof_agent_first = np.zeros((num_box, fut_ts, 6))  # [dx, dy, dz, dyaw, dpitch, droll]
            for i, anno in enumerate(annotations):
                instance_token = anno['instance_token']
                # Get 2D trajectory (original)
                fut_traj_local = predict_helper.get_future_for_agent(
                    instance_token, 
                    sample['token'], 
                    seconds=fut_ts/2, 
                    in_agent_frame=True
                )
                if fut_traj_local.shape[0] > 0:
                    box = boxes[i]
                    trans = box.center
                    rot = Quaternion(matrix=box.rotation_matrix)
                    fut_traj_scene = convert_local_coords_to_global(fut_traj_local, trans, rot)
                    valid_step = fut_traj_scene.shape[0]
                    # assert valid_step==1 
                    gt_fut_trajs[i, 0] = fut_traj_scene[0] - box.center[:2]
                    gt_fut_trajs[i, 1:valid_step] = fut_traj_scene[1:] - fut_traj_scene[:-1]
                    gt_fut_masks[i, :valid_step] = 1
                    
                # Get 6DOF trajectory: get future annotation records and build 4x4 matrices
                # Similar to ego trajectory logic
                fut_annotations = predict_helper.get_future_for_agent(
                    instance_token,
                    sample['token'],
                    seconds=fut_ts/2,
                    in_agent_frame=False,
                    just_xy=False
                )
                
                

                if len(fut_annotations) > 0:
                    # Build 4x4 transformation matrices for agent future poses (in global frame)
                    agent_fut_poses_global = []
                    for ann_record in fut_annotations:
                        agent_pose_global = np.eye(4)
                        agent_pose_global[:3, :3] = Quaternion(ann_record['rotation']).rotation_matrix
                        agent_pose_global[:3, 3] = ann_record['translation']
                        agent_fut_poses_global.append(agent_pose_global)
                    
                    # Get agent's first frame pose (in global frame) as the reference
                    agent_first_pose_global_annotation = predict_helper.get_sample_annotation(
                        instance_token,
                        sample['token']
                    )
                    agent_first_pose_global = np.eye(4)
                    agent_first_pose_global[:3, :3] = Quaternion(agent_first_pose_global_annotation['rotation']).rotation_matrix
                    agent_first_pose_global[:3, 3] = agent_first_pose_global_annotation['translation']
                    # Build transformation matrices (equivalent to ego2lidar and global2ego)
                    # Step 1: global -> agent_first_frame (equivalent to global2ego)
                    agent_first_rot_mat = agent_first_pose_global[:3, :3].T  # inverse rotation
                    agent_first_trans = agent_first_pose_global[:3, 3]
                    global2agent_first = np.eye(4)
                    global2agent_first[:3, :3] = agent_first_rot_mat
                    global2agent_first[:3, 3] = -agent_first_rot_mat @ agent_first_trans
                    
                    # Step 2: agent_first_frame -> lidar (equivalent to ego2lidar)
                    # Get current agent pose in lidar frame (this is agent_first in lidar coordinates)
                    trans = box.center
                    rot = Quaternion(matrix=box.rotation_matrix)
                    # Build lidar2agent_first
                    agent_first_rot_mat_lidar = rot.rotation_matrix.T
                    agent_first_trans_lidar = np.array(trans)
                    lidar2agent_first = np.eye(4)
                    lidar2agent_first[:3, :3] = agent_first_rot_mat_lidar
                    lidar2agent_first[:3, 3] = -agent_first_rot_mat_lidar @ agent_first_trans_lidar
                    
                    # Get agent_first2lidar by inverting lidar2agent_first (equivalent to ego2lidar)
                    agent_first2lidar = np.linalg.inv(lidar2agent_first)
                    
                    # Transform agent poses to lidar frame (same pattern as ego)
                    agent_fut_poses_local = []
                    agent_fut_poses_agent_first = []  # Store poses in agent_first frame
                    for pose_global in agent_fut_poses_global:
                        # First: global to agent_first (equivalent to global to ego)
                        pose_agent_first = global2agent_first @ pose_global
                        agent_fut_poses_agent_first.append(pose_agent_first.copy())
                        # Second: agent_first to lidar (equivalent to ego to lidar)
                        pose_local = agent_first2lidar @ pose_agent_first
                        agent_fut_poses_local.append(pose_local)
                        
                    # Extract positions for translation offset
                    agent_positions_local = np.array([pose[:3, 3] for pose in agent_fut_poses_local])
                    agent_positions_agent_first = np.array([pose[:3, 3] for pose in agent_fut_poses_agent_first])
                    
                    # Compute 6DOF offsets in agent_first coordinate frame (same logic as ego trajectory)
                    valid_6dof_step = valid_step
                    for t in range(valid_6dof_step):
                        # Translation offset in agent_first frame
                        if t == 0:
                            # First frame: offset from initial position (which is origin in agent_first frame)
                            gt_agent_fut_poses_6dof_agent_first[i, t, :3] = agent_positions_agent_first[t] - 0
                        else:
                            # Subsequent frames: offset from previous frame
                            gt_agent_fut_poses_6dof_agent_first[i, t, :3] = agent_positions_agent_first[t] - agent_positions_agent_first[t-1]
                        
                        # Rotation offset in agent_first frame
                        if t == 0:
                            # First frame: relative to identity (initial orientation)
                            pose_prev_agent_first = np.eye(4)  # Identity for first frame
                        else:
                            pose_prev_agent_first = agent_fut_poses_agent_first[t-1]
                        pose_curr_agent_first = agent_fut_poses_agent_first[t]
                        
                        # Compute relative rotation: R_relative = R_prev^T @ R_curr
                        relative_rotation_agent_first = pose_prev_agent_first[:3, :3].T @ pose_curr_agent_first[:3, :3]
                        rel_quat_agent_first = Quaternion(matrix=relative_rotation_agent_first)
                        yaw_af, pitch_af, roll_af = rel_quat_agent_first.yaw_pitch_roll
                        gt_agent_fut_poses_6dof_agent_first[i, t, 3:] = [yaw_af, pitch_af, roll_af]
                    
                    # Compute 6DOF offsets in lidar frame (same logic as ego trajectory)
                    for t in range(valid_6dof_step):
                        # Translation offset: simple position difference
                        if t == 0:
                            gt_fut_trajs_6dof[i, t, :3] = agent_positions_local[t] - box.center
                        else:
                            gt_fut_trajs_6dof[i, t, :3] = agent_positions_local[t] - agent_positions_local[t-1]

                        # Rotation offset: relative rotation from frame t-1 to frame t
                        if t == 0:
                            # For first frame, use identity as previous pose
                            pose_prev = box.rotation_matrix
                        else:
                            pose_prev = agent_fut_poses_local[t-1]
                        pose_curr = agent_fut_poses_local[t]
                        
                        # Compute relative rotation: R_relative = R_prev^T @ R_curr (same as ego)
                        relative_rotation = pose_prev[:3, :3].T @ pose_curr[:3, :3]
                        rel_quat = Quaternion(matrix=relative_rotation)
                        yaw, pitch, roll = rel_quat.yaw_pitch_roll
                        gt_fut_trajs_6dof[i, t, 3:] = [yaw, pitch, roll]
            
            # Create model input version: only keep x,y from gt, set z/yaw/pitch/roll to 0
            input_fut_trajs_6dof = np.zeros_like(gt_fut_trajs_6dof)
            input_fut_trajs_6dof[:, :, :2] = gt_fut_trajs_6dof[:, :, :2]  # Copy x,y from gt


                
                    
                    

            # motion planning annos: future trajectories offset in lidar frame and valid mask
            ego_fut_trajs = np.zeros((ego_fut_ts + 1, 3))
            ego_fut_poses = []  # Store full 4x4 transformation matrices
            ego_fut_masks = np.zeros((ego_fut_ts + 1))
            sample_cur = sample
            ego_status = get_ego_status(nusc, nusc_can_bus, sample_cur)
            for i in range(ego_fut_ts + 1):
                pose_mat = get_global_sensor_pose(sample_cur, nusc)
                ego_fut_trajs[i] = pose_mat[:3, 3]  # Keep original ego_fut_trajs extraction
                ego_fut_poses.append(pose_mat.copy())
                ego_fut_masks[i] = 1
                if sample_cur['next'] == '':
                    ego_fut_trajs[i+1:] = ego_fut_trajs[i]
                    # Fill remaining poses with the last pose
                    for j in range(i+1, ego_fut_ts + 1):
                        ego_fut_poses.append(pose_mat.copy())
                    break
                else:
                    sample_cur = nusc.get('sample', sample_cur['next'])
            
            
            # Build transformation matrices for two-step conversion (same as ego_fut_trajs)
            # Step 1: global to ego
            ego_rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_trans = np.array(pose_record['translation'])
            global2ego = np.eye(4)
            global2ego[:3, :3] = ego_rot_mat
            global2ego[:3, 3] = -ego_rot_mat @ ego_trans
            
            # Step 2: ego to lidar
            lidar_rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            lidar_trans = np.array(cs_record['translation'])
            ego2lidar = np.eye(4)
            ego2lidar[:3, :3] = lidar_rot_mat
            ego2lidar[:3, 3] = -lidar_rot_mat @ lidar_trans
            
            # Transform all future poses to current lidar frame (two-step: global→ego→lidar)
            #  ego_fut_poses 的]
            ego_fut_poses_lidar = []
            ego_fut_poses_ego = []  # Store poses in ego coordinate frame
            for pose_global in ego_fut_poses:
                # First: global to ego
                pose_ego = global2ego @ pose_global
                ego_fut_poses_ego.append(pose_ego.copy())
                # Second: ego to lidar
                pose_lidar = ego2lidar @ pose_ego
                ego_fut_poses_lidar.append(pose_lidar)
            
            # Extract 6dof from transformed poses
            # For translation: use simple position difference (same as ego_fut_trajs)
            # For rotation: compute relative rotation in current frame's coordinate
            ego_fut_positions_lidar = np.array([pose[:3, 3] for pose in ego_fut_poses_lidar])
            
            ego_fut_trajs_6dof = np.zeros((ego_fut_ts,6))  # Will store offsets in lidar frame
            # Store 6DOF offsets in ego coordinate frame: translation offset (3) + rotation offset (3) = 6
            gt_ego_fut_poses_6dof_ego = np.zeros((ego_fut_ts, 6))  # [dx, dy, dz, dyaw, dpitch, droll]
            
            # Extract ego positions in ego frame
            ego_fut_positions_ego = np.array([pose[:3, 3] for pose in ego_fut_poses_ego])
            
            for i in range(ego_fut_ts):
                # Translation offset in ego frame
                gt_ego_fut_poses_6dof_ego[i, :3] = ego_fut_positions_ego[i+1] - ego_fut_positions_ego[i]
                
                # Rotation offset in ego frame: relative rotation from frame i to frame i+1
                pose_cur_ego = ego_fut_poses_ego[i]
                pose_next_ego = ego_fut_poses_ego[i+1]
                relative_rotation_ego = pose_cur_ego[:3, :3].T @ pose_next_ego[:3, :3]
                rel_quat_ego = Quaternion(matrix=relative_rotation_ego)
                yaw_e, pitch_e, roll_e = rel_quat_ego.yaw_pitch_roll
                gt_ego_fut_poses_6dof_ego[i, 3:] = [yaw_e, pitch_e, roll_e]
                
                # Translation offset: simple position difference in lidar frame (for backward compatibility)
                ego_fut_trajs_6dof[i, :3] = ego_fut_positions_lidar[i+1] - ego_fut_positions_lidar[i]
                
                # Rotation offset: relative rotation from frame i to frame i+1
                pose_cur = ego_fut_poses_lidar[i]
                pose_next = ego_fut_poses_lidar[i+1]
                # Compute relative rotation: R_relative = R_cur^T @ R_next

                #计算方式1:在lidar坐标系下相对前一帧在相对旋转量
                relative_rotation = pose_cur[:3, :3].T @ pose_next[:3, :3]
                
                # Extract rotation offset as euler angles
                rel_quat = Quaternion(matrix=relative_rotation)
                yaw, pitch, roll = rel_quat.yaw_pitch_roll
                #计算方式2:在lidar坐标系下相对前一阵在雷达坐标系下的相对旋转量，不适用方式2的原因是可能出现突变
                # yaw_cur, pitch_cur, roll_cur = Quaternion(matrix=pose_cur[:3, :3]).yaw_pitch_roll
                # yaw_next, pitch_next, roll_next = Quaternion(matrix=pose_next[:3, :3]).yaw_pitch_roll
                # roll_2 = roll_next - roll_cur
                # pitch_2 = pitch_next - pitch_cur
                # yaw_2    = yaw_next - yaw_cur
                # assert abs(yaw - yaw_2) < 1e-1 and abs(pitch - pitch_2) < 1e-1 and abs(roll - roll_2) < 1e-1, \
                #     f"Two methods for rotation offset calculation differ: {(yaw, pitch, roll)} vs {(yaw_2, pitch_2, roll_2)}"
                ego_fut_trajs_6dof[i, 3:] = [yaw, pitch, roll]
            
            # Create model input version for ego: only keep x,y from gt, set z/yaw/pitch/roll to 0
            # After first offroad/collision event, use linear extrapolation for x,y (only using pre-event info)
            input_ego_fut_trajs_6dof = np.zeros_like(ego_fut_trajs_6dof)
            input_ego_fut_trajs_6dof[:, :2] = ego_fut_trajs_6dof[:, :2]
            # Apply linear extrapolation after first event (if found), using only pre-event information
            input_ego_fut_trajs_6dof = apply_linear_extrapolation_after_event(nusc, sample, input_ego_fut_trajs_6dof, ego_fut_ts)
            
            # Keep original ego_fut_trajs computation path unchanged
            # global to ego
            ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            # ego to lidar
            ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            
            # drive command according to final fut step offset
            if ego_fut_trajs[-1][0] >= 2:
                command = np.array([1, 0, 0])  # Turn Right
            elif ego_fut_trajs[-1][0] <= -2:
                command = np.array([0, 1, 0])  # Turn Left
            else:
                command = np.array([0, 0, 1])  # Go Straight
            # get offset for xy trajectory (backward compatibility)
            ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]
            # 6dof offsets are already computed as relative transformations

            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag
            info['instance_inds'] = instance_inds
            info['gt_agent_fut_trajs'] = gt_fut_trajs.astype(np.float32)
            info['gt_agent_fut_trajs_6dof'] = gt_fut_trajs_6dof.astype(np.float32)  # 6dof offsets in lidar frame for agents
            info['input_agent_fut_trajs_6dof'] = input_fut_trajs_6dof.astype(np.float32)  # Model input: only x,y, rest are 0
            info['gt_agent_fut_poses_6dof_agent_first'] = gt_agent_fut_poses_6dof_agent_first.astype(np.float32)  # 6dof offsets in agent_first coordinate frame
            assert gt_fut_trajs_6dof.shape[0]==gt_fut_trajs.shape[0]
            assert gt_fut_trajs_6dof.shape[0]==gt_boxes.shape[0]
            info['gt_agent_fut_masks'] = gt_fut_masks.astype(np.float32)
            info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)
            info['gt_ego_fut_trajs_6dof'] = ego_fut_trajs_6dof.astype(np.float32)  # 6dof offsets in lidar frame
            info['input_ego_fut_trajs_6dof'] = input_ego_fut_trajs_6dof.astype(np.float32)  # Model input: only x,y, rest are 0
            info['gt_ego_fut_poses_6dof_ego'] = gt_ego_fut_poses_6dof_ego.astype(np.float32)  # 6dof offsets in ego coordinate frame
            info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)
            info['gt_ego_fut_cmd'] = command.astype(np.float32)
            info['ego_status'] = ego_status

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos

def apply_linear_extrapolation_after_event(nusc, sample, input_ego_fut_trajs_6dof, ego_fut_ts):
    """
    Apply linear extrapolation for ego trajectory after first offroad/collision event.
    Uses only information before the event. Returns modified trajectory array.
    
    Two cases:
    1. Event at current sample's sample_data: Use previous sample's velocity for current and all future steps
    2. Event at future sample's sample_data: Use the velocity from the step before the event
    
    Args:
        nusc: NuScenes dataset instance
        sample: Current sample record
        input_ego_fut_trajs_6dof: Input trajectory array [ego_fut_ts, 6]
        ego_fut_ts: Number of future timesteps
        
    Returns:
        np.ndarray: Modified trajectory array with linear extrapolation applied [ego_fut_ts, 6]
    """
    # Create a copy to avoid modifying the input
    result = input_ego_fut_trajs_6dof.copy()
    # print(result.shape)
    # Check each future step (0 to ego_fut_ts-1) to find first event
    first_event_step = None
    sample_check = sample
    
    for step_idx in range(ego_fut_ts):
        # Check all sample_data for this sample (current or future)
        lidar_token_check = sample_check['data']['LIDAR_TOP']
        sd_rec_check = nusc.get('sample_data', lidar_token_check)
        
        # Traverse all sample_data in this sample's timeline
        has_event = False
        is_first_sd = True  # Flag to track if we're checking the first sample_data
        
        while sd_rec_check:
            token_str_check = sd_rec_check['token']
            if 'ego_collision' in token_str_check:
                # Event found
                if is_first_sd:
                    # Event at the keyframe (first sample_data of this sample)
                    # This means we need to use velocity from step_idx-1 (or previous sample if step_idx==0)
                    first_event_step = step_idx
                else:
                    # Event at a non-keyframe sample_data (next of the keyframe)
                    # This means we should use velocity from current step (step_idx)
                    first_event_step = step_idx + 1
                has_event = True
                break
            
            is_first_sd = False  # After first iteration, we're in the 'next' sample_data
            
            if sd_rec_check['next'] == '':
                break
            next_sd_rec_check = nusc.get('sample_data', sd_rec_check['next'])
            if next_sd_rec_check['is_key_frame']:
                break
            sd_rec_check = next_sd_rec_check
        
        if has_event:
            break
        
        # Move to next sample
        if sample_check['next'] == '':
            break
        sample_check = nusc.get('sample', sample_check['next'])
    
    # Apply extrapolation if event was found
    if first_event_step is not None:
        sample_rec = nusc.get('sample', sample["token"])
        lidar_token = sample_rec['data']['LIDAR_TOP']
        # assert 'scene_change' not in lidar_token
        print(lidar_token)
        assert first_event_step > 0
        if first_event_step == 1:
            # Case 1: Event at current sample (step 0)
            # Use velocity from previous sample (which we need to compute)
            # if sample['prev'] != '':
            #     # Get previous sample and compute its last velocity
            #     # Since we're at the current sample and the event is here,
            #     # we should use the velocity that brought us here (from prev to current)
            #     # But this is not in input_ego_fut_trajs_6dof yet
            #     # Use zero velocity as fallback
            #     last_velocity = np.zeros(2)
            # else:
            last_velocity = result[first_event_step - 1, :2]
            
            # Extrapolate from the event step onwards
            # for step_idx in range(first_event_step-1, ego_fut_ts):
            #     result[step_idx, :2] = last_velocity
        else:
            # Case 2: Event at future sample (step > 0)
            # Use the velocity from the step before the event (step_idx - 1)
            last_velocity = result[first_event_step - 2, :2]
            
            # Extrapolate from the event step onwards
        for step_idx in range(first_event_step-1, ego_fut_ts):
            result[step_idx, :2] = last_velocity

    return result

def get_ego_status(nusc, nusc_can_bus, sample):
    ego_status = []
    ref_scene = nusc.get("scene", sample['scene_token'])
    try:
        pose_msgs = nusc_can_bus.get_messages(ref_scene['name'],'pose')
        steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')
        pose_uts = [msg['utime'] for msg in pose_msgs]
        steer_uts = [msg['utime'] for msg in steer_msgs]
        ref_utime = sample['timestamp']
        pose_index = locate_message(pose_uts, ref_utime)
        pose_data = pose_msgs[pose_index]
        steer_index = locate_message(steer_uts, ref_utime)
        steer_data = steer_msgs[steer_index]
        ego_status.extend(pose_data["accel"]) # acceleration in ego vehicle frame, m/s/s
        ego_status.extend(pose_data["rotation_rate"]) # angular velocity in ego vehicle frame, rad/s
        ego_status.extend(pose_data["vel"]) # velocity in ego vehicle frame, m/s
        ego_status.append(steer_data["value"]) # steering angle, positive: left turn, negative: right turn
    except:
        ego_status = [0] * 10
    
    return np.array(ego_status).astype(np.float32)

def get_global_sensor_pose(rec, nusc):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

    pose_record = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    cs_record = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])

    ego2global = transform_matrix(pose_record["translation"], Quaternion(pose_record["rotation"]), inverse=False)
    sensor2ego = transform_matrix(cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=False)
    pose = ego2global.dot(sensor2ego)

    return pose

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep

def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
