# 得到一个
import numpy as np
from pyquaternion import Quaternion
A = []
for i in range(7):
    #随机添加一个4*4的位姿矩阵
    pose = np.eye(4)
    pose[:3, :3] = Quaternion.random().rotation_matrix
    pose[:3, 3] = np.random.randn(3)
    A.append(pose)

A_position = np.array([pose[:3, 3] for pose in A])
A_rotation = A
#计算相对6dof的增量

relative_6dof = np.zeros((6,6))
for i in range(6):
    relative_6dof[i, :3] = A_position[i+1] - A_position[i]
                    

    pose_cur = A_rotation[i]
    pose_next = A_rotation[i+1]


    #计算方式1:在lidar坐标系下相对前一帧在相对旋转量
    relative_rotation =  pose_cur[:3, :3].T @ pose_next[:3, :3]  

    pose_next_test = pose_cur[:3, :3] @ relative_rotation  
    print("Test rotation reconstruction error:", np.linalg.norm(pose_next_test - pose_next[:3, :3]))

    # Extract rotation offset as euler angles
    rel_quat = Quaternion(matrix=relative_rotation)
    yaw, pitch, roll = rel_quat.yaw_pitch_roll

    #yaw pitch, roll 转换回矩阵
    offset_quat = Quaternion(axis=[1, 0, 0], angle=roll) * \
                  Quaternion(axis=[0, 1, 0], angle=pitch) * \
                  Quaternion(axis=[0, 0, 1], angle=yaw)
    print("Reconstruction rotation error:", np.linalg.norm(offset_quat.rotation_matrix - relative_rotation))
    relative_6dof[i, 3:] = [yaw, pitch, roll]





