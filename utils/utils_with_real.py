import copy

import numpy as np
import open3d as o3d

# import pybullet as pb
import pyrealsense2
import torch
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema

import utils.pytorch3d_transforms as pytorch3d_transforms

# def get_eef_velocity_from_robot(robot: Robot):
#     eef_vel = []
#     for i in range(2):
#         eef_vel.append(
#             pb.getJointState(
#                 robot.robot_uid, robot.gripper_joint_ids[i], physicsClientId=robot.cid
#             )[1]
#         )

#     # mean over the two gripper points.
#     vel = sum(eef_vel) / len(eef_vel)

#     return vel


def get_eef_velocity_from_trajectories(trajectories):
    trajectories = np.stack([trajectories[0]] + trajectories, axis=0)
    velocities = trajectories[1:] - trajectories[:-1]

    V = np.linalg.norm(velocities[:, :3], axis=-1)
    W = np.linalg.norm(velocities[:, 3:6], axis=-1)

    velocities = np.concatenate(
        [velocities, [velocities[-1]]],
        # [velocities[[0]], velocities],
        axis=0,
    )
    accelerations = velocities[1:] - velocities[:-1]

    A = np.linalg.norm(accelerations[:, :3], axis=-1)

    return V, W, A


def gripper_state_changed(trajectories):
    trajectories = np.stack([trajectories[0]] + trajectories, axis=0)
    openess = trajectories[:, -1]
    changed = openess[:-1] != openess[1:]

    return np.where(changed)[0]


def keypoint_discovery(trajectories, buffer_size=5):
    """Determine way point from the trajectories.

    Args:
        trajectories: a list of 1-D np arrays.  Each array is
            7-dimensional (x, y, z, euler_x, euler_y, euler_z, opene).
        stopping_delta: the minimum velocity to determine if the
            end effector is stopped.

    Returns:
        an Integer array indicates the indices of waypoints
    """
    # print(np.array(trajectories).shape)
    V, W, A = get_eef_velocity_from_trajectories(trajectories)

    # waypoints are local minima of gripper movement
    _local_max_A = argrelextrema(A, np.greater)[0]
    topK = np.sort(A)[::-1][int(A.shape[0] * 0.2)]
    large_A = A[_local_max_A] >= topK
    _local_max_A = _local_max_A[large_A].tolist()

    local_max_A = [_local_max_A.pop(0)]
    for i in _local_max_A:
        if i - local_max_A[-1] >= buffer_size:
            local_max_A.append(i)

    # waypoints are frames with changing gripper states
    gripper_changed = gripper_state_changed(trajectories)
    one_frame_before_gripper_changed = gripper_changed[gripper_changed > 1] - 1

    # waypoints is the last pose in the trajectory
    last_frame = [len(trajectories) - 1]

    keyframe_inds = (
        local_max_A
        + gripper_changed.tolist()
        + one_frame_before_gripper_changed.tolist()
        + last_frame
    )
    keyframe_inds = np.unique(keyframe_inds)

    keyframes = [trajectories[i] for i in keyframe_inds]

    return keyframes, keyframe_inds


def get_cam_info(calib):
    intrinsics = pyrealsense2.intrinsics()
    intrinsics.width = calib["intrinsics"]["width"]
    intrinsics.height = calib["intrinsics"]["height"]
    intrinsics.fx = calib["intrinsics"]["fx"]
    intrinsics.fy = calib["intrinsics"]["fy"]
    intrinsics.ppx = calib["intrinsics"]["ppx"]
    intrinsics.ppy = calib["intrinsics"]["ppy"]
    intrinsics.coeffs = calib["intrinsics"]["coeffs"]
    intrinsics.model = pyrealsense2.distortion.inverse_brown_conrady

    # view = np.zeros((4, 4))
    # view[:3, :3] = -np.array(cam_ori)
    # view[:3, 3] = np.array(cam_pos)
    # view[3, 3] = 1.0
    # inv_view = np.linalg.inv(view)
    # print(calib)

    extrinsics = np.zeros((4, 4))
    extrinsics[:3, :3] = np.array(calib["camera_base_ori"]).T
    extrinsics[:3, 3] = -np.array(calib["camera_base_pos"])
    extrinsics[3, 3] = 1.0

    return intrinsics, extrinsics


def deproject(depth_img, intrinsics, extrinsics):
    h, w = depth_img.shape
    v, u = np.meshgrid(np.arange(h), np.arange(w))
    v, u = v.ravel(), u.ravel()

    w_factor = w / intrinsics.width
    h_factor = h / intrinsics.height

    _intrinsics = pyrealsense2.intrinsics()
    _intrinsics.width = intrinsics.width
    _intrinsics.height = intrinsics.height
    _intrinsics.fx = intrinsics.fx * w_factor
    _intrinsics.fy = intrinsics.fy * h_factor
    _intrinsics.ppx = intrinsics.ppx * w_factor
    _intrinsics.ppy = intrinsics.ppy * h_factor
    _intrinsics.coeffs = intrinsics.coeffs
    _intrinsics.model = intrinsics.model

    points = np.array(
        [pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, i, depth_img[i]) for i in zip(v, u)]
    )
    x, y, z = points.T
    y = -y

    ones = np.ones_like(z)
    cam_pos = np.stack([x, y, z, ones], axis=0)

    world_pos = extrinsics @ cam_pos
    return world_pos[:3]


def viz_pcd(pcd, cam_pos=None):
    x = pcd[:, 0]
    y = pcd[:, 1]
    z = pcd[:, 2]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z)
    # ax.scatter(cam_pos, c="red")
    plt.show()


# def deproject(depth_img, inv_view_mat, fx, fy, ppx, ppy):
#     """
#     Deprojects a pixel point to 3D coordinates
#     Args
#         depth_img: np.array; depth image used as reference to generate 3D coordinates
#     Output
#         (x, y, z): (3, npts) np.array; world coordinates of the deprojected point
#     """
#     # intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, args.fx, args.fy, args.ppx, args.ppy)
#     # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#     #     color, depth_img, convert_rgb_to_intensity=False
#     # )
#     # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

#     # return o3d.geometry.create_point_cloud_from_depth_image(depth_img, intrinsics)

#     h, w = depth_img.shape
#     u, v = np.meshgrid(np.arange(w), np.arange(h))
#     u, v = u.ravel(), v.ravel()
#     # print(u)
#     z = depth_img[v, u] / 1000
#     # print(z)

#     # fx = 640 / (2 * np.tan(np.deg2rad(87) / 2))
#     # fy = 480 / (2 * np.tan(np.deg2rad(58) / 2))
#     # x = (u - 640 // 2) * z / fx
#     # y = -(v - 480 // 2) * z / fy

#     # fx = fx / 10
#     # fy = fy / 10
#     # ppx = ppx / 1000
#     # ppy = ppy / 1000

#     x = (u - ppx) * z / fx
#     y = (v - ppy) * z / fy
#     # z = -z
#     # print(x, y, z)
#     # return
#     ones = np.ones_like(z)

#     # ones = np.ones_like(x)
#     # pix_hom = np.stack([u, v, ones])
#     # x, y, _, _ = inv_view_mat @ pix_hom

#     # z = -depth_img[v, u]
#     cam_pos = np.stack([x, y, z, ones], axis=0)
#     print(cam_pos)


#     world_pos = inv_view_mat @ cam_pos
#     # world_pos = cam_pos
#     return world_pos[:3]


def angle_between_angles(a, b):
    diff = b - a
    return (diff + np.pi) % (2 * np.pi) - np.pi


def convert_rotation(rot):
    """Convert Euler angles to Quarternion"""
    rot = torch.as_tensor(rot)
    mat = pytorch3d_transforms.euler_angles_to_matrix(rot, "XYZ")
    quat = pytorch3d_transforms.matrix_to_quaternion(mat)
    quat = quat.numpy()

    return quat


def to_relative_action(actions, robot_obs, max_pos=1.0, max_orn=1.0, clip=True):
    assert isinstance(actions, np.ndarray)
    assert isinstance(robot_obs, np.ndarray)

    rel_pos = actions[..., :3] - robot_obs[..., :3]
    if clip:
        rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos
    else:
        rel_pos = rel_pos / max_pos

    rel_orn = angle_between_angles(robot_obs[..., 3:6], actions[..., 3:6])
    if clip:
        rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn
    else:
        rel_orn = rel_orn / max_orn

    gripper = actions[..., -1:]
    return np.concatenate([rel_pos, rel_orn, gripper])
