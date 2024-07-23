import random

import numpy as np
import pyrealsense2
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema

import utils.pytorch3d_transforms as pytorch3d_transforms


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

    extrinsics = np.zeros((4, 4))
    extrinsics[:3, 0] = np.array(calib["camera_base_ori"])[:3, 2]
    extrinsics[:3, 1] = np.array(calib["camera_base_ori"])[:3, 0]
    extrinsics[:3, 2] = -np.array(calib["camera_base_ori"])[:3, 1]
    extrinsics[:3, 3] = np.array(calib["camera_base_pos"])
    extrinsics[3, 3] = 1.0

    offset = np.zeros((4, 4))
    mat = pytorch3d_transforms.euler_angles_to_matrix(torch.as_tensor([0, -0.1, 0]), "XYZ")
    offset[:3, :3] = mat.numpy()
    offset[:3, 3] = np.array([-0.045, 0, 0.08])
    offset[3, 3] = 1.0

    return intrinsics, offset @ extrinsics


def deproject(depth_img, intrinsics, extrinsics):
    h, w = depth_img.shape
    g = np.stack(np.meshgrid(np.arange(h), np.arange(w))).T.reshape((h * w, 2))

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
        [
            pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, i, depth_img[i[0], i[1]])
            for i in g
        ]
    )
    z, y, x = points.T

    cam_pos = np.stack([x, y, -z, np.ones_like(z)], axis=0)

    world_pos = extrinsics @ cam_pos
    return world_pos[:3]


def viz_pcd(pcd, rgb=None, proprio=None, cam_pos=None, idxs=None):
    fig = plt.figure(figsize=(8, 8))

    if idxs is None:
        idxs = range(len(pcd))

    for n, i in enumerate(idxs):
        ax = fig.add_subplot(1, len(idxs), n + 1, projection="3d")
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([0, 1])
        ax.scatter(0, 0, 0)

        if rgb is not None:
            colors = rgb[i].reshape(pcd[i].shape)
            colors = (colors + 1) / 2

            ax.scatter(pcd[i][:, 0], pcd[i][:, 1], pcd[i][:, 2], c=colors)
        else:
            ax.scatter(pcd[i][:, 0], pcd[i][:, 1], pcd[i][:, 2])

        if proprio is not None:
            ax.scatter(proprio[i][0], proprio[i][1], proprio[i][2], marker="X", c="g", s=400)

        if cam_pos is not None:
            # ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c="r")
            ax.quiver(
                1.4216465041442194,
                -0.012545208136006748,
                0.8585146590952618,
                -0.8373016909952649,
                0.0031583298017489603,
                -0.5467320213864694,
                length=0.3,
            )

    plt.show()


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


class Actioner:

    def __init__(
        self,
        policy=None,
        instructions=None,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        action_dim=7,
        predict_trajectory=True,
    ):
        self._policy = policy
        self._instructions = instructions
        self._apply_cameras = apply_cameras
        self._action_dim = action_dim
        self._predict_trajectory = predict_trajectory

        self._actions = {}
        self._instr = None
        self._task_str = None

        self._policy.eval()

    def load_episode(self, task_str, variation):
        self._task_str = task_str
        instructions = list(self._instructions[task_str][variation])
        self._instr = random.choice(instructions).unsqueeze(0)
        # self._task_id = torch.tensor(TASK_TO_ID[task_str]).unsqueeze(0)
        self._actions = {}

    def get_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)

        action_ls = []
        trajectory_ls = []
        for i in range(len(key_frame)):
            obs = demo[key_frame[i]]
            action_np = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
            action = torch.from_numpy(action_np)
            action_ls.append(action.unsqueeze(0))

            trajectory_np = []
            for j in range(key_frame[i - 1] if i > 0 else 0, key_frame[i]):
                obs = demo[j]
                trajectory_np.append(np.concatenate([obs.gripper_pose, [obs.gripper_open]]))
            trajectory_ls.append(np.stack(trajectory_np))

        trajectory_mask_ls = [
            torch.zeros(1, key_frame[i] - (key_frame[i - 1] if i > 0 else 0)).bool()
            for i in range(len(key_frame))
        ]

        return action_ls, trajectory_ls, trajectory_mask_ls

    def predict(self, rgbs, pcds, gripper, interpolation_length=None):
        """
        Args:
            rgbs: (bs, num_hist, num_cameras, 3, H, W)
            pcds: (bs, num_hist, num_cameras, 3, H, W)
            gripper: (B, nhist, output_dim)
            interpolation_length: an integer

        Returns:
            {"action": torch.Tensor, "trajectory": torch.Tensor}
        """
        output = {"action": None, "trajectory": None}

        rgbs = rgbs / 2 + 0.5  # in [0, 1]

        if self._instr is None:
            self._instr = torch.zeros((rgbs.shape[0], 53, 512))

        self._instr = self._instr.to(rgbs.device)
        # self._task_id = self._task_id.to(rgbs.device)

        # Predict trajectory
        if self._predict_trajectory:
            fake_traj = torch.full([1, interpolation_length - 1, gripper.shape[-1]], 0).to(
                rgbs.device
            )
            traj_mask = torch.full([1, interpolation_length - 1], False).to(rgbs.device)
            output["trajectory"] = self._policy(
                fake_traj,
                traj_mask,
                rgbs[:, -1],
                pcds[:, -1],
                self._instr,
                gripper[..., :7],
                run_inference=True,
            )
        else:
            fake_traj = torch.full([1, 1, gripper.shape[-1]], 0).to(rgbs.device)
            traj_mask = torch.full([1, 1], False).to(rgbs.device)
            output["action"] = self._policy(
                fake_traj,
                traj_mask,
                rgbs[:, -1],
                pcds[:, -1],
                self._instr,
                gripper[..., :7],
                run_inference=True,
            )

        return output

    @property
    def device(self):
        return next(self._policy.parameters()).device


def obs_to_attn(obs, camera):
    extrinsics_44 = torch.from_numpy(obs.misc[f"{camera}_camera_extrinsics"]).float()
    extrinsics_44 = torch.linalg.inv(extrinsics_44)
    intrinsics_33 = torch.from_numpy(obs.misc[f"{camera}_camera_intrinsics"]).float()
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31.float().squeeze(1)
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v


def transform(obs_dict):
    obs_rgb = []
    obs_pc = []
    rgb = torch.tensor(obs_dict["rgb"]).float().permute(2, 0, 1)
    pc = torch.tensor(obs_dict["pc"]).float().permute(2, 0, 1)

    # normalise to [-1, 1]
    rgb = rgb / 255.0
    rgb = 2 * (rgb - 0.5)

    obs_rgb += [rgb.float()]
    obs_pc += [pc.float()]
    obs = obs_rgb + obs_pc
    return torch.cat(obs, dim=0)
