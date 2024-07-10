import json

import cv2
import einops
import numpy as np
import open3d
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from utils.utils_with_real import (
    Actioner,
    convert_rotation,
    deproject,
    get_cam_info,
    keypoint_discovery,
    obs_to_attn,
    transform,
)


class Mover:
    def __init__(self, robot, arm, gripper, max_tries=1):
        self.robot = robot
        self.arm = arm
        self.gripper = gripper
        # self._last_action = None
        # self._step_id = 0
        # self._max_tries = max_tries

    def __call__(self, action):
        print(action[:3])
        self.robot.move_to_ee_pose(action[:3], None)
        # return 0
        # TODO to quat, move gripper
        # print(action[3:7])
        # self.arm.robot.move_to_ee_pose(action[:3], None)
        # self.arm.robot.move_to_ee_pose(action[:3], action[3:7])
        return 0


class RealEnv:
    def __init__(
        self,
        data_path,
        fine_sampling_ball_diameter=None,
    ):
        # setup required inputs
        self.data_path = data_path
        self.fine_sampling_ball_diameter = fine_sampling_ball_diameter

    def get_gripper_matrix_from_action(self, action):
        action = action.cpu().numpy()
        position = action[:3]
        quaternion = action[3:7]
        rotation = open3d.geometry.get_rotation_matrix_from_quaternion(
            np.array((quaternion[3], quaternion[0], quaternion[1], quaternion[2]))
        )
        gripper_matrix = np.eye(4)
        gripper_matrix[:3, :3] = rotation
        gripper_matrix[:3, 3] = position
        return gripper_matrix

    @torch.no_grad()
    def evaluate_task(
        self,
        task_str: str,
        max_steps: int,
        actioner: Actioner,
        max_tries: int = 1,
        verbose: bool = False,
        interpolation_length=50,
        num_history=0,
    ):
        device = actioner.device

        rgbs = torch.Tensor([]).to(device)
        pcds = torch.Tensor([]).to(device)
        grippers = torch.Tensor([]).to(device)

        last_ee = torch.Tensor([[0.5266, -0.0034, 0.3818, 3.1061, -0.0047, -0.7991, 1.0000]]).to(
            device
        )

        for step_id in range(max_steps):
            # Fetch the current observation, and predict one action
            rgb = torch.zeros((1, 1, 3, 128, 128)).to(device)
            pcd = torch.zeros((1, 1, 3, 128, 128)).to(device)
            gripper = last_ee.clone().detach()

            rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
            pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
            grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)

            # Prepare proprioception history
            rgbs_input = rgbs[:, -1:][:, :, :3]
            pcds_input = pcds[:, -1:]

            if num_history < 1:
                gripper_input = grippers[:, -1]
            else:
                gripper_input = grippers[:, -num_history:]
                npad = num_history - gripper_input.shape[1]
                gripper_input = F.pad(gripper_input, (0, 0, npad, 0), mode="replicate")

            # print(gripper_input)

            output = actioner.predict(
                rgbs_input,
                pcds_input,
                gripper_input,
                interpolation_length=interpolation_length,
            )

            if verbose:
                print(f"Step {step_id}")

            action = output["action"][0].cpu().numpy()
            ee_euler = Rotation.from_quat(action[:, 3:7]).as_euler("xyz")
            last_ee = torch.Tensor(
                np.concatenate((action[:, :3], ee_euler, action[:, -1:]), axis=1)
            ).to(gripper.device)
            # print(last_ee)
            # continue

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection="3d")

        poses = grippers.cpu().numpy()
        # print(poses)
        # print(poses[0].T)
        xyz = poses[0].T
        x, y, z = xyz[0], xyz[1], xyz[2]

        ax.scatter(x, y, z)

        for i in range(len(x) - 1):
            ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], [z[i], z[i + 1]], color="g")

        ax.scatter(0, 0, 0, c="yellow")
        ax.scatter(1.4216465041442194, -0.012545208136006748, 0.8585146590952618, c="red")
        plt.show()

        print(f"Finished: {task_str}")
        return 1
