import json

import cv2
import einops
import numpy as np
import open3d
import torch
import torch.nn.functional as F
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from online_evaluation_real.utils.realsense import Realsense
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
        # arm,
        # gripper,
        cam_calib_file,
        image_size=(128, 128),
        fine_sampling_ball_diameter=None,
    ):
        # setup required inputs
        self.data_path = data_path
        self.fine_sampling_ball_diameter = fine_sampling_ball_diameter
        self.image_size = image_size

        self.robot = RobotInterface(
            ip_address="10.10.10.210", enforce_version=False, port=50051
        )

        # Get reference state
        self.robot.go_home()

        # self.arm = arm
        # self.gripper = gripper

        self.cam = Realsense("cam0")
        self.cam.connect()

        with open(cam_calib_file) as json_data:
            cam_calib = json.load(json_data)
        self.cam_info = get_cam_info(cam_calib[0])

        self.traj = []
        self.cams = []

    def get_obs_action(self, keypoint):
        """
        Fetch the desired state and action based on the provided demo.
            :param obs: incoming obs
            :return: required observation and action list
        """
        state = {"rgb": [], "pc": []}
        state["rgb"] += self.rgbs[keypoint]
        state["pc"] += self.pcds[keypoint]
        action = torch.from_numpy(self.traj[keypoint].float())
        return state, action

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        state = transform(state_dict, augmentation=False)
        state = einops.rearrange(
            state, "(m n ch) h w -> n m ch h w", ch=3, n=len(self.apply_cameras), m=2
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        attns = torch.Tensor([])
        for cam in self.apply_cameras:
            u, v = obs_to_attn(obs, cam)
            attn = torch.zeros(1, 1, 1, self.image_size[0], self.image_size[1])
            if not (
                u < 0
                or u > self.image_size[1] - 1
                or v < 0
                or v > self.image_size[0] - 1
            ):
                attn[0, 0, 0, v, u] = 1
            attns = torch.cat([attns, attn], 1)
        rgb = torch.cat([rgb, attns], 2)

        return rgb, pcd, gripper

    def get_current_obs_action(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :param normalise_rgb: normalise rgb to (-1, 1)
            :return: a list of obs and action
        """
        # TODO get obs
        sensor_state = self.cam.get_sensors()

        rgb = np.array(sensor_state["rgb"])
        rgb = cv2.resize(rgb, self.image_size)

        pcd = np.array(sensor_state["depth"]) / 1000
        pcd = cv2.resize(pcd, self.image_size)
        pcd = deproject(pcd, *self.cam_info).transpose(1, 0)
        pcd = np.reshape(pcd, (*self.image_size, 3))

        ee_pos = self.arm.get_state().ee_pos
        gripper_command = (
            0
            if self.gripper.get_sensors().item()
            < (self.gripper.robot.metadata.max_width / 2)
            else 1
        )

        # From quaternion to euler angles
        ee_euler = Rotation.from_quat(ee_pos[3:7]).as_euler("xyz")

        proprio = list(
            np.concatenate(
                [
                    ee_pos[:3],
                    ee_euler,
                    gripper_command,
                ],
                axis=-1,
            )
        )

        self.traj.append(proprio)
        self.rgbs.append(proprio)
        self.pcds.append(proprio)

        key_frame = keypoint_discovery([self.traj])
        key_frame.insert(0, 0)
        state_ls = []
        action_ls = []
        for f in key_frame:
            state, action = self.get_obs_action(demo._observations[f])
            state = transform(state, augmentation=False)
            state_ls.append(state.unsqueeze(0))
            action_ls.append(action.unsqueeze(0))
        return state_ls, action_ls

    def get_current_obs(self):
        sensor_state = self.cam.get_sensors()

        rgb = np.array(sensor_state["rgb"])
        rgb = cv2.resize(rgb, self.image_size)
        rgb = rgb / 255.0 * 2 - 1  # map RGB to [-1, 1]
        rgb = einops.rearrange(rgb, "h w d -> d h w")[None, None, :, :, :]

        pcd = np.array(sensor_state["depth"]) / 1000
        pcd = cv2.resize(pcd, self.image_size)
        pcd = deproject(pcd, *self.cam_info).transpose(1, 0)
        pcd = np.reshape(pcd, (*self.image_size, 3))
        pcd = einops.rearrange(pcd, "h w d -> d h w")[None, None, :, :, :]

        # ee_pos = self.robot.get_state().ee_pos
        ee_pos = torch.cat(self.robot.get_ee_pose())
        gripper_command = [0.0]
        # [
        #     (
        #         0
        #         if self.gripper.get_sensors().item()
        #         < (self.gripper.robot.metadata.max_width / 2)
        #         else 1
        #     )
        # ]

        # From quaternion to euler angles
        ee_euler = Rotation.from_quat(ee_pos[3:7]).as_euler("xyz")

        # print(ee_pos)
        # print(ee_euler)
        # print(gripper_command)

        proprio = np.concatenate([ee_pos[:3], ee_euler, gripper_command])[None, :]
        # print(np.array(proprio).shape)
        return (
            torch.tensor(rgb, dtype=torch.float32),
            torch.tensor(pcd, dtype=torch.float32),
            torch.tensor(proprio, dtype=torch.float32),
        )

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

    # def __del__(self):
    #     self.cam.close()

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

        success_rate = 0
        num_valid_demos = 0
        total_reward = 0

        rgbs = torch.Tensor([]).to(device)
        pcds = torch.Tensor([]).to(device)
        grippers = torch.Tensor([]).to(device)

        # move = Mover(self.robot, self.arm, self.gripper, max_tries=max_tries)
        reward = 0.0
        max_reward = 0.0

        for step_id in range(max_steps):
            # Fetch the current observation, and predict one action TODO
            # obs = self.get_current_obs_action()
            # rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
            rgb, pcd, gripper = self.get_current_obs()
            rgb = rgb.to(device)
            pcd = pcd.to(device)
            gripper = gripper.to(device)

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

            # print(rgbs_input.shape)
            # print(pcds_input.shape)
            # print(gripper_input.shape)

            output = actioner.predict(
                rgbs_input,
                pcds_input,
                gripper_input,
                interpolation_length=interpolation_length,
            )

            if verbose:
                print(f"Step {step_id}")

            # TODO Execute action
            action = output["trajectory"]
            action[..., -1] = torch.round(action[..., -1])
            action = action[-1].detach().cpu().numpy()
            action = action[-1]

            act_pos = action[:3]
            act_quat = action[3:6]
            act_grp = action[-1]

            curr = gripper_input[0][-1].cpu().numpy()
            curr_pos = curr[:3]
            curr_euler = curr[3:6]
            curr_quat = convert_rotation(curr_euler)
            curr_grp = curr[-1]

            print(f"action: {curr_pos} -> {act_pos}")

            corr_pos = curr_pos + np.clip(act_pos - curr_pos, -0.05, 0.05)

            print(f"corrected: {curr_pos} -> {corr_pos}")

            self.robot.move_to_ee_pose(corr_pos, None)
            # self.robot.move_to_ee_pose(ee_pos, None)
            # self.robot.move_to_ee_pose(action[:3], None)

            # print(f"{gripper_input[0][-1].cpu().numpy()}->{action}")

            # max_reward = max(max_reward, reward)
            # if reward == 1:
            #     break

        self.cam.close()
        print(f"Finished: {task_str}")

        return 1
