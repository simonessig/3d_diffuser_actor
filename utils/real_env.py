import einops
import numpy as np
import open3d
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils_with_real import Actioner, keypoint_discovery, obs_to_attn
from utils.utils_with_rlbench import transform


class Mover:
    def __init__(self, disabled=False, max_tries=1):
        self._last_action = None
        self._step_id = 0
        self._max_tries = max_tries
        self._disabled = disabled

    def __call__(self, action, collision_checking=False):
        # TODO:move
        return None, 0, True


class RealEnv:
    def __init__(
        self,
        data_path,
        image_size=(128, 128),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        apply_cameras=("front"),
        fine_sampling_ball_diameter=None,
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras
        self.fine_sampling_ball_diameter = fine_sampling_ball_diameter
        self.image_size = image_size

    def get_obs_action(self, obs):
        """
        Fetch the desired state and action based on the provided demo.
            :param obs: incoming obs
            :return: required observation and action list
        """

        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

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
            if not (u < 0 or u > self.image_size[1] - 1 or v < 0 or v > self.image_size[0] - 1):
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
        key_frame = keypoint_discovery(demo)
        key_frame.insert(0, 0)
        state_ls = []
        action_ls = []
        for f in key_frame:
            state, action = self.get_obs_action(demo._observations[f])
            state = transform(state, augmentation=False)
            state_ls.append(state.unsqueeze(0))
            action_ls.append(action.unsqueeze(0))
        return state_ls, action_ls

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

        success_rate = 0
        num_valid_demos = 0
        total_reward = 0

        rgbs = torch.Tensor([]).to(device)
        pcds = torch.Tensor([]).to(device)
        grippers = torch.Tensor([]).to(device)

        move = Mover(max_tries=max_tries)
        reward = 0.0
        max_reward = 0.0

        for step_id in range(max_steps):
            # Fetch the current observation, and predict one action TODO
            obs = self.get_current_obs_action()
            rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
            rgb, pcd, gripper = self.get_current_obs()
            rgb = rgb.to(device)
            pcd = pcd.to(device)
            gripper = gripper.to(device)

            rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
            pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
            grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)

            # Prepare proprioception history
            rgbs_input = rgbs[:, -1:][:, :, :, :3]
            pcds_input = pcds[:, -1:]
            if num_history < 1:
                gripper_input = grippers[:, -1]
            else:
                gripper_input = grippers[:, -num_history:]
                npad = num_history - gripper_input.shape[1]
                gripper_input = F.pad(gripper_input, (0, 0, npad, 0), mode="replicate")

            output = actioner.predict(
                rgbs_input, pcds_input, gripper_input, interpolation_length=interpolation_length
            )

            if verbose:
                print(f"Step {step_id}")

            # TODO Execute action
            continue

            terminate = True

            # Update the observation based on the predicted action
            try:
                trajectory = output["trajectory"][-1].cpu().numpy()
                trajectory[:, -1] = trajectory[:, -1].round()

                # execute
                for action in tqdm(trajectory):
                    reward, terminate = move(action)

                max_reward = max(max_reward, reward)

                if reward == 1:
                    success_rate += 1
                    break

                if terminate:
                    print("The episode has terminated!")

            except Exception as e:
                print(task_str, step_id, success_rate, e)
                reward = 0

        total_reward += max_reward
        if reward == 0:
            step_id += 1

        print(
            task_str,
            "Reward",
            f"{reward:.2f}",
            "max_reward",
            f"{max_reward:.2f}",
            f"SR: {success_rate}",
            f"SR: {total_reward:.2f}",
            "# valid demos",
            num_valid_demos,
        )

        return success_rate
