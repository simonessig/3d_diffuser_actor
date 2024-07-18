import abc
import json
from typing import Tuple

import cv2
import einops
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from utils.utils_with_real import deproject, get_cam_info


class EnvInterface(abc.ABC):
    def __init__(self, image_size, cam_calib_file) -> None:
        super().__init__()
        self.image_size = image_size

        with open(cam_calib_file) as json_data:
            cam_calib = json.load(json_data)
        self.cam_info = get_cam_info(cam_calib[0])

    def connect(self) -> None:
        pass

    def get_obs(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        pass

    def move(self, action: torch.tensor) -> None:
        pass

    def close(self) -> None:
        pass

    def _prepare_obs(
        self, rgb_img, depth_img, ee_pos, gripper_command
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        rgb = np.array(rgb_img)
        rgb = cv2.resize(rgb, self.image_size)
        rgb = rgb / 255.0 * 2 - 1  # map RGB to [-1, 1]
        rgb = einops.rearrange(rgb, "h w d -> d h w")[None, None, :, :, :]

        pcd = np.array(depth_img) / 1000
        pcd = cv2.resize(pcd, self.image_size)
        pcd = deproject(pcd, *self.cam_info).transpose(1, 0)
        pcd = np.reshape(pcd, (*self.image_size, 3))
        pcd = einops.rearrange(pcd, "h w d -> d h w")[None, None, :, :, :]

        # From quaternion to euler angles
        ee_euler = Rotation.from_quat(ee_pos[3:7]).as_euler("xyz")

        proprio = np.concatenate([ee_pos[:3], ee_euler, [gripper_command]])[None, :]
        return (
            torch.tensor(rgb, dtype=torch.float32),
            torch.tensor(pcd, dtype=torch.float32),
            torch.tensor(proprio, dtype=torch.float32),
        )
