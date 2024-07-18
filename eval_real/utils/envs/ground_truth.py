from typing import Tuple

import numpy as np
import torch
from PIL import Image

from eval_real.utils.envs.env_interface import EnvInterface


class GroundTruthInterface(EnvInterface):
    def __init__(self, image_size, cam_calib_file, data_dir) -> None:
        super().__init__(image_size, cam_calib_file)
        self._step = 0

        self._ee_poses = torch.load(f"{data_dir}/ee_pos.pt").numpy()
        self._gripper_commands = torch.load(f"{data_dir}/gripper_command.pt").numpy()

        self._gripper_commands = (self._gripper_commands > 0).astype(np.float32)

        rgb_dir = data_dir / "img" / "cam0_rgb"
        rgb_path_gen = sorted(rgb_dir.glob("*.png"), key=lambda x: int(x.name[:-4]))
        self._rgb_imgs = []
        for path in rgb_path_gen:
            self._rgb_imgs.append(Image.open(path))

        depth_dir = data_dir / "img" / "cam0_d"
        depth_path_gen = sorted(depth_dir.glob("*.png"), key=lambda x: int(x.name[:-4]))
        self._depth_imgs = []
        for path in depth_path_gen:
            self._depth_imgs.append(Image.open(path))

    def get_obs(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        return self._prepare_obs(
            self._rgb_imgs[self._step],
            self._depth_imgs[self._step],
            self._ee_poses[self._step],
            self._gripper_commands[self._step],
        )

    def move(self, action: torch.tensor) -> None:
        if self._step == 0:
            self._step += 1
        self._step += 1
