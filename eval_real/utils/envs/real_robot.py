from typing import Tuple

import numpy as np
import torch
from polymetis import RobotInterface  # type: ignore

from eval_real.utils.envs.env_interface import EnvInterface
from eval_real.utils.real_robot.realsense import Realsense


class RealRobotInterface(EnvInterface):
    def __init__(self, image_size, cam_calib_file) -> None:
        super().__init__(image_size, cam_calib_file)
        self.robot = RobotInterface(ip_address="10.10.10.210", enforce_version=False, port=50051)
        self.cam = Realsense("cam0")

    def connect(self) -> None:
        self.robot.go_home()
        self.cam.connect()

    def get_obs(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        ee_pos = torch.cat(self.robot.get_ee_pose())
        gripper_command = [
            (
                0.0
                if self.gripper.get_sensors().item() < (self.gripper.robot.metadata.max_width / 2)
                else 1.1
            )
        ]
        sensor_state = self.cam.get_sensors()

        return self._prepare_obs(
            sensor_state["rgb"], sensor_state["depth"], ee_pos, gripper_command
        )

    def move(self, action) -> None:
        act_pos = action[:3]
        act_euler = action[3:6]
        act_grp = action[-1]

        self.robot.move_to_ee_pose(act_pos, None)

    def close(self) -> None:
        self.cam.close()
