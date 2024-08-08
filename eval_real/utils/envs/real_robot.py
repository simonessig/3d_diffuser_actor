import concurrent
import time
from typing import Tuple

import numpy as np
import torch
from polymetis import GripperInterface  # type: ignore
from polymetis import RobotInterface  # type: ignore

from eval_real.utils.envs.env_interface import EnvInterface
from eval_real.utils.real_robot.azure import Azure
from eval_real.utils.real_robot.realsense import Realsense


class RealRobotInterface(EnvInterface):
    def __init__(self, image_size, cam_calib_file) -> None:
        super().__init__(image_size, cam_calib_file)
        self.robot = RobotInterface(ip_address="10.10.10.210", enforce_version=False, port=50051)
        self.gripper = GripperInterface(ip_address="10.10.10.210", port=50052)
        self.cam = Azure("cam0", self.image_size, self.cam_info)
        # self.cam = Realsense("cam0", self.image_size, self.cam_info)

        self.max_width = 0.0
        self.min_width = 0.0

        self.within_grasp_action = False

        self.pool = concurrent.futures.ThreadPoolExecutor(1)

    def prepare(self, _):
        self._grasp()
        self._open()
        self.robot.move_to_joint_positions(
            [-0.1271, -0.0062, 0.1186, -2.2072, 0.0027, 2.1948, 0.7662]
        )
        # self._grasp()
        # time.sleep(5)
        # self._open()

    def connect(self) -> None:
        if self.gripper.metadata:
            self.max_width = self.gripper.metadata.max_width
        elif self.gripper.get_state().max_width:
            self.max_width = self.gripper.get_state().max_width
        else:
            self.max_width = 0.085

        # self.robot.go_home()

        self.cam.connect()

    def get_obs(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        rgb, pcd = self.cam.get_obs()

        ee_pos = torch.cat(self.robot.get_ee_pose())

        gripper_state = self.gripper.get_state()
        gripper_command = 0.0 if gripper_state.is_grasped else 1.0

        ee_pos[3:7] += torch.tensor([0, 0.38, 0, 0])
        ee_pos[3:7] = ee_pos[[6, 3, 4, 5]]

        proprio = np.concatenate([ee_pos[:7], [gripper_command]])[None, :]
        return (
            rgb,
            pcd,
            torch.tensor(proprio, dtype=torch.float32),
        )

    def move(self, action) -> None:
        act_pos = action[:3]
        act_quat = action[3:7]
        act_grp = action[-1]

        if act_grp > 0:
            self._open()
        else:
            self._grasp()

        self.robot.move_to_ee_pose(act_pos, None)

    def close(self) -> None:
        self.cam.close()

    def _open(self):
        state = self.gripper.get_state()
        if state.is_moving:
            return

        if state.is_grasped:
            self.gripper.goto(self.max_width, 0.1, 0.1)
            # self.pool.submit(self.gripper.goto, self.max_width, 0.1, 0.1)

    def _grasp(self):
        if self.within_grasp_action:
            return

        state = self.gripper.get_state()
        if state.is_moving:
            return

        if not state.is_grasped:
            self.grasp_helper(0.1, 0.1)
            # self.pool.submit(self.grasp_helper, 0.1, 0.1)

    def grasp_helper(self, speed, force):
        self.within_grasp_action = True

        self.gripper.grasp(speed, force)

        state = self.gripper.get_state()
        while not state.is_grasped:
            state = self.gripper.get_state()
            time.sleep(0.1)
        while state.is_moving:
            state = self.gripper.get_state()
            time.sleep(0.1)

        self.within_grasp_action = False
