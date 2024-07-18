import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import trange

from eval_real.utils.envs.env_interface import EnvInterface
from utils.utils_with_real import Actioner


class RealEnv:
    def __init__(self, interface: EnvInterface) -> None:
        self.interface = interface
        self.interface.connect()

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

        actions = []

        for step_id in trange(max_steps):
            rgb, pcd, gripper = self.interface.get_obs()
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

            output = actioner.predict(
                rgbs_input,
                pcds_input,
                gripper_input,
                interpolation_length=interpolation_length,
            )

            if verbose:
                print(f"Step {step_id}")
                print(action)

            action = output["action"][0]
            action = action.detach().cpu().numpy()[0]
            # print(action)

            act_euler = Rotation.from_quat(action[3:7]).as_euler("xyz")
            action = np.concatenate([action[:3], act_euler, [np.round(action[-1])]])
            actions.append(action)

            self.interface.move(action)

        rgb, pcd, gripper = self.interface.get_obs()
        rgb = rgb.to(device)
        pcd = pcd.to(device)
        gripper = gripper.to(device)

        rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1).detach().cpu().numpy()[0]
        pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1).detach().cpu().numpy()[0]
        grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1).detach().cpu().numpy()[0]
        actions = np.array(actions)

        self.interface.close()

        # self.draw_actions_3D(actions)

        self.draw_diff(grippers, actions)

        print(f"Finished: {task_str}")

        return 1

    def draw_actions_3D(self, actions):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection="3d")

        poses = actions.detach().cpu().numpy()
        xyz = poses[0].T
        x, y, z = xyz[0], xyz[1], xyz[2]

        ax.scatter(x, y, z)

        for i in range(len(x) - 1):
            ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], [z[i], z[i + 1]], color="g")

        ax.scatter(0, 0, 0, c="yellow")
        ax.scatter(1.4216465041442194, -0.012545208136006748, 0.8585146590952618, c="red")
        plt.show()

    def draw_diff(self, grippers, actions):
        fig = plt.figure(figsize=(30, 20))
        actions = np.concatenate((np.array([[None] * 7]), actions))

        ax_info = [
            [1, "X", (0, 1.5)],
            [2, "Y", (-1, 1)],
            [3, "Z", (0, 1.5)],
            [4, "Rot X", (-np.pi, np.pi)],
            [5, "Rot Y", (-np.pi, np.pi)],
            [6, "Rot Z", (-np.pi, np.pi)],
            [8, "Gripper", (0, 1)],
        ]

        axes = []

        for i, info in enumerate(ax_info):
            ax = fig.add_subplot(3, 3, info[0])
            ax.set_title(info[1])
            ax.set_ylim(*info[2])
            ax.plot(grippers[:, i], label="Ground Truth")
            ax.plot(actions[:, i], label="Prediction")
            axes.append(ax)

        axes[2].legend()

        # # X
        # ax = fig.add_subplot(2, 3, 1)
        # ax.set_title("X")
        # ax.set_ylim(0, 1.5)
        # ax.plot(grippers[:, 0], label="Ground Truth")
        # ax.plot(actions[:, 0], label="Prediction")

        # # Y
        # ax = fig.add_subplot(2, 3, 2)
        # ax.set_title("Y")
        # ax.set_ylim(-1, 1)
        # ax.plot(grippers[:, 1], label="Ground Truth")
        # ax.plot(actions[:, 1], label="Prediction")

        # # Z
        # ax = fig.add_subplot(2, 3, 3)
        # ax.set_title("Z")
        # ax.set_ylim(0, 1.5)
        # ax.plot(grippers[:, 2], label="Ground Truth")
        # ax.plot(actions[:, 2], label="Prediction")
        # ax.legend()

        plt.show()
