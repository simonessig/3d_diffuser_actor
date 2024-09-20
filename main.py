import json
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torchvision import transforms

import interactive_guidance as ig
from eval_real.utils.real_robot.azure import Azure
from eval_real.utils.real_robot.realsense import Realsense
from utils.utils_with_real import get_cam_info


def plot_guidance(guidance):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    x_range = [0, 1]
    y_range = [-0.5, 0.5]
    z_range = [0, 1]
    n_grid = 15
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_zlim(*z_range)

    grid_x, grid_y, grid_z = np.meshgrid(
        np.linspace(*x_range, n_grid),
        np.linspace(*y_range, n_grid),
        np.linspace(*z_range, n_grid),
    )

    x = grid_x.flatten()
    y = grid_y.flatten()
    z = grid_z.flatten()
    grid_xyz = np.stack((x, y, z)).swapaxes(0, 1)

    scores = guidance.apply(
        torch.zeros(grid_xyz.shape, dtype=torch.float32),
        torch.tensor(grid_xyz, dtype=torch.float32),
    )

    u, v, w = scores.cpu().numpy().T

    ax.quiver(x, y, z, u, v, w, length=0.04, normalize=True)
    plt.show()


def main():
    # time.sleep(5)
    with open("data/real/calibration.json") as json_data:
        cam_calib = json.load(json_data)
    cam_info = get_cam_info(cam_calib[0], offset=False)

    # cam = Azure("cam0", (640, 480), cam_info)
    # cam.connect()
    # img = cam.get_sensors()["rgb"]

    # intrinsics = cam.get_intrinsics()

    # pos = cam_info[1][:3, 3]
    # rot = cam_info[1][:3, :3]
    # rot = rot[[1, 2, 0]]

    cam = Realsense("cam1", (640, 480), cam_info)
    cam.connect()
    img = cam.get_sensors()["rgb"][:, :, [2, 1, 0]]

    intrinsics = cam_info[0]
    pos = cam_info[1][:3, 3]
    rot = cam_info[1][:3, :3]
    rot = rot[[1, 2, 0]]

    intrinsics = torch.zeros((3, 4), dtype=torch.float32)
    intrinsics[0, 0] = cam_info[0].fx
    intrinsics[1, 1] = cam_info[0].fy
    intrinsics[0, 2] = cam_info[0].ppx
    intrinsics[1, 2] = cam_info[0].ppy
    intrinsics[2, 2] = 1

    guidance = ig.Guidance()

    llm_mask = None
    while llm_mask is None or input("Again? ") == "y":
        llm_mask = ig.make_llm_mask(
            Image.fromarray(img),
            "The robot can pick up the lemon or the orange. Pick up the left fruit.",
            True,
        )

    llm_guide = ig.CamGuide(llm_mask, intrinsics, pos, rot, mask_only_frame=False, mult=3)
    guidance.add(llm_guide)

    # mask = ig.start_gui(Image.fromarray(img))
    # guide = ig.CamGuide(mask, intrinsics, pos, rot, mask_only_frame=True)
    # guidance = ig.Guidance()
    # guidance.add(guide)

    # person_guide = ig.PersonGuide()
    # guidance.add(person_guide)

    # def static_call(x):
    #     zeros = torch.zeros_like(x)
    #     ones = torch.zeros_like(x)
    #     ones[:, 1] = -1
    #     return torch.where(x >= 0, ones, zeros)

    # static_guide = ig.StaticGuide(static_call, mult=5)
    # guidance.add(static_guide)

    plot_guidance(guidance)


if __name__ == "__main__":
    main()
