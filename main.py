import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torchvision import transforms

import interactive_guidance as ig
from eval_real.utils.real_robot.azure import Azure
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
    with open("data/real/calibration.json") as json_data:
        cam_calib = json.load(json_data)
    cam_info = get_cam_info(cam_calib[0])

    cam = Azure("cam0", (640, 480), cam_info)
    cam.connect()
    img = cam.get_sensors()["rgb"]

    intrinsics = cam.get_intrinsics()

    pos = cam_info[1][:3, 3]
    rot = cam_info[1][:3, :3]
    rot = rot[[1, 2, 0]]

    mask = ig.start_gui(Image.fromarray(img))
    guide = ig.CamGuide(mask, intrinsics, pos, rot, mask_only_frame=True)
    guidance = ig.Guidance()
    guidance.add(guide)

    plot_guidance(guidance)


if __name__ == "__main__":
    main()
