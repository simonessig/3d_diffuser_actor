import itertools
import json
import os
import pickle
import random
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import blosc
import cv2
import einops
import matplotlib.pyplot as plt
import numpy as np
import tap
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from utils.utils_with_real import deproject, get_cam_info, keypoint_discovery, viz_pcd


class Arguments(tap.Tap):
    data_dir: Path = Path(__file__).parent.parent / "data/real/raw"
    seed: int = 2
    task: str = "pick_box"
    split: str = "train"
    image_size: str = "128,128"  # "256,256"
    output: Path = Path(__file__).parent.parent / "data/real/packaged"


def load_episode(root_dir, episode, datas, args, cam_info):
    """Load episode and process datas

    Args:
        root_dir: a string of the root directory of the dataset
        split: a string of the split of the dataset
        episode: a string of the episode name
        datas: a dict of the datas to be saved/loaded
            - pcd: a list of nd.arrays with shape (height, width, 3)
            - rgb: a list of nd.arrays with shape (height, width, 3)
            - proprios: a list of nd.arrays with shape (8,)
    """
    data_dir = root_dir / f"episode{episode}"
    img_dim = tuple(map(int, args.image_size.split(",")))

    # sorted rgb images
    rgb_dir = data_dir / "img" / "cam0_rgb"
    rgb_path_gen = sorted(rgb_dir.glob("*.png"), key=lambda x: int(x.name[:-4]))

    # unpack rgb images
    # rgb = torch.empty((*img_dim, 3, 1))
    rgb = []
    for path in rgb_path_gen:
        img = np.array(Image.open(path))
        img = cv2.resize(img, img_dim)
        img = img / 255.0 * 2 - 1  # map RGB to [-1, 1]
        # rgb = torch.cat([rgb, torch.tensor(img)[:, :, :, None]], dim=3)
        rgb.append(img)

    # sorted depth images
    depth_dir = data_dir / "img" / "cam0_d"
    depth_path_gen = sorted(depth_dir.glob("*.png"), key=lambda x: int(x.name[:-4]))

    # unpack depth images
    # pcd = torch.empty((*img_dim, 3, 1))
    pcd = []
    for path in depth_path_gen:
        img = np.array(Image.open(path)) / 1000
        img = cv2.resize(img, img_dim)
        depth = deproject(img, *cam_info).transpose(1, 0)
        # viz_pcd(depth)
        depth = np.reshape(depth, (*img_dim, 3))
        # pcd = torch.cat([pcd, torch.tensor(depth)[:, :, :, None]], dim=3)
        pcd.append(depth)

    ee_pos = torch.load(f"{data_dir}/ee_pos.pt").numpy()
    gripper_command = torch.load(f"{data_dir}/gripper_command.pt").numpy()

    # From quaternion to euler angles
    ee_euler = Rotation.from_quat(ee_pos[:, 3:7]).as_euler("xyz")

    # Map gripper openess to [0, 1] TODO
    gripper_command = (gripper_command > 0).astype(np.float32)[:, None]

    proprio = list(
        np.concatenate(
            [
                ee_pos[:, :3],
                ee_euler,
                gripper_command,
            ],
            axis=-1,
        )
    )
    # print(proprio.shape)

    # print(np.array(rgb).shape)
    # print(pcd.shape)
    # print(proprio.shape)

    # Put them into a dict
    # datas["pcd"].append(*pcd)  # (*img_dim, 3)
    # datas["rgb"].append(*rgb)  # (*img_dim, 3)
    # datas["proprios"].append(*proprio)  # (8)
    datas["pcd"] += pcd  # (*img_dim, 3)
    datas["rgb"] += rgb  # (*img_dim, 3)
    datas["proprios"] += proprio  # (7,)
    # datas["annotation_id"].append(ann_id)  # int

    # print(np.array(datas["pcd"]).shape)
    # print(np.array(datas["rgb"]).shape)
    # print(np.array(datas["proprios"]).shape)

    return datas


def process_datas(datas, keyframe_inds):
    """Fetch and drop datas to make a trajectory

    Args:
        datas: a dict of the datas to be saved/loaded
            - pcd: a list of nd.arrays with shape (height, width, 3)
            - rgb: a list of nd.arrays with shape (height, width, 3)
            - proprios: a list of nd.arrays with shape (7,)
        keyframe_inds: an Integer array with shape (num_keyframes,)

    Returns:
        the episode item: [
            [frame_ids],
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
            [annotation_ind] # wrt frame_ids, (1,)
        ]
    """
    pcd = np.array(datas["pcd"])[:, None]  # (traj_len, H, W, 3)
    rgb = np.array(datas["rgb"])[:, None]  # (traj_len, H, W, 3)
    rgb_pcd = np.stack([rgb, pcd], axis=2)  # (traj_len, ncam, 2, H, W, 3)])
    rgb_pcd = rgb_pcd.transpose(0, 1, 2, 5, 3, 4)  # (traj_len, ncam, 2, 3, H, W)
    rgb_pcd = torch.as_tensor(rgb_pcd, dtype=torch.float32)  # (traj_len, ncam, 2, 3, H, W)

    # prepare keypose actions
    keyframe_indices = torch.as_tensor(keyframe_inds)[None, :]
    gripper_indices = torch.arange(len(datas["proprios"])).view(-1, 1)
    action_indices = torch.argmax((gripper_indices < keyframe_indices).float(), dim=1).tolist()
    action_indices[-1] = len(keyframe_inds) - 1
    actions = [datas["proprios"][keyframe_inds[i]] for i in action_indices]
    action_tensors = [torch.as_tensor(a, dtype=torch.float32).view(1, -1) for a in actions]

    # prepare camera_dicts
    camera_dicts = [{"front": (0, 0)}]

    # prepare gripper tensors
    gripper_tensors = [
        torch.as_tensor(a, dtype=torch.float32).view(1, -1) for a in datas["proprios"]
    ]

    # prepare trajectories
    trajectories = []
    for i in range(len(action_indices)):
        target_frame = keyframe_inds[action_indices[i]]
        current_frame = i
        trajectories.append(
            torch.cat(
                [
                    torch.as_tensor(a, dtype=torch.float32).view(1, -1)
                    for a in datas["proprios"][current_frame : target_frame + 1]
                ],
                dim=0,
            )
        )

    # Filter out datas
    keyframe_inds = [0] + keyframe_inds[:-1].tolist()
    keyframe_indices = torch.as_tensor(keyframe_inds)
    rgb_pcd = torch.index_select(rgb_pcd, 0, keyframe_indices)
    action_tensors = [action_tensors[i] for i in keyframe_inds]
    gripper_tensors = [gripper_tensors[i] for i in keyframe_inds]
    trajectories = [trajectories[i] for i in keyframe_inds]

    # prepare frame_ids
    frame_ids = [i for i in range(len(rgb_pcd))]

    # Save everything to disk
    state_dict = [
        frame_ids,
        rgb_pcd,
        action_tensors,
        camera_dicts,
        gripper_tensors,
        trajectories,
        datas["annotation_id"],
    ]

    return state_dict


def main(args):
    # print(480 / (2 * np.tan(np.deg2rad(58) / 2)))
    # print(np.rad2deg(2 * np.arctan(480 / (2 * 608))))
    # return
    episodes_dir = args.data_dir / args.task / args.split
    episodes = [int(ep.stem[7:]) for ep in episodes_dir.glob("episode*")]

    cam_calib_file = args.data_dir / args.task / "calibration.json"
    with open(cam_calib_file) as json_data:
        cam_calib = json.load(json_data)
    cam_info = get_cam_info(cam_calib[0])

    for ep_id in tqdm(episodes):
        datas = {
            "pcd": [],
            "rgb": [],
            "proprios": [],
            "annotation_id": [],
        }

        # Load data
        load_episode(episodes_dir, ep_id, datas, args, cam_info)

        # Get keypoints
        _, keyframe_inds = keypoint_discovery(datas["proprios"])

        # Construct save data
        state_dict = process_datas(datas, keyframe_inds)
        # print(state_dict[1][0])
        # return
        # Save
        taskvar_dir = args.output / args.split / f"{args.task}+0"
        taskvar_dir.mkdir(parents=True, exist_ok=True)
        with open(taskvar_dir / f"ep{ep_id}.dat", "wb") as f:
            f.write(blosc.compress(pickle.dumps(state_dict)))


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)
