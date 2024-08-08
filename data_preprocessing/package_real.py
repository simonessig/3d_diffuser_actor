import json
import pickle
from pathlib import Path

import blosc
import numpy as np
import tap
import torch
from PIL import Image
from tqdm import tqdm

from utils.utils_with_real import get_cam_info, process_kinect, viz_pcd


class Arguments(tap.Tap):
    data_dir: Path = Path(__file__).parent.parent / "data/real/raw"
    seed: int = 15
    task: str = "pick_box"
    split_train: float = 8
    split_test: float = 2
    split_val: float = 2
    image_size: str = "256,256"
    output: Path = Path(__file__).parent.parent / "data/real/packaged"


def load_episode(data_dir, datas, args, cam_info, ann_id):
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
    img_dim = tuple(map(int, args.image_size.split(",")))

    ee_pos = torch.load(f"{data_dir}/ee_pos.pt").numpy()
    gripper_command = torch.load(f"{data_dir}/gripper_command.pt").numpy()

    ee_pos[:, 3:7] += np.array([0, 0.38, 0, 0])
    ee_pos[:, 3:7] = ee_pos[:, [6, 3, 4, 5]]

    # Map gripper openess to [0, 1]
    gripper_command = (gripper_command > 0).astype(np.float32)[:, None]

    proprio = list(
        np.concatenate(
            [
                ee_pos[:, :7],
                gripper_command,
            ],
            axis=-1,
        )
    )

    # sorted rgb images
    rgb_dir = data_dir / "img" / "cam0_rgb"
    rgb_path_gen = sorted(rgb_dir.glob("*.png"), key=lambda x: int(x.name[:-4]))

    # sorted depth images
    depth_dir = data_dir / "img" / "cam0_d"
    depth_path_gen = sorted(depth_dir.glob("*.png"), key=lambda x: int(x.name[:-4]))

    # sorted pointclouds
    pcd_dir = data_dir / "img" / "cam0_pc"
    pcd_path_gen = sorted(pcd_dir.glob("*.pt"), key=lambda x: int(x.name[:-3]))

    rgbs = []
    pcds = []
    for rgb_path, depth_path, pcd_path in zip(rgb_path_gen, depth_path_gen, pcd_path_gen):
        rgb = np.array(Image.open(rgb_path))
        depth = np.array(Image.open(depth_path))
        pcd = torch.load(pcd_path).numpy()

        rgb, pcd = process_kinect(rgb, pcd, img_dim, cam_info, depth=depth)

        rgbs.append(rgb)
        pcds.append(pcd)

    # viz_pcd(pcd=pcds, rgb=rgbs, proprio=proprio, idxs=[0, 1])
    # return

    # Put them into a dict
    datas["pcd"] += pcds  # (*img_dim, 3)
    datas["rgb"] += rgbs  # (*img_dim, 3)
    datas["proprios"] += proprio  # (8,)
    datas["annotation_id"].append(ann_id)  # int

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
    episodes_dir = args.data_dir / args.task / "episodes"
    data_dirs = np.array([[], [], []]).T

    split_strings = np.array(
        ["train"] * args.split_train + ["test"] * args.split_test + ["val"] * args.split_val
    )

    for var in episodes_dir.glob("var*"):
        episodes = np.array([ep for ep in var.glob("episode*")])
        ann_id = np.full_like(split_strings, var.stem[3:])
        np.random.shuffle(episodes)
        data_dirs = np.concatenate(
            (data_dirs, np.stack((episodes, split_strings, ann_id)).T), axis=0
        )

    cam_calib_file = args.data_dir / args.task / "calibration.json"
    with open(cam_calib_file) as json_data:
        cam_calib = json.load(json_data)
    cam_info = get_cam_info(cam_calib[0])

    for data_dir, split, ann_id in tqdm(data_dirs):
        ep_id = data_dir.stem[7:]

        datas = {
            "pcd": [],
            "rgb": [],
            "proprios": [],
            "annotation_id": [],
        }

        # Load data
        load_episode(data_dir, datas, args, cam_info, ann_id)

        # Only keypoints are captured during demos
        keyframe_inds = np.array([1, 3, 4])

        # Construct save data
        state_dict = process_datas(datas, keyframe_inds)
        # print(state_dict[4][0])
        # return
        # Save
        taskvar_dir = args.output / split / f"{args.task}+0"
        taskvar_dir.mkdir(parents=True, exist_ok=True)
        with open(taskvar_dir / f"ep{ep_id}.dat", "wb") as f:
            f.write(blosc.compress(pickle.dumps(state_dict)))


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)
