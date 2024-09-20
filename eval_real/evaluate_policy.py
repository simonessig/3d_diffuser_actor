"""Online evaluation script on Real Robot."""

import json
import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tap
import torch
from PIL import Image

import interactive_guidance as ig
import wandb
from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
from eval_real.utils.envs.ground_truth import GroundTruthInterface
from eval_real.utils.envs.real_robot import RealRobotInterface

# from eval_real.utils.ig import InteractiveGuidance
from eval_real.utils.real_env import RealEnv
from eval_real.utils.real_robot.azure import Azure
from eval_real.utils.real_robot.realsense import Realsense
from interactive_guidance.guides import PointGuide
from utils.common_utils import get_gripper_loc_bounds
from utils.utils_with_real import Actioner, get_cam_info


class Arguments(tap.Tap):
    checkpoint: Path = ""
    seed: int = 2
    device: str = "cuda"
    num_episodes: int = 1
    headless: int = 0
    max_tries: int = 10
    tasks: Optional[Tuple[str, ...]] = None
    instructions: Optional[Path] = "instructions.pkl"
    variations: Tuple[int, ...] = (-1,)
    data_dir: Path = Path(__file__).parent / "demos"
    cameras: Tuple[str, ...] = "front"
    image_size: str = "128,128"
    verbose: int = 0
    output_file: Path = Path(__file__).parent / "eval.json"
    max_steps: int = 25
    test_model: str = "3d_diffuser_actor"
    collision_checking: int = 0
    gripper_loc_bounds: str = None
    gripper_loc_bounds_buffer: float = 0.04
    predict_trajectory: int = 0

    robot_ip: str = "10.10.10.210"
    arm_port: int = 50051
    gripper_port: int = 50052

    # 3D Diffuser Actor model parameters
    diffusion_timesteps: int = 100
    num_history: int = 3
    fps_subsampling_factor: int = 5
    lang_enhanced: int = 0
    dense_interpolation: int = 1
    interpolation_length: int = 2
    relative_action: int = 0

    # Shared model parameters
    action_dim: int = 8
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 0
    rotation_parametrization: str = "6D"
    quaternion_format: str = "xyzw"


def load_models(args):
    device = torch.device(args.device)

    print("Loading model from", args.checkpoint, flush=True)
    # time.sleep(5)
    guidance = ig.Guidance()

    # with open(Path(args.data_dir) / "calibration.json") as json_data:
    #     cam_calib = json.load(json_data)
    # cam_info = get_cam_info(cam_calib[0])

    # cam = Azure("cam0", (640, 480), cam_info)
    # cam.connect()
    # img = cam.get_sensors()["rgb"]

    # intrinsics = cam.get_intrinsics()

    # cam = Realsense("cam1", (640, 480), cam_info)
    # cam.connect()
    # img = cam.get_sensors()["rgb"][:, :, [2, 1, 0]]

    # intrinsics = torch.zeros((3, 4), dtype=torch.float32)
    # intrinsics[0, 0] = cam_info[0].fx
    # intrinsics[1, 1] = cam_info[0].fy
    # intrinsics[0, 2] = cam_info[0].ppx
    # intrinsics[1, 2] = cam_info[0].ppy
    # intrinsics[2, 2] = 1

    # pos = cam_info[1][:3, 3]
    # rot = cam_info[1][:3, :3]
    # rot = rot[[1, 2, 0]]

    def point_cond():
        # print(RealEnv.STEP_ID == 0)
        return RealEnv.STEP_ID != 2

    # llm_mask = None
    # while llm_mask is None or input("Again? ") == "y":
    #     llm_mask = ig.make_llm_mask(
    #         Image.fromarray(img),
    #         "The robot can pick up the lemon or the orange. Pick up the right fruit.",
    #         True,
    #     )

    # llm_guide = ig.CamGuide(
    #     llm_mask, intrinsics, pos, rot, mask_only_frame=False, mult=5, condition=point_cond
    # )
    # guidance.add(llm_guide)

    # person_guide = ig.PersonGuide(condition=point_cond)
    # guidance.add(person_guide)

    # mask = ig.start_gui(Image.fromarray(img))
    # cam_guide = ig.CamGuide(mask, intrinsics, pos, rot, mask_only_frame=False, mult=5)
    # guidance.add(cam_guide)

    # point_guide = PointGuide(torch.tensor([0.45, 0.5, -0.1]), 1, mult=5, condition=point_cond)
    # guidance.add(point_guide)
    # return

    def static_call(x):
        zeros = torch.zeros_like(x)
        ones = torch.zeros_like(x)
        ones[:, 1] = -1
        return torch.where(x >= 0, ones, zeros)

    static_guide = ig.StaticGuide(static_call, mult=5, condition=point_cond)
    guidance.add(static_guide)

    if args.test_model == "3d_diffuser_actor":
        model = DiffuserActor(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            use_instruction=bool(args.use_instruction),
            fps_subsampling_factor=args.fps_subsampling_factor,
            gripper_loc_bounds=args.gripper_loc_bounds,
            rotation_parametrization=args.rotation_parametrization,
            quaternion_format=args.quaternion_format,
            diffusion_timesteps=args.diffusion_timesteps,
            nhist=args.num_history,
            relative=bool(args.relative_action),
            lang_enhanced=bool(args.lang_enhanced),
            guidance=guidance,
        ).to(device)
    else:
        raise NotImplementedError

    # Load model weights
    model_dict = torch.load(args.checkpoint, map_location="cpu")
    model_dict_weight = {}
    for key in model_dict["weight"]:
        _key = key[7:]
        model_dict_weight[_key] = model_dict["weight"][key]
    model.load_state_dict(model_dict_weight)
    model.eval()

    return model


if __name__ == "__main__":
    # Arguments
    args = Arguments().parse_args()
    args.cameras = tuple(x for y in args.cameras for x in y.split(","))
    print("Arguments:")
    print(args)
    print("-" * 100)
    # Save results here
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # gripper_loc_bounds = np.array([[0.3, -0.28, 0.1], [0.67, 0.28, 0.5]]) * 1.0

    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[0.3, -0.3, 0.1], [0.7, 0.3, 0.5]]) * 1.0
    else:
        args.gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds,
            task=args.tasks[0] if len(args.tasks) == 1 else None,
            buffer=args.gripper_loc_bounds_buffer,
        )

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load models
    model = load_models(args)

    # instruction = load_instructions(args.instructions)

    # ifc = GroundTruthInterface(
    #     tuple(int(x) for x in args.image_size.split(",")),
    #     Path(args.data_dir) / "raw" / args.tasks[0] / "calibration.json",
    #     Path(args.data_dir / "raw" / f"{args.tasks[0]}"),
    # )

    ifc = RealRobotInterface(
        tuple(int(x) for x in args.image_size.split(",")),
        Path(args.data_dir) / "calibration.json",
    )

    env = RealEnv(ifc)

    actioner = Actioner(
        policy=model,
        instructions=None,
        apply_cameras=args.cameras,
        action_dim=args.action_dim,
        predict_trajectory=bool(args.predict_trajectory),
    )
    task_success_rates = {}

    for task_str in args.tasks:
        success_rate = env.evaluate_task(
            task_str,
            max_steps=args.max_steps,
            actioner=actioner,
            max_tries=args.max_tries,
            interpolation_length=args.interpolation_length,
            verbose=bool(args.verbose),
            num_history=args.num_history,
        )
