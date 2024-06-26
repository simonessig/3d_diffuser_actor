"""Online evaluation script on Real Robot."""

import json
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tap
import torch

from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
from online_evaluation_real.ig import InteractiveGuidance
from utils.common_utils import get_gripper_loc_bounds, round_floats
from utils.real_env import RealEnv
from utils.utils_with_real import Actioner


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
    gripper_loc_bounds_file: str = None
    gripper_loc_bounds_buffer: float = 0.04
    single_task_gripper_loc_bounds: int = 0
    predict_trajectory: int = 1

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
    use_instruction: int = 1
    rotation_parametrization: str = "6D"
    quaternion_format: str = "xyzw"


def load_models(args):
    device = torch.device(args.device)

    print("Loading model from", args.checkpoint, flush=True)

    # Gripper workspace is the union of workspaces for all tasks
    if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
        task = args.tasks[0]
    else:
        task = None
    print("Gripper workspace")

    if args.gripper_loc_bounds is None:
        gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    else:
        gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds_file,
            task=task,
            buffer=args.gripper_loc_bounds_buffer,
        )

    ig = InteractiveGuidance(device)

    if args.test_model == "3d_diffuser_actor":
        model = DiffuserActor(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            use_instruction=bool(args.use_instruction),
            fps_subsampling_factor=args.fps_subsampling_factor,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=args.rotation_parametrization,
            quaternion_format=args.quaternion_format,
            diffusion_timesteps=args.diffusion_timesteps,
            nhist=args.num_history,
            relative=bool(args.relative_action),
            lang_enhanced=bool(args.lang_enhanced),
            ig=ig,
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

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load models
    model = load_models(args)

    # Load RLBench environment
    env = RealEnv(
        data_path=args.data_dir,
        image_size=[int(x) for x in args.image_size.split(",")],
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=args.cameras,
    )

    actioner = Actioner(
        policy=model,
        instructions=None,
        apply_cameras=args.cameras,
        action_dim=args.action_dim,
        predict_trajectory=bool(args.predict_trajectory),
    )
    task_success_rates = {}

    for task_str in args.tasks:
        var_success_rates = env.evaluate_task(
            task_str,
            max_steps=args.max_steps,
            actioner=actioner,
            max_tries=args.max_tries,
            interpolation_length=args.interpolation_length,
            verbose=bool(args.verbose),
            num_history=args.num_history,
        )
        print()
        print(f"{task_str} variation success rates:", round_floats(var_success_rates))
        print(f"{task_str} mean success rate:", round_floats(var_success_rates["mean"]))

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
