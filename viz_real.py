import torch

from eval_real.utils.envs.real_robot import RealRobotInterface
from tools.visualize_keypose_frames import visualize_actions_and_point_clouds_video


def main():

    ifc = RealRobotInterface(
        (256, 256),
        "data/real/calibration.json",
    )
    ifc.connect()
    # ifc.prepare(0)
    rgbs, pcds, proprios = ifc.get_obs()
    rgbs = rgbs / 2 + 0.5

    visualize_actions_and_point_clouds_video(pcds, rgbs, torch.zeros((1, 8)), proprios, video=False)


if __name__ == "__main__":
    main()
