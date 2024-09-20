import numpy as np
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

    proprios[:, 3:7] += np.array([0, 0.38, 0, 0])
    proprios[:, 3:7] = proprios[:, [6, 3, 4, 5]]

    # print(ifc.cam.get_intrinsics())

    visualize_actions_and_point_clouds_video(pcds, rgbs, torch.zeros((1, 8)), proprios, video=False)


if __name__ == "__main__":
    main()
