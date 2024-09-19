import pickle

import blosc
import torch

from tools.visualize_keypose_frames import visualize_actions_and_point_clouds_video


def main():
    # data_dir = "data/real_demo/mouse_on_pad+0/ep0.dat"
    data_dir = "data/real/packaged/train/pick_fruit+0/ep0.dat"
    ep = pickle.loads(blosc.decompress(open(data_dir, "rb").read()))
    rgbpcds = ep[1]
    rgbs = torch.as_tensor(rgbpcds[:, :, 0] / 2 + 0.5)
    pcds = torch.as_tensor(rgbpcds[:, :, 1])
    proprios = torch.cat(ep[4], dim=0)
    actions = torch.cat(ep[2], dim=0)

    visualize_actions_and_point_clouds_video(pcds, rgbs, actions, proprios, video=False)


if __name__ == "__main__":
    main()
