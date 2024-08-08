import time
from typing import Tuple

import cv2
import einops
import numpy as np
import pyrealsense2 as rs
import torch

from utils.utils_with_real import deproject


class Realsense:
    def __init__(self, name: str, image_size, cam_info):
        self.name = name
        self.pipeline = None
        self.image_size = image_size
        self.cam_info = cam_info

    def connect(self) -> bool:
        print("Connecting to {}: ".format(self.name))
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            return True
        except Exception as e:
            self.device = None
            print("Failed with exception: ", e)
            return False

    def get_sensors(self):
        if not self.pipeline:
            raise Exception(f"Not connected to {self.name}")

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        rgb_frame = aligned_frames.get_color_frame()
        d_frame = aligned_frames.get_depth_frame()

        rgb_img = np.asanyarray(rgb_frame.get_data())
        depth_img = np.asanyarray(d_frame.get_data())
        timestamp = time.time()

        return {"time": timestamp, "rgb": rgb_img, "depth": depth_img}

    def get_obs(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        sensor_state = self.get_sensors()

        rgb = np.array(sensor_state["rgb"])
        h, w = rgb.shape[0], rgb.shape[1]
        rgb = rgb[:, int((w - h) / 2) : int((w + h) / 2)]  # crop to square
        rgb = cv2.resize(rgb, self.image_size)
        rgb = rgb / 255.0 * 2 - 1  # map RGB to [-1, 1]
        rgb = einops.rearrange(rgb, "h w d -> d h w")[None, None, :, :, :]

        pcd = np.array(sensor_state["depth"]) / 1000
        pcd = pcd[:, int((w - h) / 2) : int((w + h) / 2)]  # crop to square
        pcd = cv2.resize(pcd, self.image_size)
        pcd[pcd > 2] = 2.0
        pcd = cv2.medianBlur(pcd.astype(np.float32), 5)
        pcd = cv2.medianBlur(pcd.astype(np.float32), 5)
        pcd = cv2.medianBlur(pcd.astype(np.float32), 5)
        pcd = deproject(pcd, *self.cam_info).transpose(1, 0)
        pcd = np.reshape(pcd, (*self.image_size, 3))
        pcd = einops.rearrange(pcd, "h w d -> d h w")[None, None, :, :, :]

        return (
            torch.tensor(rgb, dtype=torch.float32),
            torch.tensor(pcd, dtype=torch.float32),
        )

    def close(self):
        self.pipeline.stop()


if __name__ == "__main__":
    cam = Realsense(name="1")
    cam.connect()

    for i in range(100):
        img = cam.get_sensors()
        rgb = img["rgb"]
        depth = img["depth"]
        # rgb = cv2.resize(img["rgb"], (256, 256))
        # depth = cv2.resize(img["depth"], (256, 256))
        cv2.imshow("rgb", rgb)
        cv2.imshow("depth", depth)
        cv2.waitKey(1)
        time.sleep(0.1)

    cam.close()
