import logging
import time

import cv2
import einops
import numpy as np
import pykinect_azure as pykinect
import torch

from utils.utils_with_real import process_kinect


class Azure:
    def __init__(self, name: str, image_size, cam_info):
        self.name = name
        self.image_size = image_size
        self.cam_info = cam_info

        self.__set_device_configuration()  # sets self.device_config
        self.device = None

    def connect(self) -> bool:
        print("Connecting to {}: ".format(self.name))
        try:
            pykinect.initialize_libraries()
            self.device = pykinect.start_device(config=self.device_config)
            print("Connected.")
            return True
        except Exception as e:
            self.device = None
            print("Failed with exception: ", e)
            return False

    def get_sensors(self):
        if not self.device:
            raise Exception(f"Not connected to {self.name}")

        s_rgb = False
        s_depth = False
        s_pc = False
        while not s_rgb or not s_depth or not s_pc:
            capture = self.device.update(5000)
            s_rgb, rgb = capture.get_color_image()
            s_depth, depth = capture.get_transformed_depth_image()
            s_pc, pcd = capture.get_transformed_pointcloud()
            timestamp = time.time()

        rgb = np.resize(np.asanyarray(rgb), (1080, 1920, 4))[:, :, [2, 1, 0]]
        depth = np.resize(np.asanyarray(depth), (1080, 1920))
        pcd = np.resize(np.asanyarray(pcd, dtype=np.float64), (1080, 1920, 3))

        return {
            "time": timestamp,
            "rgb": rgb,
            "depth": depth,
            "pcd": pcd,
        }

    def get_obs(self):
        sensor_state = self.get_sensors()

        rgb = np.array(sensor_state["rgb"])
        depth = np.array(sensor_state["depth"])
        pcd = np.array(sensor_state["pcd"])

        rgb, pcd = process_kinect(rgb, pcd, self.image_size, self.cam_info, depth=depth)

        rgb = einops.rearrange(rgb, "h w d -> d h w")[None, None, :, :, :]
        pcd = einops.rearrange(pcd, "h w d -> d h w")[None, None, :, :, :]

        return (
            torch.tensor(rgb, dtype=torch.float32),
            torch.tensor(pcd, dtype=torch.float32),
        )

    def close(self):
        self.device.close()
        self.device = None

    def __set_device_configuration(self):
        self.device_config = pykinect.default_configuration
        self.device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED


if __name__ == "__main__":
    cam = Azure(device_id=0)
    cam.connect()

    for i in range(100):
        sensor = cam.get_sensors()
        rgb = sensor["rgb"]
        depth = sensor["depth"]
        depth = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("rgb", rgb)
        cv2.imshow("depth", depth)
        cv2.waitKey(1)
        time.sleep(0.1)

    cam.close()
