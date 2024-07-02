import logging
import time

import cv2
import numpy as np
import pyrealsense2 as rs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Realsense:
    def __init__(self, name: str):
        self.name = name
        self.pipeline = None

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
