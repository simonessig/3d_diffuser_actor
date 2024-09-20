# from dephai-python examples

import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import depthai as dai
import numpy as np
import torch

from ..guides import Guide

__all__ = ["PersonGuide"]


class PersonGuide(Guide):
    def __init__(
        self,
        mult: float = 1,
        condition: Optional[Callable[[], bool]] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(mult, condition, device)
        nnPath = str(Path("models/mobilenet-ssd_openvino_2021.4_6shave.blob").resolve().absolute())

        # MobilenetSSD label texts
        self.labelMap = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        nn = self.pipeline.create(dai.node.MobileNetDetectionNetwork)
        xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        nnOut = self.pipeline.create(dai.node.XLinkOut)
        nnNetworkOut = self.pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")
        nnOut.setStreamName("nn")
        nnNetworkOut.setStreamName("nnNetwork")

        # Properties
        camRgb.setPreviewSize(300, 300)
        camRgb.setInterleaved(False)
        camRgb.setFps(40)
        # Define a neural network that will make predictions based on the source frames
        nn.setConfidenceThreshold(0.5)
        nn.setBlobPath(nnPath)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)

        # Linking
        camRgb.preview.link(xoutRgb.input)

        camRgb.preview.link(nn.input)
        nn.out.link(nnOut.input)
        nn.outNetwork.link(nnNetworkOut.input)

        self.detections = []

        self.device = dai.Device(self.pipeline)
        self._get_detection()

    def _get_score(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.zeros_like(x)
        s[:, 1] = self._eval_detection(x[:, 1])
        # print(m.shape)
        return s
        # return self._coord_to_mask(x[:, :3])

    def _get_detection(self):
        # frame = None
        for i in range(10):
            qRgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            qDet = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            # frame = None
            # detections = []

            # inRgb = qRgb.tryGet()
            # inDet = qDet.tryGet()
            # inRgb = qRgb.get()
            qRgb.get()
            inDet = qDet.tryGet()

            # if inRgb is not None:
            #     frame = inRgb.getCvFrame()

            if inDet is not None:
                self.detections = inDet.detections
                print(self.detections)
                break

        # print(y)
        # print(detections)
        # if frame is not None:
        #     self.displayFrame("rgb", frame, detections)
        #     cv2.waitKey(0)

        # for d in detections:
        #     if self.labelMap[d.label] == "person":
        #         ones = torch.ones(y.shape, dtype=torch.float32, device=y.device)
        #         zeros = torch.zeros(y.shape, dtype=torch.float32, device=y.device)
        #         # print(y < 0)
        #         if d.xmax - d.xmin < 0.5:
        #             return torch.where(y >= 0, ones, zeros)
        #         else:
        #             return torch.where(y <= 0, ones, zeros)

        # return torch.zeros(y.shape, dtype=torch.float32, device=y.device)
        # if np.abs(0.5 - d.xmin) > np.abs(0.5 - d.xmax):
        #     pass  # 0 -> np.max(d.xmax, 0.5)
        # else:
        #     pass  # np.min(d.xmin, 0.5) -> 1

    def _eval_detection(self, y):
        for d in self.detections:
            if self.labelMap[d.label] == "person":
                ones = torch.ones(y.shape, dtype=torch.float32, device=y.device)
                zeros = torch.zeros(y.shape, dtype=torch.float32, device=y.device)
                if d.xmax + d.xmin < 1:
                    return torch.where(y >= 0, -ones, zeros)
                else:
                    return torch.where(y <= 0, ones, zeros)

        return torch.zeros(y.shape, dtype=torch.float32, device=y.device)

    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(self, name, frame, detections):
        color = (255, 0, 0)
        for detection in detections:
            bbox = self.frameNorm(
                frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            )
            cv2.putText(
                frame,
                self.labelMap[detection.label],
                (bbox[0] + 10, bbox[1] + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                color,
            )
            cv2.putText(
                frame,
                f"{int(detection.confidence * 100)}%",
                (bbox[0] + 10, bbox[1] + 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                color,
            )
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)
