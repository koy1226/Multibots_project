"""
 Copyright (c) 2019-2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "common/python"))
from images_capture import open_images_capture


class MulticamCapture:
    def __init__(self, serial_numbers):
        self.pipeline = rs.pipeline()
        self.configs = []
        self.fps = []

        for serial_number in serial_numbers:
            config = rs.config()
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.configs.append(config)

        # Configure the pipeline with both camera configurations
        self.pipeline.start()
        for config in self.configs:
            self.pipeline.stop()
            self.pipeline.start(config)
            profile = self.pipeline.get_active_profile()
            video_stream = profile.get_stream(rs.stream.color)
            self.fps.append(video_stream.as_video_stream_profile().fps())

    def get_frames(self):
        frames = []
        for _ in self.configs:
            frameset = self.pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame()
            if color_frame:
                frame = np.asanyarray(color_frame.get_data())
                frames.append(frame)

        return len(frames) == len(self.configs), frames

    def get_num_sources(self):
        return len(self.configs)

    def get_fps(self):
        return self.fps


class NormalizerCLAHE:
    def __init__(self, clip_limit=0.5, tile_size=16):
        self.clahe = cv.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)
        )

    def __call__(self, frame):
        for i in range(frame.shape[2]):
            frame[:, :, i] = self.clahe.apply(frame[:, :, i])
        return frame
