import os, sys
import numpy as np
import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


class LeRobotNILSDataset(LeRobotDataset):
    def __init__(self, data_id, image_key='image', sampling_rate=1, name="lerobot_dataset", **kwargs):
        super().__init__(data_id, episodes=[0, 10, 11, 23])

        self.paths = []
        self.frame_names = []
        self.trajectories = []

        self.sampling_rate = sampling_rate

        for i, episode_idx in enumerate(self.episodes):
            self.paths.append(self.meta.get_data_file_path(episode_idx))

            start_idx = self.episode_data_index['from'][i].item()
            end_idx = self.episode_data_index['to'][i].item()
            frames = []
            frame_names = []

            for t in range(start_idx, end_idx, self.sampling_rate):
                image = super().__getitem__(t)[f'observation.images.{image_key}'] * 255.0
                image = image.numpy().astype(np.uint8)
                frames.append(image)
                frame_names.append(f"episode_{episode_idx}_frame_{t}.jpg")
                frames.append(image)
                frame_names.append(f"episode_{episode_idx}_frame_{t}.jpg")
            self.trajectories.append(
                frames
            )
            self.frame_names.append(frame_names)

        self.name = name

    def __getitem__(self, idx):        
        frames = self.trajectories[idx]
        
        data = {}
        data["rgb_static"] = frames
        data["path"] = np.array([self.paths[idx]] * len(frames))
        data["frame_names"] = np.array(self.frame_names[idx])

        return data

    def __len__(self):
        return len(self.episodes)