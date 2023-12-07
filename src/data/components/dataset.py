"""
All the __getitem__ functions will return a triplet
image, label, meta_data, with meta_data containing
paths to the non-transformed images or other necessary
information
"""

from typing import List, Optional

import numpy as np
import torch
from albumentations import Compose
from astrovision.data import SatelliteImage
from s3fs import S3FileSystem
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    Dataset class for segmentation.
    """

    def __init__(
        self,
        patchs: List,
        labels: List,
        n_bands: int,
        fs: S3FileSystem,
        transform: Optional[Compose] = None,
    ):
        """
        Constructor.
        """
        self.patchs = patchs
        self.labels = labels
        self.n_bands = n_bands
        self.fs = fs
        self.transform = transform

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        si = SatelliteImage.from_raster(
            file_path=f"s3://{self.patchs[idx]}", dep=None, date=None, n_bands=self.n_bands
        )

        label = torch.LongTensor(np.load(self.fs.open(f"s3://{self.labels[idx]}")))

        sample = self.transform(image=np.transpose(si.array, [1, 2, 0]), label=label)

        transformed_image = sample["image"]
        transformed_label = sample["label"]

        metadata = {"path_image": self.patchs[idx], "path_label": self.labels[idx]}

        return transformed_image, transformed_label, metadata

    def __len__(self):
        return len(self.patchs)
