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
        transforms: Optional[Compose] = None,
    ):
        """
        Constructor.
        """
        return

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
            file_path=self.patchs[idx], dep=None, date=None, n_bands=self.n_bands
        )

        # TODO: Le transpose sera intégré dans asrovision donc à virer
        si.array = np.transpose(si.array, [1, 2, 0])
        img = si.array

        label = torch.LongTensor(np.load(self.labels[idx]))

        if self.transforms:
            sample = self.transforms(image=img, label=label)
            img = sample["image"]
            label = sample["label"]

        metadata = {"path_image": self.patchs[idx], "path_label": self.labels[idx]}

        return img, label, metadata

    def __len__(self):
        return len(self.list_paths_images)
