from torch.utils.data import Dataset
from glob import glob
import torch
import numpy as np
import traceback
import os
import cv2


class SampleLoader(Dataset):
    def __init__(self, mode, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.input = []

    def __len__(self):
        return len(self.input)

    def __getitem__(self, key):
            return {"image": 1,
                    "label": 1}

    def get_loader(self, shuffle=True):

        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


if __name__ == "__main__":
    dataloader = SampleLoader('train', 16, 0)
    print(dataloader[11])
