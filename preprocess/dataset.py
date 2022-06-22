import os

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ValenceArousalWithClassesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir,
                                self.img_labels.iloc[idx, 0])
        image = cv.imread(img_path)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        label = self.img_labels.iloc[idx, 1]
        label = np.array([label])
        if self.transform:
            image = self.transform(Image.fromarray(image))

        return image, label
