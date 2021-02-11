import cv2
import torch
from torchvision import transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import pandas as pd
import os

class Dataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = pd.read_csv(csv_file, sep=';')
        self.root_dir = root_dir
        self.transform = transform
        self.TransformToTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[idx, 0])
        image = cv2.imread(img_name)
        pil_img = Image.fromarray(image)
        if self.transform:
            pil_img = self.transform(pil_img)
        image = np.asarray(pil_img).astype(np.float32)
        image = image - 127.5
        image = image * 0.00078125
        image = self.TransformToTensor(image)
        label = self.frame.iloc[idx, 1]

        return image, label
#2 6 11 12 13