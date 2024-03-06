from torch.utils.data import Dataset
#import numpy as np
#import io
from PIL import Image
#import os
#from os.path import join
#import json
#import torch
#import numpy as np
# import random
# import data.distortion as distortion
# import torch.nn.functional as F
# import cv2
# import torch.distributed as dist


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class McDataset(Dataset):
    """
    json_path:'DATASET/paths/'
    real_name:ff_real
    fake_name:ff_fake

    """

    def __init__(self, transform=None):
        self.transform = transform

        self.fake_txt_path = "/home/lizg/lizg/SelfBlendedImages/data/CelebAHQ_HFGI_smile.txt"
        self.real_txt_path = "/home/lizg/lizg/SelfBlendedImages/data/CelebAHQ_WM.txt"
        imgs = []
        fh1 = open(self.fake_txt_path, "r")
        fh2 = open(self.real_txt_path, "r")
        for line1 in fh1:
            line1 = line1.rstrip()
            imgs.append(line1)
        fh1.close()
        for line2 in fh2:
            line2 = line2.rstrip()
            imgs.append(line2)
        fh2.close()
        self.imgs = imgs



    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if index < 3000:  ### fake
            label = 1
            path = self.imgs[index]

        else:  ### real
            label = 0
            path = self.imgs[index]

        img = Image.open(path)
        img = self.transform(img)

        return img, label


