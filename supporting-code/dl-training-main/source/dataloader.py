import csv
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
class MyDataset(Dataset):
    """Create an image and mask dataset based on a list."""
    """Dataset created is for Lung Ultrasounds"""

    def __init__(self, img_dir, mask_dir, transform=None, holdout_list=None, holdout_set=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        try:
            holdout_list_file = open(holdout_list)
            self.holdout_list = holdout_list_file.read().splitlines()
        except:
            self.holdout_list = []


        if not holdout_set:
            self.images = [
                os.path.join(img_dir, patient, video, image)
                    for patient in os.listdir(img_dir)
                        if patient not in self.holdout_list
                            for video in os.listdir(os.path.join(img_dir, patient))
                                for image in os.listdir(os.path.join(img_dir, patient, video))
                ]
        else:
            self.images = [
                os.path.join(img_dir, patient, video, image)
                    for patient in os.listdir(img_dir)
                        if patient in self.holdout_list
                            for video in os.listdir(os.path.join(img_dir, patient))
                                for image in os.listdir(os.path.join(img_dir, patient, video))
                ]

        self.classes = {
            # all grayscale values
            # background, class 0
            0: 0,
            1: 0,
            # liver - merged into background class
            117: 0,
            # spleen - merged into background class
            74: 0,
            # ribs, class 1
            76: 1,
            77: 1,
            # pleura, class 2
            105: 2,
            106: 2,
            # a-lines, class 3
            179: 3,
            # confluent b-lines, class 4
            135: 4,
            136: 4,
            # consolidations, class 5
            150: 5,
            53: 5,
            # effusion, class 6
            29: 6,
            30: 6,
            59: 6,
            60: 6,
        }
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        mask_path = self.images[idx].replace("images", "masks")
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)

        for grayValue in self.classes:
            mask[mask == grayValue] = self.classes[grayValue]

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
