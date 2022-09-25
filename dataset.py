"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

def pad_to_square(image, pad_value):
    c, h, w = image.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    image = F.pad(image, pad, "constant", value=pad_value)

    return image, pad
def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

class YOLODataset(Dataset):
    def __init__(
        self,
        txt_file,
        image_dir,
        transform=True,
    ):
        self.label_paths = []
        f = open(txt_file)
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace("images", "labels").replace("jpg", "txt")
            self.label_paths.append(line)
        self.image_dir = image_dir
        self.transform = transform
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        # print("label_path:{}".format(label_path))
        bboxes = torch.from_numpy(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)) # cls x y w h 
        image_path = os.path.join(self.image_dir, label_path.split("/")[-1].replace("txt", "jpg"))
        image = transforms.ToTensor()(Image.open(image_path).convert("RGB"))
        _, h, w = image.shape
        h_factor, w_factor = (h, w)
        # Pad to square resolution
        image, pad = pad_to_square(image, 0)
        _, padded_h, padded_w = image.shape
        image = resize(image, config.IMAGE_SIZE)
        
        targets = torch.zeros((len(bboxes), 6)) # ? cls cx, cy, w h 
        # cls x y w h 
        # Extract coordinates for unpadded + unscaled image
        try:
            x1 = w_factor * (bboxes[:, 1] - bboxes[:, 3] / 2)
            y1 = h_factor * (bboxes[:, 2] - bboxes[:, 4] / 2)
            x2 = w_factor * (bboxes[:, 1] + bboxes[:, 3] / 2)
            y2 = h_factor * (bboxes[:, 2] + bboxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # boxes[:, 0] => class id
            # Returns (xc, yc, w, h)
            bboxes[:, 1] = ((x1 + x2) / 2) / padded_w        # newer center x
            bboxes[:, 2] = ((y1 + y2) / 2) / padded_h        # newer center y
            bboxes[:, 3] *= w_factor / padded_w              # newer width
            bboxes[:, 4] *= h_factor / padded_h              # newer height
            
            targets[:, 1:] = bboxes

            # Apply augmentations
            if self.transform:
                if np.random.random() < 0.5:
                    image, targets = horisontal_flip(image, targets)
        except Exception as ex:
            print("\n exception:{}".format(ex))
            print("bboxes:{}".format(bboxes))
            print("label_path:{}".format(label_path))


        return image_path, image, targets # target: nbboxes 6
    def collate_fn(self, batch):
        paths, images, targets = list(zip(*batch))
        # targets: b nb_bboxes (image_idx_in_batch cls cx, cy, w h )
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None] # boxes: nb_bboxes (image_idx_in_batch cls cx, cy, w h )
        # Add sample index to targets
        for i, boxes in enumerate(targets): # boxes: nb_bboxes (image_idx_in_batch cls cx, cy, w h )
            boxes[:, 0] = i
        targets = torch.cat(targets, 0) # nb_boxes_this_batch (image_idx_in_batch cls cx, cy, w h )
        images = torch.stack(images)
        return paths, images, targets

