import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
# from utils import seed_everything

DATASET = 'yolov3/PyTorch-YOLOv3-code/data/custom'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 8
IMAGE_SIZE = 416
NUM_CLASSES = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False 
SAVE_MODEL = True
CHECKPOINT_DIR = "YOLOv3/checkpoint"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"
TRAIN_TXT_PTH = "yolov3/PyTorch-YOLOv3-code/data/custom/train.txt"
TEST_TXT_PTH = "yolov3/PyTorch-YOLOv3-code/data/custom/test.txt"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]


# scale = 1.1
train_transforms = A.Compose(
    [
        A.augmentations.geometric.resize.Resize (int(IMAGE_SIZE), int(IMAGE_SIZE), interpolation=1, always_apply=False, p=1),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.IAAAffine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = A.Compose(
    [
        A.augmentations.geometric.resize.Resize (int(IMAGE_SIZE), int(IMAGE_SIZE), interpolation=1, always_apply=False, p=1),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)
MASK_CLASSES = ["face", "mask"]

