import os
import sys

import numpy as np

from skimage import io
from skimage import transform
from tqdm import tqdm

import configs

# Image size config.
IMG_HEIGHT = configs.IMG_HEIGHT
IMG_WIDTH = configs.IMG_WIDTH
IMG_CHANNELS = configs.IMG_CHANNEL

DEBUG_MODE = configs.DEBUG_MODE
DEBUG_TRAIN_SIZE = 10
DEBUG_TEST_SIZE = 2

# Directory config.

# Train data dir with such structure:
# ├── 00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552
# │   ├── images
# │   │   └── 00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552.png
# │   └── masks
# │       ├── 07a9bf1d7594af2763c86e93f05d22c4d5181353c6d3ab30a345b908ffe5aadc.png
# │       ├── 0e548d0af63ab451616f082eb56bde13eb71f73dfda92a03fbe88ad42ebb4881.png
# │       ├── ...
# ├── 003cee89357d9fe13516167fd67b609a164651b21934585648c740d2c3d86dc1
# │...
# Contains no files other than from stage1_train.zip.
TRAIN_PATH = configs.TRAIN_PATH
# Test data dir with such structure:
# ├── 0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5
# │   └── images
# │       └── 0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5.png
# ├── 003cee89357d9fe13516167fd67b609a164651b21934585648c740d2c3d86dc1
# │...
# Contains no files other than from stage1_test.zip.
TEST_PATH = configs.TEST_PATH
IMAGES_DIR = 'images'
MASKS_DIR = 'masks'


def read_image(img_id, path_under_id):
    """Read, resize one image.

    Returns:
        (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8
    """
    img = io.imread(
        os.path.join(path_under_id, img_id, IMAGES_DIR,
                     img_id + '.png'))[:, :, :IMG_CHANNELS]
    img = transform.resize(
        img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    return img.astype(np.uint8)


def read_mask(img_id, path_under_id):
    """Read, resize, merge masks for one image.

    Returns:
        (IMG_HEIGHT, IMG_WIDTH), dtype=np.bool
    """
    masks_path = os.path.join(path_under_id, img_id, MASKS_DIR)
    # resize() produce np.float array.
    merged_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float)
    for mask_file in os.listdir(masks_path):
        mask_img = io.imread(os.path.join(masks_path, mask_file))
        mask_img = transform.resize(
            mask_img, (IMG_HEIGHT, IMG_WIDTH),
            mode='constant',
            preserve_range=True)
        merged_mask = np.maximum(merged_mask, mask_img)
    return merged_mask.astype(np.bool)


def read_train_data():
    """Read train images and masks.

    Return:
        images: (number of images, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            dtype=np.uint8
        masks: (number of images, IMG_HEIGHT, IMG_WIDTH), dtype=np.bool
    """
    img_ids = os.listdir(TRAIN_PATH)
    if DEBUG_MODE:
        img_num = DEBUG_TRAIN_SIZE
        img_ids = img_ids[:img_num]
    images = np.zeros(
        (img_num, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    masks = np.zeros((img_num, IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)
    sys.stdout.flush()
    for n, img_id in tqdm(enumerate(img_ids), total=img_num):
        images[n] = read_image(img_id, TRAIN_PATH)
        masks[n] = read_mask(img_id, TRAIN_PATH)
    return images, masks


def read_test_data():
    """Read test images."""
    img_ids = os.listdir(TEST_PATH)
    if DEBUG_MODE:
        img_num = DEBUG_TEST_SIZE
        img_ids = img_ids[:img_num]
    images = np.zeros(
        (img_num, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    for n, img_id in tqdm(enumerate(img_ids), total=img_num):
        images[n] = read_image(img_id, TEST_PATH)
    return images
