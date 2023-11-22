import cv2
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


# /path/to/REDS/
#         │
#         ├── train/
#         │   ├── 720/
#         │   │   ├── 000/
#         │   │   ├── ...
#         │   │   └── 239/
#         │   │        ├── 00000000.png
#                      ├── ...
#                      └── 00000099.png
# to
# /path/to/REDS/
#         │
#         ├── train/
#         │   ├── 360/
#         ...


def downscale(path, degradations=['360', '180'], num_threads=8):
    # path to reds
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')

    train_720 = os.path.join(train_path, '720')
    val_720 = os.path.join(val_path, '720')

    for degradation in degradations:
        train_degradation = os.path.join(train_path, degradation)
        if not os.path.exists(train_degradation):
            os.makedirs(train_degradation)
        val_degradation = os.path.join(val_path, degradation)
        if not os.path.exists(val_degradation):
            os.makedirs(val_degradation)
        folderlist = sorted(os.listdir(train_720))

        for folder in folderlist:
            train_folder = os.path.join(train_720, folder)
            train_degradation_folder = os.path.join(train_degradation, folder)
            if not os.path.exists(train_degradation_folder):
                os.makedirs(train_degradation_folder)
            val_folder = os.path.join(val_720, folder)
            val_degradation_folder = os.path.join(val_degradation, folder)
            if not os.path.exists(val_degradation_folder):
                os.makedirs(val_degradation_folder)
            train_filelist = sorted(os.listdir(train_folder))
            val_filelist = sorted(os.listdir(val_folder))
            train_filelist = [os.path.join(train_folder, file) for file in train_filelist]
            val_filelist = [os.path.join(val_folder, file) for file in val_filelist]
            train_degradation_filelist = [os.path.join(train_degradation_folder, file) for file in train_filelist]
            val_degradation_filelist = [os.path.join(val_degradation_folder, file) for file in val_filelist]
            with Pool(num_threads) as p:
                list(tqdm(p.imap(downscale_image, zip(train_filelist, train_degradation_filelist)), total=len(train_filelist)))
                list(tqdm(p.imap(downscale_image, zip(val_filelist, val_degradation_filelist)), total=len(val_filelist)))

def downscale_image(filelist):
    file, degradation_file = filelist
    img = cv2.imread(file)
    img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(degradation_file, img)

