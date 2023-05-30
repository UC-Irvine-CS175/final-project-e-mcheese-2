import sys
sys.path.append(".")
from torchvision import transforms, utils
import src.dataset.bps_dataset as bps_dataset
import boto3
import cv2
from botocore.config import Config
from botocore import UNSIGNED
import numpy as np
from matplotlib import pyplot as plt
from src.dataset.augmentation import NormalizeBPS, ResizeBPS
from src.models.watershed import Watershed
import os
import pandas as pd
from src.data_utils import get_bytesio_from_s3



def _generate_masks():
    # GETTING THE MASKS AND SAVING
    dataset = bps_dataset.BPSMouseDataset(csv_file, csv_dir, s3_client, bucket_name, is_watershed = True, file_on_prem = False, transform=transforms.Compose([NormalizeBPS(), ResizeBPS(256, 256)]))
    myWatershed = Watershed()

    for i in range(len(dataset)):
        data = dataset[i][0]
        mask = myWatershed.get_mask(data)
        ## BECAREFUL FOR FILE PATH WHEN TESTING BETWEEN WINDOWS AND UNIX BASED SYSTEMS
        mask_path = os.path.join(mask_dir, str(i) + '.txt')
        np.savetxt(mask_path, mask)

def _generate_images():
    # GETTING THE IMAGES AND SAVING
    dataset = bps_dataset.BPSMouseDataset(csv_file, csv_dir, s3_client, bucket_name, is_watershed = False, file_on_prem = False)

    for i in range(len(dataset)):
        data = dataset[i][0]
        ## BECAREFUL FOR FILE PATH WHEN TESTING BETWEEN WINDOWS AND UNIX BASED SYSTEMS
        image_path = os.path.join(image_dir, str(i) + '.tif')
        cv2.imwrite(image_path, data)
    
def _rename_files():
    path = os.path.join(csv_dir, csv_file)
    path = path.replace("\\", "/")
    df = pd.read_csv(get_bytesio_from_s3(s3_client, bucket_name, path))

    for i in range(len(df)):
        file_name = df.iloc[i]["filename"]
        img_src = os.path.join(image_dir, f'{i}.tif')
        img_dst = os.path.join(image_dir, f'{file_name}')
        os.rename(img_src, img_dst)

        mask_src = os.path.join(mask_dir, f'{i}.txt')
        mask_dst = os.path.join(mask_dir, f'{file_name[:-4]}.txt')
        os.rename(mask_src, mask_dst)

def generate_all_data():
    _generate_images()
    _generate_masks()
    _rename_files()


if __name__ == "__main__":
    csv_file = 'meta.csv'
    csv_dir = 'Microscopy/train'
    bucket_name = 'nasa-bps-training-data'
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    image_dir = ("data\\raw")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    mask_dir = ("data\\masks")
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # This only needs to be run once to download all the raw/mask data
    generate_all_data()
