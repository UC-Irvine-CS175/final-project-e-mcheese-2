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


def generate_masks():
        


    # GETTING THE MASKS AND SAVING
    dataset = bps_dataset.BPSMouseDataset(csv_file, csv_dir, s3_client, bucket_name, is_watershed = True, file_on_prem = False,transform=transforms.Compose([NormalizeBPS(),ResizeBPS(256, 256),]))
    norm = NormalizeBPS()
    myWatershed = Watershed()
    if not os.path.exists("data\\masks"):
        os.makedirs("data\\masks")
    for i in range(len(dataset)):
        data = dataset[i][0]
        mask = myWatershed.get_mask(data)
        ## BECAREFUL FOR FILE PATH WHEN TESTING BETWEEN WINDOWS AND UNIX BASED SYSTEMS
        cv2.imwrite(f'data\\masks\\{i}.png', mask)

def generate_images():
    # GETTING THE IMAGES AND SAVING
    dataset = bps_dataset.BPSMouseDataset(csv_file, csv_dir, s3_client, bucket_name, is_watershed = False, file_on_prem = False,transform=transforms.Compose([NormalizeBPS(),ResizeBPS(256, 256),]))
    norm = NormalizeBPS()
    myWatershed = Watershed()
    if not os.path.exists("data\\images"):
        os.makedirs("data\\images")
    for i in range(len(dataset)):
        data = dataset[i][0]
        ## BECAREFUL FOR FILE PATH WHEN TESTING BETWEEN WINDOWS AND UNIX BASED SYSTEMS
        cv2.imwrite(f'data\\images\\{i}.png', data)
    

if __name__ == "__main__":
    csv_file = 'meta.csv'
    csv_dir = 'Microscopy/train'
    bucket_name = 'nasa-bps-training-data'
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    generate_images()
    generate_masks()
