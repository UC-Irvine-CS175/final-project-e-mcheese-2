"""
This module contains the BPSMouseDataset class which is a subclass of torch.utils.data.Dataset.
"""

from src.dataset.augmentation import (
    NormalizeBPS,
    ResizeBPS,
    VFlipBPS,
    HFlipBPS,
    RotateBPS,
    RandomCropBPS,
    ToTensor
)

import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from torchvision import transforms, utils

import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
from matplotlib import pyplot as plt
from pyprojroot import here

root = here()

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.errorfactory import ClientError
import io
from io import BytesIO

import sys
sys.path.append(str(here()))

from src.data_utils import get_bytesio_from_s3
from src.models.watershed import Watershed


""" 
Note about Tensor:
        PyTorch expects tensors to have the following dimensions:
        (batch_size, channels, height, width)
        A numpy array has the following dimensions:
        (height, width, channels)
"""


class BPSMouseDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset class for the BPS microscopy data.

    args:
        meta_csv_file (str): name of the metadata csv file
        meta_root_dir (str): path to the metadata csv file
        bucket_name (str): name of bucket from AWS open source registry.
        transform (callable, optional): Optional transform to be applied on a sample.

    attributes:
        meta_df (pd.DataFrame): dataframe containing the metadata
        bucket_name (str): name of bucket from AWS open source registry.
        train_df (pd.DataFrame): dataframe containing the metadata for the training set
        test_df (pd.DataFrame): dataframe containing the metadata for the test set
        transform (callable): The transform to be applied on a sample.

    raises:
        ValueError: if the metadata csv file does not exist
    """

    def __init__(
            self,
            meta_csv_file:str,
            meta_root_dir:str,
            s3_client: boto3.client = None,
            bucket_name: str = None,
            transform=None,
            file_on_prem:bool = True,
            get_masks:bool = True):
        
        self.meta_root_dir = meta_root_dir
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.transform = transform
        self.file_on_prem = file_on_prem
        self.get_masks = get_masks
        self.watershed = Watershed()

        # formulate the full path to metadata csv file
        # if the file is not on the local file system, use the get_bytesio_from_s3 function
        # to fetch the file as a BytesIO object, else read the file from the local file system.
        
        path = os.path.join(meta_root_dir, meta_csv_file)
        full_path = os.path.normpath(path)

        if file_on_prem:
            if os.path.isfile(full_path):
                self.full_path = full_path
                self.meta_df = pd.read_csv(full_path)
            else: 
                raise ValueError("The file specified was not found on the local file system")
        else:
            if s3_client and bucket_name:
                try: 
                    full_path = full_path.replace("\\", "/")
                    s3_client.head_object(Bucket = bucket_name, Key = full_path)
                    self.meta_df = pd.read_csv(get_bytesio_from_s3(s3_client, bucket_name, full_path))

                except ClientError as e:
                    raise ValueError("The file specified does not exist")
            else:
                raise ValueError("S3 client and bucket name were not given") 

        self.meta_df = pd.get_dummies(self.meta_df, columns=["particle_type"])     
            

    def __len__(self):
        """
        Returns the number of images in the dataset.

        returns:
          len (int): number of images in the dataset
        """
        return len(self.meta_df)

    def __getitem__(self, idx):
        """
        Fetches the image and corresponding label for a given index.

        Args:
            idx (int): index of the image to fetch

        Returns:
            img_tensor (torch.Tensor): tensor of image data
            label (int): label of image
        """

        # get the bps image file name from the metadata dataframe at the given index
        row = self.meta_df.iloc[idx]
        file_name = row["filename"]


        # Fetch one hot encoded labels for all classes of particle_type as a Series
        particle_type_tensor = row[['particle_type_Fe', 'particle_type_X-ray']]
        # Convert Series to numpy array
        particle_type_tensor = particle_type_tensor.to_numpy().astype(np.bool_)
        
        # Convert One Hot Encoded labels to tensor
        particle_type_tensor = torch.from_numpy(particle_type_tensor)
        # Convert tensor data type to Float
        particle_type_tensor = particle_type_tensor.type(torch.FloatTensor)

        # formulate path to image/mask given the root directory (note meta.csv is in the
        # same directory as the images)
        file_path = os.path.join(self.meta_root_dir, file_name)
        file_path = os.path.normpath(file_path)
        

        # If on_prem is False, then fetch the image from s3 bucket using the get_bytesio_from_s3
        # function, get the contents of the buffer returned, and convert it to a  numpy array
        # with datatype unsigned 16 bit integer used to represent microscopy images.
        # If on_prem is True load the image from local. 
        
        im_data = None
        cv2_flag = cv2.IMREAD_ANYDEPTH

        if self.get_masks:
            cv2_flag =cv2.IMREAD_COLOR

        if self.file_on_prem:
            if self.get_masks:
                file_path = file_path[:-4] + '.txt'
                mask = np.loadtxt(file_path)
                return mask, particle_type_tensor
            
            im_data = cv2.imread(file_path, cv2_flag)
        else:
            im_bytesio = get_bytesio_from_s3(self.s3_client, self.bucket_name, file_path)

            im_bytes = np.asarray(bytearray(im_bytesio.read()))
            im_data = cv2.imdecode(np.frombuffer(im_bytes, dtype = np.uint8), cv2_flag)

        # apply tranformation if available
        image = im_data
        if self.transform:
            image = self.transform(im_data)

        # return the image and one hot encoded tensor
        if not self.get_masks:
            return image, particle_type_tensor
        
        # return the mask and one hot encoded tensor
        mask = self.watershed.get_mask(image)
        return mask, particle_type_tensor


def show_label_batch(image: torch.Tensor, label: str):
    """Show image with label for a batch of samples."""
    images_batch, label_batch = \
            image, label
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    # grid is a 4 dimensional tensor (channels, height, width, number of images/batch)
    # images are 3 dimensional tensor (channels, height, width), where channels is 1
    # utils.make_grid() takes a 4 dimensional tensor as input and returns a 3 dimensional tensor
    # the returned tensor has the dimensions (channels, height, width), where channels is 3
    # the returned tensor represents a grid of images
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title(f"Label: {label}")
    plt.savefig('test_grid_1_batch.png')



def main():
    """main function to test PyTorch Dataset class (Make sure the directory structure points to where the data is stored)"""
    bucket_name = "nasa-bps-training-data"
    s3_path = "Microscopy/train"
    s3_meta_csv_path = f"{s3_path}/meta.csv"

    #### testing get file functions from s3 ####

    local_file_path = "../data/raw"
    local_train_csv_path = "../data/processed/meta_dose_hi_hr_4_post_exposure_train.csv"

    print(root)


    #### testing dataset class ####
    train_csv_path = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    training_bps = BPSMouseDatasetLocal(train_csv_path, '../data/processed', transform=None, file_on_prem=True)
    print(training_bps.__len__())
    print(training_bps.__getitem__(0))

    transformed_dataset = BPSMouseDataset(train_csv_path,
                                           '../data/processed',
                                           transform=transforms.Compose([
                                               NormalizeBPS(),
                                               ResizeBPS(224, 224),
                                               VFlipBPS(),
                                               HFlipBPS(),
                                               RotateBPS(90),
                                               RandomCropBPS(200, 200),
                                               ToTensor()
                                            ]),
                                            file_on_prem=True
                                           )

    # Use Dataloader to package data for batching, shuffling, 
    # and loading in parallel using multiprocessing workers
    # Packaging is image, label
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=2)

    for batch, (image, label) in enumerate(dataloader):
        print(batch, image, label)

        if batch == 5:
            show_label_batch(image, label)
            print(image.shape)
            break

if __name__ == "__main__":
    main()
#The PyTorch Dataset class is an abstract class that is used to provide an interface for accessing all the samples
# in your dataset. It inherits from the PyTorch torch.utils.data.Dataset class and overrides two methods:
# __len__ and __getitem__. The __len__ method returns the number of samples in the dataset and the __getitem__

