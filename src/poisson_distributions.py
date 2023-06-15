import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np
import cv2
import pickle

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from torchvision import transforms

from dataset.bps_dataset import BPSMouseDataset
from dataset.augmentation import NormalizeWatershed, ResizeBPS
from data_utils import export_subset, save_tiffs_local_from_s3
from src.models.watershed import Watershed
from count_foci import foci_counts

from scipy.stats import poisson



### Data Helper Functions ###
def prepare_subset_data(meta_csv_path, subset_csv_dir):
    csv_filenames = []
    for particle_type in ['X-ray', 'Fe']:
        for dose in ['low', 'med', 'hi']:
            for hr in [4, 24, 48]:
                csv_filename, _ = export_subset(dose, particle_type, hr, meta_csv_path, subset_csv_dir)
                csv_filenames.append(csv_filename)

    bucket_name = 'nasa-bps-training-data'
    s3_path = 'Microscopy/train'
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    for fname in csv_filenames:
        save_tiffs_local_from_s3(
            s3_client, 
            bucket_name,
            s3_path,
            fname,
            "../data/raw"
        )
    return csv_filenames

def pickle_masks(local_csv_path, local_data_dir, local_masks_path, watershed, transform = None):
    df = pd.read_csv(local_csv_path, header=0)
    img_paths = [f"{local_data_dir}/{fname}" for fname in df["filename"]]
    #print(f"   # of images in set: {len(img_paths)}")
    imgs = [cv2.imread(img_path, cv2.IMREAD_COLOR) for img_path in img_paths]
    #print("   augmenting")
    if transform:
        imgs = [transform(img) for img in imgs]
    #print("   masking")
    masks = [watershed.get_mask(img) for img in imgs]
    #print(f"   # of masks: {len(masks)}")
    pickle_file = open(local_masks_path, "wb")
    pickle.dump(masks, pickle_file)

def get_saved_masks(mask_dir, particle_type, dose, hr):
    mask_path = os.path.join(mask_dir, f"masks_dose_{dose}_type_{particle_type}_hr_{hr}_post_exposure.p")
    pickle_file = open(mask_path, "rb")
    return pickle.load(pickle_file)

### Plotting Helper Functions ###
def style_plot(dose, particle_type, hr, mu):
    plt.title(f"{particle_type}, {dose} dose {hr} hrs post exposure\nPoisson distribution mean: \u03BB = {np.round(mu, 4)}")
    plt.xlabel("x = number of foci")
    plt.ylabel("y = probability of an image from this subset having x foci")
    plt.legend()

def plot_num_foci(counts):
    weights = np.ones_like(counts) / len(counts)
    plt.hist(counts, weights = weights, range = (0,20), bins = 20, label="Real Data")

def plot_poisson_dist(counts):
    if len(counts) == 0: return 0
    x = np.arange(0, 20, 1)
    mu = np.mean(counts)
    y = poisson.pmf(x, mu=mu)
    plt.plot(x, y, label="Estimated Poisson Dist.")
    return mu

### Plot Function ###
def visualize_subset(dose, particle_type, hr_post_exposure, mask_dir):
    masks = get_saved_masks(mask_dir, particle_type, dose, hr_post_exposure)
    counts = foci_counts(masks)
    plot_num_foci(counts)
    mu = plot_poisson_dist(counts)
    style_plot(dose, particle_type, hr_post_exposure, mu)
    plt.show()


def main():
    # To run the code below:
    # 1. save the meta.csv file locally using data_utils
    # 2. use prepare_subset_data() to create csv files and download images for each data subset
    # 3. use pickle_masks() to create a pickled array of watershed masks for each data subset
    # 4. change the directory "../data/masks" below to the directory you chose to save the pickled masks in

    for particle_type in ['X-Ray', 'Fe']:
        for dose in ['low', 'med', 'hi']:
            for hr in [4, 24, 48]:
                visualize_subset(dose, particle_type, hr, "../data/masks/")

if __name__ == '__main__':
    main()