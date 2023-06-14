import os
import random 

import torch
from torchvision import transforms

import boto3
from botocore import UNSIGNED
from botocore.config import Config

import matplotlib.pyplot as plt
import cv2
import numpy as np

import pyprojroot
import sys
sys.path.append(str(pyprojroot.here()))

from src.data_utils import get_file_from_s3, save_tiffs_local_from_s3, train_test_split_subset_meta_dose_hr
from src.dataset.bps_dataset import BPSMouseDataset
from src.dataset.bps_datamodule import BPSDataModule
from src.dataset.augmentation import NormalizeWatershed, ResizeBPS
from src.models.watershed import Watershed
from src.models.unsupervised.autoencoder_v1 import LitAutoEncoder, train_autoencoder, Encoder, Decoder
from src.models.unsupervised.pca_tsne import preprocess_images, perform_pca, perform_tsne, create_tsne_cp_df
from src.vis_utils import plot_2D_scatter_plot, plot_3D_scatter_plot


def initial_download():
    '''## Function to download meta csv files and all corresponding data'''

    data_dir = r"./data"
    processed_dir = r"./data/processed"

    bucket_name = 'nasa-bps-training-data'
    s3_path = 'Microscopy/train'
    s3_meta_csv_path = f'{s3_path}/meta.csv'
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    file_path = os.path.join(processed_dir, "meta.csv")
    file_path = os.path.normpath(file_path)
    if not os.path.isfile(file_path):
        get_file_from_s3(
            s3_client=s3_client,
            bucket_name=bucket_name,
            s3_file_path=s3_meta_csv_path,
            local_file_path=processed_dir)
    
    save_tiffs_local_from_s3(
        s3_client=s3_client,
        bucket_name=bucket_name,
        s3_path=s3_path,
        local_fnames_meta_path=file_path,
        save_file_path=processed_dir)
    
    train_test_split_subset_meta_dose_hr(
        subset_meta_dose_hr_csv_path=file_path,
        test_size=0.2,
        out_dir_csv=processed_dir,
        random_state=42,
        stratify_col="particle_type")
    

def generate_watershed(csv_dir = None, csv_file = None, file_on_prem: bool = True, save_local = None):
    '''Function to generate watershed masks given a csv file
    file_on_prem: True if the raw images are on the local machine,
    save_local: Output directory to save watershed masks to, None to not save anything'''
    
    if not csv_dir or not csv_file:
        return

    bucket_name = 'nasa-bps-training-data'
    s3_path = 'Microscopy/train'
    s3_client = None
    if not file_on_prem: 
        s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    dataset = BPSMouseDataset(csv_file, csv_dir, s3_client, bucket_name, \
                              transform = transforms.Compose([ResizeBPS(256, 256)]), 
                              file_on_prem = file_on_prem, get_masks = True)
    
    if save_local and not os.path.exists(save_local):
        os.makedirs(save_local)

    for mask, file_name in dataset:
        if not save_local:
            plt.imshow(mask)
            plt.show()
        else:
            save_path = os.path.join(save_local, file_name[:-4] + '.png')
            print(save_path)
            plt.imsave(save_path, mask)


def generate_autoencoder_output(model_weights = None, csv_dir = None, csv_file = None, save_local = None):
    '''This function loads a trained autoencoder model and generates output
    images of the raw images in the given csv file
    model_weights: The path to the trained model
    csv_dir: The directory that contains the target csv file
    csv_file: The name of the csv file contained in csv_dir with the target images
    save_local: Output directory to save watershed masks to, None to not save anything'''

    bps_dm = BPSDataModule(None, None, 
                           val_csv_file=csv_file, 
                           val_dir=csv_dir, 
                           resize_dims=(256, 256), 
                           batch_size=1, 
                           convertToFloat=True
                           )
    
    # Set up the test dataloader
    bps_dm.setup(stage='validate')
    validate = bps_dm.val_dataloader()

    if save_local and not os.path.exists(save_local):
        os.makedirs(save_local)

    model = LitAutoEncoder.load_from_checkpoint(model_weights)
    for (x, _, file_name) in validate:    
        image = x.cuda()
        y_hat = model(image)

        output = np.array(y_hat.cpu().data[0, :, :, :].permute(1, 2, 0))
        if not save_local:
            plt.imshow(output)
            plt.show()
        else:
            output_path = os.path.join(save_local, file_name[0])
            cv2.imwrite(output_path, output)


def generate_tsne_plots(csv_dir = None, csv_file = None, file_name = None):
    '''This function generates the tsne outputs based on the the csv file in the given csv directory
    Make sure the files you are trying to evaluate are in the directory, as this will only evaluate
    the files in the csv that are in this directory.
    csv_dir: The directory that contains the target csv file
    csv_file: The name of the csv file contained in csv_dir with the target images
    file_name: What the output plots will be named '''

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    bps_datamodule = BPSDataModule(None, None, 
                                   val_csv_file=csv_file,
                                   val_dir=csv_dir,
                                   resize_dims=(256, 256),
                                   batch_size=2,
                                   num_workers=1)

    # Setup BPSDataModule which will instantiate the BPSMouseDataset objects
    # to be used for training and validation depending on the stage ('train' or 'val')
    bps_datamodule.setup(stage='validate')

    image_stream_1d, all_labels = preprocess_images(lt_datamodule=bps_datamodule.val_dataloader)

    # Project the flattened images onto the principal components
    IMAGE_SHAPE = (64, 64)
    N_ROWS = 5
    N_COLS = 7
    N_COMPONENTS = N_ROWS * N_COLS

    # Perform PCA on the flattened images and specify the number of components to keep as 35
    pca, X_pca = perform_pca(X_flat=image_stream_1d, n_components=N_COMPONENTS)

    # Perform t-SNE on the flattened images before reducing the dimensionality using PCA
    X_tsne_direct = perform_tsne(X_reduced_dim=image_stream_1d, perplexity=30, n_components=2)

    # Perform t-SNE on the flattened images after reducing the dimensionality using PCA
    X_tsne_pca = perform_tsne(X_reduced_dim=X_pca, perplexity=30, n_components=2)
    tsne_df_direct = create_tsne_cp_df(X_tsne_direct, all_labels, 1000)
    tsne_df_pca = create_tsne_cp_df(X_tsne_pca, all_labels, 1000)
    plot_2D_scatter_plot(tsne_df_direct, file_name)
    plot_2D_scatter_plot(tsne_df_pca, file_name)


def plot_all_images(process_dir = None, watershed_dir = None, autoencoder_dir = None, csv_file = None):
    '''This function plots all the raw, watershed, and autoencoder images in one image.
    The process_dir should have the csv_file of interest. 
    This assumes all other directories have all of the respective images of the csv saved.
    process_dir: Holds the raw images and the csv_file (This should be done by initial_download())
    watershed_dir: Holds all the watershed images (This should be done by generate_watershed())
    autoencoder_dir: Holds all the autoencoder output images (This should be done by generate_autoencoder_output())'''

    training_bps = BPSMouseDataset(csv_file, process_dir, transform=transforms.Compose([ResizeBPS(256, 256)]), file_on_prem=True)
    for i in training_bps:
        image, _, file_name = i
        print(image)

        watershed_path = os.path.join(watershed_dir, file_name)
        auto_path = os.path.join(autoencoder_dir, file_name)

        figure, axis = plt.subplots(1, 3)

        axis[0].imshow(image)
        axis[0].set_title('Original Image')

        axis[1].imshow(cv2.imread(watershed_path, cv2.IMREAD_ANYDEPTH))
        axis[1].set_title('Watershed Masks')
        
        axis[2].imshow(cv2.imread(auto_path, cv2.IMREAD_ANYDEPTH))
        axis[2].set_title('Autoencoder 32')
        plt.show()


def main():
    initial_download()

    generate_watershed(csv_dir = r'C:\Users\Jarrod\Documents\UCI\Spring_2023\CS_175\Main_Repo\final-project-e-mcheese-2\data\processed', 
                       csv_file = 'meta.csv', file_on_prem = True, 
                       save_local = r'C:\Users\Jarrod\Documents\UCI\Spring_2023\CS_175\Main_Repo\final-project-e-mcheese-2\data\watershed')
    
    train_autoencoder(csv_dir = r'C:\Users\Jarrod\Documents\UCI\Spring_2023\CS_175\Main_Repo\final-project-e-mcheese-2\data\processed', \
                       train_file = 'meta_dose_hi_hr_4_post_exposure_train.csv', val_file = 'meta_dose_hi_hr_4_post_exposure_test.csv', num_workers = 1)
    
    generate_autoencoder_output(model_weights = r'lightning_logs\version_2\checkpoints\epoch=99-step=1543600.ckpt', 
                                csv_dir = r'C:\Users\Jarrod\Documents\UCI\Spring_2023\CS_175\Main_Repo\final-project-e-mcheese-2\data\processed',
                                csv_file = 'meta.csv', 
                                save_local = r'C:\Users\Jarrod\Documents\UCI\Spring_2023\CS_175\Main_Repo\final-project-e-mcheese-2\data\autoencoder')
    
    generate_tsne_plots(csv_dir = r'C:\Users\Jarrod\Documents\UCI\Spring_2023\CS_175\Main_Repo\final-project-e-mcheese-2\data\watershed', 
                       csv_file = 'meta_dose_hi_hr_4_post_exposure_train.csv', file_name='TSNE Plots')
    
    plot_all_images(process_dir = r'C:\Users\Jarrod\Documents\UCI\Spring_2023\CS_175\Main_Repo\final-project-e-mcheese-2\data\processed',
                    watershed_dir = r'C:\Users\Jarrod\Documents\UCI\Spring_2023\CS_175\Main_Repo\final-project-e-mcheese-2\data\watershed', 
                    autoencoder_dir = r'C:\Users\Jarrod\Documents\UCI\Spring_2023\CS_175\Main_Repo\final-project-e-mcheese-2\data\autoencoder', 
                    csv_file = 'meta_dose_hi_hr_4_post_exposure_train.csv')

if __name__ == '__main__':

    main()