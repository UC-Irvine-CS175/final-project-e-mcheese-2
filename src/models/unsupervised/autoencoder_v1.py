import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.errorfactory import ClientError

import pyprojroot
from pyprojroot import here
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import sys
sys.path.append(str(root))

from src.dataset.bps_dataset import BPSMouseDataset 
from src.dataset.bps_datamodule import BPSDataModule


class Encoder(nn.Module):
    def __init__(self,
        latent_dim,
        num_channels,
        dim_x,
        dim_y,
        filter_1=8,
        filter_2=16,
        filter_3=32,
        filter_4=64,
        ks=3,):
        super().__init__()
        pad = int((ks - 1) / 2)

        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=filter_1,
            kernel_size=(ks, ks),
            stride=(1, 1),
            padding=(pad, pad),
        )

        #self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.bn1 = nn.BatchNorm2d(filter_1, track_running_stats=False)
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=filter_1,
            out_channels=filter_2,
            kernel_size=(ks, ks),
            stride=(1, 1),
            padding=(pad, pad),
        )
        self.bn2 = nn.BatchNorm2d(filter_2, track_running_stats=False)
        self.conv3 = nn.Conv2d(
            in_channels=filter_2,
            out_channels=filter_3,
            kernel_size=(ks, ks),
            stride=(1, 1),
            padding=(pad, pad),
        )
        self.bn3 = nn.BatchNorm2d(filter_3, track_running_stats=False)
        self.conv4 = nn.Conv2d(
            in_channels=filter_3,
            out_channels=filter_4,
            kernel_size=(ks, ks),
            stride=(1, 1),
            padding=(pad, pad),
        )
        self.bn4 = nn.BatchNorm2d(filter_4, track_running_stats=False)
        self.flat = nn.Flatten()
        self.last = nn.Linear(
            in_features=(int(filter_4 * np.floor(dim_x / 16) * np.floor(dim_y / 16))),
            out_features=latent_dim,
        )
    def forward(self, x):
        #return self.l1(x)
        x1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        x2 = self.pool(F.relu(self.bn2(self.conv2(x1))))
        x3 = self.pool(F.relu(self.bn3(self.conv3(x2))))
        x4 = self.pool(F.relu(self.bn4(self.conv4(x3))))
        x5 = self.flat(x4)
        x_out = self.last(x5)
        # If you want to get output of any other layer on the way, you can add them to return below,
        ## e.g., return x_4, x_out
        return x_out


class Decoder(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
    def __init__(
        self,
        latent_dim,
        num_channels,
        dim_x,
        dim_y,
        filter_1=8,
        filter_2=16,
        filter_3=32,
        filter_4=64,
        ks=3,
    ):
        super().__init__()
        pad = int((ks - 1) / 2)
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_channels = num_channels
        self.filter_4 = filter_4
        self.fc = nn.Linear(
            in_features=latent_dim,
            out_features=int(filter_4 * np.floor(dim_x / 16) * np.floor(dim_y / 16)),
        )
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=filter_4,
            out_channels=filter_4,
            kernel_size=(ks, ks),
            stride=(1, 1),
            padding=(pad, pad),
        )
        self.bn1 = nn.BatchNorm2d(filter_4, track_running_stats=False)
        self.up1 = nn.Upsample(size=(int(dim_x / 8), int(dim_y / 8)))
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=filter_4,
            out_channels=filter_3,
            kernel_size=(ks, ks),
            stride=(1, 1),
            padding=(pad, pad),
        )
        self.bn2 = nn.BatchNorm2d(filter_3, track_running_stats=False)
        self.up2 = nn.Upsample(size=(int(dim_x / 4), int(dim_y / 4)))
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=filter_3,
            out_channels=filter_2,
            kernel_size=(ks, ks),
            stride=(1, 1),
            padding=(pad, pad),
        )
        self.bn3 = nn.BatchNorm2d(filter_2, track_running_stats=False)
        self.up3 = nn.Upsample(size=(int(dim_x / 2), int(dim_y / 2)))
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=filter_2,
            out_channels=filter_1,
            kernel_size=(ks, ks),
            stride=(1, 1),
            padding=(pad, pad),
        )
        self.bn4 = nn.BatchNorm2d(filter_1, track_running_stats=False)
        self.up4 = nn.Upsample(size=(int(dim_x), int(dim_y)))
        self.convlast = nn.Conv2d(
            in_channels=filter_1,
            out_channels=num_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )

        
    def forward(self, x):
        #return self.l1(x)
        x1 = F.relu(self.fc(x))
        x1 = torch.reshape(
            x1,
            (
                -1,
                self.filter_4,
                int(np.floor(self.dim_x / 16)),
                int(np.floor(self.dim_y / 16)),
            ),
        )
        x2 = self.up1(F.relu(self.bn1(self.deconv1(x1))))
        x3 = self.up2(F.relu(self.bn2(self.deconv2(x2))))
        x4 = self.up3(F.relu(self.bn3(self.deconv3(x3))))
        x5 = self.up4(F.relu(self.bn4(self.deconv4(x4))))
        x6 = self.convlast(x5)
        x_out = torch.reshape(x6, (-1, self.num_channels, self.dim_x, self.dim_y))

        # If you want to get output of any other layer on the way, you can add them to return below,
        ## e.g., return x_4, x_out
        return x_out
    
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_function = torch.nn.MSELoss(reduction='none')


    def forward(self,x):
        return self.decoder.forward(self.encoder.forward(x))


    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        # loss = F.mse_loss(x, x_hat, reduction="none")
        # loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        loss = self.loss_function(x, x_hat).sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    def training_step(self, batch, batch_idx):
        # # training_step defines the train loop.
        # x, y = batch
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        # return loss
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)
    
    




def main():
    bucket_name = "nasa-bps-training-data"
    s3_path = "Microscopy/train"
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3_meta_fname = "meta.csv"


    data_dir = root / 'data'
    # testing get file functions from s3
    local_train_dir = data_dir / 'processed'

    # testing PyTorch Lightning DataModule class ####
    train_csv_file = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    train_dir = data_dir / 'processed'
    validation_csv_file = 'meta_dose_hi_hr_4_post_exposure_test.csv'
    validation_dir = data_dir / 'processed'
    bps_dm = BPSDataModule(train_csv_file=train_csv_file,
                           train_dir=train_dir,
                           val_csv_file=validation_csv_file,
                           val_dir=validation_dir,
                           resize_dims=(256, 256),
                           meta_csv_file = s3_meta_fname,
                           meta_root_dir=s3_path,
                           s3_client= None,
                           bucket_name=bucket_name,
                           s3_path=s3_path,
                           convertToFloat = True,
                           num_workers=1
                           )
    # Setup train and validate dataloaders
    bps_dm.setup(stage='train')
    bps_dm.setup(stage='validate')

    #torch.multiprocessing.set_start_method('spawn')

    # model
    # add encoder arguments!!!!
    #autoencoder = LitAutoEncoder(Encoder(32, 1, 256, 256), Decoder(32, 1, 256, 256))

    # train model
    #trainer = pl.Trainer(accelerator = "gpu", devices = 1, max_epochs=10)
    #trainer.fit(model=autoencoder, train_dataloaders=bps_dm.train_dataloader(), val_dataloaders=bps_dm.val_dataloader())

    #autoencoder = LitAutoEncoder(Encoder(), Decoder())
    #optimizer = autoencoder.configure_optimizers()
    validate = bps_dm.val_dataloader()
    model = LitAutoEncoder.load_from_checkpoint(r"lightning_logs\\version_22\\checkpoints\\epoch=9-step=17730.ckpt")
    import matplotlib.pyplot as plt
    for x, _ in validate:
        x = x.cuda()
        y_hat = model(x)

        plt.imshow(y_hat.cpu().data[0, :, :, :].permute(1, 2, 0))
        plt.show()


if __name__ == "__main__":
    main()