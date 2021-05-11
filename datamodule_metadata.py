# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:20:08 2021

@author: IvanTower
"""
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import os
import pandas as pd

from load_images_metadata import DatasetWithMetadata

class DataModuleMetadata(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.data_train = DatasetWithMetadata(img_dir=self.cfg['img_dir'], metadata=self.cfg['metadata_train'], return_metadata = self.cfg['get_metadata'])
            self.data_val = DatasetWithMetadata(img_dir=self.cfg['img_dir'], metadata=self.cfg['metadata_val'], return_metadata = self.cfg['get_metadata'])
        else:
            self.data_test = DatasetWithMetadata(img_dir=self.cfg['img_dir'], metadata=self.cfg['metadata_test'], return_metadata = self.cfg['get_metadata'])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.cfg['batch_size'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.cfg['batch_size'], shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.cfg['batch_size'], shuffle=False)


if __name__ == '__main__':
    
    #  path to the images, it should also contain the metadata csv file. The metadata file entries and images need to coincide
    images_path = r"Image Dataset"    
    metadata_file_name = "metadata_images.csv"
    metadata_path = os.path.join(images_path,metadata_file_name)
    
    #  read the metadata
    metadata = pd.read_csv(metadata_path)
    #  change the DateTime column to a pandas datetime
    metadata["DateTime"] = pd.to_datetime(metadata['DateTime'])
    
    #  select parts of the metadata for training, validation and testing
    metadata_train = metadata[(metadata["DateTime"].dt.day==11) & (metadata["DateTime"].dt.month==2)]
    metadata_val = metadata[(metadata["DateTime"].dt.day==12) & (metadata["DateTime"].dt.month==2)]
    metadata_test = metadata[(metadata["DateTime"].dt.day==13) & (metadata["DateTime"].dt.month==2)]
    
    # create the config file for the datamodule:
    '''
    img_dir - the path where the top path where all the images are
    metadata_train - the training part of the metadata, which would be used to get the correct images
    metadata_val - the validation part of the metadata
    metadata_test - the testing part of the metadata
    get_metadata - boolean which signals if you want to return metadata as part of the dataloader. 
        If it is True - the dataloader returns a tuple of 3 items - (tensor of images, folders of images, metadata string of the images)
        If it is False - the dataloader returns a tuple of 2 items - (tensor of images, folders of images)
    batch_size - amount of images to return
    '''
    cfg = {
       'img_dir': images_path,
       'metadata_train': metadata_train,
       'metadata_val': metadata_val,
       'metadata_test': metadata_test,
       'get_metadata': True,
       'batch_size': 16,
    }
    
    # instantiate the class and give it the cfg dictionary, call the setup
    dm = DataModuleMetadata(cfg)
    dm.setup()


    fixed_x = next(iter(dm.train_dataloader()))