# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:13:16 2021

@author: Ivan
"""
import cv2
import os
import numpy as np
import pandas as pd

import torch
import torch.utils.data

from load_images_metadata import DatasetWithMetadata
from datetime import datetime,timedelta

# get every n-th element - which is dictated by subset_size. It gives the closest possible normal sampling to the requred subset_size
#  if the subset_size is larger than or equal to the size of the dataset just return the dataset
def get_subset(metadata, subset_size):
    metadata_size = len(metadata)
    
    subset_size = subset_size if subset_size < metadata_size else metadata_size
    
    every = int(metadata_size/subset_size)
    
    metadata = metadata.iloc[::every, :]
    
    return metadata

#  Example of experimental metadata separation that can be used to load train, val, test data
def experiment_metadataQueries(metadata, experiment):
    if experiment == "half_day":
        metadata_train = metadata[ (metadata["DateTime"].dt.day == 10) & (metadata["DateTime"].dt.month == 1) & (metadata["DateTime"].dt.hour > 6) & (metadata["DateTime"].dt.hour < 18)]
        metadata_val = metadata[ (metadata["DateTime"].dt.day == 11) & (metadata["DateTime"].dt.month == 1) & (metadata["DateTime"].dt.hour > 6) & (metadata["DateTime"].dt.hour < 18)]
        metadata_test = metadata[(~metadata["DateTime"].isin(metadata_train["DateTime"])) & (~metadata["DateTime"].isin(metadata_val["DateTime"]))].dropna()
    elif experiment == "one_day":
        metadata_train = metadata[ (metadata["DateTime"].dt.day == 10) & (metadata["DateTime"].dt.month == 1)]
        metadata_val = metadata[ (metadata["DateTime"].dt.day == 11) & (metadata["DateTime"].dt.month == 1)]
        metadata_test = metadata[(~metadata["DateTime"].isin(metadata_train["DateTime"])) & (~metadata["DateTime"].isin(metadata_val["DateTime"]))].dropna()
    elif experiment == "one_week":
        
        dates = []
        for day in np.arange(10,17,1):
            dates.append(datetime(2021,1,day).date())
        metadata_train = metadata[metadata["DateTime"].dt.date.isin(dates)].dropna()
        
        dates = []
        for day in np.arange(17,24,1):
            dates.append(datetime(2021,1,day).date())
        metadata_val = metadata[metadata["DateTime"].dt.date.isin(dates)].dropna()
        
        metadata_test = metadata[(~metadata["DateTime"].isin(metadata_train["DateTime"])) & (~metadata["DateTime"].isin(metadata_val["DateTime"]))].dropna()
    elif experiment == "one_month":
        metadata_train = metadata[ (metadata["DateTime"].dt.month == 2) ]
        metadata_val = metadata[ (metadata["DateTime"].dt.month == 3) ]
        metadata_test = metadata[(~metadata["DateTime"].isin(metadata_train["DateTime"])) & (~metadata["DateTime"].isin(metadata_val["DateTime"]))].dropna()
        
    return metadata_train, metadata_val, metadata_test

  
    
    

if __name__ == '__main__':
    
    
    images_path = r"Image Dataset"    
    metadata_file_name = "metadata_images.csv"
    metadata_path = os.path.join(images_path,metadata_file_name)
    
    metadata = pd.read_csv(metadata_path)
    metadata["DateTime"] = pd.to_datetime(metadata['DateTime'], dayfirst = True)
    
    
   
    
    #  example metadata query for training, validation and testing - training and validation from 1 day from february
    #  testing data is all the other data left
    # metadata_train = metadata[ (metadata["DateTime"].dt.day == 1) & (metadata["DateTime"].dt.month == 2)]
    # metadata_val = metadata[ (metadata["DateTime"].dt.day == 2) & (metadata["DateTime"].dt.month == 2)]
    # metadata_test = metadata[(~metadata["DateTime"].isin(metadata_train["DateTime"])) & (~metadata["DateTime"].isin(metadata_val["DateTime"]))].dropna()
    
    metadata_train_od, metadata_val_od, metadata_test_od = experiment_metadataQueries(metadata,"one_day")
    
    metadata_train_om, metadata_val_om, metadata_test_om = experiment_metadataQueries(metadata,"one_month")
    
    print(len(get_subset(metadata_train_od, 2000)))
    print(len(get_subset(metadata_train_om, 2000)))
    
    # dataset_train = DatasetWithMetadata(img_dir=images_path, metadata = metadata_train, check_data = True, return_metadata = True)
    # dataset_val = DatasetWithMetadata(img_dir=images_path, metadata = metadata_val, check_data = True, return_metadata = True)
    # dataset_test = DatasetWithMetadata(img_dir=images_path, metadata = metadata_test, check_data = True, return_metadata = True)