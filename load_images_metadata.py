# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:00:02 2021

@author: IvanTower
"""

import cv2
import os
import numpy as np
import pandas as pd

import re

import torch
import torch.utils.data


#  Dataset class for image data depending on the metadata

'''
For initializing the class the inputs are:
    img_dir - the top directory containing all the images
    metadata - pandas dataframe containing references to the selected images
    return_metadata - boolean deciding if metadata would be returned as part of __getitem__ calls
    pattern_img - the type of extension that is being searched for
'''

class DatasetWithMetadata(torch.utils.data.Dataset):
    def __init__(self, img_dir, metadata, return_metadata = True, check_data = False, pattern_img = '.jpg'):
        
        self.metadata = metadata
        
        #  create a list of image directories from the metadata entries
        self.image_list = img_dir + "\\" + self.metadata["Folder name"].astype(str) + "\\" + self.metadata["Clip Name"].astype(str) + "\\" + "img_" +  self.metadata["Image Number"].astype(str) + pattern_img
        self.return_metadata = return_metadata
        
        # check if list is empty
        if not len(self.image_list)>0:
            print("No images found - check metadata")
        
        
        if check_data:
            # check if there are paths from the image_list where the path does not exist/incorrect
            check_path = [not os.path.exists(path) for path in self.image_list]
            # get incorrect parths
            wrong_paths = self.image_list[check_path] 
            
            #  signal that there are incorrect paths
            if len(wrong_paths)>0:
                print(f"Warning {len(wrong_paths)} image paths do not exist!")
                print(wrong_paths)
        
    # function for loading a specific image and its path
    def load_image(self, image_path):
        
        # print(image_path)
        curr_image = cv2.imread(image_path)
        self.image_h, self.image_w, _ = curr_image.shape
        curr_image = curr_image[:,:,0]
        return curr_image, image_path
    
    
    # function for getting the metadata for the specified image
    def get_img_metadata(self, image_path):
        img_path_split = image_path.split("\\")
        img_folder = img_path_split[-3]
        
        clip_file = img_path_split[-2].split(".")[0]
        
        img_file = img_path_split[-1].split(".")[0].split("_")[1]
        
        # print(f" {img_folder}          {img_file}")
        curr_metadata = self.metadata[(self.metadata["Folder name"]== int(img_folder))&(self.metadata["Clip Name"]== clip_file)&(self.metadata["Image Number"]== int(img_file))].dropna()

        return curr_metadata
    
    #  __getitem__ function that loads an image from an index, normalizes it between 0 and 1, makes it into a tensor of float and unsqueezes it
    def __getitem__(self, idx):
        img, path = self.load_image(self.image_list.iloc[idx])
        
        
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.float()
        img = torch.unsqueeze(img, 0) # (HxW -> CxHxW)
        
        # if metadata is returned for the selected images then make the metadata into a string, because dataloaders do not like pandas dataframes
        if self.return_metadata:
            metadata_curr = self.get_img_metadata(self.image_list.iloc[idx])
            metadata_curr['DateTime'] = metadata_curr['DateTime'].dt.strftime("%Y-%m-%d %H:%M:%S")
            output_metadata_list = metadata_curr.squeeze().values.tolist()
            output_metadata_str = ','.join(map(str, output_metadata_list))
            
            # print(metadata_curr)
            
            return img, path, output_metadata_str 
        
        return img, path

    def __len__(self):
        return len(self.image_list)
    
    
if __name__ == '__main__':   
    
    images_path = r"Image Dataset"    
    metadata_file_name = "metadata_images.csv"
    metadata_path = os.path.join(images_path,metadata_file_name)
    
    metadata = pd.read_csv(metadata_path)
    
    metadata["DateTime"] = pd.to_datetime(metadata['DateTime'], dayfirst = True)
    
    metadata_selected = metadata[(metadata["DateTime"].dt.day>=1) & (metadata["DateTime"].dt.day<=7) & (metadata["DateTime"].dt.month==2)]
    
    dataset = DatasetWithMetadata(img_dir=images_path, metadata = metadata_selected, return_metadata = True)
    
    # trial = next(iter(dataset))
    
    # for i, sample in enumerate(dataset):
    #     img, path, metadata = sample
        
    #     print(metadata)
    
    

