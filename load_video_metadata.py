# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:05:34 2021

@author: Ivan
"""


import os
from sys import platform
import numpy as np
import pandas as pd
from decord import VideoReader
from decord import cpu, gpu

import torch
import torch.utils.data


# decord is required for loading the videos. I have seen that it's quite fast at reading video data and the API is easy to use
# If people have problems getting it to work, we can just switch another option.

class VideoClipLoader(torch.utils.data.Dataset):
    def __init__(self, video_dir, metadata, pattern_vid = ".mp4", every = 25 ):

        self.every = every
        self.metadata = metadata

        #  make a list of all the videos available from the given metadata dataset
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            self.video_list = video_dir + "/" + metadata["Folder name"].astype(str) + "/" + metadata["Clip Name"].astype(str)  + pattern_vid
        else:
            self.video_list = video_dir + "\\" + metadata["Folder name"].astype(str) + "\\" + metadata["Clip Name"].astype(str)  + pattern_vid

    # load the full video and separate into images
    def load_video_as_images(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)

        start = 0
        end = len(vr)



        image_list = []

        for index in range(start, end):  # go through the whole video
            frame = vr[index]  # read a frame

            if index % self.every == 0: # check if the frame is the every other one
                frame = frame.asnumpy() # change to numpy
                frame = frame[:,:,0] # get only first dimension, the are the same
                image_list.append(frame)


        image_array = np.array(image_list)
        return image_array, video_path

    # get the metadata for the video clip
    def get_video_metadata(self, video_path):

        #  get the video path and slipt it
        video_path_split = video_path.split("\\")

        #  extract the folder name and clip name
        video_folder = video_path_split[-2]
        video_file = video_path_split[-1].split(".")[0]

        #  select the specified metadata row
        curr_metadata = self.metadata[(self.metadata.iloc[:,0]== int(video_folder))&(self.metadata.iloc[:,1]== video_file)].dropna()



        return curr_metadata


    #  get the new n images a video clip
    def __getitem__(self, idx):

        #  get images and path
        img, path = self.load_video_as_images(self.video_list.iloc[idx])

        #  get metadata
        metadata_curr = self.get_video_metadata(self.video_list.iloc[idx])

        # transform the images
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.float()
        img = torch.unsqueeze(img, 0)
        img = img.permute(1, 0, 2, 3)

        # transform the metadata to a string
        output_metadata_list = metadata_curr.squeeze().values.tolist()
        output_metadata_list[2] = output_metadata_list[2].strftime("%Y-%m-%d %H:%M:%S")

        output_metadata_str = ','.join(map(str, output_metadata_list))

        return img, path ,output_metadata_str

    def __len__(self):
        return len(self.video_list)



if __name__ == '__main__':
    video_path = r"Data"
    video_path = os.path.normpath(video_path)

    metadata_name = "metadata.csv"
    metadata_path = os.path.join(video_path, metadata_name)


    metadata = pd.read_csv(metadata_path)

    metadata["DateTime"] = pd.to_datetime(metadata['DateTime'], dayfirst = True)

    #  example of a query for 1 week from february - from 1st to 7th
    metadata_week = metadata[(metadata["DateTime"].dt.day>=1) & (metadata["DateTime"].dt.day<=7) & (metadata["DateTime"].dt.month==2)]

    dataset = VideoClipLoader(video_dir=video_path, metadata = metadata_week)
