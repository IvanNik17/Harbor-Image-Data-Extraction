# -*- coding: utf-8 -*-
"""
Created on Wed May 19 09:47:33 2021

@author: Ivan
"""
import numpy as np
import pandas as pd
import os

from scipy.spatial.distance import pdist, squareform

# furthest point sampling
def getGreedyPerm(x,n_points=1024):
    """
    A Naive O(N^2) algorithm to do furthest points sampling
    Parameters
    ----------
    x : ndarray (N, N)
        An NxN distance matrix for points
    Return
    ------
    tuple (list, list)
        (permutation (N-length array of indices),
        lambdas (N-length array of insertion radii))
    """
    N = x.shape[0]
    #print(x.shape)
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = x[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, x[idx, :])
    #print(perm[:])
    return perm[:n_points]

# Function to get data for coldest, hottest and median months
# Add the full metadata and select - min for coldest, max for hottest, median for median one

def get_experiment_data(metadata, which = "min"):
    
    mask = []
    
    if which == "min":
        mask = metadata["Temperature"] == metadata["Temperature"].min()
    elif which =="max":
        mask = metadata["Temperature"] == metadata["Temperature"].max()
    elif which == "median":
        mask = metadata["Temperature"] == metadata["Temperature"].median()
        
    # get the day number based on mask - here using "Folder name", as it is the date
    needed_day_number = metadata[mask].iloc[0]["Folder name"]
    # get all rows that have the same Folder name
    needed_day_frames = metadata[metadata["Folder name"] == needed_day_number]
    num_frames_needed_day = len(needed_day_frames)

    # get the week number using the isocalendar().week, on the DateTime column
    needed_week_number = metadata[mask]["DateTime"].dt.isocalendar().week.iloc[0]
    #get all rows that have the same week number
    needed_week_frames = metadata[metadata["DateTime"].dt.isocalendar().week == needed_week_number]
    num_frames_needed_week = len(needed_week_frames)

    # get the month number using the month call on the DateTime column
    needed_month_number = metadata[mask]["DateTime"].dt.month.iloc[0]
    #  get all rows that have the same month number
    needed_month_frames = metadata[metadata["DateTime"].dt.month == needed_month_number]
    num_frames_needed_month = len(needed_month_frames)
    
    # reset indices to start from 0
    needed_day_frames = needed_day_frames.reset_index(drop=True)
    needed_week_frames = needed_week_frames.reset_index(drop=True)
    needed_month_frames = needed_month_frames.reset_index(drop=True)
        
    return needed_day_frames, needed_week_frames, needed_month_frames






def extract_furthest_frames_metadata(metadata, features, num_frames = 100, reserved_at_ends = 5):
    """
    Function that takes the day, week, month sub-datasets,
    the features that are needed and gets farthest point frames
    ----------
    metadata : pandas dataframe, containing at minimum "Image Number" column,
        plus the columns specified in the features list
    
    features : columns from the metadata dataframe to be used in the distance calculation
        example ["Temperature"] or ["Temperature", "Humidity"]
    
    num_frames : how many frames to be extracted from the furthest point sampling
    
    reserved_at_ends : how many points at both ends of each clip to be reserved and not sampled
    
    Return
    ------
    pandas dataframe containing the sampled frame metadata
    """
    
    # Add the "Image Number" to the features
    features.append('Image Number')
    # Make a sub-dataframe for the calculations
    metadata_small = metadata[features]
    

    # make a new column containing the indices, before cutting the reserved ones
    metadata_small["index_metadata"] = metadata.index
    
    # get all the unique Image Numbers
    unique_frame_nums = metadata_small['Image Number'].unique()
    
    # get the reserved_at_ends from the front and back of the array 
    reserved_frame_nums = unique_frame_nums[:reserved_at_ends]
    reserved_frame_nums = np.concatenate((reserved_frame_nums, unique_frame_nums[-reserved_at_ends:]), axis=0)

    # remove them from the sub-dataframe
    metadata_small = metadata_small[(~metadata_small["Image Number"].isin(reserved_frame_nums))]

    # separate the index_metadata so it is not used for the distance calc
    indices_before_cutting = metadata_small["index_metadata"]
    metadata_small = metadata_small.drop(["index_metadata"], axis=1)
    
    # normalize the columns between the minimum and maximum
    for column in metadata_small.columns:
        metadata_small[column] = (metadata_small[column] - np.min(metadata_small[column]))/np.ptp(metadata_small[column])
    
    # use the scipy pdist to calculate the distances between all elements
    metadata_small_dist = pdist(metadata_small)
    #  make the square distance matrix
    metadata_small_dist_square = squareform(metadata_small_dist)
    
    #  use the greedyPerm to get the sampled frame indices
    indices_search = getGreedyPerm(metadata_small_dist_square, num_frames)
    #  get the indices in the context of the metadata
    indices_metadata = indices_before_cutting.iloc[indices_search].to_numpy()

    # get the sampled frame metadata
    return metadata.iloc[indices_metadata]



if __name__ == '__main__':
    
    images_path = r"Image Dataset"    
    metadata_file_name = "metadata_images.csv"
    metadata_path = os.path.join(images_path,metadata_file_name)
    
    metadata = pd.read_csv(metadata_path)

    metadata['DateTime'] = pd.to_datetime(metadata['DateTime'], dayfirst = True)
    
    # Coldest day, week, month
    cold_day, cold_week, cold_month = get_experiment_data(metadata,"min")

    cold_day_frames = extract_furthest_frames_metadata(cold_day, ["Temperature"])
    cold_week_frames = extract_furthest_frames_metadata(cold_week, ["Temperature"])
    # cold_month_frames = extract_furthest_frames_metadata(cold_month, ["Temperature"])
    
    # Hottest day, week, month
    hot_day, hot_week, hot_month = get_experiment_data(metadata,"max")

    hot_day_frames = extract_furthest_frames_metadata(hot_day, ["Temperature"])
    hot_week_frames = extract_furthest_frames_metadata(hot_week, ["Temperature"])
    # hot_month_frames = extract_furthest_frames_metadata(hot_month, ["Temperature"])
    
    # Median day, week, month
    mid_day, mid_week, mid_month = get_experiment_data(metadata,"median")
    
    mid_day_frames = extract_furthest_frames_metadata(mid_day, ["Temperature"])
    mid_week_frames = extract_furthest_frames_metadata(mid_week, ["Temperature"])
    # mid_month_frames = extract_furthest_frames_metadata(mid_month, ["Temperature"])
    