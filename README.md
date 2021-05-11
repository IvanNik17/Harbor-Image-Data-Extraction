# Image-Data-Extraction
 Extract Images from the Video Clip data and the metadata


# Working order:

1. Download the video dataset and extract it into the Data folder (both daily video directories and the metadata.csv)
2. Run extract_image_datasets.py - setup what part of the videos you want to transform into image sequences, how many images to be extracted from each video clip, etc.
3. To load image data use load_images_metadata.py - here a subset of images can be selected depending on the metadata and used as a dataset
4. To create a datamodule containing testing, training, validation data use datamobule_metdata.py - here metadata queries need to be given as input to the datamodule for testing, training, validation.
5. The script setup_experiments_metadata.py contains a function that already separates the metadata into train, test, validation depending on experiments:
  - half day
  - one day
  - one week
  - one month

The for the experiments script is from February and March.
