#!/usr/bin/env python3
import sys
from zipfile import PyZipFile
for zip_file in sys.argv[1:]:
    pzf = PyZipFile(zip_file)
    pzf.extractall()



#import tensorflow_datasets as tfds
#import os

#dataset_dir = '/scratch1/09111/mbbm/ilsvrc2012'  # directory where you downloaded the tar files to
#temp_dir = '/tmp'  # a temporary directory where the data will be stored intermediately

#download_config = tfds.download.DownloadConfig(
#    extract_dir=os.path.join(temp_dir, 'extracted'),
#    manual_dir=dataset_dir
#)

#tfds.builder("imagenet2012").download_and_prepare(download_config=download_config)
