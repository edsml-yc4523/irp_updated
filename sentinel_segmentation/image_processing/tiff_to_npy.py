"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import rasterio
import numpy as np
import os


def tiff_to_npy(tiff_file, output_dir):
    with rasterio.open(tiff_file) as src:
        data = src.read()
        data = np.transpose(data, (1, 2, 0)) 
    base_name = os.path.splitext(os.path.basename(tiff_file))[0]
    npy_file = os.path.join(output_dir, base_name + '.npy')
    np.save(npy_file, data)
    print(f"Saved {npy_file}")


def batch_convert_tiff_to_npy(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            tiff_file = os.path.join(input_dir, filename)
            tiff_to_npy(tiff_file, output_dir)
