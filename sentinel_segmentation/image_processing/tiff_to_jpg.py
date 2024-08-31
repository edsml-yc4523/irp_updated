"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import cv2
import numpy as np
from osgeo import gdal


def normalize_band(band):
    """
    Normalize a single band from the TIFF image to the range 0-255.

    Args:
        band (numpy.ndarray): The band data to normalize.

    Returns:
        numpy.ndarray: The normalized band as an 8-bit unsigned integer array.
    """
    band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
    band_min = np.min(band)
    band_max = np.max(band)
    print(f"Band min: {band_min}, Band max: {band_max}")
    
    if band_max > band_min:
        normalized_band = (band - band_min) / (band_max - band_min) * 255
        return normalized_band.astype(np.uint8)
    else:
        return np.zeros_like(band, dtype=np.uint8)


def tif_jpg_transform(file_path_name, bgr_savepath_name, band_indices):
    """
    Transform a TIFF image into a JPG image using specified band indices.

    Args:
        file_path_name (str): The path to the input TIFF file.
        bgr_savepath_name (str): The path to save the output JPG file.
        band_indices (list of int): List of band indices
                                    to extract and process.
    """
    try:
        dataset = gdal.Open(file_path_name, gdal.GA_ReadOnly)
        if dataset is None:
            raise Exception("Failed to open file.")
        
        if len(band_indices) > dataset.RasterCount:
            raise Exception("Not enough bands in the input file.")

        bands = []
        for index in band_indices:
            band = dataset.GetRasterBand(index + 1)
            band_data = band.ReadAsArray()
            print(f"Band {index} data range: {np.min(band_data)} - {np.max(band_data)}")
            bands.append(band_data)

        # For multiple bands RGB or single bands like NIR, NDVI, NDWI
        if len(bands) > 1:
            combined_img = np.stack(bands, axis=-1)
            for i in range(combined_img.shape[2]):
                combined_img[:, :, i] = normalize_band(combined_img[:, :, i])
            cv2.imwrite(bgr_savepath_name, combined_img)
        else:
            single_band_img = normalize_band(bands[0])
            cv2.imwrite(bgr_savepath_name, single_band_img[:, :, np.newaxis])

        print(f"Successfully converted {file_path_name} to {bgr_savepath_name}")

    except Exception as e:
        print(f"Failed to convert {file_path_name}. Error: {e}")
