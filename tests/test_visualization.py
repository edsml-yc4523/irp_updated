"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import numpy as np
import pandas as pd
from sentinel_segmentation.visualization import process_image, process_images_in_folder, update_lat_lon, generate_map
import os
import cv2


def test_process_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[10:20, 10:20] = [0, 0, 255]  # Simulate a "harbor"
    results = process_image(image)
    assert len(results) > 0, "No landmarks detected"
    assert results[0]['label'] == 'harbor', "Incorrect label detected"


def test_process_images_in_folder(tmpdir):
    folder = tmpdir.mkdir("test_images")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[10:20, 10:20] = [0, 0, 255]
    image_path = os.path.join(folder, "test_image_2023-01-01.png")
    cv2.imwrite(image_path, image)
    df = process_images_in_folder(str(folder))
    assert not df.empty, "Dataframe is empty"
    assert df.iloc[0]['detected_label'] == 'harbor', "Incorrect label in dataframe"


def test_update_lat_lon():
    # Create a DataFrame with the necessary columns
    df = pd.DataFrame({
        'centroid_x': [50],
        'centroid_y': [50],
        'date': ['2023-01-01']  # Add a 'date' column
    })
    top_left_lat, top_left_lon = -0.594651, 73.217103
    bottom_right_lat, bottom_right_lon = -0.600996, 73.225116
    original_width, original_height = 90, 72

    # Update lat/lon values
    updated_df = update_lat_lon(
        df, top_left_lat, top_left_lon,
        bottom_right_lat, bottom_right_lon,
        original_width, original_height
    )

    # Check that latitude and longitude were added
    assert 'latitude' in updated_df.columns, "Latitude column should be added"
    assert 'longitude' in updated_df.columns, "Longitude column should be added"
    assert not updated_df[['latitude', 'longitude']].isnull().any().any(), "Latitude and longitude should not contain NaN values"


def test_generate_map(tmpdir):
    df = pd.DataFrame({
        'latitude': [-0.596],
        'longitude': [73.221],
        'status': ['existing'],
        'true_label': ['harbor'],
        'date': ['2023-01-01'],
        'date_start': [pd.Timestamp('2023-01-01')],
        'date_end': [pd.Timestamp('2023-02-01')],
    })
    map_file = tmpdir.join("test_map.html")
    generate_map(df, str(map_file))
    assert os.path.exists(str(map_file)), "Map file was not created"
