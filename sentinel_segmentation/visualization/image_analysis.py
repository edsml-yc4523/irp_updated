"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import os
import cv2
import numpy as np
import pandas as pd


def process_image(image):
    """
    Process a single image to detect features
    such as harbors, jetties, and resorts.

    Args:
        image (numpy.ndarray): The image to process.

    Returns:
        list of dict: A list of dictionaries containing detected label,
        centroid coordinates, and area.
    """
    color_to_label = {
        (0, 0, 255): 'harbor',
        (0, 255, 0): 'jetty',
        (255, 0, 0): 'resort'
    }

    results = []

    for color, label in color_to_label.items():
        lower = np.array(color, dtype="uint8")
        upper = np.array(color, dtype="uint8")
        mask = cv2.inRange(image, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for i in range(1, num_labels):  
            x, y = centroids[i]
            area = stats[i, cv2.CC_STAT_AREA]
            results.append({
                'label': label,
                'centroid': (int(x), int(y)),
                'area': int(area)
            })

    return results


def process_images_in_folder(folder_path):
    """
    Process all images in a specified folder and
    compile the results into a DataFrame.

    Args:
        folder_path (str): Path to the folder containing the images.

    Returns:
        pandas.DataFrame: DataFrame containing the detected labels,
        centroids, and areas for each image.
    """
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    columns = ['date', 'detected_label', 'true_label', 'centroid_x',
               'centroid_y', 'area']
    df = pd.DataFrame(columns=columns)

    for image_file in image_files:
        date = image_file.split('_')[1]
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        processed_results = process_image(image)

        if not processed_results:
            new_row = {
                'date': date,
                'detected_label': np.nan,
                'true_label': np.nan,
                'centroid_x': np.nan,
                'centroid_y': np.nan,
                'area': np.nan
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            for result in processed_results:
                new_row = {
                    'date': date,
                    'detected_label': result['label'],
                    'true_label': np.nan,  
                    'centroid_x': result['centroid'][0],
                    'centroid_y': result['centroid'][1],
                    'area': result['area']
                }
                df = pd.concat([df, pd.DataFrame([new_row])],
                               ignore_index=True)

    return df


def calculate_lat_lon(top_left_lat, top_left_lon,
                      bottom_right_lat, bottom_right_lon,
                      original_width, original_height,
                      i_prime, j_prime,
                      resized_width=224, resized_height=224):
    """
    Calculate the latitude and longitude for a given pixel in an image.

    Args:
        top_left_lat (float): Latitude of the top-left corner of the image.
        top_left_lon (float): Longitude of the top-left corner of the image.
        bottom_right_lat (float): Latitude of the bottom-right corner of the image.
        bottom_right_lon (float): Longitude of the bottom-right corner of the image.
        original_width (int): Width of the original image.
        original_height (int): Height of the original image.
        i_prime (int): Row index (y-coordinate) in the resized image.
        j_prime (int): Column index (x-coordinate) in the resized image.
        resized_width (int, optional): Width of the resized image. Defaults to 224.
        resized_height (int, optional): Height of the resized image. Defaults to 224.

    Returns:
        tuple: Calculated latitude and longitude corresponding to the pixel coordinates.
    """
    # Map the pixel position from the resized image to the original image
    i = (i_prime / (resized_height - 1)) * (original_height - 1)
    j = (j_prime / (resized_width - 1)) * (original_width - 1)

    # Calculate the latitude and longitude for the pixel position in the original image
    lat_range = bottom_right_lat - top_left_lat
    lon_range = bottom_right_lon - top_left_lon

    lat = top_left_lat + (i / (original_height - 1)) * lat_range
    lon = top_left_lon + (j / (original_width - 1)) * lon_range

    return lat, lon


def update_lat_lon(df, top_left_lat, top_left_lon, bottom_right_lat,
                   bottom_right_lon, original_width, original_height):
    """
    Update the DataFrame with latitude and longitude for each detected feature.

    Args:
        df (pandas.DataFrame): DataFrame containing detected features with pixel coordinates.
        top_left_lat (float): Latitude of the top-left corner of the image.
        top_left_lon (float): Longitude of the top-left corner of the image.
        bottom_right_lat (float): Latitude of the bottom-right corner of the image.
        bottom_right_lon (float): Longitude of the bottom-right corner of the image.
        original_width (int): Width of the original image.
        original_height (int): Height of the original image.

    Returns:
        pandas.DataFrame: Updated DataFrame with latitude and longitude columns.
    """
    df[['latitude', 'longitude']] = df.apply(
        lambda row: calculate_lat_lon(top_left_lat, top_left_lon,
                                      bottom_right_lat, bottom_right_lon,
                                      original_width, original_height,
                                      row['centroid_x'], row['centroid_y']),
        axis=1, result_type='expand'
    )
    df.sort_values(by=['date'], inplace=True)
    return df
