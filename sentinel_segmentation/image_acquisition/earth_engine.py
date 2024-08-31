"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import ee
import time


def initialize_earth_engine():
    """
    Initialize the Google Earth Engine.

    Attempts to authenticate and initialize the Earth Engine. If
    initialization fails, an exception is raised.
    """
    try:
        ee.Authenticate()
        ee.Initialize()
    except Exception as e:
        print("The Earth Engine was not initialized.")
        raise e


def get_images(lat1, lon1, lat2, lon2,
               start_date="2014-01-01", end_date="2024-05-31"):
    """
    Retrieve a collection of satellite images from the Sentinel-2 dataset.

    Filters images based on the specified geographic region, date range,
    and cloud cover percentage.

    Args:
        lat1 (float): Latitude of the first corner of the bounding box.
        lon1 (float): Longitude of the first corner of the bounding box.
        lat2 (float): Latitude of the opposite corner of the bounding box.
        lon2 (float): Longitude of the opposite corner of the bounding box.
        start_date (str): Start date for the image collection (YYYY-MM-DD).
        end_date (str): End date for the image collection (YYYY-MM-DD).

    Returns:
        ee.ImageCollection: Processed image collection containing selected
        bands and additional computed indices (NDVI, NDWI).
    """
    rectangle = ee.Geometry.Rectangle([lon1, lat1, lon2, lat2])
    collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
        .filterBounds(rectangle) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))

    def process_image(image):
        selected_bands = image.select(['B4', 'B3', 'B2', 'B8']).toFloat()
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI').toFloat()
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI').toFloat()
        combined_image = selected_bands.addBands([ndvi, ndwi])
        clipped_image = combined_image.clip(rectangle)
        return clipped_image

    processed_collection = collection.map(process_image)

    return processed_collection


def export_images(collection, region, folder_name,
                  area_name, scale=10, batch_size=20):
    """
    Export images from an Earth Engine image collection to Google Drive.

    Args:
        collection (ee.ImageCollection): The image collection to export.
        region (ee.Geometry): The region of interest for image export.
        folder_name (str): The name of the folder in Google Drive to save
        the images.
        area_name (str): A prefix for the exported image file names.
        scale (int): The scale in meters for export resolution. Defaults to 10.
        batch_size (int): Number of images to export in a batch. Defaults to 20.
    """
    count = collection.size().getInfo()
    print(f"Total images to export: {count}")

    images_list = collection.toList(count)

    for i in range(0, count, batch_size):
        end = i + batch_size if i + batch_size < count else count
        for j in range(i, end):
            image = ee.Image(images_list.get(j))
            date = image.date().format('YYYY-MM-dd').getInfo()
            description = f"{folder_name}_{area_name}_{date}_{j}"
            file_name_prefix = f"{area_name}_{date}_{j}"
            full_folder_name = f"Images_for_detection/{folder_name}"

            export_params = {
                'description': description,
                'scale': scale,
                'region': region,
                'folder': full_folder_name,
                'fileNamePrefix': file_name_prefix,
                'fileFormat': 'GeoTIFF'
            }

            task = ee.batch.Export.image.toDrive(image, **export_params)
            task.start()
            print(
                f"Started export task {description} in folder {full_folder_name}."
                )
        time.sleep(10)
