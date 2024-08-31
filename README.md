# Sentinel Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/)

## Introduction

**Sentinel Segmentation** is a Python package specifically designed for segmenting coastal infrastructure, such as harbours, jetties, and resorts, from Sentinel-2 remote sensing images using deep learning models with PyTorch. This package provides tools for data processing, model training, evaluation, and visualization of results, including interactive map visualizations using Folium. It simplifies the process for researchers and developers to build, fine-tune, and deploy segmentation models tailored to the unique challenges of analyzing coastal environments.

## Documentation

For detailed documentation, please visit the [documentation](https://github.com/edsml-yc4523/irp_updated/blob/main/_build/html/index.html).

## Dataset Description

The dataset is organized into the following structure:

```
dataset/
│
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

- **Source**: The dataset is sourced from [Sentinel-2](https://sentinel.esa.int/en/web/sentinel/missions/sentinel-2/overview) satellite imagery.
- **Channels**: The dataset integrates several types of data into a six-channel `'.npy'` file:
  - **RGB**: Red, Green, Blue bands, typically used for visual interpretation.
  - **NIR**: Near-Infrared band, useful for vegetation analysis.
  - **NDWI**: Normalized Difference Water Index, helps in identifying water bodies.
  - **NDVI**: Normalized Difference Vegetation Index, useful for assessing vegetation health. 
  
  You can quickly view the dataset through this [Google Drive link](https://drive.google.com/drive/folders/16Nxdb45-A6DEh5_cKQVO2OuXc7msD6xG?usp=drive_link).

## Code Structure

```
project_root/
│
├── exploration/
│   ├── change_detection.ipynb
│   ├── Image_acquisition.ipynb
│   ├── DeepLabV3Plus_Resnet50.ipynb
│   ├── Unet_Resnet34.ipynb
│   ├── Unet_Resnet50.ipynb
│   ├── UnetPlusPlus_Resnet50.ipynb
│
├── notebook/
│   ├── download.ipynb
│   ├── mapping.ipynb
│   ├── model_inference.ipynb
│   ├── UnetPlusPlus_ResNet_50.ipynb
│
├── sentinel_segmentation/
│   ├── __init__.py
│   ├── data_loader/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   ├── download/
│   │   ├── __init__.py
│   │   └── download_files.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── Deeplabv3plus_resnet50.py
│   │   ├── unet_resnet34.py
│   │   ├── unet_resnet50.py
│   │   ├── unetplusplus_resnet50.py
│   ├── loss_function/
│   │   ├── __init__.py
│   │   └── losses.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── show_prediction.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   ├── image_analysis.py
│   │   ├── map_visualization.py
│   │   └── load_image.py
│   ├── image_acquisition/
│   │   ├── __init__.py
│   │   ├── earth_engine.py
│   ├── image_processing/
│   │   ├── __init__.py
│   │   ├── tiff_conversion.py
│   │   ├── tiff_to_jpg.py
│   │   ├── image_resize.py
│   │   ├── image_fusion.py
│   │   └── data_split.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   ├── test_image_processing.py
│   ├── test_visualization.py
│   ├── test_loss_function.py
│
├── folium_map
│   ├── final_result.xlsx
│   ├── original_result.xlsx
│   ├── map_with_timeline.html
│ 
├── example
│   ├── change_detection
│   │   ├── jpg
│   │   ├── npy
│   ├── map
│   │   ├── demp_final_result.xlsx
│   │   ├── demo_original_result.xlsx
│   │   ├── demo_map_with_timeline.html
│
├── config
├── setup.py
├── LICENCE
└── README.md
```

## Installation

### 1. Prerequisites

Ensure you have Python 3.6 or higher installed.

### 2. Using `pip`

You can install the package and its dependencies using the `setup.py`:

```sh
git clone https://github.com/edsml-yc4523/irp_updated.git
cd irp_updated
pip install .
```

## Quick Start

Here's how you can quickly get started with training a model using `sentinel_segmentation`:

### 1. Prepare your dataset

Ensure your dataset is organized as described above. You can also use my dataset to get started quickly. Use the following code to download the Sentinel-2 dataset of this project:

```python
from sentinel_segmentation.download import authenticate_gdrive, download_folder

# Paths to credentials and token
credentials_path = "../config/credentials.json"
token_path = "../config/token.json"

# Google Drive folder ID for the dataset
folder_id = '16Nxdb45-A6DEh5_cKQVO2OuXc7msD6xG'  
destination = './downloads/dataset'  

# Authenticate and download the dataset
service = authenticate_gdrive(credentials_path, token_path)
download_folder(service, folder_id, destination)
```

### 2. Model training and evaluation

This section outlines the steps to train and evaluate a segmentation model using the `sentinel_segmentation` package. If you don’t want to train a model from scratch, you can download a pre-trained model using this code：

```python
from sentinel_segmentation.download import authenticate_gdrive, download_folder

# Paths to credentials and token
credentials_path = "../config/credentials.json"
token_path = "../config/token.json"

# Google Drive folder ID for the pretrained models
folder_id = '1_rdSK2WrTDIja-VAJmi3BHQw3KPhNmhO'  
destination = './downloads'  

# Authenticate and download the pretrained models
service = authenticate_gdrive(credentials_path, token_path)
download_folder(service, folder_id, destination)
```

For a detailed implementation, refer to the [UnetPlusPlus_ResNet50.ipynb](https://github.com/edsml-yc4523/irp_updated/blob/main/notebook/UnetPlusPlus_ResNet_50.ipynb).

### 3. Model inference

This notebook shows how to use the `sentinel_segmentation` to for change detection. For a detailed implementation, refer to the [model_inference.ipynb](https://github.com/edsml-yc4523/irp_updated/blob/main/notebook/model_inference.ipynb). The data used in this notebook can be seen in the ['example/change_detection'](https://github.com/edsml-yc4523/irp_updated/blob/main/example/change_detection) folder.

### 4. Folium visualization

This notebook shows how to use the `sentinel_segmentation` to extract information from the mask and make the map through folium. For a detailed implementation, refer to the [mapping.ipynb](https://github.com/edsml-yc4523/irp_updated/blob/main/notebook/model_inference.ipynb). The excel files used in this notebook and generated map can be seen in the ['example/map'](https://github.com/edsml-yc4523/irp_updated/blob/main/example/map) folder.

If you want to see more map visualization results, you can refer to [folium_map](https://github.com/edsml-yc4523/irp_updated/blob/main/folium_map) folder.

## Acknowledgements

I would like to thank my supervisors, **Dr. Yves Plancherel** and **Ms. Myriam Prasow-Emond**, for their invaluable guidance and support throughout this project. I am fortunate to have had the opportunity to work under their supervision, and their mentorship has played a significant role in my academic and professional growth.
