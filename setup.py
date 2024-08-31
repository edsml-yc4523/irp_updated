from setuptools import setup, find_packages

setup(
    name="sentinel_segmentation",
    version="0.1.0",
    description="A package for Sentinel-2 remote sensing images \
                segmentation using PyTorch",
    author="Yibing Chen",
    author_email="yc4523@ic.ac.uk",
    url="https://github.com/edsml-yc4523/irp_updated.git",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'torchvision',
        'segmentation-models-pytorch',
        'scikit-learn',
        'matplotlib',
        'Pillow',
        'opencv-python',
        'gdal',
        'folium',
        'google-auth-oauthlib',
        'google-auth',
        'google-api-python-client',
        'albumentations',
        'earthengine-api',
        'rasterio',
        'pytest',
        'openpyxl'
    ],
    python_requires='>=3.6',
)
