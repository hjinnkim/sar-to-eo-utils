from setuptools import setup, find_packages

setup(
    name='sar-to-eo-utils',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'rasterio',
        'numpy',
        'tqdm',
        'matplotlib',
        
    ],
)