# Place all your constants here
import os

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)
SERVER_PATH = os.path.join('/Users', 'romero61', '..', '..', 'capstone', 'pyforest')

# The larger study area to use for earth engine this study uses the western region of paraguay
STUDY_BOUNDARY_PATH = os.path.join(SERVER_PATH,'ml_data', 'study_boundary', 'study_boundary.shp')

# raw earth engine data
HANSEN_TREECOVER_FILEPATH = [
    os.path.join(SERVER_PATH, 'ml_data', 'raw_hansen', 'treecover2000.tif')
]
# raw earth engine data /Users/romero61/../../capstone/pyforest/ml_data/raw_hansen
HANSEN_LOSSYEAR_FILEPATHS = [
    os.path.join(SERVER_PATH, 'ml_data', 'raw_hansen', 'lossyear.tiff')
]

# loss year disaggragated by year and into binary for year of deforestation
DEFORESTATION_YEAR_PATHS = [
    os.path.join(SERVER_PATH, 'ml_data', 'raw_hansen', 'deforestation_by_year_binary' f'deforestation_{i}.tif')
    for i in range(11, 22)
]

# INFONA DATA
PROPERTIES_SHAPEFILE_PATHS = [
    os.path.join(SERVER_PATH, 'ml_data', 'active_inactive_subsets', f'active_inactive_{i}.gpkg')
    for i in range(11, 22)
]

# INFONA DATA
LUP_YEAR_PATHS = [
    os.path.join(SERVER_PATH, 'ml_data', 'lup_subsets',  f'lup_{i}.gpkg')
    for i in range(11, 22)
]

# Where to save outputs
OUTPUT_PATH = [
    os.path.join(SERVER_PATH, 'ml_data', 'output')
]

# Base feature stacks
MASKED_RASTERS_DIR = [
    os.path.join(OUTPUT_PATH[0], 'masked_rasters')
]
