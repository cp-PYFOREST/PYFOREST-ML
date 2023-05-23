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
# raw earth engine data 
HANSEN_LOSSYEAR_FILEPATHS = [
    os.path.join(SERVER_PATH, 'ml_data', 'raw_hansen', 'lossyear.tiff')
]

# loss year in binary for year of deforestation 2011-2020
DEFORESTATION_1120_PATH = os.path.join(SERVER_PATH, 'ml_data', 'output', 'deforestation-cumulative_0110', 'deforestation11_20.tif')


# treecover in 2010
TREECOVER_2010 = os.path.join(SERVER_PATH, 'ml_data', 'output', 'deforestation-cumulative_0110', 'treecover2010.tif')

# INFONA DATA
PROPERTIES_SHAPEFILE_PATHS = [
    os.path.join(SERVER_PATH, 'ml_data', 'active_inactive_subsets', f'active_inactive_{i}.gpkg')
    for i in range(11, 22)
]

# INFONA DATA
LUP_YEAR = [
    os.path.join(SERVER_PATH, 'ml_data', 'lup_subsets',  'lup_10.gpkg')
]

# River Data 
ROAD_PATH = [
    os.path.join(SERVER_PATH, 'ml_data', 'features', 'dissolved_road', 'dissolved_road.gpkg')
]

# Road Data
RIVER_PATH = [
    os.path.join(SERVER_PATH, 'ml_data', 'features', 'river_buffer', 'river_buffer.gpkg')
]

# Soil Data
SOIL_RASTER = os.path.join(SERVER_PATH, 'ml_data','features', 'soil',  'merged_soil.tif')


# Where to save outputs
OUTPUT_PATH = [
    os.path.join(SERVER_PATH, 'ml_data', 'output')
]

# Rasters Not yet masked and cropped w/ nodata value of -1
LUP_LUT_RASTER = os.path.join(SERVER_PATH, 'ml_data','output', 'processed_rasters', 'land_use_type', 'lup_10_land_use_type_raster.tif')

ROAD_DISTANCE_RASTER =  os.path.join(SERVER_PATH, 'ml_data','output', 'processed_rasters', 'road_raster', 'road_raster.tif')

RIVER_DISTANCE_RASTER =  os.path.join(SERVER_PATH, 'ml_data','output', 'processed_rasters', 'river_raster', 'river_raster.tif')

DEFORESTATION_0110_PATH = os.path.join(SERVER_PATH, 'ml_data', 'output', 'deforestation-cumulative_0110', 'deforestation1_10.tif')

TREECOVER_PERCENTAGE_10 =  os.path.join(SERVER_PATH, 'ml_data','output', 'tree_cover_10_percent_and_above_00', 'tree_cover_10_percent_and_above_00.tif')

# The folder masked rasters contains the required files for the machine learning model. deforestation11_20_masked.tif is alway the 'y' target variable

# Base feature stacks
MASKED_RASTERS_DIR = [
    os.path.join(OUTPUT_PATH[0], 'masked_rasters')
    ]