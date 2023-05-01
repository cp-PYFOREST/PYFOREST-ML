# Place all your constants here
import os

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)
SERVER_PATH = os.path.join('/Users', 'romero61', '..', '..', 'capstone', 'pyforest')


SHAPEFILE_PATH = os.path.join(PROJECT_PATH,'src', 'data_loading', 'study_boundary', 'study_boundary.shp')

HANSEN_FILEPATHS = [
    os.path.join(SERVER_PATH, 'ml_data', 'hansen', f'Hansen_GFC-2021-v1.9_lossyear_{lat}{lon}.tif')
    for lat, lon in [('20S', '_070W'), ('20S', '_060W'), ('10S', '_070W'), ('10S', '_060W')]
]

PROPERTIES_SHAPEFILE_PATHS = [
    os.path.join(SERVER_PATH, 'ml_data', 'active_inactive_subsets', f'active_inactive_{i}.gpkg')
    for i in range(11, 22)
]

DEFORESTATION_YEAR_PATHS = [
    os.path.join(SERVER_PATH, 'ml_data', 'hansen', 'deforestation_by_year' f'deforestation_{i}.tif')
    for i in range(11, 22)
]