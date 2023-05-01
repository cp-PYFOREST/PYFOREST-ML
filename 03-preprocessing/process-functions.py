import os
import math
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon, MultiPolygon

'''mosaic_hansen_tiles is a function that takes a list of filepaths to Hansen tiles 
and combines them into a single raster. 
The function first opens each file using rasterio.open and stores the 
opened files in a list. Then, it uses the merge function from rasterio.merge 
to create the mosaicked raster. The transformation info and metadata from the
input rasters are also returned. The function ensures that the source files are 
closed after processing.'''

# This function mosaics multiple Hansen tiles into a single raster.
# It takes a list of filepaths as input and returns the mosaicked raster data,
# the transformation info, and the metadata of the input rasters.

def mosaic_hansen_tiles(filepaths):
    # Open each file in the filepaths list using rasterio
    src_files_to_mosaic = [rasterio.open(fp) for fp in filepaths]
    
    # Merge the source files using the merge function from rasterio.merge
    mosaic, out_transform = merge(src_files_to_mosaic)
    
    # Copy the metadata of the first raster in the list
    out_meta = src_files_to_mosaic[0].meta.copy()
    
    # Close the source files
    for src in src_files_to_mosaic:
        src.close()
    
    # Return the mosaicked raster data, transformation info, and metadata
    return mosaic, out_transform, out_meta



'''Returns pixels encoded with value of 1 and zeros as NaN.
if `year_pixels[year_pixels == 0] = np.nan` is removed then will return [0 1]. '''
    # Extract pixels corresponding to each year (2011-2021)
'''def extract_pixels_by_year(raster_data, start_year, end_year):
    year_data = {}
    for year in range(start_year, end_year + 1):
        year_pixels = (raster_data == year).astype(int) 
        year_data[year] = year_pixels

        # Print unique values for each year
        unique_values = np.unique(year_pixels)
        print(f"Unique values for year {year + 2000}: {unique_values}") # Add 2000 to the year to get the correct year values
    return year_data'''

'''
Returns pixels encoded with value of corresponding year(11,12,13...) and zeros as NaN.
if `year_pixels[year_pixels == 0] = np.nan` is removed then will return [0 11].'''

def extract_pixels_by_year(raster_data, start_year, end_year):
    year_data = {}
    for year in range(start_year, end_year + 1):
        year_pixels = (raster_data == year).astype(int) * year
        year_data[year] = year_pixels

        # Print unique values for each year
        unique_values = np.unique(year_pixels)
        print(f"Unique values for year {year + 2000}: {unique_values}") # Add 2000 to the year to get the correct year values
             
    return year_data

'''The extract_polygon_features function takes a Shapely Polygon or MultiPolygon object as input 
and calculates various shape features for the largest polygon in the input. 
These shape features include the number of sides, aspect ratio, area, perimeter, and compactness. 
The function returns a dictionary containing these shape features, 
which can be used for further analysis or machine learning tasks.'''

#extract features  area, perimeter, number of sides, aspect ratio, and compactness:
def extract_polygon_features(polygon):
    # Check if the input geometry is a Polygon or a MultiPolygon
    if isinstance(polygon, Polygon):
        largest_polygon = polygon
    elif isinstance(polygon, MultiPolygon):
        # For MultiPolygon, select the largest polygon by area
        largest_polygon = max(polygon.geoms, key=lambda x: x.area)
    else:
        raise ValueError("Unsupported geometry type")

    # Calculate the number of sides for the largest polygon
    num_sides = len(largest_polygon.exterior.coords) - 1

    # Calculate the bounding box and aspect ratio for the largest polygon
    minx, miny, maxx, maxy = largest_polygon.bounds
    width = maxx - minx
    height = maxy - miny
    aspect_ratio = width / height

    # Calculate the area and perimeter for the largest polygon
    area = largest_polygon.area
    perimeter = largest_polygon.length

    # Calculate the compactness for the largest polygon
    compactness = (4 * math.pi * area) / (perimeter ** 2)

    # Return the calculated features as a dictionary
    return {
        'num_sides': num_sides,
        'aspect_ratio': aspect_ratio,
        'area': area,
        'perimeter': perimeter,
        'compactness': compactness
    }


'''This function reads an input vector file (GeoPackage or Shapefile), 
optionally adds shape features, converts the attribute column to numerical values 
or sets it to a single value if provided, and prepares the metadata for creating an 
output raster file. '''

'''- input_vector: The path to the input vector file (GeoPackage or Shapefile).

- output_raster: The path to the output raster file.

- attribute: The name of the attribute column in the input vector file that should be used 
as the pixel value in the output raster.

- study_area_bounds: The bounds of the study area as a tuple (minx, miny, maxx, maxy).

- single_value (optional, default: None): If provided, all the features in the input 
vector will be encoded with this single value in the output raster. If not provided, 
the attribute column values will be used.

- resolution (optional, default: 30): The spatial resolution of the output raster in 
the same units as the study area bounds.

- dtype (optional, default: 'uint16'): The data type of the output raster pixel values.

- add_shape_features (optional, default: False): If True, shape features such as area, 
perimeter, number of sides, aspect ratio, and compactness will be added to the GeoDataFrame 
before rasterizing.'''

def vector_to_raster(input_vector, output_raster, attribute, study_area_bounds, single_value=None, resolution=30, dtype='uint16', add_shape_features=False):
    print("Entering vector_to_raster")
    # Read the input vector file (GeoPackage or Shapefile) into a GeoDataFrame
    gdf = gpd.read_file(input_vector)
    
    # Reproject the GeoDataFrame to WGS 84 coordinate reference system (EPSG:4326)
    gdf = gdf.to_crs(epsg=4326)

    # Add shape features (area, perimeter, number of sides, aspect ratio, compactness) to the GeoDataFrame if desired
    if add_shape_features:
        shape_features = gdf['geometry'].apply(extract_polygon_features)
        shape_features_df = pd.DataFrame(shape_features.tolist())
        gdf = pd.concat([gdf, shape_features_df], axis=1)

    # If single_value is None, convert the attribute column to numerical values
    # If single_value is provided, set the attribute column to the provided single_value
    if single_value is None:
        gdf[attribute], codes = gdf[attribute].astype('category').cat.codes, gdf[attribute].astype('category').cat.categories
        print(f"Category codes for {attribute}:")
        print(dict(enumerate(codes)))
    else:
        gdf[attribute] = single_value

    # Use the study area bounds to define the dimensions and transform of the output raster
    minx, miny, maxx, maxy = study_area_bounds
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    out_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    # Define the metadata for the output raster file
    out_meta = {
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'count': 1,
        'dtype': dtype,
        'crs': 'EPSG:4326',
        'transform': out_transform
    }    

    # Create an empty array with the specified dimensions, data type, and initial value (0)
    out_array = np.zeros((height, width), dtype=dtype)

    # Rasterize the input vector file into the output array
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[attribute]))
    out_array = rasterize(shapes, out_shape=(height, width), fill=0, transform=out_transform, dtype=dtype)

    # Write the output array to a new raster file
    with rasterio.open(output_raster, 'w', **out_meta) as out:
        print(f"Writing to {output_raster}")
        out.write_band(1, out_array)

'''
This function, process_columns, is designed to process multiple columns from an input vector file 
(shapefile or geopackage) and create raster files for each column. 
It takes the following parameters:

input_vector: The path to the input vector file (shapefile or geopackage).
output_dir: The path to the output directory where the raster files will be saved.
study_area_bounds: The bounds of the study area as a tuple (minx, miny, maxx, maxy).
columns: A list of column names to process. If not provided, it will process all columns 
in the input vector file except the 'geometry' column.
single_value: An optional parameter to set a single value for all features in the raster. 
If not provided, the function will use the attribute values from the input vector file.
The function first reads the input vector file using GeoPandas and, 
if no columns are specified, it gets all columns except the 'geometry' column. 
It then iterates through each column and calls the vector_to_raster function to create a raster 
file for each column. The output raster files are saved to the specified output directory with 
the column name as part of the file name.'''

def process_columns(input_vector, output_dir, study_area_bounds, columns=None, single_value=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if columns is None:
        gdf = gpd.read_file(input_vector)
        columns = ['value']

    for column in columns:
        output_raster = f"{output_dir}/{column}_raster.tif"
        vector_to_raster(input_vector, output_raster, column, study_area_bounds, single_value=single_value)

# Code is to transform polygons to lines but we have roads and rivers already as lines

""" def convert_polygons_to_lines(input_vector, output_vector):
    gdf = gpd.read_file(input_vector)
    gdf['geometry'] = gdf['geometry'].boundary
    gdf.to_file(output_vector)

convert_polygons_to_lines("rivers.gpkg", "rivers_lines.gpkg")
convert_polygons_to_lines("roads.gpkg", "roads_lines.gpkg") """

'''calculate_distance_rasters is a function that calculates the distance 
from deforestation pixels to the nearest river and road pixels. 
The function takes filepaths to deforestation, river, and road rasters, 
and output filepaths for the resulting distance rasters. 
It reads the input raster data and creates a mask for deforestation pixels. 
Then, it calculates the distance from deforestation pixels to the nearest river 
and road pixels using the Euclidean distance transform. 
Finally, the function writes the distance rasters to the specified output files.'''

# This function calculates the distance from deforestation pixels to the nearest river and road pixels.
# It takes filepaths to deforestation, river, and road rasters and output filepaths for the resulting distance rasters.

def calculate_distance_rasters(deforestation_raster, river_raster, road_raster, river_distance_output, road_distance_output):
    print("Entering calculate_distance_rasters")
    # Read deforestation raster data and metadata
    with rasterio.open(deforestation_raster) as src:
        deforestation_data = src.read(1)
        out_meta = src.meta

    # Read river raster data
    with rasterio.open(river_raster) as src:
        river_data = src.read(1)

    # Read road raster data
    with rasterio.open(road_raster) as src:
        road_data = src.read(1)

    # Create a mask for deforestation pixels (pixels with values greater than 0)
    deforestation_mask = deforestation_data > 0

    # Calculate the distance from deforestation pixels to the nearest river pixels
    river_distance = distance_transform_edt(river_data == 0) * deforestation_mask
    
    # Calculate the distance from deforestation pixels to the nearest road pixels
    road_distance = distance_transform_edt(road_data == 0) * deforestation_mask

    # Update the metadata dtype to float32
    out_meta.update({"dtype": "float32"})

    # Write the river distance raster to the specified output file
    with rasterio.open(river_distance_output, 'w', **out_meta) as dst:
        dst.write(river_distance.astype('float32'), 1)

    # Write the road distance raster to the specified output file
    with rasterio.open(road_distance_output, 'w', **out_meta) as dst:
        dst.write(road_distance.astype('float32'), 1)