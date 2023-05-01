

'''The process_year function takes a single argument, year, 
which represents the deforestation year being processed. 
The function sets the input deforestation raster file name 
based on the year and sets the output file names for the river and road 
distance rasters. It prints a message to show progress for the current year, 
and then calls the calculate_distance_rasters function with the 
appropriate input and output file names. 
The calculate_distance_rasters function is responsible for 
computing the distance rasters for rivers and roads.'''


def process_year(year):
    # Set the input deforestation raster file name based on the year
    deforestation_raster = f"/Users/romero61/../../capstone/pyforest/ml_data/deforestation_by_year/deforestation_{year}.tif"
    print(deforestation_raster)
    # Set the output file names for the river and road distance rasters
    river_distance_output = f"output_rasters/deforestation_river_distance_{year}.tif" 
    road_distance_output = f"output_rasters/deforestation_road_distance_{year}.tif" 
    print(river_distance_output)
    print(road_distance_output)
    # Print a message to show progress for the current year
    print(f"Processing deforestation data for year {year + 2000}")

    # Call the calculate_distance_rasters function
    #with the appropriate input and output file names
    calculate_distance_rasters(
        deforestation_raster,
        "/Users/romero61/github/PYFOREST-ML/src/data_loading/output_rasters/rivers_value_raster.tif",
        "/Users/romero61/github/PYFOREST-ML/src/data_loading/output_rasters/road_value_raster.tif",
        river_distance_output,
        road_distance_output
    )