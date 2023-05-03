<h1 align="center">

PYFOREST

</h1>

<h2 align="center">

Informing Forest Conservation Regulations in Paraguay

</h2>

<h2 align="center">

<img src="https://github.com/cp-PYFOREST/Land-Use-Plan-Simulation/blob/main/img/pyforest_hex_sticker.png" alt="Banner" width="200">

</h2>

<h2 align="center">

[Land-Use-Assesment](https://github.com/cp-PYFOREST/Land-Use-Assessment) | [Land-Use-Plan-Simulation](https://github.com/cp-PYFOREST/Land-Use-Plan-Simulation) | [PYFOREST-ML](https://github.com/cp-PYFOREST/PYFOREST-ML) | [PYFOREST-Shiny](https://github.com/cp-PYFOREST/PYFOREST-Shiny)

</h2>

# PYFOREST-ML


## Description
We predicted deforestation patterns in the undeveloped region of the Paraguayan Chaco by training a machine learning model. Utilizing a random forest algorithm from the sci-kit-learn library, we incorporate historical deforestation patterns, LUP data, and relevant environmental features such as distance to rivers, road proximity, soil type, and precipitation. This model enables more accurate predictions of deforestation patterns, accounting for variations in LUP laws and the complex interplay of socio-economic and environmental factors driving land use change.

### Usage

01-requirements
A google earth engine account is required to reproduce the data retrieval, but a possible alternative will require the user to download the data and determine the appropriate processing manually. 
The README.md of this folder provides three methods to recreate the required Python environment. The environment can be created using conda, pip, or manually installing the provided list of required packages. 

02-data-loading
The Jupyter notebook hansen-ee.ipynb  is a workflow to retrieve the Hansen et al. (2013) Global Forest Change dataset used in this analysis, which is cropped to our previously defined study boundary shapefile. 
The Hansen et al. (2013) Global Forest Change dataset  results from a time-series analysis of Landsat images in characterizing global forest extent and change, at 30 meters resolution, globally, between 2000 and 2021. 
Our analysis only uses two bands from this dataset, ‘treecover2000’ and ‘lossyear.’ Trees are defined as vegetation taller than 5 meters (m) m in height and are expressed as a percentage per output grid cell as ‘2000 Percent Tree Cover’.  ‘Forest Loss Year’ is a disaggregation of total ‘Forest Loss’ to annual time scales. ‘Forest Cover Loss’ is defined as a stand-replacement disturbance, or a change from a forest to a non-forest state, during the 2000–2021
The  Hansen et al. dataset can also be manually downloaded from Global Forest Change (storage.googleapis.com) for the desired study region.

03-preprocessing
If the dataset is manually downloaded, we provide the necessary preprocessing to mosaic and clip the tiles to the desired study boundary in mosaic-crop-hansen.ipynb. This approach is less desirable as it may introduce complications in the projected coordinate reference system when integrating with other datasets at a later time. External checks in open-source software QGIS are highly recommended.
extract-pixels-hanse.ipynb will take the 'lossyear' image from the Hansen et al. (2013) dataset and creates a .tiff file deforestation_year for each year desired. 'lossyear' and 'treecover2000' also need to be cropped so that pixels are only within the boundary of the active property of that year. 
The land use plans and additional features derived from shapefiles need to be converted into raster format. The process-shapefile.ipynb will take a shapefile with a categorical column and create a raster that encodes the categorical variable within the pixel as a numerical value. This is similar to one hot encoding but differs in that this will produce a .tiff file that can be inspected with its spatial location.

## Contributors
[Atahualpa Ayala](Atahualpa-Ayala),  [Dalila Lara](https://github.com/dalilalara),  [Alexandria Reed](https://github.com/reedalexandria),  [Guillermo Romero](https://github.com/romero61)

Any advise for common problems or issues.

## License

This project is licensed under the Apache-2.0 License - see the LICENSE.md file for details
