

# Instruction for required environment and package
Though it is possible to recreate the environment from the .yml file we by running

`conda env create -f environment.yml`
or
`pip install -r environment.txt`
 and then `conda activate pyforest`
 we've found that  creating an environment in this way can fail. 
 
 What follows are the steps to create the environment from scratch. This process allows for flexibility to upgrade to newer versions of the packages. New packages run the risk of breaking functionality but we find those issues are easier to overcome than having no environment at all. 

If you'd like to delete an environment, you can run `conda remove env -n pyforest`

1. `conda create -n pyforest python=3.11`
2. `conda activate pyforest`
3. `conda install numpy` 
4. `conda install matplotlib`
5. `conda install pandas`
6. `conda install --channel conda-forge geopandas`
or `pip install geopandas` 
8. `conda install -c anaconda scikit-learn`
8. `conda install rasterio=1.3.6`
9. `conda install jupyter ipykernel`
 or `python -m ipykernel install --user --name pyforest --display-name "pyforest`
10. `conda update jupyter` This might be optional dependig on which version conda has in its repository
11. `conda install seaborn -c conda-forge1
or `pip install seaborn`
or `conda install seaborn` 
12. `conda install -c conda-forge earthengine-api`
13. `conda install -c conda-forge google-cloud-sdk` This worked on the server but wouldn't run on my local though it has in the past. Earth Engine and geemap still worked fine afterwards.
14. `pip install geemap`
17. `pip install imbalanced-learn`
15. `pip install xarray` - optional
16. `pip install rioxarray` - optional



