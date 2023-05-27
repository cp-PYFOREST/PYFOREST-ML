#!/usr/bin/env python
# coding: utf-8

# # Import Libraries and Constants

# In[3]:


import os
import re
import sys
import numpy as np
import pandas as pd
import tempfile
import shutil
import matplotlib.pyplot as plt
import rasterio

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                             precision_score, recall_score, f1_score, roc_auc_score, 
                             precision_recall_curve, roc_curve, auc)
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, RandomizedSearchCV)
from scipy.stats import randint as sp_randint

from imblearn.ensemble import BalancedRandomForestClassifier
import joblib
from joblib import dump


# In[22]:


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)


# In[4]:


# Get the current working directory
current_dir = os.path.abspath('')

# Search for the 'constants.py' file starting from the current directory and moving up the hierarchy
project_root = current_dir
while not os.path.isfile(os.path.join(project_root, 'constants.py')):
    project_root = os.path.dirname(project_root)

# Add the project root to the Python path
sys.path.append(project_root)



# In[5]:


from constants import SERVER_PATH, OUTPUT_PATH, MASKED_RASTERS_DIR, FEATURES_DIR


# In[6]:


#output- update this for subsequent runs
output_folder = os.path.join(OUTPUT_PATH[0], 'basic_rf_model')




# # Create Stack

# In[7]:


# helper function to read tiff files
def read_tiff_image(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)


# In[8]:


# List of paths to the raster files to be used as features
feature_files = [os.path.join(FEATURES_DIR[0], 'lup_10_treemask.tif')]

# Then you can use this list of feature_files to create feature_data_arrays and feature_data_flat:
feature_data_arrays = [read_tiff_image(file_path) for file_path in feature_files]
feature_data_flat = [data_array.flatten() for data_array in feature_data_arrays]

# Path to the y_file
y_file = os.path.join(MASKED_RASTERS_DIR[0], 'deforestation11_20_masked.tif')


# In[9]:


feature_files


# In[10]:


y_file


# In[11]:


# Find the dimensions of all the raster data arrays
raster_shapes = [raster_data.shape for raster_data in feature_data_arrays]

# Check if all raster data arrays have the same dimensions
if len(set(raster_shapes)) > 1:
    print("There are mismatching dimensions:")
    for file_path, raster_shape in zip(raster_files, raster_shapes):
        print(f"File: {file_path}, Shape: {raster_shape}")
else:
    print("All raster data arrays have the same dimensions.")
    # Check the dimensions of all the raster data arrays
    for i, data_array in enumerate(feature_data_arrays):
        print(f"Raster {i}: {data_array.shape}")


# # Stack and Flatten Data

# In[12]:


# NoData Value
no_data_value = -1

# Stack the flattened raster data
X_flat = np.column_stack(feature_data_flat)

# Use the y_file obtained from the find_deforestation_file function
y = read_tiff_image(y_file).flatten()

# Remove rows with NoData values
'''checks each row in X_flat and creates a boolean array (valid_rows_X) that has the same number of elements 
as the number of rows in X_flat. Each element in valid_rows_X is True if there is no NoData value in 
the corresponding row of X_flat and False otherwise.'''
valid_rows_X = ~(X_flat == no_data_value).any(axis=1)

'''checks each element in the y array and creates a boolean array (valid_rows_y) that has the same number of 
elements as y. Each element in valid_rows_y is True if the corresponding element in y is not 
equal to the NoData value and False otherwise.'''
valid_rows_y = y != no_data_value

'''checks each element in the y array and creates a boolean array (valid_rows_y) 
that has the same number of elements as y. Each element in valid_rows_y is True if the corresponding element 
in y is not equal to the NoData value and False otherwise.'''
valid_rows = valid_rows_X & valid_rows_y

'''creates a new array X_cleaned by selecting only the rows in X_flat that 
correspond to the True elements in valid_rows.'''
X_cleaned = X_flat[valid_rows]

'''creates a new array y_cleaned by selecting only the elements in y that correspond 
to the True elements in valid_rows.'''
y_cleaned = y[valid_rows]
 


# To ensure your data cleaning steps have been applied correctly, you can check the following:
# 
# **NoData values have been removed:** You should confirm that there are no NoData values in your cleaned data. This can be done by asserting that there are no occurrences of no_data_value in X_cleaned and y_cleaned.

# In[13]:


assert not (X_cleaned == no_data_value).any()
assert not (y_cleaned == no_data_value).any()


# These assertions will throw an error if there is a NoData value in X_cleaned or y_cleaned
# 
# **Dimensions are correct:** The shapes of X_cleaned and y_cleaned should match along the row dimension (the first dimension for 2D array X_cleaned and the only dimension for 1D array y_cleaned).

# In[14]:


print("Shape of X_cleaned:", X_cleaned.shape)
print("Shape of y_cleaned:", y_cleaned.shape)


# Make sure the number of rows in X_cleaned equals the number of elements in y_cleaned.
# 
# **Confirm that the valid rows have been correctly identified:** You can do this by checking that the number of True values in valid_rows equals the number of rows in X_cleaned (or the number of elements in y_cleaned).

# In[15]:


assert valid_rows.sum() == X_cleaned.shape[0]


# # Split the data into training and testing sets
# 

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.3, random_state=42, stratify=y_cleaned)


# In[17]:


print("Shape of y_test:", y_test.shape)


# # Class Balance Check

# In[18]:


# Create pandas Series from your labels
y_train_series = pd.Series(y_train)
y_test_series = pd.Series(y_test)
y_cleaned_series = pd.Series(y_cleaned)

# Print balance of classes in training, testing, and whole dataset
print("Training data balance:\n", y_train_series.value_counts(normalize=True))
print("Testing data balance:\n", y_test_series.value_counts(normalize=True))
print("Whole dataset balance:\n", y_cleaned_series.value_counts(normalize=True))


# The balance of your dataset seems to be roughly the same in both the training and testing sets, with about 80.6% of the instances belonging to class 0 (no deforestation) and 19.3% to class 1 (deforestation). This shows that the classes are quite imbalanced. Machine learning algorithms, including Random Forest, may have a bias towards the majority class in such situations, which could be one of the reasons why your model is not performing well on the minority class.



# # Random Forest model using BalancedRandomForestClassifier:

# In[20]:


brfc = BalancedRandomForestClassifier(random_state=42, class_weight= 'balanced', sampling_strategy='not majority')
'''# Define new hyperparameter options
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20, 30],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}'''

# Define a basic parameter grid
param_grid = {
    'n_estimators': [40, 50, 60],      # number of trees in the forest
    'max_depth': [150, 200, 250],        # maximum depth of the tree
    'min_samples_split': [4, 6, 8],    # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 3],     # Minimum number of samples required to split a node
    'bootstrap': [False]      # Method of selecting samples for training each tree
}



# Set scoring metrics
scoring = {
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator = brfc,
    param_distributions=param_grid,
    scoring=scoring,
    refit='f1',  # because we are interested in maximizing f1_score
    cv=5,
    n_jobs=30,
    verbose=0,
    n_iter=15,  # number of parameter settings that are sampled
    random_state=42  # for reproducibility
)

# Create the GridSearchCV object
'''random_search = GridSearchCV(
    estimator = brfc,
    param_grid=param_grid,
    scoring=scoring,
    refit='f1',  # because we are interested in maximizing f1_score
    cv=5,
    n_jobs=19,
    verbose=0
)'''



# In[23]:


# Fit GridSearch to the BalancedRandomForestClassifier data
#random_search.fit(X_train, y_train)
random_search.fit(X_train, y_train)

#Fitting 5 folds for each of 12 candidates, totalling 60 fits




# In[2]:


random_search.score


# In[3]:


# Get the best parameters and the corresponding score
best_params = random_search.best_params_
best_score = random_search.best_score_

best_estimator = random_search.best_estimator_

cv_results = random_search.cv_results_

cv_results_df = pd.DataFrame(random_search.cv_results_)

scorer = random_search.scorer_

refit_time = random_search.refit_time_


# In[ ]:


print("Best parameters:", best_params)
print("Best cross-validation score:", best_score)
print("Best estimator:", best_estimator)
print("CV Results:",cv_results_df)
print("Scorer function:", scorer)
print("Refit time (seconds):", refit_time)



# # Model evaluation performance  metrics 
# e.g., confusion matrix, classification report, accuracy, F1-score, etc.

# In[ ]:


best_model = random_search.best_estimator_


# In[ ]:


# Predictions for test data
y_pred = best_model.predict(X_test)


# # TESTING DATA Classificatin Report-Confusion Matrix

# In[ ]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate F1-score (use 'weighted' or 'macro' depending on your problem)
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1-score:", f1)

# Print classification report
report = classification_report(y_test, y_pred)
print("Classification report:\n", report)

# # TRAINING DATA Classificatin Report-Confusion Matrix

# In[ ]:


# Predictions for train data
y_pred_train = best_model.predict(X_train)


# In[ ]:


# Confusion matrix and classification report for train data
train_cm = confusion_matrix(y_train, y_pred_train)
train_cr = classification_report(y_train, y_pred_train)
print("Training confusion matrix:")
print(train_cm)
print("Training classification report:")
print(train_cr)


# In[ ]:


# Calculate feature importances and the standard deviation for those importances
importances = best_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)


 # list of feature names corresponding to the input bands of your raster stack
feature_names = ['LUP_10', 'Precipitation', 'River','Road', 'Soil']
# Create a sorted list of tuples containing feature names and their importances:
sorted_features = sorted(zip(feature_names, importances, std), key=lambda x: x[1], reverse=True)






# # Probabilities for deforestation

# In[ ]:


y_proba_curve = best_model.predict_proba(X_test)[:, 1]


# In[ ]:


print("Shape of y_proba_curve:", y_proba_curve.shape)


# In[ ]:


# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_proba_curve)


print(f"Area under Precision-Recall curve: {auc(recall, precision)}")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba_curve)

print(f"Area under ROC curve: {auc(fpr, tpr)}")


# In[ ]:


# Predict probabilities for deforestation events
y_proba = best_model.predict_proba(X_cleaned)[:, 1]


# In[ ]:


# Predicts the 
# Create a probability raster by filling in the valid pixel values
prob_raster = np.full(y.shape, no_data_value, dtype=np.float32)
prob_raster[valid_rows] = y_proba
prob_raster = prob_raster.reshape(feature_data_arrays[0].shape)


# In[ ]:


print(y_proba.shape)
try:
    joblib.dump(random_search, 'random_search_results.pkl')
    joblib.dump(best_params, 'best_params.pkl')
    joblib.dump(best_score, 'best_score.pkl')
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(cv_results, 'cv_results.pkl')
    joblib.dump(cv_results_df, 'cv_results_df.pkl')
    joblib.dump(scorer, 'scorer.pkl')
    joblib.dump(refit_time, 'refit_time.pkl')
    joblib.dump(report, 'report.pkl')
except Exception as e:
    print(f"An error occurred: {e}")

# In[ ]:


# Save the probability raster as a GeoTIFF file
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file = os.path.join(output_folder, "deforestation_prob_brfc.tiff")

with rasterio.open(y_file) as src:
    profile = src.profile
    profile.update(dtype=rasterio.float32, count=1)

prob_raster_reshaped = prob_raster.reshape((1, prob_raster.shape[0], prob_raster.shape[1]))

with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write_band(1, prob_raster_reshaped[0])


# Report
model_report = f'''Balanced Random Forest Classifier Model Report

# Summary 
The Balanced Random Forest Classifier performed reasonably well on this task, 
with an accuracy of  {accuracy} and an F1-score of {f1}. 
However, there is room for improvement, particularly in the precision and recall for class 1. 
Future work could explore different models, additional feature engineering, or further hyperparameter tuning to improve performance.

# Model Selection
We chose to use a Balanced Random Forest Classifier for this task. 
This model is an ensemble method that combines the predictions of several base estimators 
built with a given learning algorithm in order to improve generalizability and robustness over a single estimator. 
It also handles imbalanced classes, which is a common problem in many machine learning tasks.

Hyperparameter Tuning
We used RandomizedSearchCV for hyperparameter tuning. 
This method performs a random search on hyperparameters, which is more efficient than an exhaustive search like GridSearchCV.

The hyperparameters we tuned were:

'n_estimators': The number of trees in the forest.
'max_depth': The maximum depth of the tree.
'min_samples_split': The minimum number of samples required to split a node.
'min_samples_leaf': The minimum number of samples required at a leaf node.
'bootstrap': Whether bootstrap samples are used when building trees.

{param_grid}

# Model Performance
The best parameters found by RandomizedSearchCV were:

Best parameters:, {best_params}



With these parameters, the model achieved the following performance metrics:
Best cross-validation score: {best_score}
Best model:, {best_estimator}
Scorer function:, {scorer}
Refit time (seconds): {refit_time}
Accuracy:, {accuracy}
F1-score: {f1}
# Testing Data
Classification report:
 {report}

#  TRAINING DATA Classificatin Report-Confusion Matrix

Training confusion matrix:
{train_cm}
Training classification report:
{train_cr}


This indicates that the model correctly classified [1,1] instances of class 0 and [2,2] instances of class 1, 
while misclassifying [1,2] instances of class 0 and [2,1] instances of class 1.

CV Results:
{cv_results_df}

'''
# Write the report to a Quarto markdown file
with open('model_report.qmd', 'w') as f:
    f.write(model_report)

# # Tuning Strategies

# In[ ]:


# Randomized Search
# Set the range of values for each hyperparameter
'''param_dist = {
    "n_estimators": sp_randint(100, 300),
    'criterion': ['gini',],
    'max_features': ['sqrt', None],
    "max_depth": sp_randint(1, 20),
    "min_samples_split": sp_randint(2, 11),
    "min_samples_leaf": sp_randint(1, 11),
    "bootstrap": [True],
    'class_weight': ['balanced']
}

# Instantiate the RandomForestClassifier
clf = RandomForestClassifier(random_state=0)

# Set up the RandomizedSearchCV
random_search = RandomizedSearchCV(
    clf, param_distributions=param_dist, n_iter=20, cv=5, random_state=0, n_jobs=19
)'''

