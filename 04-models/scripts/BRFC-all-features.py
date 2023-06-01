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
output_folder = os.path.join(OUTPUT_PATH[0], 'brfc-features-model')




# # Create Stack

# In[7]:


# helper function to read tiff files
def read_tiff_image(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)


# In[8]:


# List of paths to the raster files to be used as features
feature_files = [os.path.join(FEATURES_DIR[0], file_name) for file_name in os.listdir(FEATURES_DIR[0])]

# Then you can use this list of feature_files to create feature_data_arrays and feature_data_flat:
feature_data_arrays = [read_tiff_image(file_path) for file_path in feature_files]
feature_data_flat = [data_array.flatten() for data_array in feature_data_arrays]

# Path to the y_file
y_file = os.path.join(MASKED_RASTERS_DIR[0], 'deforestation11_20_masked.tif')



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
 



# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.9, random_state=42, stratify=y_cleaned)






# # Random Forest model using BalancedRandomForestClassifier:

# In[20]:


brfc = BalancedRandomForestClassifier(random_state=42, class_weight= 'balanced', sampling_strategy='not majority')

# Define the parameter search space
search_space = {
    'n_estimators': Integer(10, 200),  # number of trees in the forest
    'max_depth': Integer(1, 200),  # maximum depth of the tree
    'min_samples_split': Integer(2, 20),  # minimum number of samples required to split an internal node
    'min_samples_leaf': Integer(1, 20),  # minimum number of samples required to be at a leaf node
    'bootstrap': Categorical([True, False])  # whether bootstrap samples are used when building trees
}




# Create the BayesSearchCV object
bayes_search = BayesSearchCV(
    estimator=brfc,
    search_spaces=search_space,
    n_iter=10,
    cv=5,
    scoring='f1',
    refit=True,  # refit an estimator using the best found parameters on the whole dataset
    n_jobs=30,
    verbose=3
)

# In[23]:

# Perform Bayesian optimization
bayes_search.fit(X_train, y_train)

joblib.dump(bayes_search, 'bayes_search_results.pkl')


# In[2]:


bayes_search.score


# In[3]:


# Get the best parameters and the corresponding score
best_params = bayes_search.best_params_
best_score = bayes_search.best_score_

best_estimator = bayes_search.best_estimator_

cv_results = bayes_search.cv_results_

cv_results_df = pd.DataFrame(bayes_search.cv_results_)

scorer = bayes_search.scorer_

refit_time = bayes_search.refit_time_


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


best_model = bayes_search.best_estimator_


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
feature_names =  [ 'SOIL', 'ROAD', 'LUP_10', 'PRECIPITATION', 'RIVER', 'CITIES', 'PORTS' ]
# Create a sorted list of tuples containing feature names and their importances:
sorted_features = sorted(zip(feature_names, importances, std), key=lambda x: x[1], reverse=True)

print(sorted_features)




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
    joblib.dump(bayes_search, 'bayes_search_results.pkl')
    joblib.dump(best_params, 'best_params.pkl')
    joblib.dump(best_score, 'best_score.pkl')
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(cv_results, 'cv_results.pkl')
    joblib.dump(cv_results_df, 'cv_results_df.pkl')
    joblib.dump(scorer, 'scorer.pkl')
    joblib.dump(refit_time, 'refit_time.pkl')
    joblib.dump(sorted_features, 'sorted_features.pkl')
    joblib.dump(report, 'report.pkl')
    
except Exception as e:
    print(f"An error occurred: {e}")

# In[ ]:


# Save the probability raster as a GeoTIFF file
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file = os.path.join(output_folder, "brfc-features-tuning.tiff")

with rasterio.open(y_file) as src:
    profile = src.profile
    profile.update(dtype=rasterio.float32, count=1)

prob_raster_reshaped = prob_raster.reshape((1, prob_raster.shape[0], prob_raster.shape[1]))

with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write_band(1, prob_raster_reshaped[0])


# Report
model_report = f'''

Balanced Random Forest Classifier Model Report

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


This indicates that the model correctly classified [1,1] instances of class 0 
and [2,2] instances of class 1, 

while misclassifying [1,2] instances of class 0 and [2,1] instances of class 1.

# Feature Importance

{sorted_features}

CV Results:
{cv_results_df}

'''
# Write the report to a Quarto markdown file
with open('brfc-model_tuning-report.qmd', 'w') as f:
    f.write(model_report)


