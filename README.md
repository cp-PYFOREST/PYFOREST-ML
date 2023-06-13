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

# Documentation
 For more detailed information about our project, including our methodologies, data sources, and technical specifications, please refer to our [technical documentation](https://bren.ucsb.edu/projects/informing-forest-conservation-regulations-paraguay).
  
## Table of Contents
- [Description](#description)
- [Methods](#methods)
- [Machine Learning Approach](#machine-learning-approach)
- [Datasets](#datasets)
- [Training and Validation](#training-and-validation)
- [Prediction](#prediction)
- [Results](#results)
- [Training & Validation](#training-&-validation)
- [Predictions](#predictions)
- [Visual Interpretation](#visual-interpretation)
- [Data Information](#data-information)
- [Contributors](#contributors)
- [License](#license)

## Description
To determine how policy changes will affect deforestation across the Paraguayan Chaco in the next ten years, we first established the pattern of the actual occurrence of deforestation and its relationship to LUPs in the developed region of the Paraguayan Chaco. The mock properties and simulated LUPs of Objective 2 provide the underlying framework for predicting deforestation under different alternative forest laws over the next decade. The simulated LUPs provide a means to determine potential deforestation patterns in areas without registered LUPs and examine how these patterns may be influenced by changes in forest laws.

# Methods

## Machine Learning Approach

The application of a Random Forest machine learning algorithm was decided on to predict deforestation patterns in the Paraguayan Chaco over the next decade. The choice of this algorithm is justified by several of its characteristics that are particularly relevant to the task at hand. Random Forest is capable of managing large datasets with a multitude of variables. This is a critical feature given the complex nature of deforestation, which is influenced by a wide array of factors. The algorithm is also robust to overfitting, a quality that enhances the reliability of its predictions, even in the presence of noise and outliers in the data. This robustness is achieved by averaging the results of many individual decision trees, each trained on a different subset of the data. Another advantage of Random Forest is its interpretability. It provides measures of variable importance, which can be instrumental in identifying the factors that exert the most influence on deforestation. Furthermore, Random Forest is adept at modeling complex, non-linear relationships between variables. Given the intricate interplay of factors contributing to deforestation, this ability to capture non-linear relationships is likely to be beneficial. Finally, Random Forest is known for its strong performance in predictive tasks. It has been demonstrated to provide accurate and reliable predictions across a range of contexts.

## Datasets

The datasets utilized in this study are integral to the analysis and the subsequent predictions of deforestation patterns. The first dataset comprises LUPs that were active up until 2010. This dataset serves as a cornerstone of our analysis, providing a historical context for land use and its potential impact on deforestation. The second dataset of significance is the Hansen dataset of forest cover change (Hansen, 2013). We utilized the 'treecover2000' variable, where the pixels with greater than an amount 10% of tree cover were selected and then masked  with the deforestation that occurred from 2000 to 2010, this manipulation of the dataset establishes what we refer to as 'treecover2010'; This provides a binary snapshot of the state of tree cover at the end of the first decade of the 21st century. The binary ‘treecover2010’ was used as a mask to all other datasets so that the variables reflect where there is the possibility of deforestation. To capture the geographical and environmental characteristics of the region, the features of distance to roads and rivers (Appendix A, Table 1), travel time to ports and cities (Nelson, 2019), soil types (ISRIC - World Soil Information, 2023), and average annual precipitation from 2000 to 2010 (Funk et al., 2015) were included. These datasets provide important context for understanding the physical and environmental factors influencing deforestation patterns. The target variable of our model is the deforestation data from 2011 to 2020, sourced from the Hansen dataset. This dataset provides a record of deforestation events over this decade, which we aim to predict based on our selected features.

## Training and Validation

For all model runs, we utilized the Balanced Random Forest Classifier from the imblearn library. This model is an ensemble method that combines the predictions of several base estimators built with a given learning algorithm to improve generalizability and robustness over a single estimator. We employed RandomizedSearchCV for hyperparameter tuning, which performs a random search on hyperparameters, offering more efficiency than an exhaustive search like GridSearchCV.
The training and validation of our machine learning model is a two-step process designed to establish the predictive power of our features and optimize the model's performance. In the first step, we train the model using only the LUPs active up to and including 2010 as the feature to predict the target variable, the deforestation data from 2011 to 2020. This initial model allows us to assess the predictive power of the LUPs and establish a baseline for further analysis.
Following this, we proceed to the second step, incorporating all features into the model. These features include the LUPs of the developed region, geographical and environmental variables such as distance to roads and rivers, travel time to ports and cities, soil types, and average annual precipitation from 2000 to 2010. By training the model with this comprehensive set of features, we aim to enhance its predictive accuracy and reliability.
After training and validating the model with all the features, we identify the best-performing model based on its predictive performance. This model is then saved for future use, ensuring that the insights gained from the training and validation process can be effectively applied to predict future deforestation patterns in the Paraguayan Chaco under different scenarios of forest law changes.

## Prediction
The predictive model, trained and validated on data from the developed region, is now applied to the undeveloped region of the Paraguayan Chaco. The first step in this process involves using the simulated LUPs for the undeveloped region. Although deforestation data is not used as a target in this case, the LUPs serve as an essential feature, providing a basis for the initial deforestation predictions.
Subsequently, we generate the same set of features for the undeveloped region as we did for the developed region. These features include the simulations, reflecting current and alternative laws, and geographical and environmental variables such as distance to roads and rivers, travel time to ports and cities, soil types, and average annual precipitation from 2000 to 2010.
With these features and the simulated LUPs, we employ the trained model to predict deforestation in the undeveloped region. We use both the 'predict' and 'predict_proba' methods of the model. The 'predict' method provides the most likely outcome (deforestation or no deforestation), while the 'predict_proba' method gives the probabilities for each possible outcome. This dual approach allows us to obtain a nuanced understanding of the potential deforestation patterns in the undeveloped region under different scenarios of forest law changes.

# Results
## Training & Validation
In the initial assessment of the predictive power of the LUPs and establishing a baseline for further analysis, the Balanced Random Forest Classifier performed reasonably well, achieving an accuracy of 0.66 and an F1-score of 0.68. A simple parameter space was chosen to increase model run time as performance was expected to improve with the inclusion of additional features.

The final model was a Balanced Random Forest Classifier with the best parameters found by RandomizedSearchCV being 'n_estimators': 50 and 'max_depth': None. With these parameters, the model achieved a best cross-validation score of 0.88, an accuracy of 0.94, and an F1-score of 0.94.

The model demonstrated robust performance on the training data, correctly classifying 2167159 instances of class 0 and 731728 instances of class 1, while misclassifying 10113 instances of class 0 and 2289 instances of class 1. The model's performance on the testing data was also robust, with high precision and recall scores for both classes.

The hyperparameters we tuned were 'n_estimators' and 'max_depth', with the options being [50, 100, 200] for 'n_estimators' and [None, 5, 10, 20] for 'max_depth'. The RandomizedSearchCV results indicated that the model with 'n_estimators': 50 and 'max_depth': None achieved the highest mean test ROC AUC of 0.975002, ranking first among all models.

## Predictions
For each of the simulations created in Objective 2 a corresponding map of  pixel-wise probabilities (Fig.11), forecasting deforestation patterns over a 10 year period under current and alternative laws was created. The pixel-wise probabilities determine the amount of area predicted to be deforested and the remaining probability for the pixel is designated as non-deforested. 

<h2 align="center">

<img src="https://github.com/cp-PYFOREST/.github/blob/main/img/obj3-prediction-map25.png" alt="Pixel-wise probability of deforestation in the undeveloped Chaco region">

</h2>

*Pixel-wise probability of deforestation in the undeveloped Chaco region*

The bins created for deforested and non-deforested create eight divisions,  as opposed to simply the four categorical divisions of the simulations. As an example, the one categorical variable of paddocks in the simulation has two division for the prediction, the area designated as paddocks that we predict as deforested and the remaining area of paddocks designated as non-deforested. 

The eight divisions of the prediction are  illustrated in the figure 12, organized in a left to right fashion of non-deforested to deforested. The organization of the charts allows for distributions of estimated and predicted totals to be compared. The attached table to each illustration applies a two bin approach to the simulations of non-deforested and deforested where paddocks are the only variable within the deforested bin. The two bin approach allows for a different perspective when considering the implications of possible scenario.

The deforestation rate is calculated as the ratio of the deforested area to the total area (i.e., the sum of the deforested and non-deforested areas) multiplied by 100.

**Current Forest Law:** The total deforested area is 1,696,318 ha, the non-deforested area is 7,503,372 ha, and the deforestation rate is 18.43%. This means that under the Current Forest Law scenario, about 18.43% of the total area is expected to be deforested.

**Law Ambiguity:** The total deforested area is 1,841,201 ha, the non-deforested area is 7,357,490 ha, and the deforestation rate is 20.01%. This means that under the Law Ambiguity scenario, about 20.01% of the total area is expected to be deforested.

**Prioritize Cattle Production:** The total deforested area is 1,841,214 ha, the non-deforested area is 7,359,220 ha, and the deforestation rate is 20.01%. This means that under the Prioritize Cattle Production scenario, about 20.01% of the total area is expected to be deforested.

**Promotes Forest Conservation:** The total deforested area is 1,507,326 ha, the non-deforested area is 7,692,862 ha, and the deforestation rate is 16.38%. This means that under the Promotes Forest Conservation scenario, about 16.38% of the total area is expected to be deforested.

These results provide a comparative overview of the expected deforestation rates under different policy scenarios. They can help policymakers understand the potential impacts of different policies on deforestation and make informed decisions.

<h2 align="center">

<img src="https://github.com/cp-PYFOREST/.github/blob/main/img/obj3-stats.png" alt="Comparison of Deforestation and Areas under Different Policy Scenarios by Land Use Types. Riparian Corridor Deforestation is insignificant, Non-deforested Riparian Corridor remains relatively constant.">

</h2>

*Comparison of Deforestation and Areas under Different Policy Scenarios by Land Use Types. Riparian Corridor Deforestation is insignificant, Non-deforested Riparian Corridor remains relatively constant.*


**Table 3: Comparison of Deforestation Rates and Areas under Different Policy Scenarios**

| Policy Scenario | Simulation Non-Deforested | Predicted Non-Deforested | Simulation Deforested | Predicted Deforested | Total Area | Deforestation Rate |
|---------|----------|----------|---------|----------|----------| ---------- | 
| Current Forest Law  | 3,702,454	| 7,503,372 | 5,497,236 | 1,696,318 | 9,199,690 | 18.43| 
| Promotes Forest Conservation |	5,589,018 |	7,692,862 |	3,611,169 |	1,507,326 |	9,200,187 |	16.38 |
| Prioritize Cattle Production |	2,401,457 |	7,359,220 |	2,401,457 |	6,798,97 |	1,841,214 |	9,200,435 |	20.01 |
| Law Ambiguity |	2,233,436 |	7,357,490 |	2,233,436 |	6,965,255 |	1,841,201 |	9,198,691 |	20.01 |	

## Visual Interpretation
The results of the LUP assessment for Objective 1 reveal that properties not in compliance with their LUPs are predominantly located in the western region of the study boundary. The analysis of forest cover and deforestation rates throughout the Paraguayan Chaco confirms both reduced forest cover and lower rates of deforestation in the Department of Presidente Hayes. This study's machine learning model takes these facts into account, aiming to understand the lack of LUPs in regions at higher risk of deforestation and how policy changes could potentially shape the landscape.

Examining the pixel-wise probabilities of deforestation in the western region (see Figure 13, left panel), the risk of deforestation escalates significantly, with various pockets indicating near-certain probability of occurrence. Conversely, in the right panel of Figure 13, which depicts the southern region of the Department of Presidente Hayes, we see the trends of deforestation and forest cover mirrored in the low to near-zero probabilities of deforestation.

<h2 align="center">

<img src="https://github.com/cp-PYFOREST/.github/blob/main/img/obj3-depts.png" alt="Left, western edge of study boundary of Department Boqueron with higher probabilities of deforestation. Right, southern tip of study boundary of Department Presidente Hayes with low to zero probabilities of deforestation.">

</h2>

*Left, western edge of study boundary of Department Boqueron with higher probabilities of deforestation. Right, southern tip of study boundary of Department Presidente Hayes with low to zero probabilities of deforestation.*


The patterns observed suggest a potential high spatial autocorrelation within the trained model. Feature importance analysis on the trained model identifies distance to rivers and roads as the most influential features, accounting for approximately 20-25% of the model's predictive power, followed by precipitation. This is strongly reflected in the southern region, which boasts a dense network of rivers and experiences higher annual average precipitation. To further understand the influence of LUPs on the model's predictive power, refer to Figure 14 a-d, which contrasts against the pixel-wise probability raster of the corresponding simulation.

<h2 align="center">

<img src="https://github.com/cp-PYFOREST/.github/blob/main/img/obj3-lups.png" alt="Contrasting pixel-wise probability rasters of deforestation against corresponding simulations to illustrate the influence of land use plans on the model's predictive power. Each panel (a-d) represents a different scenario, highlighting the variability in deforestation probabilities across different land use plan categories.">
   
</h2>

*Contrasting pixel-wise probability rasters of deforestation against corresponding simulations to illustrate the influence of land use plans on the model's predictive power. Each panel (a-d) represents a different scenario, highlighting the variability in deforestation probabilities across different land use plan categories.*


In a more detailed examination of the influence of the LUP, we can observe distinct patterns across different categories. In all scenarios, the outlines of the various categories within the LUP are discernible, and they generally exhibit higher probabilities in areas designated for deforestation. However, these areas also display a range of probabilities, with some spots nearing a probability of 1, indicating almost certain deforestation, and others hovering around 0.33, suggesting a lower but still significant risk.

Interestingly, lower probabilities are observed in protected areas, reflecting the effectiveness of these zones in mitigating deforestation. Additionally, we notice patterns that suggest the influence of geographical and environmental factors. For instance, there appear to be buffers that could represent a certain distance or travel time from key features such as roads or rivers. This could imply that accessibility plays a role in deforestation risk. Similarly, patterns in soil pixels are distinguishable, particularly in areas designated for deforestation. These patterns could be indicative of certain soil types or conditions that are more conducive to deforestation, although further investigation would be required to confirm this. This nuanced view underscores the complexity of the factors influencing deforestation and the importance of considering these factors in land use planning and policy development.

In conclusion, we present a pair of compelling examples where, by chance, we generated a simulated LUP in an area previously without one where the pattern closely mirrored the actual deforestation that occurred in the area. In some instances, the simulated plan even predicted very high probabilities of deforestation. 

However, it's crucial to clarify that our model is not designed to predict specific locations where deforestation will occur. Instead, its strength lies in its predictive power to understand the potential outcomes of simulated LUPs based on different policy scenarios.

This underscores the potential of our model as a tool for policy simulation and planning. By providing insights into the potential impacts of different land use policies, our model can support more informed and effective policy development.

<h2 align="center">

<img src="https://github.com/cp-PYFOREST/.github/blob/main/img/obj3-hansen.png" alt="Two representative examples demonstrating the model's predictive power in simulating land use plans. These simulations, created in an area previously devoid of any land use plan, align closely with the actual deforestation patterns from Hansen (2013), underscoring the model's potential in forecasting outcomes under various policy scenarios">
   
</h2>

*Two representative examples demonstrating the model's predictive power in simulating land use plans. These simulations, created in an area previously devoid of any land use plan, align closely with the actual deforestation patterns from Hansen (2013), underscoring the model's potential in forecasting outcomes under various policy scenarios*

## Data Information
Soon on Zenodo

Funk, C., Peterson, P., Landsfeld, M., Pedreros, D., Verdin, J., Shukla, S., Husak, G., Rowland, J., Harrison, L., Hoell, A., & Michaelsen, J. (2015). The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. Scientific Data, 2, 150066. https://doi.org/10.1038/sdata.2015.66

Hansen, M. C., Potapov, P. V., Moore, R., Hancher, M., Turubanova, S. A., Tyukavina, A., Thau, D., Stehman, S. V., Goetz, S. J., Loveland, T. R., Kommareddy, A., Egorov, A., Chini, L., Justice, C. O., & Townshend, J. R. G. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. Science, 342, 850-853. https://doi.org/10.1126/science.1244693. Data available on-line at: https://glad.earthengine.app/view/global-forest-change.

ISRIC - World Soil Information. (2023). SoilGrids: A system for global soil information. Retrieved from https://soilgrids.org/ 

Nelson, A. (2019). Travel time to cities and ports in the year 2015 [Data set]. Figshare. https://doi.org/10.6084/m9.figshare.7638134.v4


## Contributors
[Atahualpa Ayala](Atahualpa-Ayala),  [Dalila Lara](https://github.com/dalilalara),  [Alexandria Reed](https://github.com/reedalexandria),  [Guillermo Romero](https://github.com/romero61)

Any advise for common problems or issues.

## License

This project is licensed under the Apache-2.0 License - see the LICENSE.md file for details
