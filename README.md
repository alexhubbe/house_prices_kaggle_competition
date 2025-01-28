# House Prices - Advanced Regression Techniques (Kaggle Competition)

This repository contains the solution and analysis for the Kaggle competition **["House Prices: Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)**. The objective of this competition is to predict the final sale price of homes based on a variety of features, such as square footage, number of rooms, location.

![](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/reports/images/sale_prices.png)

My main goal was to improve [AutoGluon](https://autogluon.mxnet.io/)’s prediction score through data engineering and configuration optimization.

## Key Findings:
- Improved AutoGluon's baseline prediction score by **10.1%** (Table 1).
- Achieved a **top 3% ranking** among nearly 4,200 competitors in the Kaggle competition.

### Table 1: Approaches Employed to Improve Model Performance

| Data                        | AutoGluon Presets, Time (min) | Score (RMSE) | Score Improvement (%) |
|-----------------------------|-------------------------------|--------------|-----------------------|
| Original data               | Default                       | 0.12748      | -                     |
| Original data               | Good, 30                      | 0.12036      | 5.6                   |
| - Feature 'Id'              | Good, 30                      | 0.11632      | 8.8                   |
| + NaN Treatment             | Good, 30                      | 0.11539      | 9.5                   |
| + Median Price (25 Closest) | Good, 30                      | 0.11459      | 10.1                  |

---

## Project Overview

Below is a succinct description of the steps developed in this project.

## [Adding Latitude and Longitude to the Dataset](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/01_ah_merging_datas.ipynb)  
- Combined latitude and longitude information from [R's tidymodels](https://www.tmwr.org/ames) dataset with Kaggle's dataset. While this did not improve the test dataset score (Table 2), it enabled the calculation of the median sale price of the 25 closest houses.
- Produced visualizations (see [here](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/ah_appendix_1.ipynb)).

## [Exploratory Data Analysis and Data Engineering](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/02_ah_EDA.ipynb)  
At this stage, I went through several procedures:  
1. **Sanity check**
- Inspected for duplicate entries, standardized text data, and analyzed missing values.
- Dropped the 'Id' feature, improving the test dataset score by **3.2%** (Table 1).
- Tested dropping categorical features that lacked a second representative category in the training dataset, but this approach did not improve the score on the test dataset (Table 2).

2. **Missing values treatment**
- Implemented a custom approach for handling missing values, resulting in a **0.7%** improvement over AutoGluon's default method (Table 1).
  
3. **Categorical features**
- Tested combining underrepresented categories and grouping neighborhoods, but neither approach improved the score (Table 2).

4. **Numeric features**
- Experimented with delta features, seasonal features, and area-related features, but none improved the score (Table 2).

5. **Numeric vs. non-numeric features**
- Explored an alternative encoding method for ordinal features (see [here](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/ah_appendix_2.ipynb)), but this did not improve the score (Table 2).
  
6. **Outliers**
- Evaluated the impact of removing extreme outliers, but this did not improve the score (Table 2).
  
7. **New feature based on location**
- Introduced a feature representing the median sale price of the 25 closest houses, with the number of neighboring houses determined through experimentation (values tested: 5, 25, 50, 75, and 100). This feature yielded a **0.6%** improvement (Table 1).

8. **New feature based on economic data**
- Tested two economic indexes related to housing prices, but these features did not improve model performance (Table 2).

9. **Interaction between original features**
- Experimented with creating interaction features by combining pairs of the top numeric/ordinal features (e.g., 'GrLivArea' * 'OverallQual') based on AutoGluon’s feature importance ranking. I tested combinations of the top 3, 5, and 10 features, but this approach did not improve the test dataset score (Table 2). A caveat is that AutoGluon's feature importance ranking is intended for use on the test dataset; however, since I do not have the SalePrice for the test dataset, I applied it to the training dataset instead. 

## [Machine Learning](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/03_ah_MODEL.ipynb)  
- I used **AutoGluon** to perform the machine learning analyses.  
- The analyses were conducted using the `presets='good_quality'` and `time_limit=30 min` (the "good_quality" preset typically completes in under 30 minutes). The `medium_quality`, `high_quality`, and `best_quality` presets did not improve the score on the test dataset (see Table 2).  
- Neither the **AutoGluon pseudolabel model refit** nor **model distillation** improved the score on the test dataset (Table 2).  

### Table 2: Proposed Treatments That Did Not Improve Scores

| **Treatments**                                                                 |
|--------------------------------------------------------------------------------|
| Removing 'Street', 'Utilities', 'RoofMatl', 'Condition2', 'PoolQC'             |
| Longitude and Latitude                                                         |
| Combining Underrepresented Classes                                             |
| Grouping Neighborhoods into Larger Areas                                       |
| Defining Seasons of the Year                                                   |
| Delta Features for Construction, Remodeling, and Selling Years                |
| Total Area, Finished Area, High-Quality Area, and Total Bathrooms              |
| Ordinal Encoding                                                               |
| Removing Extreme Outliers                                                      |
| Case-Shiller U.S. National Home Price Index                                    |
| All-Transactions House Price Index for Ames, IA                                |
| Interaction Features Among Top Features                                        |
| AutoGluon Presets: `medium_quality`, `high_quality`, `best_quality`            |
| AutoGluon Pseudolabel Model Refit and Model Distillation                       |

---

## Tools and Technologies:
- **Libraries**: AutoGluon, Featuretools, Matplotlib, Numpy, Pandas, Seaborn, Scikit-Learn, 

---

## Project organization

```
├── .gitignore                         <- Files and directories to be ignored by Git
|
├── LICENSE                            <- License type 
|
├── README.md                          <- Main README explaining the project
|
├── data                               <- Project data files
|   ├── ames_dataset.csv               <- Original dataset from R's tidymodels
|   ├── ATNHPIUS11180Q.csv             <- Data for Ames' House Price Index
|   ├── clean_data.csv                 <- Dataset prepared for machine learning analysis
|   ├── CSUSHPINSA.csv                 <- Data for the USA's Case-Shiller Index
|   ├── original_plus_lat_lon.csv      <- Kaggle dataset with added longitude and latitude from tidymodels
|   ├── test.csv                       <- Original test dataset from Kaggle
|   ├── train.csv                      <- Original train dataset from Kaggle
|
├── environments                       <- Requirements files to reproduce the analysis environments
|   ├── autogluon_environment.yml      <- Environment for running '03_ah_MODEL.ipynb' 
|   ├── eda_environment.yml            <- Environment for running '01_ah_merging_datas.ipynb', '02_ah_EDA.ipynb', and 'ah_appendix_2.ipynb' 
|   ├── maps_environment.yml           <- Environment for running 'ah_appendix_1.ipynb'
|
├── models                             <- Trained and serialized models, model predictions, or model summaries
|
├── notebooks                          <- Jupyter notebooks
|   ├── 01_ah_merging_datas.ipynb      <- Adding latitude and longitude information to Kaggle's dataset
|   ├── 02_ah_EDA.ipynb                <- Exploratory data analysis
|   ├── 03_ah_MODEL.ipynb              <- Machine learning approach
|   ├── ah_appendix_1.ipynb            <- Creating maps
|   ├── ah_appendix_2.ipynb            <- Exploring an alternative dataset transformation
|   └── src                            <- Source code used in this project
|      ├── __init__.py                 <- Makes this a Python module
|      ├── auxiliaries.py              <- Scripts to compute nearest houses and median sale prices
|      ├── config.py                   <- Basic project configuration
|      ├── eda.py                      <- Scripts for exploratory data analysis and visualizations
|
├── references                         <- Data dictionaries, manuals, and other explanatory materials
|   ├── data_description.txt           <- Description of the dataset as presented on Kaggle
|
├── reports                            <- Generated analyses in HTML, PDF, LaTeX, etc., and results
|   ├── best_kaggle_prediction.csv     <- Best prediction from Kaggle competition
│   └── images                         <- Images used in the project
|      ├── sale_prices_train_test_plot.png  <- Map showing houses with and without prices 
|      ├── sale_prices.png                  <- Map displaying house prices
```

## Contributing
All contributions are welcome!

### Issues
Submit issues for:
- Recommendations or improvements
- Additional analyses or models
- Feature enhancements
- Bug reports

### Pull Requests
- Open an issue before starting work.
- Fork the repository and clone it.
- Create a branch and commit your changes.
- Push your changes and open a pull request for review.