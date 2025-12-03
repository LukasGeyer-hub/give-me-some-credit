# Give-Me-Some-Credit – Credit Default Prediction

## Overview

This project focuses on predicting the risk of credit default within the next two years. The work is based on the **“Give Me Some Credit” dataset** from Kaggle:

Kaggle Dataset: (https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset/data)

The dataset contains historical financial, income, and credit behavior information of borrowers. The goal is to develop models that estimate the probability of default (`SeriousDlqin2yrs`).

## Objective

The objective is to build a robust machine learning workflow to:

- Quantitatively assess credit risk  
- Identify key influencing factors  
- Evaluate and compare different model variants  
- Improve performance through data-driven optimization

**Target variable:** `SeriousDlqin2yrs`  
- 1 = Credit default within 2 years  
- 0 = No default

## Dataset

The original file `cs-training.csv` includes features such as:

- RevolvingUtilizationOfUnsecuredLines  
- age  
- DebtRatio  
- MonthlyIncome  
- NumberOfOpenCreditLinesAndLoans  
- Number of 30/60/90-day late payments  
- NumberOfDependents  

The dataset contains missing values and outliers, particularly in `MonthlyIncome`.

## Data Preparation

### Steps

1. **Drop irrelevant columns**  
   - The index column `Unnamed: 0` was removed.

2. **Handle missing values**  
   - Rows with missing `NumberOfDependents` were dropped.  
   - `MonthlyIncome` was imputed using **KNN Imputer** (k=5) based on the following features:
     - MonthlyIncome  
     - NumberOfDependents  
     - age  
     - DebtRatio  
     - NumberOfOpenCreditLinesAndLoans  

   This approach minimizes data loss while reconstructing missing income values based on similar observations.

3. **Save processed datasets for modeling and visualization**  
   - `train_clean.csv`  
   - `metrices.csv`  
   - `importances.csv`

## Exploratory Data Analysis (EDA)

Performed:

- Basic statistics (`df.describe()`)  
- Feature correlations and heatmap  
- Feature importance via **Random Forest**

Top features influencing `SeriousDlqin2yrs`:

1. RevolvingUtilizationOfUnsecuredLines  
2. DebtRatio  
3. MonthlyIncome  
4. age  
5. NumberOfTimes90DaysLate  

## Modeling Approach

Three model variants were implemented:

### 1. Baseline Model
**Random Forest Classifier**  
- n_estimators = 200  
- Default parameters  
- 80/20 train-test split  

**Baseline Results:**
- Accuracy: 0.9350  
- ROC-AUC: 0.8349  
- F1-Score: 0.2755  
- Precision: 0.5317  
- Recall: 0.1859  

High overall accuracy but low recall for the minority class (defaults).

### 2. Grid Search Model
Hyperparameter tuning using Grid Search:
- n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features  

**Best parameters:**
- max_depth = 10  
- max_features = "sqrt"  
- min_samples_leaf = 4  
- min_samples_split = 10  
- n_estimators = 300  

**Results (Grid Search):**
- Accuracy: 0.9369  
- ROC-AUC: 0.8557  
- F1-Score: 0.2560  
- Precision: 0.5925  
- Recall: 0.1632  

Significant improvement in ROC-AUC; still low recall due to rare event class.

### 3. Threshold Tuning
Adjusted the classification threshold to improve F1-score for rare defaults.  

**Best threshold:** 0.2  
**Best F1-Score:** 0.3953

**Results (Threshold-Tuned Model):**
- Accuracy: 0.8954  
- ROC-AUC: 0.8557  
- F1-Score: 0.3953  
- Precision: 0.3210  
- Recall: 0.5144  

Recall improved substantially, detecting more defaults, while overall accuracy decreased.

## Visualization and Reporting

A Power BI dashboard was created:

- Feature distributions    
- Model performance comparison  
- Feature importance plots  

Screenshots (`Power_BI_screenshot_1.png`, `Power_BI_screenshot_2.png`) are included.

## Conclusion

Through systematic data preparation, KNN imputation for missing incomes, EDA, and iterative model improvements, robust models for credit default prediction were developed.

Key insights:

- Credit risk can be effectively modeled (ROC-AUC ≈ 0.85)  
- Threshold tuning increases sensitivity for rare events  
- Debt ratios, income, and payment history are the most critical risk factors  

