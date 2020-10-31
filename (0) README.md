# Credit-Card-Fraud-Detection-with-XGboost-Comparison-b-w-GridSearchCV-RandomizedSearchCV
![Hits](https://hitcounter.pythonanywhere.com/count/tag.svg?url=https%3A%2F%2Fgithub.com%2Fajaychouhan-nitbhopal%2FCredit-Card-Fraud-Detection-with-XGboost-and-Comparison-b-w-GridSearchCV-and-RandomizedSearchCV)

This is a XGboost classifier model for Credit Card Fraud Detection on Anonymized features dataset. Hyperparameters of the models are tuned with GridSearchCV and RandomizedSearchCV and Comparison is also done between them.

## Dataset Overview (by Kaggle)
The datasets contains transactions made by credit cards in September 2013 by european cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Overview

This is the code of XGboost Classifier which is implemented on Credit Card Fraud Detection Anonymized features dataset. Hyperparameter tuning is done with GridSearchCV and RandomizedSearchCV. Comparison with visual representation is done on all results.

Train set is of 227845 transactions and Test set is of 56962 transactions. f1-score of 0.88 is achieved by both GridSearchCV and RandomizedSearchCV Hyperparameter tuning on X_test set, But GridSearchCV takes half time in comparison with RandomizedSearchCV and alsoa GridSearchCV took only 4 Hyperparameters' values while RandomizedSearchCV took 13 Hyperparameters' values.

You can find abovementioned dataset [here](https://www.kaggle.com/jayfaldu/creditcard-fraud-detection)

## Dependencies

[numpy](https://numpy.org/)

[pandas](https://pandas.pydata.org/)

[XGboost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)

[scikt-learn](https://scikit-learn.org/stable/)

[matplotlib](https://matplotlib.org/)

[seaborn](https://seaborn.pydata.org/)

Install missing dependencies with [pip](https://pip.pypa.io/en/stable/)

## Usage
1. Credit_Card_Fraud_Detection_with_XGboost_&_Comparision_of_Hyperparameter_tuning_with_GridSearchCV_&_RandomizedSearchCV.ipynb is Jupyter Notebook which contains classifier model and comparision of GridSearchCV & RandomizedSearchCV Models.
2. credit_card_fraud_detection_with_xgboost_&_comparision_of_hyperparameter_tuning_with_gridsearchcv_&_randomizedsearchcv.py is Python file which contains python code of point no. 1 Jupyter notebook.
3. Dataset.txt is a txt file which contains link of kaggle dataset which is used in this classifier.
4. Final_conclusion.csv contains outputs of xgboost models of notebook. This file is made by me and I used it to plot various conclusive plots in jupyter notebook.
5. Correlation matrix of all features is a png file which contains Correlation matrix of all columns of the dataset to understand and visualize the data in a better way.

Install jupyter [here](https://jupyter.org/install).

## Credits
This problem is taken from [Kaggle.com](https://www.kaggle.com/jayfaldu/creditcard-fraud-detection)
