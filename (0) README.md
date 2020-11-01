# Credit-Card-Fraud-Detection-with-XGboost-Comparison-b-w-GridSearchCV-RandomizedSearchCV
![Hits](https://hitcounter.pythonanywhere.com/count/tag.svg?url=https%3A%2F%2Fgithub.com%2Fajaychouhan-nitbhopal%2FCredit-Card-Fraud-Detection-with-XGboost-and-Comparison-b-w-GridSearchCV-and-RandomizedSearchCV)

This is a XGboost classifier model for Credit Card Fraud Detection on Anonymized features dataset. Hyperparameters of the models are tuned with GridSearchCV and RandomizedSearchCV and Comparison is also done between them.

## Dataset Overview (by Kaggle)
The datasets contains transactions made by credit cards in September 2013 by european cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Overview

This is the code of XGboost Classifier which is implemented on Credit Card Fraud Detection Anonymized features dataset. Hyperparameter tuning is done with GridSearchCV and RandomizedSearchCV. Comparison with visual representation is done on all results.

Train set is of 227845 transactions and Test set is of 56962 transactions. The f1-score of 0.89 is achieved by RandomizedSearchCV Hyperparameter tuning on X_test set.

Randomized Search CV approach of Hyperparameter tuning gave better results but in more time (with 13 parameters distribution). Grid Search CV takes more time and computation power when grid is bigger, so I used only 5 numbers of parameters in parameter grid. Although Randomized Search CV took more time but it covers more parameters.

With the same grid size of parameters, Grid Search CV takes very much time, so this is prohibited for bigger grids.

RandomizedSearchCV always gives different results. I ran this 4 times and 0.89 was the best f1-score I got on the x_test set.

In this notebook I focused more on Understanding the data with visualization and RandomizedSearchCV & GridSearchCV approaches of Hyper parameters tuning. Although better f1-score can be achieved by providing more parameters to these two approaches but it will be cost more time and computational power.

You can find abovementioned dataset [here](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Dependencies

[numpy](https://numpy.org/)

[pandas](https://pandas.pydata.org/)

[XGboost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)

[scikt-learn](https://scikit-learn.org/stable/)

[matplotlib](https://matplotlib.org/)

[seaborn](https://seaborn.pydata.org/)

Install missing dependencies with [pip](https://pip.pypa.io/en/stable/)

## Usage
1. XGboost Classifier model with Hyperparameter tuning notebook.ipynb is Jupyter Notebook which contains classifier model and comparision of GridSearchCV & RandomizedSearchCV Models.
2. conclusion.csv contains outputs of xgboost models of notebook. This file is made by me and I used it to plot various conclusive plots in jupyter notebook.
3. Dataset.txt is a txt file which contains link of kaggle dataset which is used in this classifier.
4. Correlation matrix of all features is a png file which contains Correlation matrix of all columns of the dataset to understand and visualize the data in a better way.

Install jupyter [here](https://jupyter.org/install).

## Credits
This problem is taken from [Kaggle.com](https://www.kaggle.com/mlg-ulb/creditcardfraud)
