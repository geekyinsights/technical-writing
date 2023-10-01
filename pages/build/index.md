---
title: How to build a random Forest with scikit-learn and pandas
---
# {% $markdoc.frontmatter.title %}

**Create a notebook**
In your notebook folder, create a new file. 

- Right click the file name. 
- Click New file.
- Name it 'regressor.ipynb' . ipynb is the Jupyter notebook file extension.


We are also using the [Cookie Cutter Data science file structure](https://drivendata.github.io/cookiecutter-data-science/#starting-a-new-project). See article here for more info

**Import the packages**
Import the packages required to run your code. If you followed our tutorial on setting up your dev environment, then all of these paackages should already be installed in the default anaconda distribution. 

We import the following packages: 
- Pandas: Used to format the data for the model
- Matplotlib: Used to visualize the data
- joblib: Used to save the model
- scikit - learn  : Used to run the model

```py
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import PredictionErrorDisplay
```

**Save the output**

We create variables to store file locations. This way we don't have to change the file location in multiple pieces of code if we make changes. 
It is good practice to create a place to store your visuals and a place to store your models. This allows you to rerun test or provided the visual when needed.

```py
MODEL_DIR = '../models/'
IMG_DIR = '../reports/figures/'
```

**Import your data**

Kaggle is a great place to get practice datasets. It is a competition website that allos you to practiec your skills on real-world data. 
We are using the data from the [House Prices-Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) Project. 

Download the train.csv file for this example. Upload that file to your data folder. 

```py
#read dataset to variable
train = pd.read_csv("../data/raw/house_prices_train.csv")
test = pd.read_csv("../data/raw/house_prices_test.csv"
```

Check that the file was loaded. The .head() method will show the first 5 rows of the dataset. 

```py
train.head()
```

**Feature Engineering**

It's important to know which columns have missing data. We will need to fill these.

```py
pd.set_option('display.max_rows', None)
train.isnull().sum().sort_values(ascending = False)
```

We need to fill in any column that has mising values. We add all of those column names to a list.
```py
columns = [ 'MSZoning', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
        'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2',  'Heating',
       'HeatingQC', 'CentralAir', 'Electrical',
        'KitchenQual',
        'Functional', 'FireplaceQu', 'GarageType',
        'GarageFinish',   'GarageQual',
       'GarageCond', 'PavedDrive', 
         'PoolQC',
       'Fence', 'MiscFeature', 'SaleType', 'YearRemodAdd',
       'SaleCondition', ]
```

We create  a function that will fill the missing data. We fill the data with -1. However, it could be better to fill data with mean, mode, mode, or other advanced techniques. 
This fiunction also change any categories into their numerical representation. Regression models can not handle strings so everything must be numerical. The [pd.factorize()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.factorize.html) method will hand this for use. 

```py
def fill_missing(filename): 
    file = filename.fillna(-1)
    for col in columns:
        file[col], unique =pd.factorize(file[col])
    return file

# call the function created abocve on the data
train = fill_missing(train)
```

**extracting the target column from the data**

We want to remove the column we want to predict, or target column, from the data set. It is customary to save this variable as y and the remaining data as X. 
The .drop() will remove the column ( axis=1) from the data. If axis=0 it will try to remove a row.
```py
# save target column
y = train['SalePrice'] 
# remove target column from dataset
X = train.drop('SalePrice', axis=1)
X_test = test
```


** Training the model**

Model training is trains data on historical information so that it can predict potential future outcomes. We later predict on the test data on the historical trained data to see how close it is to that. That outcome is the predicion. 

We will use scikit-learn's [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) implementation. 

We save the hyperparamters of the model to a dictionary. this way if we make changes or run multiple experiments we change everything in one place.
We have 3 hyperparameters:
- n_estimaters: number of trees will create in our forest
- max_depth: how big should the tree grow. 
- random_state: controls the randomness of the bootstropioing sample

```py
param = {'n_estimators': 200, 'max_depth':10, 'random_state':42}
```

Runn the model on the training data

```py
regr = RandomForestRegressor().set_params(**param).fit(X, y)
```

Predict the model on the test data. We can validate our results by uploading to Kaggle
```py
y_pred = regr.predict(X_test)
```

**Save your model**
Running models repeatedly cost money. Instead just save it. Then load it when you need it. 
```py

dump(regr, f'{MODEL_DIR}/rf_regressor.joblib') 
```

**FEature importance**

On method of checking your model is to look at the feature importances. 
```py
#Format the graph size and background color
fig, ax = plt.subplots(figsize=(25, 25), constrained_layout = True)
ax.barh(X_test.columns, regr.feature_importances_[regr.feature_importances_.argsort()])
ax.set_xlabel('Degree of importance', fontsize=14)
ax.set_ylabel('Dataset Columns/Features', fontsize=14)
ax.set_title('What re the most important features in the regression decision?', fontsize=28)

plt.savefig(f'{IMG_DIR}/feature_importance/feature_importance.png')
```

We also want to check our Rscored and the predicve interval thes values occur on. More on that later.
```py
disp = PredictionErrorDisplay.from_predictions(y_test, y_pred)
plt.savefig(f'{IMG_DIR}/prediction_error/prediction_error.png')
print(f"R2score: \n",r2_score(y_test, y_pred))
print(f"MAE: \n", mean_absolute_error(y_test, y_pred))
```
