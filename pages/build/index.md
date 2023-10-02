---
title: How to build a Random Forest with Scikit-learn and Pandas
---
# {% $markdoc.frontmatter.title %}

**Create a notebook**

- Right click the file name. 
- Click New file.
- Name it 'regressor.ipynb'. (ipynb is the Jupyter notebook file extension.)


You are also using the Cookie Cutter Data science file structure. See article [here](https://drivendata.github.io/cookiecutter-data-science/#starting-a-new-project). 

**Import the packages**
If you folloyoud our tutorial on setting up your dev environment, then all of these packages should already be installed in the default Anaconda distribution. 

Now that you have created your environment, you will need to import the following packages: 
- Pandas: Used to format the data for the model
- Matplotlib: Used to visualize the data
- Joblib: Used to save the model
- Scikit - learn  : Used to run the model

```py
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import PredictionErrorDisplay
```

**Save the output**

You need to create variables to store file locations. Doing it this way means you don't have to change the file location in multiple pieces of code when you make changes. 

It is a good practice to create a place to store your visuals and a place to store your models. You can seperate these in to diffent file locations. One to start models and one to store visuals. 

```py
MODEL_DIR = '../models/'
IMG_DIR = '../reports/figures/'
```

**Import your data**

Kaggle is a great place to get practice datasets. It is a competition youbsite that allows you to practice your skills on real-world data. 
You will use the data from the [House Prices-Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) Project. 

Download the *train.csv* file for this example. Upload that file to your data folder. 

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

It's important to know which columns have missing data. You may need to fill these.

```py
pd.set_option('display.max_rows', None)
train.isnull().sum().sort_values(ascending = False)
```

You need to fill in any column that has mising values. you add all of those column names to a list.
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

You will create can a function that will fill the missing data. you fill the data with -1. However, it could be better to fill data with mean, mode, mode, or other advanced techniques.

The function you create will also change any columns that are filled with strings into a numbers. Regression models can not handle strings so everything must be numerical. The [pd.factorize()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.factorize.html) method will hand this for us. It turns the strings into categories. Then those categories are turned into numbers. 

For example, if a column has only the words 'cat' or 'dog'. It would be turned into 1 or 2. Check the documentation above for more information. 

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

You want to remove the column you want to predict, or target column, from the data set. It is customary to save this variable as y and the remaining data as X. 

The .drop() will remove the column ( axis=1) from the data. If axis=0 it will try to remove a row.
```py
# save target column
y = train['SalePrice'] 
# remove target column from dataset
X = train.drop('SalePrice', axis=1)
X_test = test
```


** Training the model**

Historical information is used to train the data so that it can predict potential future outcomes. Later, you will use he test use the test data to predict on the trained data. That outcome of the comparison between the two is the prediction. 

You will use scikit-learn's [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) implementation. 

You can save the hyperparamters of the model to a dictionary. This allows you to easily make changes or run multiple experiments you change everything in one place.

There are 3 hyperparameters:
- n_estimaters: number of trees will create in our forest
- max_depth: how big should the tree grow. 
- random_state: controls the randomness of the bootstropping sample

```py
param = {'n_estimators': 200, 'max_depth':10, 'random_state':42}
```

Run the model on the training data.

```py
regr = RandomForestRegressor().set_params(**param).fit(X, y)
```

Predict the model on the test data. you can validate our results by uploading to Kaggle.

```py
y_pred = regr.predict(X_test)
```

**Save your model**
Running models repeatedly cost money. Instead, just save it, then load it when you need it. 

```py

dump(regr, f'{MODEL_DIR}/rf_regressor.joblib') 
```

**Feature importance**

One method of checking your model is to look at the feature importances. Feature importances is a measure of how much each feature, or column contributed to the prediction. There's plenty of math behind this too. You can read more [here.](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

```py
#Format the graph size and background color
fig, ax = plt.subplots(figsize=(25, 25), constrained_layout = True)
ax.barh(X_test.columns, regr.feature_importances_[regr.feature_importances_.argsort()])
ax.set_xlabel('Degree of importance', fontsize=14)
ax.set_ylabel('Dataset Columns/Features', fontsize=14)
ax.set_title('What re the most important features in the regression decision?', fontsize=28)

plt.savefig(f'{IMG_DIR}/feature_importance/feature_importance.png')
```

You also want to check your Rrscore and the predictive interval thes values occur on. More on that later.

```py
disp = PredictionErrorDisplay.from_predictions(y_test, y_pred)
plt.savefig(f'{IMG_DIR}/prediction_error/prediction_error.png')
print(f"R2score: \n",r2_score(y_test, y_pred))
print(f"MAE: \n", mean_absolute_error(y_test, y_pred))
```
