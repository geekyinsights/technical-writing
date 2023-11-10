---
title: How to build a Random Forest with Scikit-learn and Pandas
---
# {% $markdoc.frontmatter.title %}

**Create a notebook**

In your notebook folder, create a new file by:
- Right clicking the file name 
- Clicking New file
- Giving the file a name -- 'regressor.ipynb' (.ipynb is the Jupyter notebook file extension)


In order to streamline the creation of the file structure you will be using the Cookie Cutter Data Science.[Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/#starting-a-new-project). 

**Importing packages**

To run your code, you will need a dev environment with the following imported packages. 
 
- Pandas: To format the data for the model
```py
import pandas as pd
```
- Matplotlib: To visualize the data
```py
import matplotlib.pyplot as plt
```

- Joblib: To save the model
```py
from joblib import dump, load
```
- Scikit - learn: To run the model
```py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import PredictionErrorDisplay
```

**Saving the output**

You need to create variables to be used for storing file locations. By using a variable, you won't need to change the file location in multiple sections of code when you make changes. It is a good practice to create a place to store your visuals and a place to store your models. You can seperate these into different file locations. One to store models and one to store visuals. This allows you to rerun tests or provide a visual when needed.

Create the MODEL_DIR variable for your models.
```py
MODEL_DIR = '../models/'
```
Create the IMG_DIR variable for your images.
```py
IMG_DIR = '../reports/figures/'
```
**Import your data**

Kaggle is a great place to get practice datasets. It is a competition website that allows you to practice your skills on real-world data. In this guide you will be using the data from the [House Prices-Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) Project. 

Once you are on the site you can locate the data set by clicking on the Data tab.Then scroll down the page to locate the download link for the train.csv file. 

Once the download is complete upload that file to your data folder using the code below. 

```py
#read dataset to variable
train = pd.read_csv("../data/raw/house_prices_train.csv")
test = pd.read_csv("../data/raw/house_prices_test.csv"
```

Verify that the file has loaded correctly by using the .head() method which will show the first five rows of the dataset. 

```py
train.head()
```
Once the rows display, note te data features(column heading) to congirm that there are no problems.
**Feature Engineering**

It's important to know which columns have missing data. These columns will need to be filled in. Use the following code. 

```py
pd.set_option('display.max_rows', None)
train.isnull().sum().sort_values(ascending = False)
```
Now use the train function to train your data by locating the null values then sorting them in descending order. You need to fill in any column that have missing values by placing those column names in a list. To do this you must use the following code.
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

You will create a function that will fill the missing data. you fill the data with -1. Depending on your goal, it could be better to fill in the missing data with the mean, mode, mode, or other advanced techniques. This function also changes categories into their numerical representation. Everything must be numerical because regression models cannot handle strings. Pandas will allow you to use the  pd.factorize() method to obtain a numeric representation of your array. The [pd.factorize()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.factorize.html) method will handle this for you.

For example, if a column has only the words 'cat' or 'dog'. It would be turned into a 1 or a 2. Check the documentation above for more information. 

```py
def fill_missing(filename): 
    file = filename.fillna(-1)
    for col in columns:
        file[col], unique =pd.factorize(file[col])
    return file

# call the function created abocve on the data
train = fill_missing(train)
```

**Extracting the target column from the data**

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
