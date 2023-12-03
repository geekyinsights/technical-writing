---
title: Refactoring your code
---
# {% $markdoc.frontmatter.title %}


There are 'pythonic' ways of coding using the python language. There are also varying ways to implement different types of code. Some implementations are better than others. Coding pythonically is generally time efficient. Therefore, the computation time required will be less and thereby making the code more cost effective.   

Creating a Random Forest model using Scikit-learn and Pandas will put the principle of efficient pythonic coding into practice.

**Create a new notebook.**

To begin you will need to create a notebook and import three packages. Using a import module code below you will add numpy, pandas and matplotlib to your notebook.

```py
#standard import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
Now you will import your data.

```py
from joblib import dump, load
```
As you will be using the Random Forest Model the RandomForestRegressor from scikit-learn also needs to be imported.

```py
from sklearn.ensemble import RandomForestRegressor
```
The purpose of a Random Forest Model is to predict the most likely value to place in missing data. Therefore, you need to import PredictionErrorDisplay and make_column_selector so that you can more precisely predict the values for the missing data.

```py
from sklearn.metrics import PredictionErrorDisplay
from sklearn.compose import make_column_selector
```
Next you will need to import the OrdinalEncoder and the SimpleImputer. 

```py
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
```

Copy all of the following cells from the previous tutorial into the new notebook. 

```py
MODEL_DIR = '../models/'
IMG_DIR = '../reports/figures/'
```
Use the code below to read the dataset to the variable.

```py
#read dataset to variable
train = pd.read_csv("../data/raw/house_prices_train.csv")
test = pd.read_csv("../data/raw/house_prices_test.csv")
```

```py
train.head()
```

Next, instead of locating the columns with missing values or column types, Scikit-learn has a handy method called [make_column_selector](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html) which will choose columns of a specific type for you. This method can be more precise and time efficient than other methods.

```py
cat_selector = make_column_selector(dtype_include=object)
num_selector = make_column_selector(dtype_include=np.number)
```

Next, you can just fill it them using an [Ordinal Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) for categorical values. You can also use a [Simple Imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer) to replace with the mean in the numerical columns. 

```py
cat_processor = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    encoded_missing_value=-2,
)
num_processor = SimpleImputer(strategy="mean", add_indicator=True)
```

In order to be able to use the values they must be transformed into numerical values. You can use [column transformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html) to transform all columns using the code below. 

```py

preprocessor = make_column_transformer(
     (num_processor, num_selector),(cat_processor, cat_selector)
)
```
Now you need to save your transformed data to a new column. To begin, you will need to set y as the variable for training your data.

```py
# save target column
y = train['SalePrice'] 
# remove target column from dataset
X = train.drop('SalePrice', axis=1)
X_test = test

```

It is common to want to test different hyperparameters. The code below will allow you try the model with 4 different hyperparamters. 

```py
param_1 = {'n_estimators': 200, 'max_depth':10, 'random_state':42}
param_2 = {'n_estimators': 300, 'max_depth':10, 'random_state':42}
param_3 = {'n_estimators': 200, 'n_jobs':10, 'max_depth':10, 'random_state':42}
param_4 = {'n_estimators': 100, 'n_jobs':10, 'max_depth':10, 'random_state':42}

#Put parameters in a list to be called
params = [param_1, param_2, param_3, param_4]
```

To run your model, export the images, and save each model file you need to create the function.  Begin with using the [make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) function. This model allows you to run the preprocessing and the model together in one line of code. It will process the data then it will run the model. 

```py
# run regressor
def run_model(param, index):
    rf_pipeline = make_pipeline(tree_preprocessor, RandomForestRegressor().set_params(**param))
    regr = rf_pipeline.fit(X_test, y_test)
    y_pred = regr.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(25, 25), constrained_layout = True)
    ax.barh(X_test.columns, regr.steps[1][1].feature_importances_[regr.steps[1][1].feature_importances_.argsort()][:80])
    ax.set_xlabel('Degree of importance', fontsize=14)
    ax.set_ylabel('Dataset Columns/Features', fontsize=14)
    ax.set_title('What re the most important features in the regression decision?', fontsize=28)
    
    disp = PredictionErrorDisplay.from_predictions(y_test, y_pred)
    plt.savefig(f'{IMG_DIR}/prediction_error/{index}_prediction_error_refactored.png')
    
    plt.savefig(f'{IMG_DIR}/feature_importance/{index}_feature_importance_refactored.png')
    print(f"R2score: \n",r2_score(y_test, y_pred))
    print(f"MAE: \n", mean_absolute_error(y_test, y_pred))
    
    
    dump(regr, f'{MODEL_DIR}/{index}_rf_regressor_refactored.joblib') 
```


Lastly, you will then loop through the params list and run the function. To make this possible, you will need to create an index variable in which to save the index number so that you can specify each in the model’s name. To get the index to increment properly you need to take into account that in Python indexes begin with 0. This  will require you to begin by adding 1 to your counter. If you do not it would return 0, 1, 2, 3 instead of 1, 2, 3, 4. 

```py
for param in params:
    index = params.index(param) + 1
    run_model(param, index)
```
At this point you have created a Random Forest, and your missing values should have been resolved. Take a minute to verify that everything is as you needed. Congratulations!
