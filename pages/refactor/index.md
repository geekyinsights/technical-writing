---
title: Building a random Forest with scikit-learn and pandas
---
# {% $markdoc.frontmatter.title %}

## Refactoring your code. 

There are 'pythonic' ways of coding in the python language. There are also better ways to implement different types of code. This matters for two reasons: 
- computation time required to run the code cause money
- you are more likely to get hired if you code well. see point one. 

Create a new notebook. 

We've added a few more methods to be imported

```py
#standard import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import PredictionErrorDisplay
from sklearn.compose import make_column_selector

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
```

Copy all of the folliw
```py
MODEL_DIR = '../models/'
IMG_DIR = '../reports/figures/'
```

```py
#read dataset to variable
train = pd.read_csv("../data/raw/house_prices_train.csv")
test = pd.read_csv("../data/raw/house_prices_test.csv")
```

```py
train.head()
```

Insteead of finding out which columns are missing values or the columns types.

Scikit-learn has a handy method called [make_column_selector](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html) which will choose columns of a specific type for you. 

```py
cat_selector = make_column_selector(dtype_include=object)
num_selector = make_column_selector(dtype_include=np.number)
```

Next instead of finding the missing values we can just fill it them using an [Ordinal Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) for categorical values. We can also use a [Simple Imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer) to replace with the mean in our numerical columns. 

```py
cat_processor = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    encoded_missing_value=-2,
)
num_processor = SimpleImputer(strategy="mean", add_indicator=True)
```

After that we are going to use a [column transformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html) to transform all of the values for use. 
```py


preprocessor = make_column_transformer(
     (num_processor, num_selector),(cat_processor, cat_selector)
)
```

```py
# save target column
y = train['SalePrice'] 
# remove target column from dataset
X = train.drop('SalePrice', axis=1)
X_test = test

```

We are going to get a bit fancy here. We're going to run the model with 4 different hyperparamters. 
```py
param_1 = {'n_estimators': 200, 'max_depth':10, 'random_state':42}
param_2 = {'n_estimators': 300, 'max_depth':10, 'random_state':42}
param_3 = {'n_estimators': 200, 'n_jobs':10, 'max_depth':10, 'random_state':42}
param_4 = {'n_estimators': 100, 'n_jobs':10, 'max_depth':10, 'random_state':42}

#Put parameters in a list to be called
params = [param_1, param_2, param_3, param_4]
```

We creae a function to run the model , export the images, and save each model file. 

To do this we start with using [make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) This lets you run the preprocessing and the model together in one line of code. It will step through to process the data thenm it will run the model. 

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



We then loop through the params list and run the function. We create an index variable to save the index number so that we can specific each in the model name. We add a +1 because python indexes start at 0. It would return 0, 1, 2, 3 instead of 1, 2, 3, 4. 

```py
for param in params:
    index = params.index(param) + 1
    run_model(param, index)
```
