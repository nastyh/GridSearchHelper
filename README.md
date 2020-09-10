# GridSearchHelper
 Class that allows to run GridSearch for several models at a time.
 
 ## Table of Content
 1. Summary
 2. Installation
 3. Available functions
 4. Use case


## 1. Summary
When working on a supervised ML problem, users often come to a point when they need to run and compare several models; find the most optimal parameters for these models (usually via [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)); retrain the models using the optimal parameters.
Unfortunately, a combination of [Pipeline and FeatureUnion](https://scikit-learn.org/0.18/modules/pipeline.html) doesn't allow to do grid search for several models and access the optimal parameters in one step. This package addresses the problem and allows to pass models and parameters as dictionaries:
```
models = {'regression': LinearRegression(),\
          'ridge': Ridge(),\
          'SVR': SVR(),\
          'tree': DecisionTreeRegressor(),\
          'Grad_Boost': GradientBoostingRegressor()}
params = {
           'regression': {},
           'ridge': {'alpha': [0.2, 0.5, 0.7, 1, 1.5, 2, 2.2], 'solver': ['svd', 'lsqr']},
           'SVR': {'C': [1, 1.2, 1.5, 2]},
           'tree': {'splitter': ['best', 'random'],
                    'min_samples_split': [2, 3, 4, 5, 8, 10],
                    'min_samples_leaf':[1, 2, 3, 4]},
           'Grad_Boost': {'learning_rate': [0.05, 0.1, 0.2],
                          'n_estimators': [100, 150, 200],
                          'min_samples_split': [2, 3],
                          'max_depth': [2, 3, 5]}   
           }
```

and returns the best combinations of parameters
```
{'learning_rate': 0.05,
 'max_depth': 2,
 'min_samples_split': 3,
 'n_estimators': 100}
```

and allows to refit the models using the optimal parameters.

Users can select various [scoring parameters](https://scikit-learn.org/stable/modules/model_evaluation.html). The current version that was built for regression purposes supports [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error), [Explained Variance Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score), [Median Absolute Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error), and [R2 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score). Later versions of the module will provide additional parameters. 

## 2. Installation
Currently it is merely a module. That said, you just need to download _GridSearchHelper.py_, put it to the same folder as your main .py/.ipynb files is stored, and import:
```
from GridSearchHelper import EstimatorSelectionHelper
```
It assumes that standard packages (pandas, numPy, scikit-learn) are already installed. 
