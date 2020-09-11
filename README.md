# GridSearchHelper
 Class that allows to run [GridSearchCV()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) for several models at a time. It reduces the amount of manual work involved and requires to pass models and parameters only once.
 
 ## Table of Content
 1. Summary
 2. Installation
 3. Supported functions and parameters
 4. Traditional and optimized used cases


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

returns the best combinations of parameters
```
{'learning_rate': 0.05,
 'max_depth': 2,
 'min_samples_split': 3,
 'n_estimators': 100}
```

and allows to refit the models using the optimal parameters.

Users can select various [scoring parameters](https://scikit-learn.org/stable/modules/model_evaluation.html) that are used when a model is trained with optimal parameters. The current version that was built for regression purposes supports [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error), [Explained Variance Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score), [Median Absolute Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error), and [R2 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score). Later versions of the module will provide additional parameters. 

## 2. Installation
Currently it is merely a module. That said, you just need to download _GridSearchHelper.py_, put it to the same folder as your main .py/.ipynb files is stored, and import:
```
from GridSearchHelper import EstimatorSelectionHelper
```
It assumes that standard packages (pandas, numPy, scikit-learn) are already installed. 

## 3. Supported functions and parameters

### 3.1 Initialization
A class instance should be initialized with two parameters, _models_ and _params_.
_models_ is a dictionary with an arbitrary key that serves as a name for a given model, and a value that is one of the scikit-learn's [supervised models](https://scikit-learn.org/stable/supervised_learning.html).

Example:
```
models = {'regression': LinearRegression(),\
             'ridge': Ridge(),\
             'Stochastic_Descent': SGDRegressor(),\
             'Decision_Tree': DecisionTreeRegressor(),\
             'Random_Forest': RandomForestRegressor()}
```

_params_ is another dictionary for models' parameters which users want to test.

Its format is a little bit more complex:
{'name of the model': list of dictionaries with parameters and their values}

Example:
```
params = {'regression': {},\
             'ridge': [{'alpha': [0.01, 0.05, 0.1, 0.3, 0.6]}],\
             'Stochastic_Descent': [{'penalty': ['l1', 'l2', 'elasticnet'], 'alpha': [0.00005, 0.0001, 0.0003, 0.0006, 0.001]}],\
             'Decision_Tree': [{'min_samples_split': [2, 3, 4, 5], 'min_samples_leaf': [1, 2, 3], 'max_features': ['auto', 'sqrt', 'log2']}],\
             'Random_Forest': [{'n_estimators': [100, 150, 200, 250], 'min_samples_split': [2, 3, 4, 5], 'min_samples_leaf': [1, 2, 3], 'max_features': ['auto', 'sqrt', 'log2']}]}
 ```
 If you don't test any parameters (like in a Regression example above), you still need to pass an empty dictionary. In all other cases, you pass a list and put dictionaries inside the list (all other models in the code snippet above).
 
 ### 3.2 fit() method
 Once you have an instance, you can run a .fit() method. It requires _X_train_ and _y_train_ to train your models on, and also accepts _**kwargs_ 
[that are supported by GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV). 
Some kwargs that you might want to pass:
1. **Scoring.** A string or a scorer callable object / function with signature scorer(estimator, X, y). If None, the score method of the estimator is used. [Scoring functions that are supported by scikit learn](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) can be passed.
2. **cv.** Determines the cross-validation splitting strategy. Possible inputs for cv are: None (will use a 5-fold cross validation), integer, to specify the number of folds in a (Stratified)KFold, [CV splitter](https://scikit-learn.org/stable/glossary.html#term-cv-splitter), an iterable yielding (train, test) splits as arrays of indices
3. **return_train_score**: bool, default=False. Computing training scores is used to get insights on how different parameter settings impact the overfitting/underfitting trade-off. However computing the scores on the training set can be computationally expensive and is not strictly required to select the parameters that yield the best generalization performance.
 
After fitting, a dictionary _best_params_ is returned and can also be accessed via 
```
<your_class_instance>.best_params
```

### 3.3 fit_with_best() method
You can retrain your models with the best parameters by calling _fit_with_best()_.
It requires _X_train, y_train, X_test, y_test_ (numpy arrays) and an optional parameter _metric_ that is used to report the performance on the provided test set. 
Currently, _metric_ supports:
* [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)
* [Explained Variance](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score)
* [Median Absolute Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error)
* [R2 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)

It returns a dictionary with keys being models' names and values that are calculated using the provided _metric_:
```
{'regression': 3006.0733911079724,
 'ridge': 3019.4609723576123,
 'Stochastic_Descent': 3958.7643420367217,
 'Decision_Tree': 6553.937887887888,
 'Random_Forest': 3875.6286939592223}
 ```
 
## Traditional and optimized used cases
The best way to see how the class helps to save time and to avoid typos is to compare two workbooks: [one that shows a traditional manual approach](https://github.com/nastyh/GridSearchHelper/blob/master/manual_way.ipynb) and [another that uses this class](https://github.com/nastyh/GridSearchHelper/blob/master/GridSearchHelper_way.ipynb). 
