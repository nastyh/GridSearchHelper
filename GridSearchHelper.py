from collections import defaultdict
import numpy as np
import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# import scipy.stats as st
# from scipy.stats import norm
import seaborn as sns
# import datetime
# import pandas_profiling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, power_transform
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, median_absolute_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.model_selection import KFold, StratifiedKFold

class EstimatorSelectionHelper:
    """
    Class that conducts Grid Search for provided models over a provided set of parameters.
    Prints out statistics and the best parameters for every model
    """
    def __init__(self, models, params):
        """
        Arguments:
        models: dictionary where a key is an arbitrary name of the method, its value is a Scikit Learn model.
        Example: {'regression': LinearRegression()}
        params: dictionary where a key is a model's name that is about to be GridSearched, its value is a list of another dictionaries in a format
        {'parameter name': [values of parameters]}
        Example:
        {
           'regression': {},
           'ridge': {'alpha': [0.2, 0.5, 0.7, 1, 1.5, 2, 2.2]}, {'solver': ['svd', 'lsqr']},
           'SVR': {'C': [1, 1.2, 1.5, 2]}
        }
        GridSearchCV for three models. 'regression' uses default parameters, the other two are sent into GridSearchCV. 
        Lengths of 'models' and 'params' should be the same
        data: numpy dataframe with features. Only required for 

        """
        self.models = models
        self.params = params
        # self.data = data
        self.keys = models.keys()
        self.grid_searches = {}
        self.best_params = defaultdict(list)
        self.best_errors = {}

        assert len(self.models) == len(self.params), 'Lengths of dictionaries with models and their parameters should be of the same length'
    

    def fit(self, X, y, **grid_kwargs):
        """
        Runs grid search and returns a dictionary with the best parameters for every model
        Arguments:
        X, y: numpy arrays, training sets
        **grid_kwargs: kwargs for GridSearchCV (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
        """
        assert isinstance(X, np.ndarray), 'X should be a numpy array'
        assert isinstance(y, np.ndarray), 'y should be a numpy array'
        assert X.shape[0] == y.shape[0], 'X and y should be of the same length'

        for key in self.keys:
            print(f'Running GridSearchCV for {key}')
            model = self.models[key]
            params = self.params[key]
            grid_search = GridSearchCV(model, params, **grid_kwargs)
            grid_search.fit(X, y)
            self.grid_searches[key] = grid_search
            self.best_params[key].append(grid_search.best_params_)
#             self.best_params[key].append(self.grid_searches[key].best_params_)
        print('Completed')
        return self.best_params

    
    def fit_with_best(self, X_train, y_train, X_test, y_test, metric = 'neg_root_mean_squared_error'):
        """
        Refits the models with the best parameters derived in 'fit' above.
        Returns a dictionary where keys are models' names defined when the class was initialized and keys are values of metrics defined in
        the variable 'metric.' If no grid search has been run for a model, the respective value will be -1.
        Arguments:
        X_train, y_train, X_test, y_test: numpy arrays 
        metric: metrics for model evaluation. Supported now: mean squared error (default), explained variance, median absolute error, R2 score
        """

        assert X_train.shape[0] == y_train.shape[0], 'Lengths of X_train and y_train should be the same'
        assert X_test.shape[0] == y_test.shape[0], 'Lengths of X_test and y_test should be the same'

        d_metrics = {'neg_root_mean_squared_error': mean_squared_error,\
                     'explained_variance': explained_variance_score,\
                      'neg_median_absolute_error': median_absolute_error,\
                      'R2': r2_score}
        # d_errors = {}
        for key in self.keys:
            print(f'Fitting model {key} with its best parameters')
            curr_model = self.grid_searches[key].best_estimator_
            curr_model.fit(X_train, y_train)
            self.best_errors[key] = d_metrics[metric](y_test, curr_model.predict(X_test))
        for k, v in enumerate(self.models):
            print(f'{metric} of the model {v} with the best parameters is {self.best_errors.get(v, -1):.2f}')
        return self.best_errors


    def FeatureImportanceAllocator(self, X_train, y_train, X_test, y_test, data, **gradient_kwargs):
        """
        Fits GradientBoostingRegressor(), makes predictions, takes the sum of predictions and allocates the number across the features
        based on their importance scores. 
        Sklearn implementation of GradientBoostingRegressor() normalizes feature importances so they add up to 1
        (https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py and
        https://github.com/scikit-learn/scikit-learn/blob/989613e5c33df49f99796e0ba17effbd28d8b831/sklearn/ensemble/_gb.py#L683; 
        thus, it's classical Gini importance or mean decrease impurity).
        Returns two sorted dictionaries: first is <feature_name>: feature importance value.
        Second is <feature_name>: contribution of the feature to the total predicted GMV; (feature_importance * sum(predicted GMV))
        Arguments:
        X_train, y_train, X_test, y_test: numpy arrays
        **gradient_kwargs: parameters that are passed into GradientBoostingRegressor(). List of parameters: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
        """
        rf = GradientBoostingRegressor(**gradient_kwargs)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

#         assert len(rf.feature_importances_) == self.X.shape[1]

        d_importances, d_importances_gmv = {}, {}
        for ix in range(data.shape[1]):
            d_importances[data.columns.to_list()[ix]] = rf.feature_importances_[ix]
        for ix in range(data.shape[1]):
            d_importances_gmv[data.columns.to_list()[ix]] = rf.feature_importances_[ix] * sum(rf_pred)
        
        return sorted(d_importances.items(), key = lambda x: x[1], reverse = True), sorted(d_importances_gmv.items(), key = lambda x: x[1], reverse = True)