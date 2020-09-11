import unittest
from GridSearchHelper import EstimatorSelectionHelper


class TestGridSearchHelper(unittest.TestCase):
    """
    Things to test and later write assert statements to the main code: 
    len(models) == len(params)

    for .fit(): X, y should be np.arrays
    X.shape[0] == y.shape[0]
    y.shape[1] == 1
    same for X_train, y_train, X_test, y_test
    """
    
