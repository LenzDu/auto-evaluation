import numpy as np
from sklearn import metrics

from .evaluation import *

class CrossEvaluator(BaseEvaluator):
    """
    Evaluator to perform cross validation on the data and provide the same functions as its corresponding evaluator.
    A model and feature data are required for this evaluator instead of Yfit.

    Args:
        X: Numpy array
        Y: Numpy array
        task: String, the task of the evaluator. Can be 'binary', 'multi' and 'regression'
        model: Model to perform the cross validation. should be sklearn model or other models that implemented methods
               'fit' and 'predict_proba' or 'predict'
        cv: Sklearn cross validator, e.g. ShuffleSplit
    """
    def __init__(self, X, y, task, model, cv):
        self.fit(Ytrue, Yfit, task, cv)

    def fit(self, X, y, task, model, cv):
        self.Ytrue = np.array([])
        self.Yfit = np.array([])
        for train_index, test_index in cv.split(X, y):
            model.fit(X[train_index], y[train_index])
            try:
                self.Yfit = np.concatenate([self.Yfit, model.predict_proba(X[test_index])])
            except AttributeError:
                self.Ytrue = np.concatenate([self.Yfit, model.predict(X[test_index])])
            
        if task == 'binary':
            self.e = BinaryClassEvaluator(self.Ytrue, self.Yfit)
        elif task == 'multi':
            self.e = MultiClassEvaluator(self.Ytrue, self.Yfit)
        elif task == 'regression':
            self.e = RegressionEvaluator(self.Ytrue, self.Yfit)

    def report(self):
        # TODO: add cv specific metrics.
        e.report()