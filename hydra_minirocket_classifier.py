import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sktime.classification import BaseClassifier
from sktime.transformations.panel.rocket import Rocket

from hydra_multivariate import HydraMultivariate
from rocket import MiniRocketTransformer


class HydraMultivariateClassifier(BaseClassifier):
    _tags = {
        "X_inner_mtype": "numpy3D",  # which type do _fit/_predict, support for X?
        #    it should be either "numpy3D" or "nested_univ" (nested pd.DataFrame)
        "capability:multivariate": True
    }

    def __init__(self, k = 8, g = 64, clf=None, batch_size = 1024, verbose=0):
        """
        Parameters
        ----------
        hydra_params : dict
            Parameters for HydraMultivariate.
        clf : sklearn classifier
            Classifier to use. If None, RidgeClassifierCV is used.
        verbose : int
            Verbosity level.
        """

        super().__init__()
        self.clf = clf
        self.k = k
        self.g = g
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.verbose = verbose
        self.batch_size = batch_size

    def _fit(self, X, y):
        if type(X) is not torch.Tensor:
            X = torch.from_numpy(X)

        X = X.float().to(self.device)

        if self.clf is None:
            scorer = make_scorer(accuracy_score, greater_is_better=True)
            self.clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), scoring=scorer, cv=10)

        self.pipeline = make_pipeline(
            HydraMultivariate(X.shape[-1], X.shape[1], k = self.k, g = self.g, batch_size=self.batch_size).to(self.device),
            StandardScaler(with_mean=False),
            self.clf
        )

        if self.verbose > 0:
            print(self.pipeline)

        self.pipeline.fit(X, y)
        return self

    def _predict(self, X):
        if type(X) is not torch.Tensor:
            X = torch.from_numpy(X)

        X = X.to(self.device)

        return self.pipeline.predict(X)

class MiniRocketMultivariateClassifier(BaseClassifier):
    _tags = {
        "X_inner_mtype": "numpy3D",  # which type do _fit/_predict, support for X?
        #    it should be either "numpy3D" or "nested_univ" (nested pd.DataFrame)
        "capability:multivariate": True
    }

    def __init__(self, num_features=10_000, clf=None, chunksize = 1024, verbose=0):
        """
        Parameters
        ----------
        mr_params : dict
            Parameters for MiniRocket.
        clf : sklearn classifier
            Classifier to use. If None, RidgeClassifierCV is used.
        verbose : int
            Verbosity level.
        """

        super().__init__()
        self.clf = clf
        self.num_features = num_features
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.chunksize = chunksize
        self.verbose = verbose

    def _fit(self, X, y):
        if type(X) is not torch.Tensor:
            X = torch.from_numpy(X)

        X = X.float().to(self.device)

        if self.clf is None:
            scorer = make_scorer(accuracy_score, greater_is_better=True)
            self.clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), scoring=scorer, cv=10)

        self.pipeline = make_pipeline(
            MiniRocketTransformer(X.shape[1], X.shape[-1], num_features=self.num_features, chunksize=self.chunksize).to(self.device),
            StandardScaler(with_mean=False),
            self.clf
        )

        if self.verbose > 0:
            print(self.pipeline)

        self.pipeline.fit(X, y)
        return self

    def _predict(self, X):
        if type(X) is not torch.Tensor:
            X = torch.from_numpy(X)
        
        X = X.float().to(self.device)

        return self.pipeline.predict(X)
   

class RocketMultivariateClassifier(BaseClassifier):
    _tags = {
        "X_inner_mtype": "numpy3D",  # which type do _fit/_predict, support for X?
        #    it should be either "numpy3D" or "nested_univ" (nested pd.DataFrame)
        "capability:multivariate": True
    }

    def __init__(self, num_features=10_000, clf=None, chunksize = 1024, verbose=0):
        super().__init__()
        self.clf = clf
        self.num_features = num_features
        self.chunksize = chunksize
        self.verbose = verbose

    def _fit(self, X, y):
        if self.clf is None:
            scorer = make_scorer(accuracy_score, greater_is_better=True)
            self.clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), scoring=scorer, cv=10)

        self.pipeline = make_pipeline(
            Rocket(num_kernels=self.num_features, n_jobs=5),
            StandardScaler(with_mean=False),
            self.clf
        )

        if self.verbose > 0:
            print(self.pipeline)

        self.pipeline.fit(X, y)
        return self

    def _predict(self, X):
        return self.pipeline.predict(X)
   