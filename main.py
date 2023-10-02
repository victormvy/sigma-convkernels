import time
from typing import cast

import numpy as np
import xgboost as xgb
from sacred import Experiment
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, RidgeCV
from sklearn.metrics import accuracy_score, make_scorer, mean_absolute_error
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.utils import shuffle

from functions import (
    compute_binary_metrics,
    compute_regression_metrics,
    fix_seeds,
    load_sigma_dataset,
    minimum_sensitivity,
    print_binary_metrics,
)
from hydra_minirocket_classifier import (
    HydraMultivariateClassifier,
    MiniRocketMultivariateClassifier,
    RocketMultivariateClassifier,
)
from inceptiontime import InceptionTimeClassifier

ex = Experiment("Sigma")


@ex.config
def cfg():
    seed = 1
    n_folds = 5
    fold = 0

    if fold >= n_folds:
        raise ValueError(
            f"Fold {fold} must be smaller than the number of folds ({n_folds})"
        )

    dataset_path = "./data/sigma_pdm.npy"

    method = "hydra"

    task = "classification"
    temporal = True

    output_paths = {"results": "./results"}

    ridge_alphas = np.logspace(-3, 3, 20).tolist()


@ex.automain
def run(
    seed,
    n_folds,
    fold,
    dataset_path,
    method,
    task,
    temporal,
    output_paths,
    ridge_alphas,
    _config,
):
    fix_seeds(seed)

    # Load data
    X, y = load_sigma_dataset(dataset_path)

    # Create train and test splits using group k-fold
    X, y = shuffle(X, y, random_state=seed)
    X = cast(np.ndarray, X)
    y = cast(np.ndarray, y)

    if task == "regression":
        gkf = GroupKFold(n_splits=n_folds)
        train_idx, test_idx = list(gkf.split(X, y=y[:, 0], groups=y[:, 1]))[fold]
        train_x, test_x, train_y, test_y = (
            X[train_idx],
            X[test_idx],
            y[train_idx, 0],
            y[test_idx, 0],
        )

        train_y = np.log(train_y + 1)
        test_y = np.log(test_y + 1)
    else:
        sgk = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        train_idx, test_idx = list(sgk.split(X, y=y[:, -1], groups=y[:, 1]))[fold]
        train_x, test_x, train_y, test_y = (
            X[train_idx],
            X[test_idx],
            y[train_idx, -1],
            y[test_idx, -1],
        )

    train_groups = y[train_idx, 1]
    test_groups = y[test_idx, 1]

    if not temporal:
        # Convert time series of shape (n_samples, n_dims, n_points) to (n_samples * n_points, n_dims)
        train_x_new = train_x.transpose((0, 2, 1)).reshape(
            train_x.shape[0] * train_x.shape[2], train_x.shape[1]
        )

        # Check that the values new array and the old matches
        for i in range(train_x.shape[0]):
            for j in range(train_x.shape[2]):
                assert np.allclose(
                    train_x_new[i * train_x.shape[2] + j], train_x[i, :, j]
                )

        train_y = np.repeat(train_y, train_x.shape[2])
        train_groups = np.repeat(train_groups, train_x.shape[2])
        train_x = train_x_new

        test_x_new = test_x.transpose((0, 2, 1)).reshape(
            test_x.shape[0] * test_x.shape[2], test_x.shape[1]
        )
        test_y = np.repeat(test_y, test_x.shape[2])
        test_groups = np.repeat(test_groups, test_x.shape[2])
        test_x = test_x_new

    print(
        f"Data loaded and splitted. Train size: {train_x.shape}, Test size: {test_x.shape}"
    )

    if task == "regression":
        scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        final_clf = RidgeCV(alphas=ridge_alphas, scoring=scorer)
    else:
        scorer = make_scorer(
            lambda y_true, y_pred: minimum_sensitivity(y_true, y_pred)
            * accuracy_score(y_true, y_pred),
            # roc_auc_score,
            greater_is_better=True,
        )
        final_clf = RidgeClassifierCV(
            alphas=ridge_alphas, scoring=scorer, class_weight="balanced"
        )
        # final_clf = xgb.XGBClassifier(tree_method='gpu_hist')

    if task == "regression":
        cv = GroupKFold(n_splits=5)
    else:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    if method == "hydra":
        ts_clf = GridSearchCV(
            estimator=HydraMultivariateClassifier(
                clf=final_clf,
                batch_size=10000,
                verbose=0,
            ),
            param_grid={"k": [2, 4, 8], "g": [4, 8, 16, 32]},
            scoring=scorer,
            n_jobs=1,
            cv=cv,
            verbose=10,
        )
    elif method == "mr":
        ts_clf = GridSearchCV(
            estimator=MiniRocketMultivariateClassifier(
                clf=final_clf, chunksize=3000, verbose=0
            ),
            param_grid={"num_features": [250, 500, 1000, 2000, 4000]},
            scoring=scorer,
            n_jobs=1,
            cv=cv,
            verbose=0,
        )
    elif method == "rocket":
        ts_clf = GridSearchCV(
            estimator=RocketMultivariateClassifier(clf=final_clf, verbose=0),
            param_grid={"num_features": [250, 500, 1000, 2000, 4000]},
            scoring=scorer,
            n_jobs=2,
            cv=cv,
            verbose=1,
        )
    elif method == "inception":
        ts_clf = InceptionTimeClassifier(
            c_in=X.shape[1], c_out=2, lr=1e-3, batch_size=2048, epochs=100, n_jobs=3
        )
    elif method == "xgb":
        ts_clf = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False),
            param_distributions={
                "colsample_bytree": uniform(0.7, 0.3),
                "gamma": uniform(0, 0.5),
                "learning_rate": uniform(0.03, 0.3),  # default 0.1
                "max_depth": randint(2, 6),  # default 3
                "n_estimators": randint(100, 150),  # default 100
                "subsample": uniform(0.6, 0.4),
            },
            scoring=scorer,
            n_jobs=1,
            cv=cv,
            verbose=10,
            n_iter=10,
        )
    elif method == "rf":
        ts_clf = RandomizedSearchCV(
            estimator=RandomForestClassifier(),
            param_distributions={
                "bootstrap": [True, False],
                "n_estimators": randint(100, 200),
                "max_depth": randint(2, 20),
                "min_samples_split": randint(5, 30),
                "min_samples_leaf": randint(5, 30),
                "max_features": ["auto", "sqrt", "log2"],
            },
            scoring=scorer,
            n_jobs=8,
            cv=cv,
            verbose=10,
            n_iter=10,
        )
    elif method == "ridge":
        ts_clf = RandomizedSearchCV(
            estimator=RidgeClassifier(class_weight="balanced"),
            param_distributions={"alpha": ridge_alphas},
            scoring=scorer,
            n_jobs=8,
            cv=cv,
            verbose=10,
            n_iter=len(ridge_alphas),
        )
    else:
        raise ValueError(f"Invalid method {method}")

    clf = ts_clf

    print(clf)

    start_time = time.time()
    clf.fit(train_x, train_y, groups=train_groups)
    fitting_time = time.time() - start_time
    print(f"Finished fitting in {fitting_time} seconds")

    train_preds = clf.predict(train_x)
    test_preds = clf.predict(test_x)

    total_time = start_time - time.time()

    if task == "regression":
        train_preds = np.exp(train_preds) - 1
        test_preds = np.exp(test_preds) - 1
        train_y = np.exp(train_y) - 1
        test_y = np.exp(test_y) - 1
        train_metrics = compute_regression_metrics(train_y, train_preds)
        test_metrics = compute_regression_metrics(test_y, test_preds)
    else:
        train_metrics = compute_binary_metrics(train_y, train_preds)
        test_metrics = compute_binary_metrics(test_y, test_preds)

    print("\nTrain metrics")
    print_binary_metrics(train_metrics)

    print("\nTest metrics")
    print_binary_metrics(test_metrics)
