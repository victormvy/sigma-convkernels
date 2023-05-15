import json
import os
import random
from pathlib import Path
from pickle import dump, load
from typing import Union, Tuple, Callable, List, Optional

import numpy as np
import pandas as pd
import pkbar
import torch
from sklearn import metrics
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, balanced_accuracy_score,
                             precision_score, f1_score, roc_auc_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit

from metrics import accuracy_off1


def minimum_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""Computes the sensitivity by class and returns the lowest value.

	Parameters
	----------
	y_true : array-like
		Target labels.
	y_pred : array-like
		Predicted probabilities or labels.

	Returns
	-------
	ms: float
		Minimum sensitivity.
	
	Examples
	--------
	>>> y_true = np.array([0, 0, 1, 1])
	>>> y_pred = np.array([0, 1, 0, 1])
	>>> minimum_sensitivity(y_true, y_pred)
	0.5
	"""

	sensitivities = recall_score(y_true, y_pred, average=None)
	return np.min(sensitivities)


def train(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: str, regression: bool = False):
    num_batches = len(dataloader)
    kbar = pkbar.Kbar(target=num_batches, width=32, always_stateful=False)
    model.train()
    mean_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        
        loss = loss_fn(pred, y)
        mean_loss += loss

        if pred.shape[1] == 1:
            # Binary
            accuracy = metrics.accuracy_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy() > 0.5)
        elif not regression:
            # Multiclass
            accuracy = (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item() / len(y)
        else:
            # Regression
            accuracy = 0.0

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update pbar
        if not regression:
            kbar.update(batch + 1, values=[("loss", loss.item()), ("accuracy", accuracy)])
        else:
            kbar.update(batch + 1, values=[("loss", loss.item())])

    mean_loss /= num_batches

    return mean_loss


def validate(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, metrics: List[Callable], device: str, regression: bool = False):
    kbar = pkbar.Kbar(target=len(dataloader), width=32, always_stateful=True)
    model.eval()
    running_loss = 0
    y_pred = torch.empty(0)
    y_true = torch.empty(0)
    val_metrics = {}

    with torch.no_grad():
        for batch, (batch_X, batch_y) in enumerate(dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y_pred = model(batch_X)
            batch_loss = loss_fn(batch_y_pred, batch_y)

            predicted_label = batch_y_pred > 0.5
            running_loss += batch_loss.item()

            y_pred = torch.cat((y_pred, predicted_label.detach().cpu()), 0)
            y_true = torch.cat((y_true, batch_y.detach().cpu()), 0)

            loss = running_loss / (batch + 1)
            values = [('val_loss', loss)]

            if not regression:
                for metric in metrics:
                    val_metrics[metric.__name__] = metric(y_true, y_pred)
                    values.append((f'val_{metric.__name__}', val_metrics[metric.__name__]))
            
            kbar.update(batch + 1, values=values)

    return loss, val_metrics


def test(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, device: str, num_classes: int = None):
    num_batches = len(dataloader)
    kbar = pkbar.Kbar(target=num_batches, width=32, always_stateful=False)
    model.eval()
    test_loss = 0
    y_pred, y_true = None, None
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Stack predictions and true labels
            if num_classes is None or num_classes == 2:
                pred_np = pred.cpu().detach().numpy()
            else:
                pred_np = pred.argmax(1).cpu().detach().numpy()

            true_np = y.cpu().detach().numpy()
            if y_pred is None:
                y_pred = pred_np
            else:
                y_pred = np.concatenate((y_pred, pred_np))
            if y_true is None:
                y_true = true_np
            else:
                y_true = np.concatenate((y_true, true_np))

            kbar.update(batch + 1, values=[("test_loss", test_loss / (batch + 1))])

    test_loss /= num_batches

    if y_pred is not None and y_true is not None:
        if num_classes is None:
            # Compute regression metrics
            metrics = compute_regression_metrics(y_true, y_pred)
            print_regression_metrics(metrics)
        elif num_classes == 2:
            # Compute binary metrics
            metrics = compute_binary_metrics(y_true, y_pred)
            print_binary_metrics(metrics)
        else:
            # Compute classification metrics
            metrics = compute_metrics(y_true, y_pred, num_classes)
            print_metrics(metrics)

    return metrics, test_loss, y_pred, y_true


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)

    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    labels = range(0, num_classes)

    # Calculate metrics
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=labels)
    ms = minimum_sensitivity(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    off1 = accuracy_off1(y_true, y_pred, labels=labels)
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        'QWK': qwk,
        'MS': ms,
        'MAE': mae,
        'CCR': acc,
        '1-off': off1,
        'Confusion matrix': conf_mat
    }

    return metrics

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    # Compute regression metrics
    metrics = {}
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    metrics['MAE'] = mae
    metrics['MSE'] = mse

    return metrics

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    # Compute binary metrics
    metrics = {}
    y_pred_binary = (y_pred > 0.5).astype(int)
    metrics['Confusion matrix'] = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    metrics['Accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['Balanced accuracy'] = balanced_accuracy_score(y_true, y_pred_binary)
    # metrics['R2'] = r2_score(y_true, y_pred_binary)
    metrics['F1'] = f1_score(y_true, y_pred_binary)
    metrics['AUC'] = roc_auc_score(y_true, y_pred_binary)
    metrics['MS'] = minimum_sensitivity(y_true, y_pred_binary)
    metrics['Precision'] = precision_score(y_true, y_pred_binary)

    return metrics

def print_metrics(metrics):
    for name, value in metrics.items():
        if type(value) not in [np.ndarray, list, tuple]:
            print(f'{name}: {value:.4f}')
        else:
            print(f'{name}:\n{value}')

print_binary_metrics = print_metrics
print_regression_metrics = print_metrics

def write_results_file(metrics: dict, results_file: str):
    results_file = Path(results_file)

    serializable_metrics = metrics
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()

    if results_file.is_dir():
        raise ValueError('Results file is a directory')
    elif results_file.is_file():
        with open(results_file, 'r') as f:
            current_data = list(json.load(f))
        current_data.append(serializable_metrics)
        with open(results_file, 'w') as f:
            json.dump(current_data, f, indent=1)
    else:
        with open(results_file, 'w') as f:
            json.dump([serializable_metrics], f, indent=1)

   


def fix_seeds(seed: int) -> None:
    """ Fix random seeds for numpy, tensorflow, random, etc.

        Parameters
        -----------
        seed : int.
            Random seed.
    """

    np.random.seed(seed)  # numpy seed
    torch.manual_seed(seed)  # torch seed
    random.seed(seed)  # random seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)



def prep_alldata_cumulative(seed: int, all_files: list, window_size: int = 144):

    frameAll =  pd.read_csv(all_files)
   
    frameAll['cycle_new'] = frameAll['machine'].astype(str)+frameAll['cycle'].astype(str)
    frameAll['cycle_new'] = frameAll['cycle_new'].astype(int)
    

    # Split train and test using group shuffle split
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    gss_splits = list(gss_test.split(X=frameAll, groups=frameAll['machine']))
    train_idx, test_idx = gss_splits[0]


    # Get train and test splits
    trainval_df, test_df = frameAll.iloc[train_idx], frameAll.iloc[test_idx]

    print(f"{trainval_df.shape=}, {test_df.shape=}")

    # Split train and validation using gss
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    gss_val_splits = list(gss_val.split(X=trainval_df, groups=trainval_df['machine']))
    train_idx, val_idx = gss_val_splits[0]

    # Get train and validation splits
    train_df, val_df = trainval_df.iloc[train_idx], trainval_df.iloc[val_idx]
    
    print(f"{train_df.shape=}, {val_df.shape=}")
    
    train_cycles = train_df['cycle_new'].unique()
    val_cycles = val_df['cycle_new'].unique()
    test_cycles = test_df['cycle_new'].unique()


    # We cannot do the undersampling here because the cumulative values change if we do that
    # train_df = train_df[train_df['RUL'] <= 8]
    #train_df['RUL'].where(train_df['RUL'] <= 8, 8, inplace=True)
    # val_df = val_df[val_df['RUL'] <= 8]
    #val_df['RUL'].where(val_df['RUL'] <= 8, 8, inplace=True)
    # test_df = test_df[test_df['RUL'] <= 8]
    #test_df['RUL'].where(test_df['RUL'] <= 8, 8, inplace=True)

    train_dfs = []
    for cycle in train_cycles:
        train_cycle_df = train_df[train_df['cycle_new'].isin([cycle])].copy()
        
        cumulative_df = train_cycle_df.cumsum()
        cumulative_df['id'] = train_cycle_df['id']
        cumulative_df['cycle'] = train_cycle_df['cycle']
        cumulative_df['cycle_new'] = train_cycle_df['cycle_new']
        cumulative_df['machine'] = train_cycle_df['machine']
        cumulative_df['RUL'] = train_cycle_df['RUL']
        train_dfs.append(cumulative_df)

    val_dfs = []
    for cycle in val_cycles:
        val_cycle_df = val_df[val_df['cycle_new'].isin([cycle])].copy()
        
        cumulative_df = val_cycle_df.cumsum()
        cumulative_df['id'] = val_cycle_df['id']
        cumulative_df['cycle'] = val_cycle_df['cycle']
        cumulative_df['cycle_new'] = val_cycle_df['cycle_new']
        cumulative_df['machine'] = val_cycle_df['machine']
        cumulative_df['RUL'] = val_cycle_df['RUL']
        val_dfs.append(cumulative_df)

    test_dfs = []
    for cycle in test_cycles:
        test_cycle_df = test_df[test_df['cycle_new'].isin([cycle])].copy()
        
        cumulative_df = test_cycle_df.cumsum()
        cumulative_df['id'] = test_cycle_df['id']
        cumulative_df['cycle'] = test_cycle_df['cycle']
        cumulative_df['cycle_new'] = test_cycle_df['cycle_new']
        cumulative_df['machine'] = test_cycle_df['machine']
        cumulative_df['RUL'] = test_cycle_df['RUL']
        test_dfs.append(cumulative_df)

    train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
    val_df = pd.concat(val_dfs, axis=0, ignore_index=True)
    test_df = pd.concat(test_dfs, axis=0, ignore_index=True)

    # Do the undersampling only on the training set after computing the cumulative values
    train_df = train_df[train_df['RUL'] <= 14]
    # Transform in binary task
    train_df['RUL'] = np.where(train_df['RUL'] <= 8, 0, 1)
    val_df['RUL'] = np.where(val_df['RUL'] <= 8, 0, 1)
    test_df['RUL'] = np.where(test_df['RUL'] <= 8, 0, 1)


    # standardization based on the train set mean and std
    for col in list(train_df.columns):
        if col != 'id' and col != 'cycle' and col != 'cycle_new' and col != 'machine' and col != 'RUL':
            train_mean = train_df[col].mean()
            train_std = train_df[col].std(ddof=0)
            train_df[col] = (train_df[col] - train_mean) / train_std
            val_df[col] = (val_df[col] - train_mean) / train_std
            test_df[col] = (test_df[col] - train_mean) / train_std

    '''preparare i set'''

    trainX = []
    trainY = []
    testX = []
    testY = []
    valX = []
    valY = []

    #Training set sliding time window processing
    for i in train_cycles:
        print(i)

        ind = np.where(train_df['cycle_new'] == i)
        ind = ind[0]
        data_temp = train_df.iloc[ind]
        for j in range(len(data_temp)- window_size + 1):
            trainX.append(np.array(data_temp.iloc[j:j + window_size,:-4]).tolist())
            train_RUL = data_temp.iloc[j + window_size-1,19]
            trainY.append(train_RUL)
            
            
    #Validation set sliding time window processing
    for i in val_cycles:
        print(i)

        ind = np.where(val_df['cycle_new'] == i)
        ind = ind[0]
        data_temp = val_df.iloc[ind]
        for j in range(int(len(data_temp)- window_size +1)):
            valX.append(np.array(data_temp.iloc[j:j + window_size,:-4]).tolist())
            val_RUL = data_temp.iloc[j + window_size-1,19]
            valY.append(val_RUL)


    #Test set sliding time window processing
    for i in test_cycles:
        print(i)

        ind = np.where(test_df['cycle_new'] == i)
        ind = ind[0]
        data_temp = test_df.iloc[ind]
        for j in range(int(len(data_temp)- window_size +1)):
            testX.append(np.array(data_temp.iloc[j:j + window_size,:-4]).tolist())
            test_RUL = data_temp.iloc[j + window_size-1,19]
            testY.append(test_RUL)


    trainX = np.array(trainX)
    testX = np.array(testX)
    valX = np.array(valX)
    trainY = np.array(trainY)
    testY = np.array(testY)
    valY = np.array(valY)


    trainY = np.expand_dims(trainY, axis=1)
    valY = np.expand_dims(valY, axis=1)
    testY = np.expand_dims(testY, axis=1)
    
    #dump([trainX, valX, testX, trainY, valY, testY], open('alldata_cum.pkl','wb'))
    
    return trainX, valX, testX, trainY, valY, testY

def fold_prep_alldata_cumulative(train_df, val_df, test_df, window_size: int = 144, window_step: int = 3):

    
    train_cycles = train_df['cycle_new'].unique()
    val_cycles = val_df['cycle_new'].unique()
    test_cycles = test_df['cycle_new'].unique()

    # Do the undersampling only on the training set after computing the cumulative values
    train_df = train_df[train_df['RUL'] <= 14]
    # Transform in binary task
    train_df['RUL'] = np.where(train_df['RUL'] <= 8, 0, 1)
    val_df['RUL'] = np.where(val_df['RUL'] <= 8, 0, 1)
    test_df['RUL'] = np.where(test_df['RUL'] <= 8, 0, 1)


    # standardization based on the train set mean and std
    # train_mean = train_df.iloc[:,:-5].mean(axis=0)
    # train_std = train_df.iloc[:,:-5].std(axis=0)
    # train_df.iloc[:,:-5] = (train_df.iloc[:,:-5] - train_mean) / train_std
    # val_df.iloc[:,:-5] = (val_df.iloc[:,:-5] - train_mean) / train_std
    # test_df.iloc[:,:-5] = (test_df.iloc[:,:-5] - train_mean) / train_std

    trainX = []
    trainY = []
    testX = []
    testY = []
    valX = []
    valY = []

    #Training set sliding time window processing
    for i in train_cycles:
        ind = np.where(train_df['cycle_new'] == i)
        ind = ind[0]
        data_temp = train_df.iloc[ind]
        for j in range(0, len(data_temp)- window_size + 1, window_step):
            trainX.append(np.array(data_temp.iloc[j:j + window_size,:-5]).tolist())
            train_RUL = data_temp.iloc[j + window_size-1,19]
            trainY.append(train_RUL)
            
            
    #Validation set sliding time window processing
    for i in val_cycles:
        ind = np.where(val_df['cycle_new'] == i)
        ind = ind[0]
        data_temp = val_df.iloc[ind]
        for j in range(0, int(len(data_temp)- window_size +1), window_step):
            valX.append(np.array(data_temp.iloc[j:j + window_size,:-5]).tolist())
            val_RUL = data_temp.iloc[j + window_size-1,19]
            valY.append(val_RUL)


    #Test set sliding time window processing
    for i in test_cycles:
        ind = np.where(test_df['cycle_new'] == i)
        ind = ind[0]
        data_temp = test_df.iloc[ind]
        for j in range(0, int(len(data_temp)- window_size +1), window_step):
            testX.append(np.array(data_temp.iloc[j:j + window_size,:-5]).tolist())
            test_RUL = data_temp.iloc[j + window_size-1,19]
            testY.append(test_RUL)


    trainX = np.array(trainX)
    testX = np.array(testX)
    valX = np.array(valX)
    trainY = np.array(trainY)
    testY = np.array(testY)
    valY = np.array(valY)

    # convert X to channels first shape
    trainX = trainX.transpose(0, 2, 1)
    testX = testX.transpose(0, 2, 1)
    valX = valX.transpose(0, 2, 1)

    trainY = np.expand_dims(trainY, axis=1)
    valY = np.expand_dims(valY, axis=1)
    testY = np.expand_dims(testY, axis=1)
    
    #dump([trainX, valX, testX, trainY, valY, testY], open('alldata_cum.pkl','wb'))
    
    return trainX, valX, testX, trainY, valY, testY

def load_sigma_dataset(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load the dataset from the given path.

    Parameters
    ----------
    path : str or path.
        Path to the dataset.

    Returns
    -------
    X : array-like
        The input data with shape (n_samples, n_dims, series_length)
    y : array-like
        The output data with shape (n_samples, 4), where the columns are
        (RUL, machine_id, cycle, binary_label).
    """

    data = np.load(path, allow_pickle=True)
    
    X = data[:, :-4, :]
    y = data[:, -4:, -1]

    # Create another column in y that combines the machine_id and cycle as string
    # y = np.hstack((y, np.char.add(y[:, 1].astype(str), y[:, 2].astype(str)).reshape(-1, 1)))

    return X, y
    
def normalize_data(train_X: np.ndarray, val_X: Optional[np.ndarray] = None, test_X: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    train_X = train_X.astype(np.float32)
    if val_X is not None:
        val_X = val_X.astype(np.float32)
    if test_X is not None:
        test_X = test_X.astype(np.float32)

    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)

    train_X = (train_X - mean) / std
    if val_X is not None:
        val_X = (val_X - mean) / std
    if test_X is not None:
        test_X = (test_X - mean) / std

    return train_X, val_X, test_X