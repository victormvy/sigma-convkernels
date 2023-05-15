import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, accuracy_score, top_k_accuracy_score


def _compute_sensitivities(y_true, y_pred, labels=None):
	if len(y_true.shape) > 1 and y_true.shape[1] > 1:
		y_true = np.argmax(y_true, axis=1)
	if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
		y_pred = np.argmax(y_pred, axis=1)

	conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

	sum = np.sum(conf_mat, axis=1)
	mask = np.eye(conf_mat.shape[0], conf_mat.shape[1])
	correct = np.sum(conf_mat * mask, axis=1)
	sensitivities = correct / sum

	sensitivities = sensitivities[~np.isnan(sensitivities)]

	return sensitivities

def minimum_sensitivity(y_true, y_pred, labels=None):
	return np.min(_compute_sensitivities(y_true, y_pred, labels=labels))

def accuracy_off1(y_true, y_pred, labels=None):
	if len(y_true.shape) > 1:
		y_true = np.argmax(y_true, axis=1)
	if len(y_pred.shape) > 1:
		y_pred = np.argmax(y_pred, axis=1)

	conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
	n = conf_mat.shape[0]
	mask = np.eye(n, n) + np.eye(n, n, k=1), + np.eye(n, n, k=-1)
	correct = mask * conf_mat

	return 1.0 * np.sum(correct) / np.sum(conf_mat)


def compute_metrics(y_true, y_pred, num_classes):
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)

    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    labels = range(0, num_classes)

    # Calculate metrics
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=labels)
    ms = minimum_sensitivity(y_true, y_pred, labels=labels)
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


def print_metrics(metrics):
		print('Confusion matrix :\n{}'.format(metrics['Confusion matrix']))
		print('QWK: {:.4f}'.format(metrics['QWK']))
		print('MS: {:.4f}'.format(metrics['MS']))
		print('MAE: {:.4f}'.format(metrics['MAE']))
		print('CCR: {:.4f}'.format(metrics['CCR']))
		print('1-off: {:.4f}'.format(metrics['1-off']))