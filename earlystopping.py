import math
from copy import deepcopy

from torch import Tensor
from torch.nn import Module


class EarlyStopping(object):
    def __init__(self, model: Module, mode: str='min', patience: int=20, threshold: float=1e-4, verbose: bool=True, restore_best_weights: bool=True) -> None:
        if mode not in ['min', 'max']:
            raise ValueError("mode argument must be one of 'min', 'max'.")

        if patience < 0:
            raise ValueError('patience must be a positive integer.')

        if threshold < 0:
            raise ValueError('threshold must be a positive float.')

        self.model = model
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights

        self._reset()

    def step(self, metric: Tensor) -> bool:
        """
            Checks if the training should stop on each step.
            Returns true if the training should be stopped.
        """

        current = float(metric)
        better = self.is_better(current)
        
        if better:
            self.best_metric = current
            self.best_epoch = self.current_epoch
            self.best_model = deepcopy(self.model.state_dict())
        else:
            if self.current_epoch > self.best_epoch + self.patience:
                if self.restore_best_weights:
                    if self.verbose:
                        print("Early stopping the training process...")
                    self._restore_best()
                return True

        self.current_epoch += 1

        return False

    def _reset(self) -> None:
        self.best_metric = math.inf if self.mode == 'min' else -math.inf
        self.best_epoch = 0
        self.best_model = None
        self.current_epoch = 0

    def is_better(self, metric: float) -> bool:
        # If no model is saved, it is always better
        if self.best_model is None:
            return True

        if self.mode == 'min':
            if metric < self.best_metric - self.threshold:
                return True
        
        if self.mode == 'max':
            if metric > self.best_metric + self.threshold:
                return True

        return False

    def _restore_best(self) -> None:
        self.model.load_state_dict(self.best_model)
        
        if self.verbose:
            print(f"Best weights restored. Epoch: {self.best_epoch}, Metric: {self.best_metric}.")


