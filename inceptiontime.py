from tsai.models.InceptionTime import InceptionTime
from sktime.classification import BaseClassifier
from torch import nn, cuda, optim, Tensor
from torch.autograd.grad_mode import no_grad
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functions import train, validate
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from copy import deepcopy
from earlystopping import EarlyStopping


class InceptionTimeClassifier(BaseClassifier):
    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True
    }

    def __init__(self, c_in, c_out, lr = 1e-3, batch_size = 128, epochs = 30, n_jobs = 1, num_classes = None, random_state=None):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.lr = lr
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        
        self.model = nn.Sequential(
            InceptionTime(c_in=self.c_in, c_out=self.c_out),
            nn.Softmax()
        )

        self.epochs = epochs
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.num_classes = num_classes
        self.random_state = random_state


    def _fit(self, X, y):
        num_classes = self.num_classes if self.num_classes is not None else len(np.unique(y))

        # Move model to gpu for fitting
        self.model = self.model.to(self.device)

        self.loss_fn = nn.BCELoss(weight=Tensor(compute_class_weight('balanced', classes = np.unique(y), y = y))).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5, verbose=True)
        # self.earlystopping = EarlyStopping(self.model, patience=20)

        # If labels are in sparse representation, convert them to one-hot
        if len(y.shape) == 1:
            y = np.eye(num_classes)[y]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
        train_idx, val_idx = next(sss.split(X, y))
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs)
        val_dataset = TensorDataset(Tensor(X_val), Tensor(y_val))
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.n_jobs)

        best_model_loss = np.inf
        best_model_weights = self.model.state_dict()
        best_epoch = 0

        for t in range(self.epochs):
            print(f"\nEpoch {t+1}/{self.epochs}\n-------------------------------")
            loss = train(train_dataloader, self.model, self.loss_fn, self.optimizer, self.device, regression=False)
            val_loss, _ = validate(val_dataloader, self.model, self.loss_fn, [], self.device, regression=False)
            self.scheduler.step(val_loss)
            # self.earlystopping.step(val_loss)

            if val_loss < best_model_loss:
                best_model_loss = val_loss
                best_model_weights = deepcopy(self.model.state_dict())
                best_epoch = t

        self.model.load_state_dict(best_model_weights)

        print(f"\nBest model was found in epoch {best_epoch+1} with a validation loss of {best_model_loss:.4f}. Loading best model...")

        # Move model back to cpu to save gpu memory
        self.model = self.model.cpu()

        return self

    def _predict(self, X):
        # Move model to gpu for prediction
        self.model = self.model.to(self.device)

        dataset = TensorDataset(Tensor(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_jobs)

        with no_grad():
            y_pred = []
            for (X,) in dataloader:
                X = X.to(self.device)

                batch_pred = self.model(X)
                batch_pred = batch_pred.cpu().detach().numpy()
                y_pred.append(batch_pred)
            
        y_pred = np.concatenate(y_pred, axis=0)
        y_pred = np.argmax(y_pred, axis=1)

        # Move model back to cpu to save gpu memory
        self.model = self.model.cpu()

        return y_pred

    def fit(self, X, y=None, **fit_params):
        return super().fit(X, y)
            