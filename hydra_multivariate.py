import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

class HydraMultivariate(nn.Module):
    """Hydra Multivariate transformer.

    Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb
    HYDRA: Competing convolutional kernels for fast and accurate time series classification
    https://arxiv.org/abs/2203.13652

    Parameters
    ----------
    input_length : int
        Length of input time series.
    num_channels : int
        Number of channels in input time series.
    k : int, default = 8
        Number of kernels per group.
    g : int, default = 64
        Number of groups.
    max_num_channels : int, default = 8
        Maximum number of channels to use in convolution.
    batch_size : int, default = 512
        Batch size for calculating features without running out of memory.

    Attributes
    ----------
    k : int
        Number of kernels per group.
    g : int
        Number of groups.
    dilations : torch.Tensor
        Dilation factors.
    num_dilations : int
        Number of dilation factors.
    paddings : torch.Tensor
        Padding factors.
    W : list of torch.Tensor
        Kernels.
    I : list of torch.Tensor
        Channels.
    batch_size : int
        Batch size for calculating features without running out of memory.

    Notes
    -----
    This is an *untested*, *experimental* extension of Hydra to multivariate input.

    References
    ----------
    [1] Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb.
    "HYDRA: competing convolutional kernels for fast and accurate time series classification."
    arXiv preprint arXiv:2203.13652 (2021).
    https://arxiv.org/abs/2203.13652

    Examples
    --------
    >>> from hydra_multivariate import HydraMultivariate
    >>> import torch
    >>> X = torch.randn(100, 3, 100)
    >>> transformer = HydraMultivariate(input_length = 100, num_channels = 3)
    >>> transformer(X).shape
    torch.Size([100, 10000])
    """

    def __init__(self, input_length, num_channels, k = 8, g = 64, max_num_channels = 8, batch_size=512):

        super().__init__()

        self.input_length = input_length
        self.num_channels = num_channels
        self.k = k # num kernels per group
        self.g = g # num groups
        self.max_num_channels = max_num_channels
        self.batch_size = batch_size

        max_exponent = np.log2((input_length - 1) / (9 - 1)) # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        self.num_dilations = len(self.dilations)

        self.paddings = torch.div((9 - 1) * self.dilations, 2, rounding_mode = "floor").int()

        # if g > 1, assign: half the groups to X, half the groups to diff(X)
        divisor = 2 if self.g > 1 else 1
        _g = g // divisor
        self._g = _g

        self.W = [self.normalize(torch.randn(divisor, k * _g, 1, 9).float()) for _ in range(self.num_dilations)]

        # combine num_channels // 2 channels (2 < n < max_num_channels)
        num_channels_per = np.clip(num_channels // 2, 2, max_num_channels)
        self.I = [torch.randint(0, num_channels, (divisor, _g, num_channels_per)) for _ in range(self.num_dilations)]

    @staticmethod
    def normalize(W):
        W -= W.mean(-1, keepdims = True)
        W /= W.abs().sum(-1, keepdims = True)
        return W

    # transform in batches of *batch_size*
    def batch(self, X, batch_size = 256):
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self(X)
        else:
            Z = []
            sample_indices = np.arange(num_examples)
            batches = np.array_split(sample_indices, num_examples // batch_size)
            for i, batch in enumerate(batches):
                Z.append(self(X[batch]))
            Z = np.concatenate(Z)
            return Z

    def forward(self, X):

        if type(X) is not torch.Tensor:
            X = torch.from_numpy(X)
        
        X = X.float().to(self.device)

        num_examples = X.shape[0]

        if self.g > 1:
            diff_X = torch.diff(X)
        else:
            print("Warning: g <= 1, diff(X) will not be used.")

        Z = []

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            # diff_index == 0 -> X
            # diff_index == 1 -> diff(X)
            for diff_index in range(min(2, self.g)):

                _Z = F.conv1d(X[:, self.I[dilation_index][diff_index]].sum(2).float() if diff_index == 0 else diff_X[:, self.I[dilation_index][diff_index]].sum(2).float(),
                              self.W[dilation_index][diff_index], dilation = d, padding = p,
                              groups = self._g) \
                      .view(num_examples, self._g, self.k, -1)

                max_values, max_indices = _Z.max(2)
                max_values = max_values.to(self.device)
                max_indices = max_indices.to(self.device)
                count_max = torch.zeros(num_examples, self._g, self.k).to(self.device)

                min_values, min_indices = _Z.min(2)
                min_values = min_values.to(self.device)
                min_indices = min_indices.to(self.device)
                count_min = torch.zeros(num_examples, self._g, self.k).to(self.device)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values).to(self.device))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        return Z.cpu().detach().numpy()

    def to(self, device):
        super().to(device)
        self.device = device

        for i, W in enumerate(self.W):
            self.W[i] = W.to(device)

        for i, I in enumerate(self.I):
            self.I[i] = I.to(device)

        self.dilations = self.dilations.to(device)
        self.paddings = self.paddings.to(device)

        return self

    def fit(self, X, y = None, **fit_params):
        return self

    def transform(self, X):
        features = self.batch(X, self.batch_size)
        return features