from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures, get_minirocket_features

class MiniRocketTransformer():
    def __init__(self, c_in, seq_len, num_features=10000, max_dilations_per_kernel=32, random_state=None, chunksize=1024, **kwargs):
        self.chunksize = chunksize
        self.mrf = MiniRocketFeatures(c_in, seq_len, num_features=num_features, max_dilations_per_kernel=max_dilations_per_kernel, random_state=random_state)

    def fit(self, X, y=None):
        self.mrf.fit(X, self.chunksize)
        return self

    def transform(self, X):
        features = get_minirocket_features(X, self.mrf, self.chunksize)
        return features.squeeze(-1)

    def to(self, device):
        self.mrf.to(device)
        return self