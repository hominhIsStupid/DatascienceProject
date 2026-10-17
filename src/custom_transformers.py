import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


class LogSkewed(BaseEstimator, TransformerMixin):
    # Áp dụng log1p cho các cột: lệch phải (skew > threshold), không âm, và không phải biến nhị phân.
    # Danh sách cột được xác định ở fit() -> chỉ dùng thông tin của tập train.
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        skew = np.nan_to_num(stats.skew(X, axis=0, nan_policy="omit"))
        n_unique = np.array([len(np.unique(X[:, j])) for j in range(X.shape[1])])
        self.mask_ = (skew > self.threshold) & (X.min(axis=0) >= 0) & (n_unique > 2)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        X[:, self.mask_] = np.log1p(X[:, self.mask_])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
