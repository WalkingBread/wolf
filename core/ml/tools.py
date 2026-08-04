import numpy as np
from sklearn.model_selection import BaseCrossValidator

class PurgedTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits: int = 5, horizon: int = 5, embargo: int = 5,
                 min_train_size: int = 250):
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo
        self.min_train_size = min_train_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        
        first_train_end = self.min_train_size
        available_test_samples = n - (first_train_end + self.horizon)
        
        if available_test_samples <= 0:
            raise ValueError(
                f"Not enough samples ({n}) for horizon={self.horizon} and "
                f"min_train_size={self.min_train_size}."
            )

        test_size = available_test_samples // self.n_splits
        if test_size <= 0:
            raise ValueError(
                f"Calculated test_size is {test_size}. Increase dataset size "
                f"or decrease min_train_size/n_splits."
            )

        indices = np.arange(n)

        for i in range(self.n_splits):
            test_start = first_train_end + self.horizon + i * test_size
            test_end = min(test_start + test_size, n)
            
            if test_start >= n:
                break

            test_idx = indices[test_start:test_end]

            purge_end = max(test_start - self.horizon, 0)
            
            train_idx = indices[:purge_end]

            if len(train_idx) < self.min_train_size:
                continue

            yield train_idx, test_idx