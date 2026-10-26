"""Feature storage and padding for precomputed audio/visual features."""
import pickle

import numpy as np


def load_feature_dict(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def pad_feature(feature, max_len):
    """Truncate or zero-pad a sequence to ``max_len`` frames."""
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim == 3:
        feature = feature.squeeze(1)
    length = min(feature.shape[0], max_len)
    if feature.shape[0] >= max_len:
        return feature[:max_len], length
    pad = np.zeros((max_len - feature.shape[0], feature.shape[1]), dtype=np.float32)
    return np.concatenate([feature, pad], axis=0), length


class FeatureList:
    """Lazily padded view over a feature dictionary."""

    def __init__(self, feature_dict, keys, max_len):
        self.feature_dict = feature_dict
        self.keys = list(keys)
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.keys)

    def get(self, index):
        return pad_feature(self.feature_dict[self.keys[index]], self.max_len)
