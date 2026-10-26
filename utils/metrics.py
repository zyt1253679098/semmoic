import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


class AverageMeter:
    """Track a running average over minibatches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count


class OIDMetrics:
    """Four metrics for open intent classification.

    ``y_true`` uses ``K`` as the open-class label, where ``K`` is the number of
    known intent classes. All returned values are fractions in [0, 1].
    """

    def __call__(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        f1_known, f1_open, f1_all = self._f_measure(cm)
        return {
            'acc': accuracy_score(y_true, y_pred),
            'f1': f1_all,
            'f1-known': f1_known,
            'f1-open': f1_open,
        }

    @staticmethod
    def _f_measure(cm):
        per_class = []
        for idx in range(cm.shape[0]):
            tp = cm[idx][idx]
            support = cm[idx].sum()
            predicted = cm[:, idx].sum()
            recall = tp / support if support else 0.0
            precision = tp / predicted if predicted else 0.0
            per_class.append(2 * recall * precision / (recall + precision)
                             if (recall + precision) else 0.0)
        return (float(np.mean(per_class[:-1]).round(4)),
                float(round(per_class[-1], 4)),
                float(np.mean(per_class).round(4)))
