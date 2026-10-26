"""Fit and evaluate the source-aware ellipsoidal boundary."""
import json
import os

import numpy as np

from utils.metrics import OIDMetrics

from .inputs import build_inputs, predict
from .solver import fit_boundary


def fit_and_evaluate(features, config, output_dir):
    inp = build_inputs(features, config.num_labels)
    K = config.num_labels
    theta, b, info = fit_boundary(inp, config)

    pred, _ = predict(inp.test_z, theta, b, info['mu'], info['delta'],
                      info['U'], inp.head_W, info['theta0'], K)
    metrics = OIDMetrics()(inp.test_true, pred)
    native = {key: float(value) * 100.0
              for key, value in metrics.items() if key in ('acc', 'f1', 'f1-known', 'f1-open')}

    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, 'boundary.npz')
    np.savez_compressed(
        state_path,
        theta=theta,
        b=b,
        mu=info['mu'],
        delta=info['delta'],
        U=info['U'],
        theta0=info['theta0'],
    )

    report = {
        'dataset': config.name,
        'metrics_percent': native,
        'state_path': os.path.abspath(state_path),
    }
    with open(os.path.join(output_dir, 'report.json'), 'w') as handle:
        json.dump(report, handle, indent=2)
    return report
