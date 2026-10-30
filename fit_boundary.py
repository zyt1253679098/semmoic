"""Stage 3: fit the source-aware ellipsoidal boundary and evaluate it."""
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import json

import numpy as np

from configs import CONFIGS
from ood_detection.pba.fit import fit_and_evaluate
from utils.functions import set_torch_seed
from utils.logging_utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='SemMoic boundary fit')
    parser.add_argument('--dataset', required=True, choices=['MIntRec', 'MIntRec2.0'])
    parser.add_argument('--inputs', required=True,
                        help='Feature .npz produced by export.py')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for boundary.npz and report.json')
    parser.add_argument('--results_dir', required=True,
                        help='Directory for the per-seed result JSON')
    parser.add_argument('--seed', type=int, required=True)
    return parser.parse_args()


def fit_seed(config, inputs, output_dir, seed):
    set_torch_seed(seed)
    with np.load(inputs, allow_pickle=False) as handle:
        features = {key: handle[key] for key in handle.files}
    report = fit_and_evaluate(features, config, output_dir)
    report['seed'] = seed
    return report


def main():
    args = parse_args()
    config = CONFIGS[args.dataset]
    setup_logger('SemMoic', 'logs', f'fit_{args.dataset}_seed{args.seed}.log')
    report = fit_seed(config, args.inputs, args.output_dir, args.seed)
    os.makedirs(os.path.join(args.results_dir, args.dataset), exist_ok=True)
    path = os.path.join(args.results_dir, args.dataset, f'seed_{args.seed}.json')
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report['metrics_percent'], indent=2))


if __name__ == '__main__':
    main()
