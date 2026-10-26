"""Stage 2: export frozen features used by the decision-boundary fit."""
import argparse
import json
import os

import numpy as np
import torch

from methods.semmoic.manager import SemMoic
from utils.functions import set_torch_seed
from utils.logging_utils import setup_logger
from utils.runtime import make_config


def parse_args():
    parser = argparse.ArgumentParser(description='SemMoic feature export')
    parser.add_argument('--dataset', required=True, choices=['MIntRec', 'MIntRec2.0'])
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run_dir', required=True,
                        help='Directory containing the selected checkpoint')
    parser.add_argument('--output', required=True,
                        help='Output .npz file')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--bert_path', default='bert-base-uncased')
    return parser.parse_args()


def export(config, data_root, seed, run_dir, output):
    set_torch_seed(seed)
    model = SemMoic(config, data_root, with_wild=False)
    model.restore(run_dir)
    features = model.export_features(data_root)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez_compressed(output, **features)
    info = {
        'dataset': config.name,
        'seed': seed,
        'output': os.path.abspath(output),
        'train_rows': len(features['train_index']),
        'test_rows': len(features['test_true']),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return info


def main():
    args = parse_args()
    config = make_config(args.dataset, args.gpu, args.num_workers, args.bert_path)
    setup_logger('SemMoic', 'logs', f'export_{args.dataset}_seed{args.seed}.log')
    info = export(config, args.data_root, args.seed, args.run_dir, args.output)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
