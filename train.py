"""Stage 1: train the representation model for one seed."""
import argparse
import json
import os

import torch

from methods.semmoic.manager import SemMoic
from utils.functions import set_torch_seed
from utils.logging_utils import setup_logger
from utils.runtime import make_config


def parse_args():
    parser = argparse.ArgumentParser(description='SemMoic representation training')
    parser.add_argument('--dataset', required=True, choices=['MIntRec', 'MIntRec2.0'])
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run_dir', required=True,
                        help='Directory for the selected checkpoint')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--bert_path', default='bert-base-uncased')
    return parser.parse_args()


def train(config, data_root, seed, run_dir):
    set_torch_seed(seed)
    model = SemMoic(config, data_root, with_wild=True)
    selected_epoch = model.train()
    os.makedirs(run_dir, exist_ok=True)
    model.save(run_dir)
    info = {
        'dataset': config.name,
        'seed': seed,
        'selected_epoch': selected_epoch,
        'train_examples': len(model.train_dataset),
    }
    with open(os.path.join(run_dir, 'train_info.json'), 'w') as handle:
        json.dump(info, handle, indent=2)
    return info, model


def main():
    args = parse_args()
    config = make_config(args.dataset, args.gpu, args.num_workers, args.bert_path)
    setup_logger('SemMoic', 'logs', f'train_{args.dataset}_seed{args.seed}.log')
    info, model = train(config, args.data_root, args.seed, args.run_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
