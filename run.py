"""Run SemMoic end to end: training, feature export, and boundary fitting."""
import os

# The boundary solver must see a single-threaded BLAS for bit-reproducible fits.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import gc
import json
import statistics

import torch

from data import clear_feature_cache
from export import export
from fit_boundary import fit_seed
from train import train
from utils.logging_utils import setup_logger
from utils.runtime import make_config


def parse_args():
    parser = argparse.ArgumentParser(description='SemMoic release')
    parser.add_argument('--dataset', required=True, choices=['MIntRec', 'MIntRec2.0'])
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--bert_path', default='bert-base-uncased')
    return parser.parse_args()


def save_seed_result(results_dir, dataset, seed, report):
    directory = os.path.join(results_dir, dataset)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'seed_{seed}.json')
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=2)
    return path


def summarize(results_dir, dataset, seeds):
    rows = []
    for seed in seeds:
        path = os.path.join(results_dir, dataset, f'seed_{seed}.json')
        with open(path) as handle:
            rows.append(json.load(handle)['metrics_percent'])
    keys = ['acc', 'f1', 'f1-known', 'f1-open']
    summary = {}
    for key in keys:
        values = [row[key] for row in rows]
        summary[key] = {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'values': values,
        }
    path = os.path.join(results_dir, dataset, 'summary.json')
    with open(path, 'w') as handle:
        json.dump({'seeds': seeds, 'metrics': summary}, handle, indent=2)

    csv_path = os.path.join(results_dir, dataset, 'summary.csv')
    with open(csv_path, 'w') as handle:
        handle.write('seed,acc,f1,f1-known,f1-open\n')
        for seed, row in zip(seeds, rows):
            handle.write(f'{seed},{row["acc"]:.4f},{row["f1"]:.4f},'
                         f'{row["f1-known"]:.4f},{row["f1-open"]:.4f}\n')
        handle.write('mean,{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(*tuple(summary[key]['mean'] for key in keys)))
        handle.write('std,{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(*tuple(summary[key]['std'] for key in keys)))
    return summary


def main():
    args = parse_args()
    logger = setup_logger('SemMoic', 'logs', f'run_{args.dataset}.log')
    config = make_config(args.dataset, args.gpu, args.num_workers, args.bert_path)
    seeds = [int(x) for x in args.seeds.split(',') if x.strip() != '']

    for seed in seeds:
        logger.info('===== %s seed %d: representation training =====', args.dataset, seed)
        run_dir = os.path.join(args.output_dir, args.dataset, f'seed_{seed}')
        train_info, model = train(config, args.data_root, seed, run_dir)
        del model
        clear_feature_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info('training done: %s', train_info)

        logger.info('===== %s seed %d: feature export =====', args.dataset, seed)
        inputs = os.path.join(run_dir, 'features.npz')
        export_info = export(config, args.data_root, seed, run_dir, inputs)
        logger.info('export done: %s', export_info)

        logger.info('===== %s seed %d: boundary fit =====', args.dataset, seed)
        report = fit_seed(config, inputs, run_dir, seed)
        report['selected_epoch'] = train_info['selected_epoch']
        path = save_seed_result(args.results_dir, args.dataset, seed, report)
        logger.info('metrics: %s', report['metrics_percent'])
        logger.info('saved %s', path)

    summary = summarize(args.results_dir, args.dataset, seeds)
    logger.info('summary (%s, seeds=%s)', args.dataset, seeds)
    for key in ('acc', 'f1', 'f1-known', 'f1-open'):
        item = summary[key]
        logger.info('  %-9s %.2f +/- %.2f', key, item['mean'], item['std'])


if __name__ == '__main__':
    main()
