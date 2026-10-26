import os

import torch

from configs import CONFIGS


def make_config(dataset, gpu='0', num_workers=2, bert_path='bert-base-uncased'):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    config = CONFIGS[dataset]
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.num_workers = int(num_workers)
    config.text_pretrained_model = bert_path
    return config
