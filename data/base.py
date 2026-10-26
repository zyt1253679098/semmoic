"""Dataset construction from the released enriched data."""
import csv
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import FeatureList, load_feature_dict

_FEATURE_CACHE: dict[str, dict] = {}


def _load_cached(path):
    key = os.path.realpath(path)
    if key not in _FEATURE_CACHE:
        _FEATURE_CACHE[key] = load_feature_dict(path)
    return _FEATURE_CACHE[key]


def clear_feature_cache():
    _FEATURE_CACHE.clear()


def _read_rows(path):
    with open(path, 'r', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        next(reader)
        return list(reader)


def _row_index(config, row):
    if config.name == 'MIntRec':
        return '_'.join([row[0], row[1], row[2]])
    return f'dia{row[0]}_utt{row[1]}'


def _source_row(directory, row):
    """Index and text of one wild source, following each source's own format."""
    if directory == 'MELD-DA':
        return f'{row[0]}_{row[1]}', row[2]
    if directory == 'MIntRec':
        return '_'.join([row[0], row[1], row[2]]), row[3]
    return f'dia{row[0]}_utt{row[1]}', row[2]


def _row_text(config, row):
    return row[3] if config.name == 'MIntRec' else row[2]


def _row_label(config, row):
    return row[4] if config.name == 'MIntRec' else row[3]


def _row_kind(index):
    if index.endswith('-DA-anti'):
        return 'anti', index[:-len('-DA-anti')]
    if index.endswith('-DA'):
        return 'da', index[:-len('-DA')]
    return 'seed', index


def _video_key(index):
    """DA/anti rows reuse the visual clip of their seed sample."""
    return _row_kind(index)[1]


@dataclass
class Split:
    indices: list[str]
    labels: list[int]
    text_feats: np.ndarray
    video: FeatureList
    audio: FeatureList


class MMDataset(Dataset):
    def __init__(self, split: Split):
        self.labels = split.labels
        self.text_feats = split.text_feats
        self.video = split.video
        self.audio = split.audio

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        video, video_len = self.video.get(index)
        audio, audio_len = self.audio.get(index)
        return {
            'label_ids': torch.tensor(self.labels[index], dtype=torch.long),
            'text_feats': torch.from_numpy(self.text_feats[index]),
            'video_feats': torch.from_numpy(video),
            'video_lengths': torch.tensor(video_len, dtype=torch.long),
            'audio_feats': torch.from_numpy(audio),
            'audio_lengths': torch.tensor(audio_len, dtype=torch.long),
        }


class _Source:
    def __init__(self, indices, text_feats, video, audio):
        self.indices = indices
        self.text_feats = text_feats
        self.video = video
        self.audio = audio


class WildDataset(Dataset):
    """Unlabeled wild samples, resampled every five epochs."""

    def __init__(self, sources: list[_Source], size: int, ood_label_id: int):
        self.sources = sources
        self.size = int(size)
        self.ood_label_id = int(ood_label_id)
        self.rows = []

    def _locate(self):
        offsets, total = [], 0
        for source in self.sources:
            offsets.append((total, total + len(source.indices)))
            total += len(source.indices)
        return offsets, total

    def resample(self):
        offsets, total = self._locate()
        count = min(self.size, total)
        selected = random.sample(list(range(total)), count)
        selected.sort()
        rows = []
        for source_id, (start, end) in enumerate(offsets):
            local = [global_id - start for global_id in selected if start <= global_id < end]
            rows.extend((source_id, local_id) for local_id in local)
        if len(rows) != count:
            raise RuntimeError(f'wild sampling produced {len(rows)} rows, expected {count}')
        self.rows = rows

    def __len__(self):
        return len(self.rows) if self.rows else self.size

    def __getitem__(self, index):
        source_id, local_id = self.rows[index]
        source = self.sources[source_id]
        video, video_len = source.video.get(local_id)
        audio, audio_len = source.audio.get(local_id)
        return {
            'label_ids': torch.tensor(self.ood_label_id, dtype=torch.long),
            'text_feats': torch.from_numpy(source.text_feats[local_id]),
            'video_feats': torch.from_numpy(video),
            'video_lengths': torch.tensor(video_len, dtype=torch.long),
            'audio_feats': torch.from_numpy(audio),
            'audio_lengths': torch.tensor(audio_len, dtype=torch.long),
        }


def _build_split(config, data_root, directory, tsv_name, tokenizer,
                 video_feats, audio_feats, allow_anti=False):
    rows = _read_rows(os.path.join(data_root, directory, tsv_name))
    indices, labels, texts = [], [], []
    for row in rows:
        index = _row_index(config, row)
        kind, _ = _row_kind(index)
        if kind == 'anti':
            if not allow_anti:
                continue
            target = _row_label(config, row)
            label = config.intent_labels.index(target[len('anti-'):]) + config.ood_label_id + 1
        else:
            label = config.intent_labels.index(_row_label(config, row))
        indices.append(index)
        labels.append(label)
        texts.append(_row_text(config, row))

    text_feats = tokenizer(texts)
    video_dict = _load_cached(os.path.join(data_root, directory, 'video_data', video_feats))
    audio_dict = _load_cached(os.path.join(data_root, directory, 'audio_data', audio_feats))
    video = FeatureList(video_dict, [_video_key(index) for index in indices], config.seq_len['video'])
    audio = FeatureList(audio_dict, indices, config.seq_len['audio'])
    return Split(indices=indices, labels=labels, text_feats=text_feats, video=video, audio=audio)


def build_train(config, data_root, tokenizer, include_anti=False):
    return _build_split(config, data_root, config.main_dir, 'train.tsv', tokenizer,
                        config.video_feats, config.audio_feats, allow_anti=include_anti)


def build_dev(config, data_root, tokenizer):
    return _build_split(config, data_root, config.main_dir, 'dev.tsv', tokenizer,
                        config.video_feats, config.audio_feats)


def build_test(config, data_root, tokenizer):
    known = _build_split(config, data_root, config.main_dir, 'test.tsv', tokenizer,
                         config.video_feats, config.audio_feats)
    rows = _read_rows(os.path.join(data_root, config.ood_dir, 'test.tsv'))
    indices = [_row_index(config, row) for row in rows]
    texts = [_row_text(config, row) for row in rows]
    text_feats = tokenizer(texts)
    video_dict = _load_cached(os.path.join(data_root, config.ood_dir, 'video_data',
                                           config.ood_video_feats))
    audio_dict = _load_cached(os.path.join(data_root, config.ood_dir, 'audio_data',
                                           config.ood_audio_feats))
    video = FeatureList(video_dict, [_video_key(index) for index in indices], config.seq_len['video'])
    audio = FeatureList(audio_dict, indices, config.seq_len['audio'])
    labels = [config.ood_label_id] * len(indices)
    return Split(indices=known.indices + indices,
                 labels=known.labels + labels,
                 text_feats=np.concatenate([known.text_feats, text_feats], axis=0),
                 video=_MergedFeatureList(known.video, video),
                 audio=_MergedFeatureList(known.audio, audio))


class _MergedFeatureList:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def get(self, index):
        if index < len(self.first):
            return self.first.get(index)
        return self.second.get(index - len(self.first))


def build_wild(config, data_root, tokenizer, train_size):
    sources = []
    for directory in config.wild_dirs:
        rows = _read_rows(os.path.join(data_root, directory, 'all.tsv'))
        pairs = [_source_row(directory, row) for row in rows]
        indices = [pair[0] for pair in pairs]
        texts = [pair[1] for pair in pairs]
        text_feats = tokenizer(texts)
        video_dict = _load_cached(os.path.join(data_root, directory, 'video_data',
                                               config.aux_video_feats))
        audio_dict = _load_cached(os.path.join(data_root, directory, 'audio_data',
                                               config.aux_audio_feats))
        video = FeatureList(video_dict, indices, config.seq_len['video'])
        audio = FeatureList(audio_dict, indices, config.seq_len['audio'])
        sources.append(_Source(indices, text_feats, video, audio))
    return WildDataset(sources, size=min(train_size, 4000), ood_label_id=config.ood_label_id)
