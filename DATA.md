# Data

All scripts read datasets from `--data_root`; the example scripts take it from
the `DATA_ROOT` environment variable (default `./datasets`). Each dataset is a
directory with tab-separated label files and precomputed feature dictionaries.
Raw videos and audio are not needed for training.

## Directory layout

```
datasets/
├── MIntRec-DA2-anti/
│   ├── train.tsv                 seed + DA + anti rows
│   ├── dev.tsv                   known intents only
│   ├── test.tsv                  known intents only
│   ├── all.tsv                   optional, used by feature extraction
│   ├── video_data/
│   │   └── swin_feats_eval.pkl   {clip_id: float32[frames, 1024]}
│   └── audio_data/
│       └── audio_feats_wavlmbase.pkl  {clip_id: float32[frames, 768]}
├── MIntRec-OOD/
│   ├── test.tsv                  open intent samples
│   ├── video_data/swin_feats_eval.pkl
│   └── audio_data/wavlm_today.pkl
├── MIntRec2.0-DA-anti/
│   ├── train.tsv
│   ├── dev.tsv
│   ├── test.tsv
│   ├── all.tsv
│   ├── video_data/swin_full_eval.pkl
│   └── audio_data/audio_feats_wavlmbase.pkl
├── MIntRec2.0-OOD/
│   ├── test.tsv
│   ├── video_data/swin_full_eval.pkl
│   └── audio_data/audio_feats_wavlmbase.pkl
├── MELD-DA/                      wild data for both experiments
│   ├── all.tsv
│   ├── video_data/swin_feats_eval.pkl
│   └── audio_data/wavlm_feats_wavlmbase.pkl
├── MIntRec/                      wild data for the MIntRec2.0 experiment
│   ├── all.tsv
│   ├── video_data/swin_feats_eval.pkl
│   └── audio_data/wavlm_feats_wavlmbase.pkl
└── MIntRec2.0/                   wild data for the MIntRec experiment
    ├── all.tsv
    ├── video_data/swin_feats_eval.pkl
    └── audio_data/wavlm_feats_wavlmbase.pkl
```

All `*.pkl` files are Python dictionaries mapping the row identifier in the TSV
file to a `numpy` array.

## TSV columns

- **MIntRec / MIntRec-OOD**: `season, episode, clip, text, label`. The row
  identifier is `season_episode_clip`.
- **MIntRec2.0 / MIntRec2.0-OOD**: `Dialogue_id, Utterance_id, Text, Label,
  ...`. The row identifier is `dia<Dialogue_id>_utt<Utterance_id>`.
- **MELD-DA**: `Dialogue_ID, Utterance_ID, Utterance, label`. The row
  identifier is `<Dialogue_ID>_<Utterance_ID>`.

In the enriched training files, a label-preserving augmentation has the suffix
`-DA` and an adversarial sample has the suffix `-DA-anti`. Their seed row is
obtained by removing the suffix. The visual feature of a `-DA` / `-DA-anti` row
is the visual feature of its seed row; the acoustic feature uses the full row
identifier.

## Dataset source

MIntRec, MIntRec2.0, and MELD-DA are public datasets. The enriched `-DA` and
`-DA-anti` splits and their precomputed features are provided with the dataset
archive accompanying this submission. Place the released dataset folders under
`datasets/` before running the examples.
