"""Fixed hyperparameters for SemMoic."""
from dataclasses import dataclass


@dataclass
class Config:
    name: str
    main_dir: str
    ood_dir: str
    wild_dirs: list[str]
    intent_labels: list[str]
    num_labels: int
    seq_len: dict[str, int]
    feat_dim: dict[str, int]

    video_feats: str
    audio_feats: str
    ood_video_feats: str
    ood_audio_feats: str
    aux_video_feats: str
    aux_audio_feats: str

    train_epochs: int
    patience: int
    min_select_epoch: int
    train_batch_size: int = 32
    eval_batch_size: int = 16
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_proportion: float = 0.1
    temperature: float = 0.07
    base_dim: int = 768
    scale: float = 16.0
    encoder_layers_v: int = 2
    encoder_layers_a: int = 2
    nheads: int = 2
    attn_dropout: float = 0.0
    relu_dropout: float = 0.1
    embed_dropout: float = 0.0
    res_dropout: float = 0.2

    rank: int = 16
    tail_max: float = 2.0
    tail_min: float = 0.5
    tol: float = 1e-4
    shape_kappa: float = 2.0
    max_iter: int = 20000
    weight_pos: float = 0.5
    weight_other: float = 0.5
    weight_da: float = 0.25
    weight_anti_abs: float = 0.5
    weight_anti_rel: float = 0.0
    weight_radius: float = 0.02
    weight_shape: float = 0.0
    b_min: float = 0.25
    b_max: float = 4.0

    device: str = 'cuda'
    num_workers: int = 0
    text_pretrained_model: str = 'bert-base-uncased'

    @property
    def ood_label_id(self) -> int:
        return self.num_labels

    @property
    def text_feat_dim(self) -> int:
        return self.feat_dim['text']

    @property
    def video_feat_dim(self) -> int:
        return self.feat_dim['video']

    @property
    def audio_feat_dim(self) -> int:
        return self.feat_dim['audio']


MINT_REC_LABELS = [
    'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize',
    'Agree', 'Taunt', 'Flaunt', 'Joke', 'Oppose',
    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange',
    'Introduce', 'Leave', 'Prevent', 'Greet', 'Ask for help',
]

MINT_REC20_LABELS = [
    'Acknowledge', 'Advise', 'Agree', 'Apologise', 'Arrange',
    'Ask for help', 'Asking for opinions', 'Care', 'Comfort', 'Complain',
    'Confirm', 'Criticize', 'Doubt', 'Emphasize', 'Explain',
    'Flaunt', 'Greet', 'Inform', 'Introduce', 'Invite',
    'Joke', 'Leave', 'Oppose', 'Plan', 'Praise',
    'Prevent', 'Refuse', 'Taunt', 'Thank', 'Warn',
]


CONFIGS: dict[str, Config] = {
    'MIntRec': Config(
        name='MIntRec',
        main_dir='MIntRec-DA2-anti',
        ood_dir='MIntRec-OOD',
        wild_dirs=['MELD-DA', 'MIntRec2.0'],
        intent_labels=MINT_REC_LABELS,
        num_labels=20,
        seq_len={'text': 30, 'video': 230, 'audio': 480},
        feat_dim={'text': 768, 'video': 1024, 'audio': 768},
        video_feats='swin_feats_eval.pkl',
        audio_feats='audio_feats_wavlmbase.pkl',
        ood_video_feats='swin_feats_eval.pkl',
        ood_audio_feats='wavlm_today.pkl',
        aux_video_feats='swin_feats_eval.pkl',
        aux_audio_feats='wavlm_feats_wavlmbase.pkl',
        train_epochs=20,
        patience=5,
        min_select_epoch=1,
        tail_max=1.5,
    ),
    'MIntRec2.0': Config(
        name='MIntRec2.0',
        main_dir='MIntRec2.0-DA-anti',
        ood_dir='MIntRec2.0-OOD',
        wild_dirs=['MELD-DA', 'MIntRec'],
        intent_labels=MINT_REC20_LABELS,
        num_labels=30,
        seq_len={'text': 50, 'video': 180, 'audio': 400},
        feat_dim={'text': 768, 'video': 1024, 'audio': 768},
        video_feats='swin_full_eval.pkl',
        audio_feats='audio_feats_wavlmbase.pkl',
        ood_video_feats='swin_full_eval.pkl',
        ood_audio_feats='audio_feats_wavlmbase.pkl',
        aux_video_feats='swin_feats_eval.pkl',
        aux_audio_feats='wavlm_feats_wavlmbase.pkl',
        train_epochs=20,
        patience=3,
        min_select_epoch=5,
        tail_max=2.0,
    ),
}
