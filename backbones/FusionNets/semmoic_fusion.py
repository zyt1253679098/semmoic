import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder

from ..SubNets.transformers_encoder.transformer import TransformerEncoder


class Fusion_ADD(nn.Module):
    def __init__(self, args):
        super().__init__()
        audio_feat_dim = args.audio_feat_dim
        video_feat_dim = args.video_feat_dim

        self.num_heads = args.nheads
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout

        self.v_encoder = self.get_transformer_encoder(video_feat_dim, args.encoder_layers_v)
        self.a_encoder = self.get_transformer_encoder(audio_feat_dim, args.encoder_layers_a)
        self.audio_proj = nn.Linear(audio_feat_dim, args.base_dim)
        self.video_proj = nn.Linear(video_feat_dim, args.base_dim)
        self.w_t = nn.Parameter(torch.zeros((2,)))
        self.w_v = nn.Parameter(torch.zeros((2,)))
        self.w_a = nn.Parameter(torch.zeros((2,)))

    def forward(self, textual, visual, acoustic):
        fusion = F.softplus(self.w_t[0]) * textual[:, 0]

        visual = visual.permute(1, 0, 2)
        visual = self.v_encoder(visual).mean(dim=0)
        visual = self.video_proj(visual)
        fusion = fusion + visual * F.softplus(self.w_v[0])

        acoustic = acoustic.permute(1, 0, 2)
        acoustic = self.a_encoder(acoustic).mean(dim=0)
        acoustic = self.audio_proj(acoustic)
        fusion = fusion + acoustic * F.softplus(self.w_a[0])

        return fusion

    def get_transformer_encoder(self, embed_dim, layers):
        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=layers,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
        )


class Fusion_BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        input_shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device)
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids)
        text_encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )
        return text_encoder_outputs[0]


class Fusion_Model(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = Fusion_BertModel(config)
        self.fusion = Fusion_ADD(args)
        self.init_weights()

    def forward(self, text, visual, acoustic):
        input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]
        textual = self.bert(input_ids, attention_mask, token_type_ids)
        return self.fusion(textual, visual, acoustic)


class Fusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = Fusion_Model.from_pretrained(args.text_pretrained_model, args=args)

    def forward(self, text_feats, video_feats, audio_feats):
        video_feats, audio_feats = video_feats.float(), audio_feats.float()
        pooled_output = self.model(text_feats, video_feats, audio_feats)
        return F.normalize(pooled_output)
