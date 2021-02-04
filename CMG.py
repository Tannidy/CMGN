# -*- coding: utf-8 -*-
# @Author: Yiding Tan
# @Last Modified by:   Yiding Tan,    Contact: 20210240297@fudan.edu.cn

from fastNLP.modules import ConditionalRandomField, allowed_transitions

from torch import nn
import torch
import torch.nn.functional as F

from GAT_encoder import Encoder


class CMG(nn.Module):
    def __init__(self, tag_vocab, embed, d_model, n_heads, d_k, d_v, n_layers, d_label=10,
                 fc_dropout=0.3, dropout=0.15, gpu=0, pos_embed=None, scale=False):
        """

        :param tag_vocab: fastNLP Vocabulary
        :param embed: fastNLP TokenEmbedding
        :param num_layers: number of self-attention layers
        :param d_model: input size
        :param n_head: number of head
        :param feedforward_dim: the dimension of ffn
        :param dropout: dropout in self-attention
        :param after_norm: normalization place
        :param attn_type: adatrans, naive
        :param rel_pos_embed: position embedding的类型，支持sin, fix, None. relative时可为None
        :param bi_embed: Used in Chinese scenerio
        :param fc_dropout: dropout rate before the fc layer
        """
        super().__init__()

        self.embed = embed
        embed_size = self.embed.embed_size


        self.in_fc = nn.Linear(embed_size, d_model)
        self.encoder = Encoder(d_model, n_heads, d_k, d_v, n_layers, d_label, dropout, feedforward_dim=int(2 * d_model))

        self.fc_dropout = nn.Dropout(fc_dropout)
        self.out_fc = nn.Linear(d_model, len(tag_vocab))

        trans = allowed_transitions(tag_vocab, include_start_end=True)
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, chars, target, attn_mask, attn_category):
        mask = chars.ne(0) # batch_size x len
        # attn_mask = mask.unsqueeze(1).expand(batch_size, len_q, len_q)
        chars = self.embed(chars)
        chars = self.in_fc(chars)
        chars, _ = self.encoder(chars, attn_mask.bool(), attn_category)
        chars = self.fc_dropout(chars)
        chars = self.out_fc(chars)
        logits = F.log_softmax(chars, dim=-1)

        if target is None:
            paths, _ = self.crf.viterbi_decode(logits, mask)
            return {'pred': paths}
        else:
            loss = self.crf(logits, target, mask)
            return {'loss': loss}

    def forward(self, chars, target, attn_mask, attn_category):
        return self._forward(chars, target, attn_mask, attn_category)

    def predict(self, chars, attn_mask, attn_category):
        return self._forward(chars, None, attn_mask, attn_category)