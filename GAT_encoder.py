# -*- coding: utf-8 -*-
# @Author: Yiding Tan
# @Last Modified by:   Yiding Tan,    Contact: 20210240297@fudan.edu.cn


import torch
import math
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask, d_k=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.dropout_layer(attn)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_label):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask, self.d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=None, d_ff=None):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_label, dropout, feedforward_dim):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v, d_label)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(feedforward_dim, d_model),
                                 nn.Dropout(dropout))
        # self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        # enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        enc_outputs = self.norm(enc_outputs)
        return enc_outputs, attn

def get_attn_pad_mask(seq_q, seq_k):
    print(seq_q.size())
    print(seq_k.size())
    batch_size, len_q, _ = seq_q.size()
    batch_size, len_k, _ = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, n_layers, d_label, dropout, feedforward_dim):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_label, dropout, feedforward_dim) for _ in range(n_layers)])
        # self.pos_embed = nn.Embedding.from_pretrained(self.get_position_embed_table(1024, d_model), freeze=True)

    def get_position_embed_table(self, seq_len, d_model):
        '''
        get positional embedding
        '''

        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_model)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(seq_len)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table)
        # return torch.FloatTensor(sinusoid_table).to(torch.device("cuda:" + str(gpu)))

    def forward(self, enc_inputs, attn_mask, attn_category): # enc_inputs : [batch_size x source_len x embed_dim]
        enc_self_attn_mask = attn_mask
        batch_size, seq_len, _ = enc_inputs.size()
        enc_self_attns = []
        # enc_outputs = enc_inputs + self.pos_embed(torch.LongTensor([list(range(seq_len))]).to(torch.device("cuda:" + str(self.gpu)))).expand(batch_size, seq_len, self.d_model)
        enc_outputs = enc_inputs
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

if __name__ == '__main__':
    batch_size = 8
    seq_len = 10
    dim = 5
    size_q = [batch_size, seq_len, dim]
    size_k = [batch_size, seq_len, seq_len, dim]
    rand_q = torch.rand(size_q)  # [0,1)内的均匀分布随机数
    rand_like_q = torch.rand_like(rand_q)  # 返回跟rand的tensor一样size的0-1随机数
    q = torch.randn(size_q)  # 返回标准正太分布N(0,1)的随机数
    print(q)

    rand_k = torch.rand(size_k)  # [0,1)内的均匀分布随机数
    rand_like_k = torch.rand_like(rand_k)  # 返回跟rand的tensor一样size的0-1随机数
    k = torch.randn(size_k)  # 返回标准正太分布N(0,1)的随机数
    v = k
    q = q.unsqueeze(1).expand(batch_size, seq_len, seq_len, dim)
    print(list(range(10)))




