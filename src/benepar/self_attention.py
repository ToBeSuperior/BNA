import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model, n_head, d_qkv, dropout,
        ):
        super().__init__()

        self.head_num = n_head # 8
        self.head_size = d_qkv # 128
        self.all_head_size = self.head_num * self.head_size 

        self.query_layer = nn.Linear(d_model, self.all_head_size) 
        self.key_layer = nn.Linear(d_model, self.all_head_size) 
        self.value_layer = nn.Linear(d_model, self.all_head_size)
        
        self.out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.head_num, int(x.size(-1)/self.head_num))
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def look_ahead_masking(self, attention_score):
        mask = (torch.triu(torch.ones(attention_score.size()[2], attention_score.size()[3])) == 1).transpose(0, 1).float()
        mask = mask.expand_as(attention_score).to(attention_score.device)
        return attention_score * mask

    def look_behind_masking(self, attention_score):
        mask = (torch.triu(torch.ones(attention_score.size()[2], attention_score.size()[3])) == 1).float()
        mask = mask.expand_as(attention_score).to(attention_score.device)
        return attention_score * mask

    def padding_masking(self, attention_score, pad_mask, mask_val=-1e4):
        pad_mask = torch.unsqueeze(pad_mask, 1)
        pad_mask = torch.unsqueeze(pad_mask, 1)
        # pad_mask = [batch_size, 1, 1, seq_len]
        return attention_score.masked_fill(pad_mask, mask_val)

    def output_matrix_reshape(self, output_matrix):
        output_matrix = output_matrix.permute(0, 2, 1, 3).contiguous()
        output_shape = output_matrix.size()[:-2] + (output_matrix.size(2)*output_matrix.size(3),) # self.all_head_size
        output_matrix = output_matrix.view(output_shape)
        return output_matrix

    def forward(self, input_matrix, padding_mask=None):

        query = self.transpose_for_scores(self.query_layer(input_matrix))
        key = self.transpose_for_scores(self.key_layer(input_matrix))
        value = self.transpose_for_scores(self.value_layer(input_matrix))

        # attention score between query and key
        attention_score = torch.matmul(query, key.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.head_size)

        if padding_mask is not None:
            attention_score = self.padding_masking(attention_score, padding_mask)
        
        attention_prob = self.dropout(F.softmax(attention_score, dim=-1))

        # update value matrix
        output_matrix = self.output_matrix_reshape(torch.matmul(attention_prob, value))
        # [bs,seq, emb/2]

        output_matrix=self.out(output_matrix)

        return output_matrix, attention_prob

class TransformerEncoder_layer(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        d_qkv,
        d_ff,
        ff_dropout=0.1,
        residual_dropout=0.1,
        attention_dropout=0.1,
        activation=nn.LeakyReLU(), 
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_head, d_qkv, attention_dropout)

        self.ff_layer_1 = nn.Linear(d_model, d_ff)
        self.ff_layer_2 = nn.Linear(d_ff, d_model)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.residual_dropout_attn = nn.Dropout(residual_dropout)
        self.dropout_ff = nn.Dropout(ff_dropout)
        self.residual_dropout_ff = nn.Dropout(residual_dropout)

        self.activation = activation

    def forward(self, x, _, padding_mask):

        residual, self_attn_score = self.self_attn(input_matrix=x, padding_mask=padding_mask)
        residual = self.residual_dropout_attn(residual)
        residual = self.layer_norm_1(x+residual)

        residual = self.ff_layer_2(self.dropout_ff(self.activation(self.ff_layer_1(residual))))
        residual = self.residual_dropout_ff(residual)
        x = self.layer_norm_2(x+residual)

        return x    #, self_attn_score
