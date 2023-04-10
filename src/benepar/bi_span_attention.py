import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class FeatureDropoutFunction(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, but got {}".format(p)
            )

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = torch.empty(
                (input.size(0), input.size(-1)),
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise[:, None, :]
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(ctx.noise), None, None, None
        else:
            return grad_output, None, None, None


class FeatureDropout(nn.Dropout):
    """
    Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.
    """

    def forward(self, x):
        return FeatureDropoutFunction.apply(x, self.p, self.training, self.inplace)


class biAttention(nn.Module):
    def __init__(
        self, d_model, n_head, d_qkv, dropout,
        ):
        super().__init__()

        self.head_num = n_head # 8
        self.head_size = d_qkv # 128
        self.all_head_size = self.head_num * self.head_size 

        self.query_layer = nn.Linear(d_model, self.all_head_size) 
        self.key_layer = nn.Linear(d_model, self.all_head_size) 
        self.value_layer = nn.Linear(d_model, int(self.all_head_size/2))
        
        self.out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.head_num, int(x.size(-1)/self.head_num))
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def look_ahead_masking(self, attention_score, mask_val=-1e4):
        mask = (torch.triu(torch.ones(attention_score.size()[2], attention_score.size()[3])) == 1).transpose(0, 1).float()
        mask = mask.expand_as(attention_score).to(attention_score.device)
        mask = torch.logical_not(mask.bool())
        return attention_score.masked_fill(mask,mask_val)

    def look_behind_masking(self, attention_score, mask_val=-1e4):
        mask = (torch.triu(torch.ones(attention_score.size()[2], attention_score.size()[3])) == 1).float()
        mask = mask.expand_as(attention_score).to(attention_score.device)
        mask = torch.logical_not(mask.bool())
        return attention_score.masked_fill(mask,mask_val) 

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

        forward_attention_score = self.look_ahead_masking(attention_score)
        backward_attention_score = self.look_behind_masking(attention_score)
        
        forward_attention_prob = self.dropout(F.softmax(forward_attention_score, dim=-1))
        backward_attention_prob = self.dropout(F.softmax(backward_attention_score, dim=-1))

        # update value matrix
        forward_output_matrix = self.output_matrix_reshape(torch.matmul(forward_attention_prob, value))
        backward_output_matrix = self.output_matrix_reshape(torch.matmul(backward_attention_prob, value))
        # [bs,seq, emb/2]

        output_matrix = torch.cat((forward_output_matrix,backward_output_matrix),-1)

        output_matrix=self.out(output_matrix)

        attention_prob = torch.cat((forward_attention_prob,backward_attention_prob),-1)

        return output_matrix, attention_prob

class BiTransformer_layer(nn.Module):
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

        self.self_attn = biAttention(d_model, n_head, d_qkv, attention_dropout)

        self.ff_layer_1 = nn.Linear(d_model, d_ff)
        self.ff_layer_2 = nn.Linear(d_ff, d_model)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.residual_dropout_attn = FeatureDropout(residual_dropout)
        self.dropout_ff = FeatureDropout(ff_dropout)
        self.residual_dropout_ff = FeatureDropout(residual_dropout)

        self.activation = activation

    def forward(self, x, _, padding_mask):

        residual, self_attn_score = self.self_attn(input_matrix=x, padding_mask=padding_mask)
        residual = self.residual_dropout_attn(residual)
        residual = self.layer_norm_1(x+residual)

        residual = self.ff_layer_2(self.dropout_ff(self.activation(self.ff_layer_1(residual))))
        residual = self.residual_dropout_ff(residual)
        x = self.layer_norm_2(x+residual)

        return x    #, self_attn_score

class SpanAttention(nn.Module):
    def __init__(
        self, d_model, n_head, d_qkv, dropout,
        ):
        super().__init__()

        self.head_num = n_head # 8
        self.head_size = d_qkv
        self.all_head_size = self.head_num * self.head_size 

        self.ngram_query_layer = nn.Linear(d_model, self.all_head_size)
        self.ngram_key_layer = nn.Linear(d_model, self.all_head_size)   
        self.ngram_value_layer = nn.Linear(d_model, self.all_head_size) 
        
        self.out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.head_num, int(x.size(-1)/self.head_num))
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def look_ahead_masking(self, attention_score, mask_val=-1e4):
        mask = (torch.triu(torch.ones(attention_score.size()[2], attention_score.size()[3])) == 1).transpose(0, 1).float()
        mask = mask.expand_as(attention_score).to(attention_score.device)
        mask = torch.logical_not(mask.bool())
        return attention_score.masked_fill(mask,mask_val) 

    def look_behind_masking(self, attention_score, mask_val=-1e4):
        mask = (torch.triu(torch.ones(attention_score.size()[2], attention_score.size()[3])) == 1).float()
        mask = mask.expand_as(attention_score).to(attention_score.device)
        mask = torch.logical_not(mask.bool())
        return attention_score.masked_fill(mask,mask_val) 

    def padding_masking(self, attention_score, pad_mask, mask_val=-1e4):
        pad_mask = torch.unsqueeze(pad_mask, 1)
        pad_mask = torch.unsqueeze(pad_mask, 1)
        # pad_mask = [batch_size, 1, 1, seq_len]
        return attention_score.masked_fill(pad_mask, mask_val)

    def key_masking(self, attention_score, pad_mask, mask_val=-1e4):
        pad_mask = torch.unsqueeze(pad_mask, 1)
        pad_mask = torch.unsqueeze(pad_mask, 3)
        # pad_mask = [batch_size, 1, seq_len, 1]
        return attention_score.masked_fill(pad_mask, mask_val)

    def output_matrix_reshape(self, output_matrix):
        output_matrix = output_matrix.permute(0, 2, 1, 3).contiguous()
        output_shape = output_matrix.size()[:-2] + (output_matrix.size(2)*output_matrix.size(3),) # self.all_head_size
        output_matrix = output_matrix.view(output_shape)
        return output_matrix

    def make_ngram_matrix(self, n, output_matrix, direction):
        # ngram = output_matrix
        ngram = output_matrix[:,1:] - output_matrix[:,:-1]

        if direction == 'forward':
            for i in range(1,n):
                ngram = torch.cat((ngram, output_matrix[:,i+1:] - output_matrix[:,:-(i+1)]),1)
        else:
            for i in range(1,n):
                ngram = torch.cat((ngram, output_matrix[:,:-(i+1)] - output_matrix[:,i+1:]),1)

        return ngram

    def make_ngram_padding(self, n, padding):
        # ngram_padding = padding
        forward_padding = padding[:,1:]

        for i in range(1,n):
            forward_padding = torch.cat((forward_padding, padding[:,i+1:]), -1)

        backward_padding = padding[:,:-1]

        for i in range(1,n):
            backward_padding = torch.cat((backward_padding, padding[:,:-(i+1)]), -1)

        return forward_padding, backward_padding

    def attention(self, query, key, value, padding_mask):

        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        # attention score between query and key
        attention_score = torch.matmul(query, key.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.head_size)

        if padding_mask is not None:
            attention_score = self.padding_masking(attention_score, padding_mask)

        attention_prob = self.dropout(F.softmax(attention_score, dim=-1))

        output_matrix = self.output_matrix_reshape(torch.matmul(attention_prob, value))

        return output_matrix

    def forward(self, input_matrix, ngram_num=3, padding_mask=None):

        ngram_query_f, ngram_query_b = torch.chunk(self.ngram_query_layer(input_matrix), 2, dim=-1)
        # [bs,seq, emb/2]

        forward_matrix, backward_matrix = torch.chunk(input_matrix, 2, dim=-1)

        forward_ngram_matrix = self.make_ngram_matrix(ngram_num, output_matrix=forward_matrix, direction='forward')
        backward_ngram_matrix = self.make_ngram_matrix(ngram_num, output_matrix=backward_matrix, direction='backward')
        # [bs,seq(1-gram+2-gram+3-gram), emb/2]
        ngram_matrix = torch.cat((forward_ngram_matrix,backward_ngram_matrix),-1)

        ngram_key_f, ngram_key_b = torch.chunk(self.ngram_key_layer(ngram_matrix), 2, dim=-1)
        # [bs,seq(1-gram+2-gram+3-gram), emb/2]

        ngram_value_f, ngram_value_b = torch.chunk(self.ngram_value_layer(ngram_matrix), 2, dim=-1)
        # [bs,seq(1-gram+2-gram+3-gram), emb/2]

        forward_padding, backward_padding = self.make_ngram_padding(ngram_num, padding_mask)
        
        forward_out_matrix = self.attention(ngram_query_f, ngram_key_f, ngram_value_f, forward_padding)
        backward_out_matrix = self.attention(ngram_query_b, ngram_key_b, ngram_value_b, backward_padding)

        output_matrix = torch.cat((forward_out_matrix,backward_out_matrix),-1)

        output_matrix=self.out(output_matrix)
        attention_prob=None

        return output_matrix, attention_prob

class SpanTransformer_layer(nn.Module):
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
        self.self_attn = SpanAttention(d_model, n_head, d_qkv, attention_dropout)

        self.ff_layer_1 = nn.Linear(d_model, d_ff)
        self.ff_layer_2 = nn.Linear(d_ff, d_model)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.residual_dropout_attn = FeatureDropout(residual_dropout)
        self.dropout_ff = FeatureDropout(ff_dropout)
        self.residual_dropout_ff = FeatureDropout(residual_dropout)

        self.activation = activation

    def forward(self, x, ngram_num, padding_mask):

        residual, self_attn_score = self.self_attn(input_matrix=x, ngram_num=ngram_num, padding_mask=padding_mask)
        residual = self.residual_dropout_attn(residual)
        residual = self.layer_norm_1(x+residual)

        residual = self.ff_layer_2(self.dropout_ff(self.activation(self.ff_layer_1(residual))))
        residual = self.residual_dropout_ff(residual)
        x = self.layer_norm_2(x+residual)

        return x    #, self_attn_score

class BiSpanTransformerEncoder(nn.Module):
    def __init__(self, n_layers, encoder_layer=None, even_layer=None, end_of_layers=1):
        super().__init__()
        # if end_of_layer:
        #     self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) if i != (n_layers-1) else copy.deepcopy(even_layer) for i in range(n_layers)])
        # else:
        #     if encoder_layer is not None and even_layer is not None:
        #         self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) if i%2==0 else copy.deepcopy(even_layer) for i in range(n_layers)])
        #     else:
        #         if encoder_layer is None:
        #             encoder_layer = even_layer
        #         self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layers)])

        
        if encoder_layer is not None and even_layer is not None:
            if end_of_layers == 0:
                self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) if i%2==0 else copy.deepcopy(even_layer) for i in range(n_layers)])
            else:
                self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) if i <= (n_layers - end_of_layers) else copy.deepcopy(even_layer) for i in range(1, n_layers+1)])
        else:
            if encoder_layer is None:
                encoder_layer = even_layer
            self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layers)])

    def forward(self, x, ngram_num, attention_mask):

        for layer in self.layers:
            x = layer(x, ngram_num, padding_mask=attention_mask)

        return x

