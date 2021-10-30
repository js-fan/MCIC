import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False,
            add_zero_attn=False, kdim=None, vdim=None, same_qk=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.same_qk = same_qk
        assert self.head_dim * num_heads == self.embed_dim

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias)
        if self.same_qk:
            self.linear_k = self.linear_q
        else:
            self.linear_k = nn.Linear(embed_dim, self.kdim, bias)

        self.linear_v = nn.Linear(embed_dim, self.vdim, bias)
        self.linear_o = nn.Linear(embed_dim, embed_dim, True)

        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.bias = bias

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.linear_q.weight)
        if not self.same_qk:
            nn.init.xavier_normal_(self.linear_k.weight)
        nn.init.xavier_normal_(self.linear_v.weight)
        nn.init.xavier_normal_(self.linear_o.weight)
        nn.init.zeros_(self.linear_o.bias)
        if self.bias:
            nn.init.zeros_(self.linear_q.bias)
            if not self.same_qk:
                nn.init.zeros_(self.linear_k.bias)
            nn.init.zeros_(self.linear_v.bias)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, only_attn_weights=False):
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)

        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        scaling = float(self.head_dim) ** -0.5
        q = q * scaling

        if self.add_bias_kv:
            raise NotImplementedError

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool, attn_mask.dtype
            assert list(attn_mask.size()) == [bsz * self.num_heads, tgt_len, src_len], attn_mask.size()

        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool, key_padding_mask.dtype
            assert list(key_padding_mask.size()) == [bsz, src_len], key_padding_mask.size()

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((bsz*self.num_heads, 1, self.head_dim), dtype=k.dtype, device=k.device)], 1)
            v = torch.cat([v, torch.zeros((bsz*self.num_heads, 1, self.head_dim), dtype=v.dtype, device=v.device)], 1)

            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len], attn_output_weights.size()

        if attn_mask is not None:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if only_attn_weights:
            attn_output_weights_ = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len).max(1)[0]
            return attn_output_weights_

        attn_output_weights_norm = F.softmax(attn_output_weights, -1)
        attn_output_weights_norm = F.dropout(attn_output_weights_norm, p=self.dropout, training=self.training)
        #attn_output_weights_norm_q = F.softmax(attn_output_weights, -2)
        
        attn_output = torch.bmm(attn_output_weights_norm, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim], attn_output.size()
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn_output = self.linear_o(attn_output)

        attn_output_weights_norm_ = attn_output_weights_norm.view(bsz, self.num_heads, tgt_len, src_len).mean(1)
        attn_output_weights_ = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len).mean(1)
        # attn_output_weights_norm_ = attn_output_weights_norm.view(bsz, self.num_heads, tgt_len, src_len)
        # attn_output_weights_ = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights_norm_, attn_output_weights_

