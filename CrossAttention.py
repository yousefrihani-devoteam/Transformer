import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, num_embed, num_heads, dropout=0.2, seq_length=None):
        super(CrossAttention, self).__init__()

        assert num_embed % num_heads == 0, "num_embed must be divisible by num_heads"
        self.embedding_dim = num_embed
        self.num_heads = num_heads
        self.d_head = self.embedding_dim // self.num_heads

        self.W_q = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_k = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_v = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_o = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.d_head).transpose(1, 2)

    def combine_heads(self, x):
        batch, head, sequence, dimension = x.shape
        return x.transpose(1, 2).contiguous().view(batch, sequence, head * dimension)

    # Aliases for backwards compatibility
    SplitHeads = split_heads
    CombineHeads = combine_heads

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V              
        return self.W_o(self.combine_heads(out))