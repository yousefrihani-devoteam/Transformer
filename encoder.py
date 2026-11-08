import torch
import torch.nn as nn
import torch.nn.functional as F

from CrossAttention import CrossAttention
from FeedForward import FeedForward

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model=256, num_heads=8, max_seq_len=512, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=d_model)
        self.positional = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=d_model)
        self.attention = CrossAttention(num_embed=d_model, num_heads=num_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_model * 4, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_len = x.size(1)

        arange = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positional = self.positional(arange)
        embeddings = self.embedding(x)
        x_processed = self.dropout(embeddings + positional)

        attn_out = self.attention(x_processed, x_processed, x_processed, mask)
        X = self.norm1(x_processed + self.dropout(attn_out))

        ff_res = self.ff(X)
        X = self.norm2(X + self.dropout(ff_res))

        return X
