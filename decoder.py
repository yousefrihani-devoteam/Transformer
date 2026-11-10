import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from CrossAttention import CrossAttention
from FeedForward import FeedForward

class Decoder(nn.Module):
    def __init__(
        self,
        Seq_length=512,
        Trt_vocab_size=None,
        d_model=256,
        dropout=0.1,
        num_heads=2,
        tgt_vocab_size=None,
        max_seq_len=None
    ):
        super(Decoder, self).__init__()
        vocab_size = Trt_vocab_size if Trt_vocab_size is not None else (tgt_vocab_size if tgt_vocab_size is not None else 50257)
        seq_len = max_seq_len if max_seq_len is not None else Seq_length

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional = nn.Embedding(num_embeddings=seq_len, embedding_dim=d_model)
        self.mask_attention = CrossAttention(num_embed=d_model, seq_length=seq_len, num_heads=num_heads)
        self.cross_attention = CrossAttention(num_embed=d_model, seq_length=seq_len, num_heads=num_heads)
        self.ff = FeedForward(d_model, d_model * 4)

        self.relu = nn.ReLU()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        seq_len = x.size(1)

        arange = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positional = self.positional(arange)
        embeddings = self.embedding(x)
        x_processed = embeddings + positional

        X = self.mask_attention.forward(x_processed, x_processed, x_processed, tgt_mask)
        Xn1 = self.dropout(self.norm1(X + x_processed))

        X = self.cross_attention(Xn1, encoder_output, encoder_output, src_mask)
        Xn2 = self.norm2(X + Xn1)

        X = self.ff.forward(Xn2)
        X = self.norm3(X + Xn2)

        return self.ln(X)
