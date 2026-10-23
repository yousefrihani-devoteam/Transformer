import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from Feedforward.py import FeedForward



class Encoder(nn.Module):
    def __init__(self, seq_len):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = seq_len, embedding_dim= 256)
        self.positional = nn.Embedding(num_embeddings = seq_len, embedding_dim= 256)
        self.attention = nn.MultiHeadAttention(embed_dim = 256, num_heads = 2, kdim = 256, vdim = 256)
        self.ff = FeedForward(256, 256 * 4)



    def forward (self, x):



