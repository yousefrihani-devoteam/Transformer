import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from Feedforward.py import FeedForward



def Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = seq_len, embedding_dim= 256)
        self.positional = nn.Embedding(num_embeddings = seq_len, embedding_dim= 256)
        self.attention = nn.MultiHeadAttention(embed_dim = 256, num_heads = 2, kdim = 256, vdim = 256)
        self.ff = FeedForward(256, 256 * 4)
        
        self.norm = nn.LayerNorm(256)


    def forward(self, x):
        seq_len = x.size(1)
        
        arange = torch.arange(0, seq_len, dtype=torch.long, device = x.device)
        positional = self.positional(arange)
        embeddings = self.embedding(x)
        x_processed = embeddings + positional 
        
        X = self.attention(x_processed)
        X = self.norm(X + x_processed)

        


        
