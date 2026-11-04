import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




from CrossAttention import CrossAttention
from FeedForward import FeedForward



def Decoder(nn.Module):
    def __init__(self, seq_len):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = seq_len, embedding_dim= 256)
        self.positional = nn.Embedding(num_embeddings = seq_len, embedding_dim= 256)
        self.mask_attention = CrossAttention(num_embed = 256, seq_length = seq_len, num_heads = 3 )
        self.cross_attention = CrossAttention(num_embed = 256, seq_length = seq_len, num_heads = 3 )
        self.ff = FeedForward(256, 256 * 4)

        
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.layerNorm(256)
        self.norm3 = nn.layerNorm(256)

        self.ln = nn.Linear(256, seq_len)



    def forward(self, x, encoder_output, src_mask, tgt_mask):
        seq_len = x.size(1)

        
        arange = torch.arange(0, seq_len, dtype=torch.long, device = x.device)
        positional = self.positional(arange)
        embeddings = self.embedding(x)
        x_processed = embeddings + positional 
        
        X = self.mask_attention.forward(x_processed, x_processed, x_processed, tgt_mask)
        Xn1 = self.norm1(X + x_processed)

        X = self.cross_attention(encoder_output, Xn1, encoder_output, src_mask)

        Xn2 = self.norm2(X + Xn1)

        X = self.ff.forward(Xn2)
        X = self.norm3(X)


        return self.ln(X)




        

        


        
