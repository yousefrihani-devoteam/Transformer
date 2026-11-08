import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




from CrossAttention import CrossAttention
from FeedForward import FeedForward



def Decoder(nn.Module):
    def __init__(self, Trt_vocab_size, d_model, dropout = 0.1, num_heads = 2):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = Trt_vocab_size, embedding_dim= d_model)
        self.positional = nn.Embedding(num_embeddings = Trt_vocab_size, embedding_dim= d_model)
        self.mask_attention = CrossAttention(num_embed = d_model, seq_length = Trt_vocab_size, num_heads = num_heads )
        self.cross_attention = CrossAttention(num_embed = d_model, seq_length = Trt_vocab_size, num_heads = num_heads )
        self.ff = FeedForward(d_model, d_model * 4)

        self.relu = nn.ReLU()


        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.ln = nn.Linear(d_model, Trt_vocab_size)



    def forward(self, x, encoder_output, src_mask, tgt_mask):
        seq_len = x.size(1)

        
        arange = torch.arange(0, seq_len, dtype=torch.long, device = x.device)
        positional = self.positional(arange)
        embeddings = self.embedding(x)
        x_processed = embeddings + positional 
        
        X = self.mask_attention.forward(x_processed, x_processed, x_processed, tgt_mask)
        Xn1 = self.dropout(self.norm1(X + x_processed))

        X = self.cross_attention(encoder_output, Xn1, encoder_output, src_mask)

        Xn2 = self.norm2(X + Xn1)

        X = self.ff.forward(Xn2)
        X = self.norm3(X)


        return self.ln(X)




        

        


        
