import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        num_heads=8,
        max_seq_len=512,
        dropout=0.1,
        pad_idx=0
    ):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        self.decoder = Decoder(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        seq_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool().unsqueeze(0).unsqueeze(1)
        
        tgt_mask = tgt_pad_mask & causal_mask
        return tgt_mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        enc_output = self.encoder(src, mask=src_mask)
        output = self.decoder(tgt, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return output
