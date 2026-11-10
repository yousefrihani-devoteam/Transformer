import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader
from tqdm import tqdm

from Transformer import Transformer 
from data import TextDataPrep

prep = TextDataPrep()
ds = prep.prepare()

train_loader = DataLoader(ds["train"], batch_size=32, shuffle=True, collate_fn=prep.collate)
val_loader   = DataLoader(ds["validation"], batch_size=32, collate_fn=prep.collate)

vocab_size = len(prep.tok)



model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=256,
    num_heads=2
)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0004)


for epoch in range(5):

    model.train()
    pbar = tqdm(train_loader, desc=f"epoch {epoch}")

    for batch in tqdm(train_loader):
        x = batch['src'].to(device)
        dec_in = batch['dec_in'].to(device)
        y = batch['labels'].to(device)
    
        y_pred = model(x, dec_in)

        loss = F.cross_entropy(
        y_pred.reshape(-1, y_pred.size(-1)),
        y.reshape(-1),
        ignore_index=-100,
    )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.3f}")



    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            src    = batch['src'].to(device)
            dec_in = batch['dec_in'].to(device)
            y      = batch['labels'].to(device)
            y_pred = model(src, dec_in)
            total += F.cross_entropy(
                y_pred.reshape(-1, y_pred.size(-1)), y.reshape(-1), ignore_index=-100
            ).item()
            count += 1
    print(f"epoch {epoch}  val loss {total / count:.3f}")




