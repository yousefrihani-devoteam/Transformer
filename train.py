import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader
import tqdm

from Transformer import Transformer 

from data import TextDataPrep



tiny = TextDataPrep("roneneldan/TinyStories", add_eos=True).prepare()


model = Transformer()

ds = tiny.set_format("torch")

train_loader = DataLoader(ds["train"], batch_size = 16, shuffle = True)



for batch in tqdm(train_loader):
    out = model 
