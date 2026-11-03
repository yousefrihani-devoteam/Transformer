<<<<<<< HEAD
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

class FeedForward(nn.Module):
    def __init__(self, embedding_size, expansion_size, dropout = 0.2):
        super(FeedForward, self).__init__()
        self.input = nn.Linear(embedding_size, expansion_size)
        self.Activation = nn.Relu()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(expansion_size, embedding_size)


        self.seq_version = nn.Sequential(
            nn.Linear(embedding_size, expansion_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_size, embedding_size)
        )

    def forward(self, input):
        x = self.input(input)
        x = self.Activation(x)
        x = self.dropout(x)
        output = self.output(x)

        return output

    def forwardSeq(self, input):
        output = self.seq_version(input)
        return output

    

        
=======
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

class FeedForward(nn.Module):
    def __init__(self, embedding_size, expansion_size, dropout = 0.2):
        super(FeedForward, self).__init__()
        self.input = nn.Linear(embedding_size, expansion_size)
        self.Activation = nn.Relu()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(expansion_size, embedding_size)


        self.seq_version = nn.Sequential(
            nn.Linear(embedding_size, expansion_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_size, embedding_size)
        )

    def forward(self, input):
        x = self.input(input)
        x = self.Activation(x)
        x = self.dropout(x)
        output = self.output(x)

        return output

    def forwardSeq(self, input):
        output = self.seq_version(input)
        return output

    

        
>>>>>>> d7f9e6ee15bcf4f821c4d6b0b19bbc3187d02b22
        