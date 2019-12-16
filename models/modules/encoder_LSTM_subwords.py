import torch.nn as nn
import torch
from torch.autograd import Variable

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size,layers,dropout=0.2):
        super(EncoderLSTM, self).__init__()
        self.num_layers=layers
        self.hidden_size = hidden_size
        self.lin_in = nn.Embedding(vocab_size,hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,num_layers=layers,dropout=dropout,bidirectional=True)

    def forward(self, input):
        input = self.lin_in(input.squeeze(1))
        output, hidden = self.lstm(input.unsqueeze(0))
        return output, hidden
