import torch.nn as nn
import torch
from torch.autograd import Variable

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,layers,dropout=0.2):
        super(EncoderLSTM, self).__init__()
        self.num_layers=layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,bidirectional=True,num_layers=layers,dropout=dropout)

    def forward(self, input):
        output, hidden = self.lstm(input.float())
        return output, hidden
