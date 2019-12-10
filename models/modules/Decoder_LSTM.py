import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLSTM(nn.Module):
    def __init__(self, input_size,hidden_size,layers,dropout=0.2):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size,num_layers=layers,dropout=dropout)

    def forward(self, input,hidden):
        output, hidden = self.lstm(input,hidden)
        return output, hidden
