import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnDecoderLSTM(nn.Module):
    def __init__(self, hidden_size,layers):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,num_layers=layers)

    def forward(self, input,hidden):
        output, hidden = self.lstm(input,hidden)
        return output, hidden
